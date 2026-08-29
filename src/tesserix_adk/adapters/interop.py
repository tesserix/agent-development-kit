"""Translate foreign tools at one validated, contextual and attributable boundary.

Translation never trusts a framework's first invocation to discover whether its schema
works. The supported JSON Schema subset is admitted and closed at import time, model
arguments are validated again before dispatch, and caller identity travels beside those
arguments rather than becoming fields a model may choose.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import re
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic import Field, JsonValue, model_validator

from tesserix_adk.core import (
    INLINE_REFS,
    AdkError,
    AdkModel,
    ApprovalPolicy,
    Idempotency,
    IdempotencyPolicy,
    ToolArgumentValidationError,
    schema_for,
)
from tesserix_adk.tools import STRICT, ArgumentPolicy, Tool, ToolContext

__all__ = [
    "ForeignTool",
    "ToolImportPolicy",
    "ToolTranslationError",
    "import_tool",
    "import_toolset",
]

_MAX_SCHEMA_BYTES = 32 * 1024
_MAX_SCHEMA_DEPTH = 16
_ANNOTATIONS = frozenset({"default", "description", "examples", "title"})
_KEYWORDS = frozenset(
    {
        *_ANNOTATIONS,
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "type",
        "uniqueItems",
    }
)
_JSON_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})


class ToolTranslationError(AdkError):
    """A foreign definition that cannot be made safe enough to register.

    Args:
        tool: The foreign tool's name, where it could be read safely.
        source: Framework or descriptor provenance.
        construct: The unsupported or missing construct.
        translated: Tools translated before a toolset failure. They remain available so
            one incompatible definition does not hide the compatible definitions.
        failures: Individual failures collected across a toolset.
    """

    def __init__(
        self,
        message: str,
        *,
        tool: str = "",
        source: str = "",
        construct: str = "",
        translated: tuple[Tool[Any, Any], ...] = (),
        failures: tuple[ToolTranslationError, ...] = (),
    ) -> None:
        self.tool = tool
        self.source = source
        self.construct = construct
        self.translated = translated
        self.failures = failures
        super().__init__(
            message,
            details={"tool": tool, "source": source, "construct": construct},
        )


class ToolImportPolicy(AdkModel):
    """Registry policy attached to every imported tool before it can be called.

    `idempotency` has no default on purpose. A foreign side effect whose repeat behaviour
    nobody declared is refused at translation, where the decision is cheap and reviewable.
    """

    timeout_seconds: float = Field(default=15.0, gt=0, le=300)
    max_concurrency: int = Field(default=1, ge=1, le=64)
    requires_approval: bool = True
    idempotency: Idempotency | None = None
    key_arguments: tuple[str, ...] = ()
    arguments: ArgumentPolicy = STRICT

    @model_validator(mode="after")
    def _keys_belong_to_a_repeatable_declaration(self) -> ToolImportPolicy:
        if self.key_arguments and self.idempotency is None:
            raise ValueError("key_arguments need an idempotency declaration")
        return self


@dataclass(frozen=True, slots=True)
class ForeignTool:
    """One toolset entry, pairing a foreign definition with its implementation."""

    definition: object
    implementation: Callable[..., object] | None = None
    schema: Mapping[str, JsonValue] | None = None
    name: str | None = None
    description: str | None = None
    context_parameter: str | None = None
    provenance: str | None = None


@dataclass(frozen=True, slots=True)
class _Resolved:
    name: str
    description: str
    schema: dict[str, JsonValue]
    returns: dict[str, Any] | None
    implementation: Callable[..., object]
    context_parameter: str
    mode: str
    origin: str


def import_tool(
    source: object,
    *,
    policy: ToolImportPolicy,
    implementation: Callable[..., object] | None = None,
    schema: Mapping[str, JsonValue] | None = None,
    name: str | None = None,
    description: str | None = None,
    context_parameter: str | None = None,
    provenance: str | None = None,
) -> Tool[..., object]:
    """Translate a callable, class tool or OpenAI function descriptor into a kit tool.

    Args:
        source: Foreign callable, class-based tool instance, OpenAI function-calling
            dictionary, or an already translated kit `Tool`.
        policy: Timeout, concurrency, approval and repeat-behaviour declarations.
        implementation: Callable paired with a descriptor that carries no implementation.
        schema: Explicit JSON Schema for a callable. Type hints are used when omitted.
        name: Local name override.
        description: Local description override.
        context_parameter: Explicit caller-context parameter for a plain callable.
        provenance: Stable source override recorded on tool-call spans.

    Returns:
        A tool whose arguments are validated before the foreign body executes.

    Raises:
        ToolTranslationError: If repeat behaviour is undeclared, caller context cannot be
            propagated, the schema is absent or unsupported, or the source has no callable.
    """
    if isinstance(source, Tool):
        return source
    if policy.idempotency is None:
        raise ToolTranslationError(
            "a foreign tool needs an explicit idempotency declaration before it can run",
            tool=name or _safe_name(source),
            source=provenance or _origin(source),
            construct="idempotency",
        )
    resolved = _resolve(
        source,
        implementation=implementation,
        schema=schema,
        name=name,
        description=description,
        context_parameter=context_parameter,
        provenance=provenance,
    )
    admitted = _admitted_schema(resolved.schema, tool=resolved.name, source=resolved.origin)
    validator = _SchemaArguments(admitted, tool=resolved.name, policy=policy.arguments)
    internal_context = _context_name(admitted)

    async def invoke(**arguments: object) -> object:
        context = cast("ToolContext", arguments.pop(internal_context))
        return await _dispatch(resolved, arguments, context)

    return Tool[..., object](
        name=resolved.name,
        description=resolved.description,
        parameters_schema=admitted,
        returns_schema=resolved.returns,
        is_async=True,
        function=invoke,
        validator=validator,
        context_parameter=internal_context,
        context_required=True,
        timeout=policy.timeout_seconds,
        parallel_safe=policy.max_concurrency > 1,
        approval=ApprovalPolicy(required=policy.requires_approval),
        idempotency=IdempotencyPolicy(
            kind=policy.idempotency,
            key_arguments=policy.key_arguments,
        ),
        returns_type=object,
        origin=resolved.origin,
        max_concurrency=policy.max_concurrency,
    )


def import_toolset(
    sources: Iterable[ForeignTool | object],
    *,
    policy: ToolImportPolicy,
    known: Collection[str] = (),
) -> tuple[Tool[..., object], ...]:
    """Translate every compatible entry and collect incompatible definitions.

    Raises:
        ToolTranslationError: After every entry has been checked. Its `translated`
            attribute carries successful translations and `failures` names every rejected
            definition. A duplicate local name fails immediately because registration
            order cannot be allowed to choose which implementation wins.
    """
    translated: list[Tool[..., object]] = []
    failures: list[ToolTranslationError] = []
    occupied = set(known)
    for source in sources:
        entry = source if isinstance(source, ForeignTool) else ForeignTool(source)
        try:
            tool = import_tool(
                entry.definition,
                policy=policy,
                implementation=entry.implementation,
                schema=entry.schema,
                name=entry.name,
                description=entry.description,
                context_parameter=entry.context_parameter,
                provenance=entry.provenance,
            )
        except ToolTranslationError as failure:
            failures.append(failure)
            continue
        if tool.name in occupied:
            raise ToolTranslationError(
                f"{tool.name!r} is already present; import order may not choose its body",
                tool=tool.name,
                source=tool.origin,
                construct="name",
                translated=tuple(translated),
            )
        occupied.add(tool.name)
        translated.append(tool)
    if failures:
        names = ", ".join(failure.tool or "<unnamed>" for failure in failures)
        raise ToolTranslationError(
            f"foreign toolset contains definitions that cannot be translated: {names}",
            construct="toolset",
            translated=tuple(translated),
            failures=tuple(failures),
        )
    return tuple(translated)


def _resolve(
    source: object,
    *,
    implementation: Callable[..., object] | None,
    schema: Mapping[str, JsonValue] | None,
    name: str | None,
    description: str | None,
    context_parameter: str | None,
    provenance: str | None,
) -> _Resolved:
    if isinstance(source, Mapping):
        return _openai(
            source,
            implementation=implementation,
            schema=schema,
            name=name,
            description=description,
            context_parameter=context_parameter,
            provenance=provenance,
        )
    body, mode = _implementation(source, implementation)
    called = name or _safe_name(source)
    origin = provenance or _origin(source)
    parameter = _accepted_context(body, context_parameter, mode, called, origin)
    declared = schema or _schema_of(source, body, parameter, called, origin)
    returns = _returns_schema(source)
    return _Resolved(
        name=called,
        description=description if description is not None else _description(source, called),
        schema=dict(declared),
        returns=returns,
        implementation=body,
        context_parameter=parameter,
        mode=mode,
        origin=origin,
    )


def _openai(
    source: Mapping[object, object],
    *,
    implementation: Callable[..., object] | None,
    schema: Mapping[str, JsonValue] | None,
    name: str | None,
    description: str | None,
    context_parameter: str | None,
    provenance: str | None,
) -> _Resolved:
    function = source.get("function") if source.get("type") == "function" else source
    if not isinstance(function, Mapping):
        raise ToolTranslationError(
            "an OpenAI function descriptor needs a function object",
            construct="function",
            source=provenance or "openai-function",
        )
    called = name or _text(function.get("name"))
    origin = provenance or f"openai-function:{called or '<unnamed>'}"
    if not called:
        raise ToolTranslationError(
            "an OpenAI function descriptor needs a non-empty name",
            source=origin,
            construct="name",
        )
    body = implementation
    if body is None:
        raise ToolTranslationError(
            f"{called!r} is a descriptor with no implementation to dispatch",
            tool=called,
            source=origin,
            construct="implementation",
        )
    parameter = _accepted_context(body, context_parameter, "keywords", called, origin)
    declared = schema or function.get("parameters")
    if not isinstance(declared, Mapping):
        raise ToolTranslationError(
            f"{called!r} has no JSON Schema for its arguments",
            tool=called,
            source=origin,
            construct="schema",
        )
    return _Resolved(
        name=called,
        description=description if description is not None else _text(function.get("description")),
        schema=cast("dict[str, JsonValue]", dict(declared)),
        returns=None,
        implementation=body,
        context_parameter=parameter,
        mode="keywords",
        origin=origin,
    )


def _implementation(
    source: object, supplied: Callable[..., object] | None
) -> tuple[Callable[..., object], str]:
    if supplied is not None:
        return supplied, "keywords"
    for attribute, mode in (
        ("ainvoke", "framework"),
        ("invoke", "framework"),
        ("run", "framework"),
    ):
        candidate = getattr(source, attribute, None)
        if callable(candidate):
            return cast("Callable[..., object]", candidate), mode
    if callable(source):
        return cast("Callable[..., object]", source), "keywords"
    raise ToolTranslationError(
        "a foreign tool definition has no callable implementation",
        tool=_safe_name(source),
        source=_origin(source),
        construct="implementation",
    )


def _accepted_context(
    body: Callable[..., object],
    explicit: str | None,
    mode: str,
    tool: str,
    origin: str,
) -> str:
    signature = inspect.signature(body)
    if mode == "framework":
        if "config" in signature.parameters:
            return "config"
        if "context" in signature.parameters:
            return "context"
    else:
        candidate = explicit or "context"
        parameter = signature.parameters.get(candidate)
        if parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            return candidate
    raise ToolTranslationError(
        f"{tool!r} has no explicit caller-context ingress; tenant authority would be lost",
        tool=tool,
        source=origin,
        construct="context",
    )


def _schema_of(
    source: object,
    body: Callable[..., object],
    context_parameter: str,
    tool: str,
    origin: str,
) -> Mapping[str, JsonValue]:
    declared = getattr(source, "args_schema", None)
    if isinstance(declared, type):
        try:
            return cast("Mapping[str, JsonValue]", schema_for(declared, dialect=INLINE_REFS))
        except Exception as failure:
            raise _translation(
                tool, origin, "schema", "argument model cannot be described"
            ) from failure
    for attribute in ("input_schema", "parameters_schema"):
        candidate = getattr(source, attribute, None)
        if isinstance(candidate, Mapping):
            return cast("Mapping[str, JsonValue]", candidate)
    try:
        return cast(
            "Mapping[str, JsonValue]",
            schema_for(body, dialect=INLINE_REFS, exclude=(context_parameter,)),
        )
    except Exception as failure:
        raise _translation(
            tool,
            origin,
            "schema",
            "it has neither a usable schema nor complete type hints",
        ) from failure


def _returns_schema(source: object) -> dict[str, Any] | None:
    for attribute in ("output_schema", "returns_schema"):
        candidate = getattr(source, attribute, None)
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return None


async def _dispatch(
    resolved: _Resolved, arguments: Mapping[str, object], context: ToolContext
) -> object:
    if resolved.mode == "framework":
        if resolved.context_parameter == "config":
            call = functools.partial(
                resolved.implementation,
                dict(arguments),
                config={"metadata": _context_metadata(context)},
            )
        else:
            call = functools.partial(resolved.implementation, dict(arguments), context=context)
    else:
        call = functools.partial(
            resolved.implementation,
            **dict(arguments),
            **{resolved.context_parameter: context},
        )
    if _async_callable(resolved.implementation):
        produced = call()
    else:
        produced = await asyncio.to_thread(call)
    return await produced if inspect.isawaitable(produced) else produced


def _context_metadata(context: ToolContext) -> dict[str, object]:
    return {
        "run_id": context.run_id,
        "tenant": context.tenant,
        "user": context.user,
        "scopes": context.scopes,
        "trace": dict(context.trace),
    }


def _async_callable(body: Callable[..., object]) -> bool:
    return inspect.iscoroutinefunction(body)


def _admitted_schema(
    schema: Mapping[str, JsonValue], *, tool: str, source: str
) -> dict[str, JsonValue]:
    try:
        encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        admitted = cast("dict[str, JsonValue]", json.loads(encoded))
    except (RecursionError, TypeError, ValueError) as failure:
        raise _translation(tool, source, "schema", "it is not bounded JSON data") from failure
    if len(encoded.encode()) > _MAX_SCHEMA_BYTES:
        raise _translation(tool, source, "schema size", f"it exceeds {_MAX_SCHEMA_BYTES} bytes")
    if admitted.get("type") != "object":
        raise _translation(tool, source, "type", "tool arguments need an object root")
    _admit_node(admitted, tool=tool, source=source, depth=1)
    return admitted


def _admit_node(schema: dict[str, JsonValue], *, tool: str, source: str, depth: int) -> None:
    if depth > _MAX_SCHEMA_DEPTH:
        raise _translation(tool, source, "nesting depth", f"it exceeds {_MAX_SCHEMA_DEPTH}")
    unknown = sorted(set(schema) - _KEYWORDS)
    if unknown:
        raise _translation(tool, source, unknown[0], "the construct is not in the safe subset")
    declared_type = schema.get("type")
    types = [declared_type] if isinstance(declared_type, str) else declared_type
    if types is not None and (
        not isinstance(types, list)
        or not types
        or any(not isinstance(value, str) or value not in _JSON_TYPES for value in types)
    ):
        raise _translation(tool, source, "type", "it names an unknown JSON type")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise _translation(tool, source, "properties", "it is not an object")
        for child in properties.values():
            if not isinstance(child, dict):
                raise _translation(tool, source, "properties", "a property is not a schema")
            _admit_node(child, tool=tool, source=source, depth=depth + 1)
    if declared_type == "object" or properties is not None:
        schema.setdefault("additionalProperties", False)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _admit_node(additional, tool=tool, source=source, depth=depth + 1)
    elif additional not in (None, False):
        raise _translation(
            tool, source, "additionalProperties", "unvalidated wildcard fields are not admitted"
        )
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise _translation(tool, source, "items", "tuple-form array schemas are unsupported")
        _admit_node(items, tool=tool, source=source, depth=depth + 1)
    for composition in ("allOf", "anyOf", "oneOf"):
        alternatives = schema.get(composition)
        if alternatives is None:
            continue
        if not isinstance(alternatives, list) or not alternatives:
            raise _translation(tool, source, composition, "it needs one or more schemas")
        for child in alternatives:
            if not isinstance(child, dict):
                raise _translation(tool, source, composition, "an alternative is not a schema")
            _admit_node(child, tool=tool, source=source, depth=depth + 1)


@dataclass(frozen=True, slots=True)
class _SchemaArguments:
    schema: dict[str, JsonValue]
    tool: str
    policy: ArgumentPolicy = STRICT

    def arguments(self, arguments: object) -> dict[str, JsonValue]:
        """Parse and validate one argument object without coercion."""
        parsed = self._parsed(arguments)
        problems: dict[str, str] = {}
        _validate(self.schema, parsed, "arguments", problems)
        if problems:
            raise ToolArgumentValidationError(
                f"{self.tool} was called with arguments that do not match its imported schema",
                tool=self.tool,
                paths=tuple(sorted(problems)),
                problems=problems,
                payload=arguments,
            )
        return parsed

    def _parsed(self, arguments: object) -> dict[str, JsonValue]:
        if isinstance(arguments, str | bytes):
            raw = arguments
            size = len(raw)
        else:
            try:
                raw = json.dumps(
                    arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
            except (TypeError, ValueError) as failure:
                raise self._refusal("arguments are not JSON data", arguments) from failure
            size = len(raw.encode())
        if size > self.policy.max_bytes:
            raise self._refusal(
                f"arguments exceed the {self.policy.max_bytes}-byte ceiling", arguments
            )
        try:
            parsed = json.loads(raw, object_pairs_hook=self._unique(arguments), parse_constant=_nan)
        except (json.JSONDecodeError, ValueError) as failure:
            raise self._refusal("arguments are not valid JSON", arguments) from failure
        if not isinstance(parsed, dict):
            raise self._refusal("arguments are not a JSON object", arguments)
        return cast("dict[str, JsonValue]", parsed)

    def _unique(self, arguments: object) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
        def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            names = [key for key, _ in pairs]
            if len(names) != len(set(names)):
                raise self._refusal("arguments repeat a JSON key", arguments)
            return dict(pairs)

        return hook

    def _refusal(self, problem: str, arguments: object) -> ToolArgumentValidationError:
        return ToolArgumentValidationError(
            f"{self.tool} was called with invalid arguments: {problem}",
            tool=self.tool,
            payload=arguments,
        )


def _validate(
    schema: Mapping[str, JsonValue], value: JsonValue, path: str, problems: dict[str, str]
) -> None:
    alternatives = [
        (name, schema.get(name)) for name in ("allOf", "anyOf", "oneOf") if name in schema
    ]
    for name, candidates in alternatives:
        assert isinstance(candidates, list)  # noqa: S101 — admitted before invocation
        matches = sum(
            _matches(cast("Mapping[str, JsonValue]", child), value) for child in candidates
        )
        if (name == "allOf" and matches != len(candidates)) or (name == "anyOf" and not matches):
            problems[path] = f"does not satisfy {name}"
        if name == "oneOf" and matches != 1:
            problems[path] = "does not satisfy exactly one oneOf branch"
    declared = schema.get("type")
    types = (
        (declared,)
        if isinstance(declared, str)
        else tuple(item for item in declared if isinstance(item, str))
        if isinstance(declared, list)
        else ()
    )
    if types and not any(_is_type(value, kind) for kind in types):
        problems[path] = f"expected {' or '.join(types)}"
        return
    if "const" in schema and not _same_json(value, schema["const"]):
        problems[path] = "does not equal const"
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(_same_json(value, choice) for choice in enum):
        problems[path] = "is not one of enum"
    if isinstance(value, dict):
        _validate_object(schema, value, path, problems)
    elif isinstance(value, list):
        _validate_array(schema, value, path, problems)
    elif isinstance(value, str):
        _validate_string(schema, value, path, problems)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        _validate_number(schema, value, path, problems)


def _validate_object(
    schema: Mapping[str, JsonValue],
    value: dict[str, JsonValue],
    path: str,
    problems: dict[str, str],
) -> None:
    required = schema.get("required", [])
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                problems[f"{path}.{name}"] = "is required"
    properties = schema.get("properties", {})
    typed = cast("dict[str, Mapping[str, JsonValue]]", properties)
    for name, child in typed.items():
        if name in value:
            _validate(child, value[name], f"{path}.{name}", problems)
    extras = set(value) - set(typed)
    additional = schema.get("additionalProperties", False)
    for name in sorted(extras):
        if additional is False:
            problems[f"{path}.{name}"] = "is not declared"
        elif isinstance(additional, dict):
            _validate(additional, value[name], f"{path}.{name}", problems)
    _range(schema, len(value), path, problems, "minProperties", "maxProperties")


def _validate_array(
    schema: Mapping[str, JsonValue],
    value: list[JsonValue],
    path: str,
    problems: dict[str, str],
) -> None:
    items = schema.get("items")
    if isinstance(items, dict):
        for index, child in enumerate(value):
            _validate(items, child, f"{path}.{index}", problems)
    if schema.get("uniqueItems") is True:
        rendered = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
        if len(rendered) != len(set(rendered)):
            problems[path] = "items are not unique"
    _range(schema, len(value), path, problems, "minItems", "maxItems")


def _validate_string(
    schema: Mapping[str, JsonValue], value: str, path: str, problems: dict[str, str]
) -> None:
    _range(schema, len(value), path, problems, "minLength", "maxLength")
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            matches = re.search(pattern, value) is not None
        except re.error:
            matches = False
        if not matches:
            problems[path] = "does not match pattern"


def _validate_number(
    schema: Mapping[str, JsonValue], value: int | float, path: str, problems: dict[str, str]
) -> None:
    checks: tuple[tuple[str, Callable[[int | float], bool]], ...] = (
        ("minimum", lambda limit: value >= limit),
        ("maximum", lambda limit: value <= limit),
        ("exclusiveMinimum", lambda limit: value > limit),
        ("exclusiveMaximum", lambda limit: value < limit),
    )
    for name, check in checks:
        limit = schema.get(name)
        if isinstance(limit, int | float) and not check(limit):
            problems[path] = f"does not satisfy {name}"


def _range(
    schema: Mapping[str, JsonValue],
    value: int,
    path: str,
    problems: dict[str, str],
    minimum: str,
    maximum: str,
) -> None:
    low, high = schema.get(minimum), schema.get(maximum)
    if isinstance(low, int) and value < low:
        problems[path] = f"is below {minimum}"
    if isinstance(high, int) and value > high:
        problems[path] = f"is above {maximum}"


def _matches(schema: Mapping[str, JsonValue], value: JsonValue) -> int:
    found: dict[str, str] = {}
    _validate(schema, value, "value", found)
    return int(not found)


def _is_type(value: JsonValue, kind: str) -> bool:
    return {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
    }[kind]


def _same_json(left: JsonValue, right: JsonValue) -> bool:
    return type(left) is type(right) and left == right


def _context_name(schema: Mapping[str, JsonValue]) -> str:
    properties = schema.get("properties")
    taken = set(properties) if isinstance(properties, dict) else set()
    name = "_tesserix_context"
    while name in taken:
        name = f"_{name}"
    return name


def _description(source: object, fallback: str) -> str:
    described = getattr(source, "description", None)
    if isinstance(described, str):
        return described
    doc = inspect.getdoc(source)
    return doc.splitlines()[0] if doc else fallback


def _safe_name(source: object) -> str:
    named = getattr(source, "name", None) or getattr(source, "__name__", None)
    return named if isinstance(named, str) else ""


def _origin(source: object) -> str:
    module = getattr(source, "__module__", type(source).__module__)
    qualified = getattr(source, "__qualname__", type(source).__qualname__)
    return f"python:{module}.{qualified}"


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _translation(tool: str, source: str, construct: str, why: str) -> ToolTranslationError:
    return ToolTranslationError(
        f"{tool or '<unnamed>'!r} cannot be imported: {why} ({construct})",
        tool=tool,
        source=source,
        construct=construct,
    )


def _nan(value: str) -> None:
    if value in {"NaN", "Infinity", "-Infinity"}:
        raise ValueError("non-finite JSON number")
