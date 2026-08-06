"""JSON Schema derived from the Python type, so the two can never disagree.

A schema written by hand is a second declaration of the same shape, and the second one is
the one nobody updates. Here the annotation is the only declaration: `schema_for` reads the
type and its docstring, and the result is both what the model is told and what the code
parses.

Providers differ in what they will accept — some close every object, some refuse `$ref`
entirely — so the dialect is a parameter rather than a caller's post-processing step. A
dialect that cannot express the type says so before the request goes out, because a schema
quietly rewritten into something looser is a schema the model can satisfy and the code
cannot.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
from typing import TYPE_CHECKING, Annotated, Any, Protocol, get_args, get_origin, get_type_hints

from pydantic import BaseModel, TypeAdapter, create_model
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError

from tesserix_adk.core.errors import CapabilityError, SchemaGenerationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "INLINE_REFS",
    "JSON_SCHEMA",
    "STRICT_SUBSET",
    "InlineRefs",
    "JsonSchema",
    "SchemaDialect",
    "StrictSubset",
    "schema_for",
    "schema_hash",
]

_REF_TEMPLATE = "#/$defs/{model}"
_ANNOTATIONS = frozenset({"default", "deprecated", "description", "examples", "readOnly"})
_SECTION = re.compile(r"^[A-Z][A-Za-z ]*:$")
_ARGUMENT = re.compile(r"^(\w+):\s*(.*)$")


class SchemaDialect(Protocol):
    """What one provider will accept, as a transformation and a list of refusals.

    A dialect is a value, applied at definition time. Its name appears in any failure it
    causes, so a refused schema says which provider refused it.
    """

    @property
    def name(self) -> str:
        """The dialect's name, as it appears in errors and in the schema hash."""
        ...

    @property
    def forbidden(self) -> frozenset[str]:
        """Schema keywords this provider will not accept, checked after `adapt`."""
        ...

    def adapt(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Rewrite a normalised schema into this provider's dialect."""
        ...


@dataclasses.dataclass(frozen=True)
class JsonSchema:
    """Draft 2020-12 as pydantic emits it, normalised. The default."""

    name: str = "json-schema"
    forbidden: frozenset[str] = frozenset()

    def adapt(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Nothing to do: normalisation has already happened."""
        return schema


@dataclasses.dataclass(frozen=True)
class StrictSubset:
    """Every object closed, for providers whose structured-output mode demands it.

    `additionalProperties: false` and nothing else: widening `required` to every field, as
    some provider guides suggest, would make the schema refuse payloads the type accepts.
    """

    name: str = "strict-subset"
    forbidden: frozenset[str] = frozenset()

    def adapt(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Close every object in the schema, including those under `$defs`."""
        return dict(_closed(schema))


@dataclasses.dataclass(frozen=True)
class InlineRefs:
    """No `$defs` and no `$ref` — every definition substituted where it is used.

    Args:
        max_depth: How many substitutions deep to go. A type that needs more, or that
            refers to itself, raises `CapabilityError` rather than being cut short: a
            truncated schema describes a shape the code will not accept.
    """

    max_depth: int = 8
    name: str = "inline-refs"
    forbidden: frozenset[str] = frozenset({"$ref"})

    def adapt(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Substitute every reference, leaving no `$defs` behind."""
        definitions = schema.get("$defs", {})
        body = {key: value for key, value in schema.items() if key != "$defs"}
        inlined = _inlined(body, definitions, (), self)
        return dict(inlined)


JSON_SCHEMA = JsonSchema()
STRICT_SUBSET = StrictSubset()
INLINE_REFS = InlineRefs()


def schema_for(
    target: type | Callable[..., Any],
    *,
    dialect: SchemaDialect = JSON_SCHEMA,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Derive a provider-ready JSON Schema from a type or a callable's signature.

    Accepts a pydantic model, a dataclass, a `TypedDict` or an annotated callable. Field
    descriptions come from `Field(description=...)` where there is one and from the
    Google-style `Args:` block of the docstring otherwise; a missing or malformed
    docstring costs descriptions, never the schema.

    Args:
        target: The type or callable to describe.
        dialect: The provider dialect to emit. Defaults to plain JSON Schema.
        max_bytes: The provider's schema size limit, where it has one.

    Raises:
        SchemaGenerationError: The type cannot be faithfully described — an unannotated
            parameter, `Any` in a required position, a type pydantic cannot render, or a
            schema past `max_bytes`. Naming the field is the point: the kit never emits a
            permissive placeholder that would let the model answer with something the code
            cannot validate.
        CapabilityError: The dialect forbids a construct this type needs, or inlining it
            would not terminate.
    """
    source = _source(target)
    described = _documented(source.schema, source)
    schema = dialect.adapt(_normalised(described))
    _refuse_permissive(schema, ())
    _refuse_forbidden(schema, dialect, ())
    _refuse_oversized(schema, max_bytes, dialect)
    return schema


def schema_hash(schema: Mapping[str, Any]) -> str:
    """A stable digest of a schema, for prompt versions and run fingerprints.

    Key order does not change it and any change to the shape does, so a renamed field
    misses a cassette recorded against the old shape instead of replaying it.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=repr)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclasses.dataclass(frozen=True)
class _Source:
    """One type, however it was declared, reduced to what the generator needs."""

    name: str
    doc: str | None
    hints: dict[str, Any]
    schema: dict[str, Any]


def _source(target: type | Callable[..., Any]) -> _Source:
    if isinstance(target, type) and issubclass(target, BaseModel):
        hints = {name: info.annotation for name, info in target.model_fields.items()}
        return _Source(target.__name__, target.__doc__, hints, _generated(target, hints))
    if isinstance(target, type) or _is_typed_dict(target):
        declared = get_type_hints(target, include_extras=True)
        return _Source(_named(target), target.__doc__, declared, _generated(target, declared))
    if callable(target):
        return _from_signature(target)
    raise SchemaGenerationError(
        f"a {type(target).__name__} is neither a type nor a callable, so there is nothing "
        f"to derive a schema from.",
        field="",
        annotation=type(target).__name__,
    )


def _generated(target: Any, hints: Mapping[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """Ask pydantic for the schema, and turn its one failure mode into a located one."""
    try:
        adapter: TypeAdapter[Any] = TypeAdapter(target)
        return adapter.json_schema(ref_template=_REF_TEMPLATE)
    except (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema) as failure:
        raise _undescribable(_named(target), hints, failure) from failure


def _from_signature(target: Callable[..., Any]) -> _Source:
    signature = inspect.signature(target)
    hints = get_type_hints(target, include_extras=True)
    fields: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise SchemaGenerationError(
                f"{_named(target)}.{name} is variadic, and no schema can say what a caller "
                f"may put in it. Declare the parameters the tool actually takes.",
                field=name,
                annotation="*args/**kwargs",
            )
        if name not in hints:
            raise SchemaGenerationError(
                f"{_named(target)}.{name} has no annotation, so the model would be told "
                f"nothing about what to send. Annotate it.",
                field=name,
                annotation="",
            )
        default = ... if parameter.default is parameter.empty else parameter.default
        fields[name] = (hints[name], default)
    declared = {name: annotation for name, annotation in hints.items() if name != "return"}
    model = create_model(_named(target), **fields)
    return _Source(_named(target), target.__doc__, declared, _generated(model, declared))


def _is_typed_dict(target: object) -> bool:
    return hasattr(target, "__required_keys__")


def _named(target: object) -> str:
    return str(getattr(target, "__name__", type(target).__name__))


def _undescribable(
    name: str, hints: Mapping[str, Any], failure: Exception
) -> SchemaGenerationError:
    """Name the field pydantic choked on: its own error names only the type."""
    text = str(failure)
    field = next(
        (field for field, annotation in hints.items() if _annotation(annotation) in text), ""
    )
    return SchemaGenerationError(
        f"{name}.{field} is annotated with a type that has no JSON Schema representation. "
        f"Model it as something a provider can describe — the kit will not emit a "
        f"placeholder that accepts anything. ({text})",
        field=field,
        annotation=text,
    )


def _annotation(annotation: object) -> str:
    return str(getattr(annotation, "__name__", annotation))


def _documented(schema: dict[str, Any], source: _Source) -> dict[str, Any]:
    """Fill in descriptions from every docstring in the type graph."""
    documented = _described(schema, source.doc, _docstring_arguments(source.doc))
    definitions = documented.get("$defs")
    if definitions:
        reachable = _reachable(source.hints)
        documented["$defs"] = {
            name: _defined(definition, reachable.get(name))
            for name, definition in definitions.items()
        }
    return documented


def _defined(definition: dict[str, Any], source: type | None) -> dict[str, Any]:
    if source is None:
        return definition
    return _described(definition, source.__doc__, _docstring_arguments(source.__doc__))


def _described(
    schema: dict[str, Any], doc: str | None, documented: Mapping[str, str]
) -> dict[str, Any]:
    """The summary line describes the type; the `Args:` block describes its fields.

    The whole docstring is never the description: pydantic uses it, and an `Args:` block
    repeated inside a field-by-field schema is tokens spent saying it twice.
    """
    described = dict(schema)
    summary = _summary(doc)
    if summary:
        described["description"] = summary
    else:
        described.pop("description", None)
    properties = described.get("properties")
    if properties:
        described["properties"] = {
            name: (
                property_schema
                if "description" in property_schema or name not in documented
                else property_schema | {"description": documented[name]}
            )
            for name, property_schema in properties.items()
        }
    return described


def _reachable(hints: Mapping[str, Any], seen: set[type] | None = None) -> dict[str, type]:
    """Every type reachable from these annotations, keyed by the name pydantic gives its `$def`."""
    seen = set() if seen is None else seen
    found: dict[str, type] = {}
    for annotation in hints.values():
        for candidate in _types_in(annotation):
            if candidate in seen:
                continue
            seen.add(candidate)
            found[candidate.__name__] = candidate
            found.update(_reachable(_fields_of(candidate), seen))
    return found


def _fields_of(candidate: type) -> dict[str, Any]:
    if issubclass(candidate, BaseModel):
        return {name: info.annotation for name, info in candidate.model_fields.items()}
    if dataclasses.is_dataclass(candidate) or _is_typed_dict(candidate):
        return dict(get_type_hints(candidate, include_extras=True))
    return {}


def _types_in(annotation: object) -> list[type]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _types_in(get_args(annotation)[0])
    if origin is not None:
        return [found for argument in get_args(annotation) for found in _types_in(argument)]
    return [annotation] if isinstance(annotation, type) else []


def _summary(doc: str | None) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).splitlines()[0].strip()


def _docstring_arguments(doc: str | None) -> dict[str, str]:
    """Read a Google-style `Args:` block. What it cannot parse it skips, rather than raising."""
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()
    if "Args:" not in lines:
        return {}
    found: dict[str, str] = {}
    current = ""
    for line in lines[lines.index("Args:") + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION.match(stripped):
            break
        match = _ARGUMENT.match(stripped)
        if match:
            current = match.group(1)
            found[current] = match.group(2).strip()
        elif current:
            found[current] = f"{found[current]} {stripped}".strip()
    return {name: text for name, text in found.items() if text}


def _normalised(schema: Any) -> Any:  # noqa: ANN401 — any JSON value, recursively
    """Drop titles and order keys, so two generations of one type are the same bytes."""
    if isinstance(schema, dict):
        return {
            key: _normalised(sorted(value) if key == "required" else value)
            for key, value in sorted(schema.items())
            if key != "title"
        }
    if isinstance(schema, list):
        return [_normalised(item) for item in schema]
    return schema


def _closed(schema: Any) -> Any:  # noqa: ANN401 — any JSON value, recursively
    if isinstance(schema, dict):
        closed = {key: _closed(value) for key, value in schema.items()}
        if closed.get("type") == "object":
            closed["additionalProperties"] = False
        return closed
    if isinstance(schema, list):
        return [_closed(item) for item in schema]
    return schema


def _inlined(
    schema: Any,  # noqa: ANN401 — any JSON value, recursively
    definitions: Mapping[str, Any],
    stack: tuple[str, ...],
    dialect: InlineRefs,
) -> Any:  # noqa: ANN401
    if isinstance(schema, list):
        return [_inlined(item, definitions, stack, dialect) for item in schema]
    if not isinstance(schema, dict):
        return schema
    reference = schema.get("$ref")
    if reference is None:
        return {key: _inlined(value, definitions, stack, dialect) for key, value in schema.items()}
    name = str(reference).rsplit("/", 1)[-1]
    if name in stack:
        raise CapabilityError(
            f"{name} refers to itself and {dialect.name} has nowhere to put a reference. "
            f"Inlining a cycle does not terminate, and a schema cut short describes a "
            f"shape the code will not accept.",
            details={"dialect": dialect.name, "type": name},
        )
    if len(stack) >= dialect.max_depth:
        raise CapabilityError(
            f"{name} sits deeper than {dialect.name} will inline "
            f"(max_depth={dialect.max_depth}). Raise the depth, or keep references.",
            details={"dialect": dialect.name, "type": name, "depth": str(len(stack))},
        )
    merged = dict(definitions[name]) | {
        key: value for key, value in schema.items() if key != "$ref"
    }
    return _inlined(merged, definitions, (*stack, name), dialect)


def _refuse_permissive(schema: Any, path: tuple[str, ...]) -> None:  # noqa: ANN401
    """An empty property schema is `Any`: the model may answer with anything at all."""
    if isinstance(schema, list):
        for index, item in enumerate(schema):
            _refuse_permissive(item, (*path, str(index)))
        return
    if not isinstance(schema, dict):
        return
    required = schema.get("required", [])
    for name, property_schema in schema.get("properties", {}).items():
        if name in required and _accepts_anything(property_schema):
            raise SchemaGenerationError(
                f"{'.'.join((*path, name))} is annotated `Any`, which describes nothing: a "
                f"model told this may answer with anything, and the code reading it can "
                f"validate none of it. Give the field a type.",
                field=name,
                annotation="Any",
            )
    for key, value in schema.items():
        _refuse_permissive(value, path if key == "properties" else (*path, key))


def _accepts_anything(schema: object) -> bool:
    """A description is not a constraint, so documenting an `Any` field does not narrow it."""
    if schema is True:
        return True
    if not isinstance(schema, dict):
        return False
    return not set(schema) - _ANNOTATIONS


def _refuse_forbidden(schema: Any, dialect: SchemaDialect, path: tuple[str, ...]) -> None:  # noqa: ANN401
    if isinstance(schema, list):
        for index, item in enumerate(schema):
            _refuse_forbidden(item, dialect, (*path, str(index)))
        return
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        if key in dialect.forbidden:
            raise CapabilityError(
                f"{dialect.name} does not accept `{key}`, which {'.'.join((*path, key))} "
                f"needs. The kit will not rewrite it into something looser that the "
                f"provider would take.",
                details={"dialect": dialect.name, "keyword": key, "path": ".".join(path)},
            )
        _refuse_forbidden(value, dialect, (*path, key))


def _refuse_oversized(
    schema: dict[str, Any], max_bytes: int | None, dialect: SchemaDialect
) -> None:
    if max_bytes is None:
        return
    size = len(json.dumps(schema, separators=(",", ":")).encode())
    if size > max_bytes:
        raise SchemaGenerationError(
            f"the schema is {size} bytes and {dialect.name} allows {max_bytes}. Truncating "
            f"it would describe a different type, so it is refused: model fewer fields, or "
            f"split the tool.",
            field="",
            annotation=f"{size} bytes",
        )
