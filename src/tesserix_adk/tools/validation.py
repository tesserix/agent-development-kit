"""What a model sent is checked against the tool's own signature before the body runs.

Tool arguments are model output, and model output is untrusted input. A payload splatted
straight into a Python function fails deep inside it where the traceback says nothing
about the call that caused it — or does not fail, and reaches a query with a field the
model invented. The schema the model was shown is derived from the signature, so the same
signature is what the call is held to: unknown fields are refused rather than dropped, no
absent field is filled in, and nothing is coerced across a type the model could have got
right.

Strict is the default because the alternative is provider-shaped behaviour: one vendor
sends `2`, another `"2"`, and a kit that quietly reads both makes the tool's contract
depend on which vendor answered. A registry that wants the coercions can say so once.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from tesserix_adk.core.errors import ToolArgumentValidationError, ToolDefinitionError
from tesserix_adk.core.schema import annotations_of

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

__all__ = ["LENIENT", "STRICT", "ArgumentPolicy", "ArgumentValidator", "ToolArgumentValidator"]

# A provider that wraps the arguments in one of these is saying the same thing twice.
_ENVELOPES = frozenset({"arguments", "args", "input", "parameters"})


@dataclass(frozen=True, slots=True)
class ArgumentPolicy:
    """How strictly a tool's arguments are read.

    Args:
        strict: Whether a value has to already be the declared type. Off, the documented
            JSON coercions apply — `"2"` for an integer, `"yes"` for a boolean — which is
            a consumer's decision to make once, not a per-provider accident.
        max_bytes: The ceiling on the raw payload, checked before it is parsed. A payload
            nobody bounded is a context window, and then a heap, spent by the model.
    """

    strict: bool = True
    max_bytes: int = 64 * 1024


STRICT = ArgumentPolicy()
LENIENT = ArgumentPolicy(strict=False)


class ArgumentValidator(Protocol):
    """What a Tool needs in order to hold model arguments to its advertised schema."""

    def arguments(self, arguments: object) -> Mapping[str, object]:
        """Return validated keyword arguments, or raise a typed refusal."""


class ToolArgumentValidator:
    """Turns whatever a provider sent into the tool's own argument types, or refuses it.

    Args:
        function: The tool's undecorated function. Its signature is the contract.
        tool: What the model calls it, used in the refusal. Defaults to the function name.
        exclude: Parameters the caller fills, which are not the model's to send — the
            injected `ToolContext` above all.
        policy: How strictly to read what arrived.

    Raises:
        ToolDefinitionError: If a parameter carries no annotation, so there is no type to
            hold the model to.
    """

    __slots__ = ("_model", "_policy", "_tool")

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        tool: str = "",
        exclude: Collection[str] = (),
        policy: ArgumentPolicy = STRICT,
    ) -> None:
        self._tool = tool or getattr(function, "__name__", "tool")
        self._policy = policy
        self._model = _model_for(function, self._tool, exclude)

    @property
    def tool(self) -> str:
        """The name the refusal is reported under."""
        return self._tool

    @property
    def model(self) -> type[BaseModel]:
        """The argument model built from the signature."""
        return self._model

    @property
    def policy(self) -> ArgumentPolicy:
        """How strictly this validator reads a payload."""
        return self._policy

    def validate(self, arguments: object) -> Any:  # noqa: ANN401 — an instance of `model`
        """Read a payload into the argument model, or refuse it.

        Args:
            arguments: What the provider sent: a mapping, or the JSON text some providers
                send instead, wrapped in a redundant envelope or not.

        Returns:
            An instance of `model`, with every field the tool's declared type.

        Raises:
            ToolArgumentValidationError: If the payload is not a JSON object, is over the
                ceiling, repeats a key, or does not match the tool's schema.
        """
        text = self._canonical(arguments)
        try:
            return self._model.model_validate_json(text, strict=self._policy.strict)
        except ValidationError as mismatch:
            raise self._refusal(
                f"{self._tool} was called with arguments that do not match its schema",
                arguments,
                problems=_problems(mismatch),
            ) from mismatch

    def arguments(self, arguments: object) -> dict[str, Any]:
        """The validated fields as keywords for the function, each one its declared type."""
        validated = self.validate(arguments)
        return {name: getattr(validated, name) for name in type(validated).model_fields}

    def _canonical(self, arguments: object) -> str:
        """One JSON object however the provider chose to send it, or a refusal."""
        parsed = self._unwrapped(self._parsed(arguments), arguments)
        if not isinstance(parsed, dict):
            raise self._refusal(
                f"{self._tool} was sent a JSON {type(parsed).__name__}, and a tool call's "
                f"arguments are a JSON object naming each one",
                arguments,
            )
        return json.dumps(parsed)

    def _parsed(self, arguments: object) -> object:
        """Whatever arrived, as Python, with the ceiling held before anything is read."""
        if isinstance(arguments, str | bytes):
            self._within_the_ceiling(len(arguments), arguments)
            return self._loaded(arguments, arguments)
        encoded = self._encoded(arguments)
        self._within_the_ceiling(len(encoded), arguments)
        return json.loads(encoded)

    def _unwrapped(self, parsed: object, arguments: object) -> object:
        """A single-key envelope no parameter answers to is the payload said twice."""
        if not isinstance(parsed, dict) or len(parsed) != 1:
            return parsed
        [(key, value)] = parsed.items()
        if key not in _ENVELOPES or key in self._model.model_fields:
            return parsed
        return self._loaded(value, arguments) if isinstance(value, str | bytes) else value

    def _encoded(self, arguments: object) -> str:
        """A mapping a provider already parsed, back to the one form everything else uses."""
        try:
            return json.dumps(arguments, default=_jsonable)
        except (TypeError, ValueError) as unserialisable:
            raise self._refusal(
                f"{self._tool} was sent arguments that are not JSON: a tool call carries "
                f"what a model can write ({unserialisable})",
                arguments,
            ) from unserialisable

    def _loaded(self, text: str | bytes, arguments: object) -> object:
        """Parse JSON text, refusing a repeated key rather than keeping whichever won."""
        try:
            return json.loads(text, object_pairs_hook=self._one_key_each(arguments))
        except json.JSONDecodeError as unparseable:
            raise self._refusal(
                f"{self._tool} was sent arguments that are not valid JSON ({unparseable})",
                arguments,
            ) from unparseable

    def _one_key_each(self, arguments: object) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
        """A hook that refuses a duplicated key: which one wins is the parser's opinion."""

        def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            seen = {key for key, _ in pairs}
            if len(seen) != len(pairs):
                repeated = sorted({key for key, _ in pairs if [k for k, _ in pairs].count(key) > 1})
                raise self._refusal(
                    f"{self._tool} was sent duplicate keys ({', '.join(repeated)}), and which "
                    f"value that means is the parser's opinion rather than the model's",
                    arguments,
                    problems=dict.fromkeys(repeated, "sent more than once"),
                )
            return dict(pairs)

        return hook

    def _within_the_ceiling(self, size: int, arguments: object) -> None:
        """Refuse an oversized payload before parsing it, which is what parsing would cost."""
        if size > self._policy.max_bytes:
            raise self._refusal(
                f"{self._tool} was sent {size} bytes of arguments, over the ceiling of "
                f"{self._policy.max_bytes}",
                arguments,
            )

    def _refusal(
        self, message: str, arguments: object, problems: Mapping[str, str] | None = None
    ) -> ToolArgumentValidationError:
        """One typed refusal, naming fields and never repeating the values they held."""
        return ToolArgumentValidationError(
            message,
            tool=self._tool,
            paths=tuple(sorted(problems or ())),
            problems=problems,
            payload=arguments,
        )


def _model_for(
    function: Callable[..., Any], called: str, exclude: Collection[str]
) -> type[BaseModel]:
    """The arguments as a model: one field per parameter the model is allowed to fill."""
    hints = annotations_of(function)
    fields: dict[str, Any] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in exclude or parameter.kind in _VARIADIC:
            continue
        if name not in hints:
            raise ToolDefinitionError(
                f"{called} takes {name} without an annotation, so there is no type to hold "
                f"the model's argument to.",
                tool=called,
                parameter=name,
            )
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (hints[name], default)
    return create_model(f"{called}_arguments", __config__=ConfigDict(extra="forbid"), **fields)


_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _problems(mismatch: ValidationError) -> dict[str, str]:
    """Every field that failed with what was wrong, never with what it held."""
    return {
        ".".join(str(part) for part in error["loc"]) or "arguments": error["msg"]
        for error in mismatch.errors()
    }


def _jsonable(value: object) -> Any:  # noqa: ANN401 — whatever a caller passed
    """A typed instance a caller passed by hand, back to the JSON a model would have sent."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"a {type(value).__name__} is not something a model can send")
