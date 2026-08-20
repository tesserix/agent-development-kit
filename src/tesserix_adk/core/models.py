"""The base every boundary model shares, and what a violation of one says.

Data crosses an agent's boundaries as provider payloads, tool arguments, config blocks and
records read back from a store. Where those stay loose dicts, the first call site to touch
a field decides what it means, and a typo becomes an AttributeError deep inside a tool
body. One strict base moves the failure to the crossing and names the field.

The strictness is deliberate on three counts: unknown fields are refused, because a
misspelt setting that is silently ignored is a setting that never took effect; values are
not coerced, because `"0"` read as `0` and `"false"` read as `True` are the same bug; and
models are frozen, because a record the next layer can edit is a record nobody can trust.

Strings are the exception the environment forces — see `parsed_from_strings`. The policy
these types implement is in `docs/models.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter, ValidationError
from typing_extensions import TypeVar

from tesserix_adk.core.errors import SchemaViolationError

__all__ = [
    "AdkModel",
    "NoOutput",
    "OutputT",
    "Sensitive",
    "parsed_from_strings",
    "telemetry_dump",
    "validated",
]

MASKED = "**********"


@dataclass(frozen=True)
class Sensitive:
    """Marks a field that must not reach telemetry.

    Applied as `Annotated[str, Sensitive()]`. The field still serialises into a
    checkpoint — a run rehydrated without its credentials rehydrates broken — but
    `telemetry_dump` drops it, and telemetry is the copy that leaves the process.
    """

    reason: str = ""


class AdkModel(BaseModel):
    """Frozen, strict, and closed to fields it does not declare.

    Every model the kit puts on a boundary derives from this, so strictness is a property
    of the kit rather than something each new model has to remember.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class NoOutput(AdkModel):
    """The answer type of an agent that answers in prose: there is not one.

    It is a type rather than `None` so that `Agent` and `Run` have one type parameter with
    one bound. A free-text run carries `output=None` typed as `NoOutput | None`, which a
    checker refuses to hand to anything expecting a real answer type.
    """


# PEP 696 defaults, so an agent that declares no answer type needs no annotation and every
# existing bare `Run` annotation still reads. Inline syntax for this is 3.13; the floor is
# 3.12, so it comes from typing_extensions until that moves. See docs/typing.md.
OutputT = TypeVar("OutputT", bound=BaseModel, default=NoOutput)


def validated[Model: BaseModel](model: type[Model], payload: object) -> Model:
    """Validate `payload` as `model`, or raise naming every field that failed.

    Pydantic's own error is precise and unreadable at a glance, and it is raised in the
    vocabulary of the library rather than the kit. This normalises it: one error, the
    model's name, every failing path at once, and the payload kept for a debugger.

    Raises:
        SchemaViolationError: If the payload does not validate. Reporting one field at a
            time is how a five-field config takes five deploys to fix.
    """
    try:
        return model.model_validate(payload)
    except ValidationError as failure:
        raise _violation(model.__name__, failure, payload) from failure


def parsed_from_strings[Value](annotation: type[Value] | Any, value: object) -> Value:  # noqa: ANN401
    """Parse `value` as `annotation`, accepting a string where the source only has strings.

    An environment variable is a string whatever it means, so the strictness that protects
    every model inside the kit has to give way exactly once, at that edge, explicitly.

    Raises:
        SchemaViolationError: If the string is not that type after all.
    """
    adapter: TypeAdapter[Value] = TypeAdapter(annotation)
    try:
        if isinstance(value, str):
            return _from_text(adapter, value)
        return adapter.validate_python(value)
    except ValidationError as failure:
        raise _violation(_named(annotation), failure, value) from failure


def _from_text[Value](adapter: TypeAdapter[Value], text: str) -> Value:
    """Read one environment string, which is JSON wherever the field is not a scalar."""
    try:
        return adapter.validate_strings(text)
    except ValidationError:
        return adapter.validate_json(text)


def telemetry_dump(model: BaseModel) -> dict[str, Any]:
    """Dump `model` for a span, a metric or a log line, without its sensitive fields.

    Field order follows the declaration, so two dumps of one model diff cleanly, and
    secrets are masked rather than printed: telemetry is the copy of a record that leaves
    the process, and it leaves under a retention policy nobody reads.
    """
    dumped: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        if any(isinstance(marker, Sensitive) for marker in field.metadata):
            continue
        dumped[name] = _safe(getattr(model, name))
    return dumped


def _safe(value: object) -> Any:  # noqa: ANN401
    if isinstance(value, SecretStr):
        return MASKED
    if isinstance(value, BaseModel):
        return telemetry_dump(value)
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    return value


def _named(annotation: object) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def _violation(model: str, failure: ValidationError, payload: object) -> SchemaViolationError:
    problems = {_path(error["loc"]): error["msg"] for error in failure.errors()}
    paths = tuple(sorted(problems))
    described = "; ".join(f"{path or '<root>'}: {problems[path]}" for path in paths)
    return SchemaViolationError(
        f"{model} rejected the payload — {described}",
        model=model,
        paths=paths,
        problems=problems,
        payload=payload,
    )


def _path(location: tuple[int | str, ...]) -> str:
    # List indices and union tags are kept: `content.0.binary.media_type` says which part.
    return ".".join(str(part) for part in location)
