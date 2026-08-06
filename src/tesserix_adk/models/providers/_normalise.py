"""What a vendor sent back becomes a tool call, or it becomes a typed refusal.

Every vendor produces tool calls its own way and none of them promises the arguments
match the schema it was given. The checks here run in the adapter, which is the last
point at which a malformed call costs nothing to stop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesserix_adk.core.capabilities import Capability
from tesserix_adk.core.errors import (
    CapabilityError,
    ModelResponseError,
    ToolArgumentValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.core.capabilities import ModelCapabilities
    from tesserix_adk.core.primitives import ToolCall
    from tesserix_adk.core.provider import ModelRequest

__all__ = ["normalised_tool_calls"]

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def normalised_tool_calls(
    calls: Sequence[ToolCall],
    *,
    request: ModelRequest,
    capabilities: ModelCapabilities,
    provider: str,
) -> tuple[ToolCall, ...]:
    """Return `calls` unchanged, having proved they are callable.

    Args:
        calls: Tool calls as the provider read them off its own wire.
        request: The request they answer, which carries the declarations they are
            checked against. A call naming a tool the request never declared is passed
            through unchecked: there is no schema to check it against, and inventing one
            refuses calls that are valid.
        capabilities: What the model declared, which is what it may exceed.
        provider: Who answered, for the error.

    Raises:
        CapabilityError: If more than one call arrived from a model that does not
            declare `parallel_tool_calls`.
        ModelResponseError: If one id was used twice in a turn. Results are matched back
            by id, so a repeated id answers the wrong call.
        ToolArgumentValidationError: If any call's arguments violate its tool's schema.
    """
    if len(calls) > 1 and not capabilities.supports(Capability.PARALLEL_TOOL_CALLS):
        raise CapabilityError(
            f"{provider} returned {len(calls)} tool calls for a model that declares one",
            capability=Capability.PARALLEL_TOOL_CALLS.value,
            provider=provider,
            model=request.model,
        )
    _unique(calls, provider=provider)
    schemas = {tool.name: tool.parameters for tool in request.tools}
    for call in calls:
        if call.name in schemas:
            _checked(call, schemas[call.name])
    return tuple(calls)


def _unique(calls: Sequence[ToolCall], *, provider: str) -> None:
    seen: set[str] = set()
    for call in calls:
        if call.id in seen:
            raise ModelResponseError(
                f"{provider} used tool call id {call.id!r} twice in one turn",
                payload=[call.model_dump() for call in calls],
                provider=provider,
            )
        seen.add(call.id)


def _checked(call: ToolCall, schema: Mapping[str, Any]) -> None:
    problems = _violations(schema, call.arguments)
    if problems:
        paths = tuple(sorted(problems))
        described = "; ".join(f"{path}: {problems[path]}" for path in paths)
        raise ToolArgumentValidationError(
            f"arguments for {call.name} do not match its schema — {described}",
            tool=call.name,
            call_id=call.id,
            paths=paths,
            problems=problems,
            payload=call.arguments,
        )


def _violations(schema: Mapping[str, Any], value: object, prefix: str = "") -> dict[str, str]:
    """Collect every field that fails, not the first: one per round trip is five deploys.

    Covers the strict subset the kit emits — `type`, `required`, `properties`, `items`,
    `enum` and closed objects. A composed schema the kit did not generate (`anyOf`, `$ref`)
    passes: refusing what this cannot read would refuse valid calls.
    """
    declared = schema.get("type")
    if isinstance(declared, str) and declared not in ("null", *_TYPES):
        return {}
    if declared == "object" or "properties" in schema:
        return _object_violations(schema, value, prefix)
    if declared == "array":
        return _array_violations(schema, value, prefix)
    return _scalar_violations(schema, value, prefix)


def _object_violations(schema: Mapping[str, Any], value: object, prefix: str) -> dict[str, str]:
    if not isinstance(value, dict):
        return {prefix or "<root>": f"expected object, got {_named(value)}"}
    properties: Mapping[str, Any] = schema.get("properties", {})
    problems: dict[str, str] = {}
    for name in schema.get("required", []):
        if name not in value:
            problems[_path(prefix, name)] = "required, and absent"
    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in properties:
                problems[_path(prefix, name)] = "not a property of this tool's schema"
    for name, field in properties.items():
        if name in value:
            problems |= _violations(field, value[name], _path(prefix, name))
    return problems


def _array_violations(schema: Mapping[str, Any], value: object, prefix: str) -> dict[str, str]:
    if not isinstance(value, list):
        return {prefix or "<root>": f"expected array, got {_named(value)}"}
    items = schema.get("items")
    if not isinstance(items, dict):
        return {}
    problems: dict[str, str] = {}
    for position, item in enumerate(value):
        problems |= _violations(items, item, f"{prefix}[{position}]")
    return problems


def _scalar_violations(schema: Mapping[str, Any], value: object, prefix: str) -> dict[str, str]:
    where = prefix or "<root>"
    declared = schema.get("type")
    if declared == "null" and value is not None:
        return {where: f"expected null, got {_named(value)}"}
    if isinstance(declared, str) and declared in _TYPES:
        allowed = _TYPES[declared]
        # A bool is an int in Python and is not one in JSON Schema; a flag read as 1 and
        # a count read as True are the same bug the strict base exists to refuse.
        wrong_type = not isinstance(value, allowed) or (
            isinstance(value, bool) and declared in ("integer", "number")
        )
        if wrong_type:
            return {where: f"expected {declared}, got {_named(value)}"}
    choices = schema.get("enum")
    if isinstance(choices, list) and value not in choices:
        return {where: f"not one of {choices}"}
    return {}


def _path(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _named(value: object) -> str:
    return "null" if value is None else type(value).__name__
