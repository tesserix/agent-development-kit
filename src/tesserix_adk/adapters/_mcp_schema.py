"""What an MCP tool's JSON Schema is held to, shared by the client and the gateway.

An MCP server declares its arguments in JSON Schema, so the kit validates them in JSON
Schema rather than translating them into something narrower and losing a keyword on the
way. What is bounded here is the schema itself — its size, its nesting, and whether it
reaches outside the document — because a schema is remote input like any other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from tesserix_adk.core import ToolArgumentValidationError, require_extra
from tesserix_adk.mcp import McpGatewayError, McpGatewayReason
from tesserix_adk.tools import STRICT, ArgumentPolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue


PROSE_KEYS = frozenset(
    {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
)
MAX_SCHEMA_BYTES = 32 * 1024
MAX_SCHEMA_DEPTH = 16


class SchemaError(Protocol):
    absolute_path: object


class JsonValidator(Protocol):
    def __init__(self, schema: dict[str, JsonValue]) -> None:
        """Compile a schema."""

    @classmethod
    def check_schema(cls, schema: dict[str, JsonValue]) -> None:
        """Refuse an invalid schema."""

    def iter_errors(self, instance: object) -> object:
        """Return validation failures."""


@dataclass(frozen=True, slots=True)
class SchemaArguments:
    schema: dict[str, JsonValue]
    tool: str
    policy: ArgumentPolicy = STRICT
    _validator: JsonValidator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        jsonschema = require_extra("mcp", "jsonschema")
        validator_for = cast(
            "Callable[[dict[str, JsonValue]], type[JsonValidator]]",
            jsonschema.validators.validator_for,
        )
        validator_type = validator_for(self.schema)
        try:
            validator_type.check_schema(self.schema)
            validator = validator_type(self.schema)
        except Exception as failure:
            raise McpGatewayError(
                "an MCP tool advertised an invalid input schema",
                reason=McpGatewayReason.DISCOVERY,
                tool=self.tool,
            ) from failure
        object.__setattr__(self, "_validator", validator)

    def arguments(self, arguments: object) -> dict[str, JsonValue]:
        """Validate JSON arguments against the schema pinned during discovery."""
        parsed = self._parsed(arguments)
        failures = list(cast("Any", self._validator.iter_errors(parsed)))
        if failures:
            paths = tuple(
                sorted(
                    ".".join(str(part) for part in cast("Any", failure.absolute_path))
                    or "arguments"
                    for failure in failures
                )
            )
            raise ToolArgumentValidationError(
                f"{self.tool} was called with arguments that do not match its MCP schema",
                tool=self.tool,
                paths=paths,
                problems=dict.fromkeys(paths, "does not match the MCP input schema"),
                payload=arguments,
            )
        return parsed

    def _parsed(self, arguments: object) -> dict[str, JsonValue]:
        if isinstance(arguments, str | bytes):
            self._within_limit(len(arguments), arguments)
            try:
                parsed = json.loads(arguments, object_pairs_hook=self._unique(arguments))
            except json.JSONDecodeError as failure:
                raise self._refusal("arguments are not valid JSON", arguments) from failure
        else:
            try:
                encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError) as failure:
                raise self._refusal("arguments are not JSON data", arguments) from failure
            self._within_limit(len(encoded.encode()), arguments)
            parsed = json.loads(encoded)
        if not isinstance(parsed, dict):
            raise self._refusal("arguments are not a JSON object", arguments)
        return cast("dict[str, JsonValue]", parsed)

    def _unique(self, arguments: object) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
        def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            keys = [key for key, _ in pairs]
            if len(keys) != len(set(keys)):
                raise self._refusal("arguments repeat a JSON key", arguments)
            return dict(pairs)

        return hook

    def _within_limit(self, size: int, arguments: object) -> None:
        if size > self.policy.max_bytes:
            raise self._refusal(
                f"arguments exceed the {self.policy.max_bytes}-byte ceiling", arguments
            )

    def _refusal(self, violation: str, arguments: object) -> ToolArgumentValidationError:
        return ToolArgumentValidationError(
            f"{self.tool} was called with invalid arguments: {violation}",
            tool=self.tool,
            payload=arguments,
        )


def bounded_schema(schema: dict[str, JsonValue], *, tool: str) -> dict[str, JsonValue]:
    depth, references = schema_shape(schema)
    try:
        encoded = json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()
    except (RecursionError, TypeError, ValueError) as failure:
        raise McpGatewayError(
            "an MCP tool advertised a schema outside the JSON boundary",
            reason=McpGatewayReason.PAYLOAD,
            tool=tool,
        ) from failure
    if len(encoded) > MAX_SCHEMA_BYTES or depth > MAX_SCHEMA_DEPTH:
        raise McpGatewayError(
            "an MCP tool advertised a schema over the configured boundary",
            reason=McpGatewayReason.PAYLOAD,
            tool=tool,
        )
    for ref in references:
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise McpGatewayError(
                "an MCP tool advertised a non-local schema reference",
                reason=McpGatewayReason.DISCOVERY,
                tool=tool,
            )
    return schema


def schema_shape(value: JsonValue) -> tuple[int, list[JsonValue]]:
    """Measure a schema iteratively so hostile nesting cannot exhaust Python's stack."""
    maximum = 0
    references: list[JsonValue] = []
    pending = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        maximum = max(maximum, depth)
        if maximum > MAX_SCHEMA_DEPTH:
            return maximum, references
        if isinstance(current, dict):
            references.extend(child for key, child in current.items() if key == "$ref")
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
    return maximum, references


def without_remote_prose(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: without_remote_prose(child)
            for key, child in value.items()
            if key not in PROSE_KEYS
        }
    if isinstance(value, list):
        return [without_remote_prose(child) for child in value]
    return value
