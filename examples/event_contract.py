"""An older consumer reading a newer publisher, and the change that would have broken it.

Runs offline.

    uv run python examples/event_contract.py
"""

from __future__ import annotations

import asyncio
import json
from typing import ClassVar

from tesserix_adk.core import Delivery, EventEnvelope, Eventing, EventType, ToolCallCompleted
from tesserix_adk.core.errors import UnsupportedEventVersionError
from tesserix_adk.core.event_schema import EventSchemaRegistry, compatibility_breaks
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.testing import FakeClock, InMemoryEventPublisher


class ToolCallCompletedV2(ToolCallCompleted):
    """Version two: one added optional attribute, which is the only additive shape."""

    version: ClassVar[int] = 2
    attempt: int = 0


async def main() -> None:
    """Publish at version one, read at version two, and diff a change that is not additive."""
    event = await _published()

    older = EventSchemaRegistry()
    older.register(ToolCallCompleted)
    print(f"version one consumer reads: {older.payload_of(event).tool}")  # noqa: T201

    newer = EventSchemaRegistry()
    newer.register(ToolCallCompleted)
    newer.register(ToolCallCompletedV2)
    newer.register_upcaster(
        EventType.TOOL_CALL_COMPLETED,
        from_version=1,
        upcast=lambda attributes: attributes | {"attempt": "1"},
    )
    upcast = newer.upcast(event)
    print(f"upcast to version {upcast.schema_version}, attempt={upcast.attributes['attempt']}")  # noqa: T201

    ahead = json.loads(event.to_json()) | {"schema_version": 99}
    try:
        newer.read(json.dumps(ahead))
    except UnsupportedEventVersionError as parked:
        print(f"parked rather than crash-looped: {parked.details['version']}")  # noqa: T201

    print(f"additive change: {compatibility_breaks(_v1(), _v1_plus_attempt())}")  # noqa: T201
    for message in compatibility_breaks(_v1(), _renamed()):
        print(f"refused: {message}")  # noqa: T201


async def _published() -> EventEnvelope:
    eventing = Eventing(InMemoryEventPublisher(), clock=FakeClock(), delivery=Delivery.GUARANTEED)
    with tenant_scope("acme", user="ada"):
        event = await eventing.emit(
            ToolCallCompleted(run_id="run_1", tool="search", tool_call_id="c1", state="ok")
        )
    if event is None:
        raise SystemExit("the envelope was not built")
    return event


def _v1() -> dict[str, dict[str, object]]:
    return {"tool_call_completed@1": {"properties": {"tool": {"type": "string"}}, "required": []}}


def _v1_plus_attempt() -> dict[str, dict[str, object]]:
    return {
        "tool_call_completed@1": {
            "properties": {"tool": {"type": "string"}, "attempt": {"type": "integer"}},
            "required": [],
        }
    }


def _renamed() -> dict[str, dict[str, object]]:
    return {
        "tool_call_completed@1": {"properties": {"tool_name": {"type": "string"}}, "required": []}
    }


if __name__ == "__main__":
    asyncio.run(main())
