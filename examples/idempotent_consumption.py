"""The same event three times, and one effect. Runs offline.

uv run python examples/idempotent_consumption.py
"""

from __future__ import annotations

import asyncio

from tesserix_adk.adapters import IdempotentConsumer
from tesserix_adk.core import Delivery, EventEnvelope, Eventing, ToolCallCompleted
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.runtime import MemoryIdempotencyStore
from tesserix_adk.testing import FakeClock


async def main() -> None:
    """Deliver one event three times, two of them at once, and count the charges."""
    charged: list[str] = []

    async def charge(event: EventEnvelope) -> None:
        await asyncio.sleep(0)
        charged.append(event.event_id)

    store = MemoryIdempotencyStore(FakeClock())
    eventing = Eventing(clock=FakeClock(), delivery=Delivery.GUARANTEED)
    with tenant_scope("acme"):
        event = await eventing.emit(
            ToolCallCompleted(run_id="run_1", tool="charge", tool_call_id="c1", state="ok")
        )
    if event is None:  # pragma: no cover — guaranteed delivery returns the envelope
        raise SystemExit("nothing was emitted")

    first = IdempotentConsumer(charge, store=store, group="billing")
    second = IdempotentConsumer(charge, store=store, group="billing")
    raced = await asyncio.gather(first.handle(event), second.handle(event), return_exceptions=True)
    await first.handle(event)

    print("charges:", len(charged))  # noqa: T201
    print("one worker was told to let it be redelivered:", [type(r).__name__ for r in raced])  # noqa: T201
    print("suppressed:", first.suppressed + second.suppressed)  # noqa: T201

    notifications = IdempotentConsumer(charge, store=store, group="notifications")
    await notifications.handle(event)
    print("another group still gets it:", notifications.handled)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
