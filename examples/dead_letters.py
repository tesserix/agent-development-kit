"""A consumer gives up on an event, an operator looks at it, and a replay puts it back.

Shows the three guardrails that matter: the listing prints no payload, the replay goes
through the same idempotent consumer as live traffic, and one audit event records who did it.

    uv run python examples/dead_letters.py
"""

from __future__ import annotations

import asyncio

from tesserix_adk.adapters import (
    DeadLetterQuery,
    IdempotentConsumer,
    InMemoryDeadLetters,
    Replayer,
)
from tesserix_adk.core import Delivery, EventEnvelope, Eventing, EventType, ToolCallCompleted
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.runtime import MemoryIdempotencyStore
from tesserix_adk.testing import FakeClock, InMemoryEventPublisher

TENANT = "acme"


async def main() -> None:
    """Bury one event under a broken consumer, then replay it under a fixed one."""
    clock = FakeClock(1_000.0)
    letters = InMemoryDeadLetters(clock=clock)
    published = InMemoryEventPublisher()
    eventing = Eventing(published, clock=clock, delivery=Delivery.GUARANTEED)

    charged: list[str] = []
    broken = True

    async def charge(event: EventEnvelope) -> None:
        if broken:
            raise ValueError("card 4111 1111 1111 1111 declined")
        charged.append(event.event_id)

    store = MemoryIdempotencyStore(clock=clock)
    consumer = IdempotentConsumer(
        charge,
        store=store,
        group="billing",
        max_attempts=1,
        dead_letter=letters.for_group("billing"),
    )

    with tenant_scope(TENANT, user="ada"):
        event = await eventing.emit(
            ToolCallCompleted(run_id="run_1", tool="charge", tool_call_id="c1", state="ok")
        )
    if event is None:
        raise RuntimeError("the publisher dropped the event this example is about")
    await consumer.handle(event)

    replayer = Replayer(letters, handler=consumer.handle, clock=clock, eventing=eventing)
    query = DeadLetterQuery(tenant=TENANT, group="billing")

    for record in await replayer.records(query):
        print("buried:", record.inspected()["event_id"], record.inspected()["last_error"])  # noqa: T201
        print("  attributes seen by the operator:", record.inspected()["attributes"])  # noqa: T201

    plan = await replayer.plan(query)
    print("dry run: would replay", plan.replayable, "| charged so far:", len(charged))  # noqa: T201

    broken = False
    report = await replayer.replay(query, operator="ada", reason="consumer_fixed")
    print("replayed:", report.replayed, "| charged:", len(charged))  # noqa: T201

    again = await replayer.replay(query, operator="ada")
    print("nothing left to replay:", again.replayed == 0)  # noqa: T201

    audit = [e for e in published.events if e.type is EventType.EVENTS_REPLAYED]
    print("audit:", [(e.attributes["approver"], e.attributes["records"]) for e in audit])  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
