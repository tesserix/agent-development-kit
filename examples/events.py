"""Agent activity as events: one contract, allowlisted bodies, an explicit delivery mode.

Run it with `uv run python examples/events.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    ALLOWED_ATTRIBUTES,
    ApprovalDecided,
    Delivery,
    EventEnvelope,
    Eventing,
    EventPublishError,
    MemoryErased,
    RunCompleted,
    RunStarted,
    TenantContext,
    ToolCallCompleted,
    ToolCallRequested,
    tenant_scope,
)
from tesserix_adk.testing import FakeClock, InMemoryEventPublisher, assert_events

RUN = "run_1"


class Unreachable:
    """A broker that is not there."""

    async def publish(self, event: EventEnvelope) -> None:
        """Fail the way a transport with no route fails."""
        del event
        raise ConnectionError("no route to the broker")

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        """Fail the same way for a batch."""
        del events
        raise ConnectionError("no route to the broker")


async def main() -> None:
    """Emit a run's worth of events, then watch both delivery modes answer an outage."""
    published = InMemoryEventPublisher()
    eventing = Eventing(published, clock=FakeClock())

    with tenant_scope(TenantContext(tenant="acme", user="ada", correlation_id="corr-1")):
        started = await eventing.emit(RunStarted(run_id=RUN, agent="planner", model="sonnet"))
        requested = await eventing.emit(
            ToolCallRequested(run_id=RUN, tool="lookup", tool_call_id="c1")
        )
        await eventing.emit(
            ToolCallCompleted(run_id=RUN, tool="lookup", tool_call_id="c1", state="ok"),
            caused_by=requested,
        )
        await eventing.emit(
            RunCompleted(run_id=RUN, iterations=2, tool_calls=1, cost_micros=1_240),
            caused_by=started,
        )

    first = published.events[0]
    print("scope nobody passed:", first.tenant, first.user, first.correlation_id)  # noqa: T201
    print("sorts by when it happened:", first.event_id)  # noqa: T201
    print("caused by:", published.events[2].causation_id == published.events[1].event_id)  # noqa: T201
    print("cost reached the dashboard:", published.events[-1].attributes["cost_micros"])  # noqa: T201
    assert_events(published.events, *[event.type for event in published.events])

    with tenant_scope("acme"):
        await eventing.emit(MemoryErased(subject="ada@example.gov", records_erased=3))
        await eventing.emit(
            ApprovalDecided(
                run_id=RUN, approval_id="a1", decision="granted", approver="ada@example.gov"
            )
        )
    print("no address travelled:", "example.gov" not in str(published.events[-1].attributes))  # noqa: T201
    print("an event may only carry:", len(ALLOWED_ATTRIBUTES), "names")  # noqa: T201

    lenient = Eventing(Unreachable(), clock=FakeClock(), delivery=Delivery.BEST_EFFORT)
    with tenant_scope("acme"):
        await lenient.emit(RunStarted(run_id=RUN, agent="planner"))
    print("best effort dropped:", lenient.dropped, "and the run continued")  # noqa: T201

    strict = Eventing(Unreachable(), clock=FakeClock(), delivery=Delivery.GUARANTEED)
    try:
        with tenant_scope("acme"):
            await strict.emit(RunStarted(run_id=RUN, agent="planner"))
    except EventPublishError as refused:
        print("guaranteed stopped the step:", refused.event_type)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
