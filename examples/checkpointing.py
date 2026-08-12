"""A run that dies mid-booking, and the four ways of carrying it on.

Four scenarios: a frontier written and read back; a payload over the cap refused; a call
whose outcome was recorded replayed rather than repeated; and a call nobody can decide
stopping the resume. Run it with `python examples/checkpointing.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Checkpoint,
    CheckpointBoundary,
    CheckpointPolicy,
    Message,
    PendingCall,
    TextPart,
    ToolCall,
    Usage,
)
from tesserix_adk.core.errors import IndeterminateToolCallError
from tesserix_adk.runtime import (
    Checkpointer,
    MemoryCheckpointStore,
    MemoryIdempotencyStore,
    claim_resume,
    plan_resume,
    refuse_if_undecidable,
)
from tesserix_adk.runtime.checkpoint import RESUME_TTL_SECONDS

TENANT = "acme"
BOOKING = ToolCall(id="c1", name="book_flight", arguments={"flight": "BA117"})


def a_frontier(*, pending: tuple[PendingCall, ...] = ()) -> Checkpoint:
    """A run nine iterations in, holding whatever calls it never got answers to."""
    return Checkpoint(
        run_id="run_1",
        tenant=TENANT,
        agent_name="planner",
        model="claude-sonnet-5",
        boundary=CheckpointBoundary.AFTER_TOOL_RESULT,
        messages=(Message(role="user", content=[TextPart(text="book me to Tokyo")]),),
        pending=pending,
        usage=Usage(input_tokens=8_400, output_tokens=1_100),
        iterations=9,
    )


async def what_a_frontier_holds() -> None:
    """Enough for another process to pick the run up where this one dropped it."""
    store = MemoryCheckpointStore()
    await Checkpointer(store).record(a_frontier())
    read = await _read(store)

    print("\n=== what a frontier holds ===")  # noqa: T201
    print(f"iterations already paid for: {read.iterations}")  # noqa: T201
    print(f"tokens already spent: {read.usage.input_tokens}")  # noqa: T201
    print(f"taken at: {read.boundary}")  # noqa: T201


async def a_frontier_too_large_to_write() -> None:
    """Refused whole. Half a frontier resumes into a conversation that never happened."""
    store = MemoryCheckpointStore()
    checkpointer = Checkpointer(store, CheckpointPolicy(max_bytes=64))
    written = await checkpointer.record(a_frontier())

    print("\n=== a frontier too large to write ===")  # noqa: T201
    print(f"written: {written}")  # noqa: T201
    print(f"why: {checkpointer.last_error}")  # noqa: T201
    nothing_stored = await store.latest("run_1", tenant=TENANT) is None
    print(f"the run carries on regardless, uncheckpointed: {nothing_stored}")  # noqa: T201


async def a_flight_that_was_already_booked() -> None:
    """The recorded outcome is replayed. Calling again is the second seat."""
    idempotency = MemoryIdempotencyStore()
    await idempotency.record("book:BA117", tenant=TENANT, outcome="PNR 7QK2ZP", ttl_seconds=600)
    frontier = a_frontier(
        pending=(PendingCall(call=BOOKING, idempotency_key="book:BA117", dispatched=True),)
    )
    plan = await plan_resume(frontier, idempotency)

    print("\n=== a flight that was already booked ===")  # noqa: T201
    print(f"safe to carry on: {plan.safe}")  # noqa: T201
    print(f"replayed rather than called: {[one.outcome for one in plan.completed]}")  # noqa: T201
    print(f"still to dispatch: {[one.call.name for one in plan.to_dispatch]}")  # noqa: T201


async def a_flight_nobody_can_decide() -> None:
    """The claim is held by a worker that is gone. Retrying is the duplicate booking."""
    idempotency = MemoryIdempotencyStore()
    await idempotency.begin("book:BA117", tenant=TENANT, ttl_seconds=RESUME_TTL_SECONDS)
    frontier = a_frontier(
        pending=(PendingCall(call=BOOKING, idempotency_key="book:BA117", dispatched=True),)
    )

    print("\n=== a flight nobody can decide ===")  # noqa: T201
    try:
        refuse_if_undecidable(await plan_resume(frontier, idempotency))
    except IndeterminateToolCallError as stopped:
        print(f"stopped: {stopped}")  # noqa: T201
        print(f"worth retrying: {stopped.retryable}")  # noqa: T201

    await claim_resume("run_1", tenant=TENANT, idempotency=idempotency)
    print("the run is now held by this worker alone")  # noqa: T201


async def _read(store: MemoryCheckpointStore) -> Checkpoint:
    """Read the frontier back, refusing to carry on if the store lost what it just took."""
    read = await store.latest("run_1", tenant=TENANT)
    if read is None:
        raise RuntimeError("run_1 was checkpointed and is not there")
    return read


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await what_a_frontier_holds()
    await a_frontier_too_large_to_write()
    await a_flight_that_was_already_booked()
    await a_flight_nobody_can_decide()


if __name__ == "__main__":
    asyncio.run(main())
