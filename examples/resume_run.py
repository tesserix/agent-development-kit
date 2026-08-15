"""A run paused on an approval, and the worker that picks it up on Monday.

Five scenarios: one worker takes the run and the second is refused by name; a lease renewed
and handed back; a key pasted into the conversation never reaching the store; an execution
that has run long enough starting again without losing the run; and the same frontier read
from a terminal. Run it with `python examples/resume_run.py`.
"""

from __future__ import annotations

import asyncio
import io

from tesserix_adk.cli import resume_main
from tesserix_adk.core import (
    DEFAULT_LEASE,
    Checkpoint,
    CheckpointBoundary,
    Message,
    RunLeaseError,
    TextPart,
    Usage,
)
from tesserix_adk.runtime import (
    Checkpointer,
    Leaseholder,
    MemoryCheckpointStore,
    MemoryLeaseStore,
    Resumer,
)
from tesserix_adk.testing import FakeClock
from tesserix_adk.workflows import DEFAULT_CONTINUATION, Journal, WorkflowState, continued

TENANT = "acme"


def a_frontier(**overrides: object) -> Checkpoint:
    """A booking run three iterations in, waiting on a person."""
    fields: dict[str, object] = {
        "run_id": "run_1",
        "tenant": TENANT,
        "agent_name": "booking",
        "boundary": CheckpointBoundary.BEFORE_APPROVAL,
        "usage": Usage(input_tokens=1_200, output_tokens=300),
        "cost_micros": 4_100,
        "iterations": 3,
        "pending_approval": "req-9",
    }
    return Checkpoint(**(fields | overrides))  # type: ignore[arg-type]


def a_resumer(store: MemoryCheckpointStore) -> tuple[Resumer, MemoryLeaseStore]:
    """A resumer over in-memory stores, sharing the clock expiry is decided on."""
    clock = FakeClock()
    leases = MemoryLeaseStore(clock)
    return Resumer(checkpoints=Checkpointer(store, clock=clock), leases=leases), leases


async def one_worker_at_a_time() -> None:
    """The second worker is refused before it ever reads the frontier."""
    store = MemoryCheckpointStore()
    await store.put(a_frontier())
    resumer, _ = a_resumer(store)

    carried = await resumer.resume("run_1", tenant=TENANT, worker="worker-7")
    if carried is None:
        raise RuntimeError("run_1 was checkpointed and is not there")

    print("\n=== one worker at a time ===")  # noqa: T201
    print(f"carried on at iteration {carried.iterations}")  # noqa: T201
    print(f"holding fence {carried.lease.fence}")  # noqa: T201
    try:
        await resumer.resume("run_1", tenant=TENANT, worker="worker-8")
    except RunLeaseError as refused:
        print(f"the second worker is told who has it: {refused.holder}")  # noqa: T201


async def a_lease_that_outlives_the_work() -> None:
    """A holder renews inside the window and hands the run back on the way out."""
    clock = FakeClock()
    leases = MemoryLeaseStore(clock)

    async with Leaseholder(leases, holder="worker-7") as holder:
        lease = await holder.acquire("run_1", tenant=TENANT)
        clock.advance(DEFAULT_LEASE.ttl_seconds - DEFAULT_LEASE.renew_within + 1)
        renewed = await holder.keep(now=clock.now())

    print("\n=== a lease that outlives the work ===")  # noqa: T201
    print(f"renewed without changing hands: {renewed is not None}")  # noqa: T201
    print(f"the fence never moved: {renewed is not None and renewed.fence == lease.fence}")  # noqa: T201
    print(f"released on the way out: {await leases.held('run_1', tenant=TENANT) is None}")  # noqa: T201


async def what_never_reaches_the_store() -> None:
    """A frontier is durable, so it is masked before it is sized."""
    store = MemoryCheckpointStore()
    said = Message(role="user", content=[TextPart(text="my key is sk-live-0123456789")])
    await Checkpointer(store).record(a_frontier(messages=(said,)))

    read = await store.latest("run_1", tenant=TENANT)
    if read is None:
        raise RuntimeError("run_1 was checkpointed and is not there")
    written = str(read.messages[0].content[0])

    print("\n=== what never reaches the store ===")  # noqa: T201
    print(f"the key is gone: {'sk-live-0123456789' not in written}")  # noqa: T201
    print(f"what was written instead: {written}")  # noqa: T201


def starting_the_execution_again() -> None:
    """The transcript crosses as a handle; the approval crosses or nothing crosses."""
    state = WorkflowState(
        run_id="run_1",
        history="h-9",
        iteration=4,
        usage=Usage(input_tokens=1_200, output_tokens=300),
        pending_approval="req-9",
        grant="grant-2",
    )
    carry = continued(state, tenant=TENANT, agent_name="booking")

    print("\n=== starting the execution again ===")  # noqa: T201
    due = DEFAULT_CONTINUATION.due(Journal(), history_bytes=2_000_000)
    print(f"due on the engine's history alone: {due}")  # noqa: T201
    print(f"the transcript travels as {carry.checkpoint.history_handle!r}")  # noqa: T201
    print(f"still waiting on {carry.checkpoint.pending_approval}")  # noqa: T201
    print(f"still acting under {carry.checkpoint.grant_id}")  # noqa: T201


async def from_a_terminal() -> None:
    """What the on-call engineer reads before deciding anything."""
    store = MemoryCheckpointStore()
    await store.put(a_frontier())
    resumer, _ = a_resumer(store)
    out = io.StringIO()

    code = await resume_main(["--resume", "run_1", "--tenant", TENANT], resumer=resumer, out=out)

    print("\n=== from a terminal ===")  # noqa: T201
    print(out.getvalue().rstrip())  # noqa: T201
    print(f"exit code: {code}")  # noqa: T201


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await one_worker_at_a_time()
    await a_lease_that_outlives_the_work()
    await what_never_reaches_the_store()
    starting_the_execution_again()
    await from_a_terminal()


if __name__ == "__main__":
    asyncio.run(main())
