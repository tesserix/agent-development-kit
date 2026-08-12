"""Run state that outlives one worker, and what happens when two of them write it.

Four scenarios: a run stored and read back; a stale write refused; ten concurrent workers
adding what they spent; and paging a tenant's runs to find abandoned work.
Run it with `python examples/state.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    RunRecord,
    RunState,
    SessionRecord,
    StateConflictError,
    StateDelta,
    StateInUseError,
    StateQuery,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import MemoryStateStore
from tesserix_adk.testing import FakeClock

TENANT = "acme"


async def what_a_store_holds() -> None:
    """A run goes in, comes back, and knows how far through the conversation it is."""
    store = MemoryStateStore(FakeClock(start=1_000.0))
    call = ToolCall(id="c1", name="pay", arguments={"api_key": "sk-live-9f3c2a71b055"})
    stored = await store.put_run(
        RunRecord(
            run_id="run_1",
            tenant=TENANT,
            agent_name="planner",
            state=RunState.RUNNING,
            message_cursor=4,
            pending_tool_calls=(call,),
        )
    )
    read = await _read(store, stored)

    print("\n=== what a store holds ===")  # noqa: T201
    print(f"version: {read.version}, cursor: {read.message_cursor}")  # noqa: T201
    print(f"stopped mid tool call: {read.mid_tool_call}")  # noqa: T201
    print(f"the key it was asked to store: {read.pending_tool_calls[0].arguments}")  # noqa: T201


async def a_write_that_lost_the_race() -> None:
    """Two workers hold the same run. The one working from a stale copy is told so."""
    store = MemoryStateStore()
    first = await store.put_run(RunRecord(run_id="run_1", tenant=TENANT, agent_name="planner"))
    await store.put_run(first.model_copy(update={"iterations": 1}))

    print("\n=== a write that lost the race ===")  # noqa: T201
    try:
        await store.put_run(first.model_copy(update={"iterations": 9}))
    except StateConflictError as refused:
        moved = f"read at {refused.expected_version}, stored is {refused.actual_version}"
        print(f"refused: {moved}")  # noqa: T201

    read = await _read(store, first)
    print(f"the first worker's iteration survived: {read.iterations}")  # noqa: T201


async def what_ten_workers_spent() -> None:
    """Additions commute, so nothing is lost. Ten workers writing totals would lose nine."""
    store = MemoryStateStore()
    run = await store.put_run(RunRecord(run_id="run_1", tenant=TENANT, agent_name="planner"))
    await asyncio.gather(
        *(
            store.patch_run(
                run.key,
                StateDelta(
                    iterations=1,
                    cost_micros=1_200,
                    usage=Usage(input_tokens=800, output_tokens=120),
                ),
            )
            for _ in range(10)
        )
    )
    read = await _read(store, run)

    print("\n=== what ten workers spent ===")  # noqa: T201
    print(f"iterations: {read.iterations}")  # noqa: T201
    print(f"cost: {read.cost_micros / 1_000_000:.4f}")  # noqa: T201
    print(f"tokens in: {read.usage.input_tokens}")  # noqa: T201


async def finding_work_nobody_finished() -> None:
    """Page every running run, oldest first, without depending on anyone's clock."""
    clock = FakeClock(start=1_000.0)
    store = MemoryStateStore(clock)
    session = await store.put_session(SessionRecord(session_id="s1", tenant=TENANT))
    for index in range(5):
        state = RunState.RUNNING if index % 2 == 0 else RunState.COMPLETED
        await store.put_run(
            RunRecord(
                run_id=f"run_{index}",
                tenant=TENANT,
                agent_name="planner",
                session_id="s1",
                state=state,
            )
        )
    clock.advance(60.0)

    abandoned: list[str] = []
    cursor: str | None = None
    while True:
        page = await store.list_runs(
            StateQuery(
                tenant=TENANT,
                state=RunState.RUNNING,
                updated_before=clock.now(),
                limit=2,
                cursor=cursor,
            )
        )
        abandoned.extend(record.run_id for record in page.records)
        cursor = page.cursor
        if cursor is None:
            break

    print("\n=== finding work nobody finished ===")  # noqa: T201
    print(f"still running after a minute: {abandoned}")  # noqa: T201
    try:
        await store.delete_session(session.key)
    except StateInUseError as refused:
        print(f"the session will not go without them: {refused}")  # noqa: T201


async def _read(store: MemoryStateStore, written: RunRecord) -> RunRecord:
    """Read a run back, refusing to carry on if the store lost what it just took."""
    read = await store.get_run(written.key)
    if read is None:
        raise RuntimeError(f"{written.run_id} was stored and is not there")
    return read


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await what_a_store_holds()
    await a_write_that_lost_the_race()
    await what_ten_workers_spent()
    await finding_work_nobody_finished()


if __name__ == "__main__":
    asyncio.run(main())
