"""The on-call engineer resuming a run that has been waiting since Friday."""

from __future__ import annotations

import io

import pytest

from tesserix_adk.cli import describe, frontier_main, resume_main
from tesserix_adk.core import (
    Checkpoint,
    CheckpointBoundary,
    CheckpointFormatError,
    HistoryUnavailableError,
    PendingCall,
    RunLeaseError,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import Checkpointer, MemoryCheckpointStore, MemoryLeaseStore, Resumer
from tesserix_adk.testing import FakeClock

pytestmark = pytest.mark.anyio


def frontier(**overrides: object) -> Checkpoint:
    """A run paused on an approval."""
    fields: dict[str, object] = {
        "run_id": "r1",
        "tenant": "acme",
        "agent_name": "booking",
        "boundary": CheckpointBoundary.BEFORE_APPROVAL,
        "usage": Usage(input_tokens=1_200, output_tokens=300),
        "cost_micros": 4_100,
        "iterations": 3,
        "pending_approval": "req-9",
    }
    return Checkpoint(**(fields | overrides))  # type: ignore[arg-type]


def resumer(store: MemoryCheckpointStore) -> Resumer:
    """A resumer over in-memory stores."""
    clock = FakeClock()
    return Resumer(checkpoints=Checkpointer(store, clock=clock), leases=MemoryLeaseStore(clock))


class Refusing:
    """A resumer that fails the way the runtime fails."""

    def __init__(self, failure: Exception) -> None:
        self.failure = failure

    async def resume(self, run_id: str, *, tenant: str, worker: str) -> None:
        assert run_id
        assert tenant
        assert worker
        raise self.failure


class TestResumingFromATerminal:
    async def test_a_resumed_run_prints_where_it_stopped(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(frontier())
        out = io.StringIO()

        code = await resume_main(
            ["--resume", "r1", "--tenant", "acme"], resumer=resumer(store), out=out
        )

        assert code == 0
        assert "iteration 3" in out.getvalue()
        assert "waiting on approval req-9" in out.getvalue()

    async def test_the_fence_it_took_is_printed_so_a_write_can_carry_it(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(frontier())
        out = io.StringIO()

        await resume_main(["--resume", "r1", "--tenant", "acme"], resumer=resumer(store), out=out)

        assert "held with fence 1" in out.getvalue()

    async def test_a_run_nobody_checkpointed_exits_one(self) -> None:
        out = io.StringIO()

        code = await resume_main(
            ["--resume", "r1", "--tenant", "acme"],
            resumer=resumer(MemoryCheckpointStore()),
            out=out,
        )

        assert code == 1
        assert "no checkpoint" in out.getvalue()

    async def test_a_command_line_it_cannot_read_exits_two(self) -> None:
        out = io.StringIO()

        code = await resume_main([], resumer=resumer(MemoryCheckpointStore()), out=out)

        assert code == 2

    async def test_a_run_another_worker_holds_exits_three(self) -> None:
        out = io.StringIO()
        refusing = Refusing(
            RunLeaseError("held", run_id="r1", holder="worker-7", requested_by="cli", fence=2)
        )

        code = await resume_main(["--resume", "r1", "--tenant", "acme"], resumer=refusing, out=out)

        assert code == 3
        assert "worker-7" in out.getvalue()

    async def test_a_frontier_this_kit_cannot_read_exits_four(self) -> None:
        out = io.StringIO()
        refusing = Refusing(
            CheckpointFormatError("newer", run_id="r1", format_version=99, readable_version=1)
        )

        code = await resume_main(["--resume", "r1", "--tenant", "acme"], resumer=refusing, out=out)

        assert code == 4
        assert "format 99" in out.getvalue()

    async def test_an_evicted_transcript_exits_four(self) -> None:
        out = io.StringIO()
        refusing = Refusing(HistoryUnavailableError("gone", run_id="r1", handle="h-1"))

        code = await resume_main(["--resume", "r1", "--tenant", "acme"], resumer=refusing, out=out)

        assert code == 4
        assert "h-1" in out.getvalue()

    async def test_an_undecidable_call_exits_four_and_names_it(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(
            frontier(
                pending=(PendingCall(call=ToolCall(id="c0", name="charge_card"), dispatched=True),)
            )
        )
        out = io.StringIO()

        code = await resume_main(
            ["--resume", "r1", "--tenant", "acme"], resumer=resumer(store), out=out
        )

        assert code == 4
        assert "charge_card" in out.getvalue()

    async def test_the_tenant_is_never_inferred(self) -> None:
        out = io.StringIO()

        code = await resume_main(
            ["--resume", "r1"], resumer=resumer(MemoryCheckpointStore()), out=out
        )

        assert code == 2


class TestLookingWithoutTouching:
    async def test_the_frontier_prints_without_a_lease_being_taken(self) -> None:
        leases = MemoryLeaseStore(FakeClock())
        out = io.StringIO()

        async def frontiers(run_id: str, tenant: str) -> Checkpoint:
            assert (run_id, tenant) == ("r1", "acme")
            return frontier()

        code = await frontier_main(["r1", "--tenant", "acme"], frontiers=frontiers, out=out)

        assert code == 0
        assert await leases.held("r1", tenant="acme") is None

    async def test_it_prints_exactly_what_the_resume_prints(self) -> None:
        store = MemoryCheckpointStore()
        await store.put(frontier())
        looked, resumed = io.StringIO(), io.StringIO()

        async def frontiers(run_id: str, tenant: str) -> Checkpoint | None:
            return await store.latest(run_id, tenant=tenant)

        await frontier_main(["r1", "--tenant", "acme"], frontiers=frontiers, out=looked)
        await resume_main(
            ["--resume", "r1", "--tenant", "acme"], resumer=resumer(store), out=resumed
        )

        assert looked.getvalue() in resumed.getvalue()

    async def test_a_run_nobody_checkpointed_exits_one(self) -> None:
        out = io.StringIO()

        async def frontiers(run_id: str, tenant: str) -> None:
            assert run_id
            assert tenant
            return

        code = await frontier_main(["r1", "--tenant", "acme"], frontiers=frontiers, out=out)

        assert code == 1

    async def test_a_command_line_it_cannot_read_exits_two(self) -> None:
        out = io.StringIO()

        async def frontiers(run_id: str, tenant: str) -> None:
            raise AssertionError(f"{run_id}/{tenant} should never have been looked up")

        assert await frontier_main([], frontiers=frontiers, out=out) == 2


class TestWhatAnOperatorReads:
    def test_a_run_waiting_on_nothing_says_so(self) -> None:
        assert "waiting on nothing" in describe(frontier(pending_approval=""))

    def test_the_ledger_is_on_the_summary_because_resuming_spends_more(self) -> None:
        assert "1200 in, 300 out" in describe(frontier())
        assert "4100 micros" in describe(frontier())
