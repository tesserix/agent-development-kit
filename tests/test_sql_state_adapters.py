"""What the PostgreSQL state store and work queue send, and what they refuse to send.

No server here: the fake answers with whatever the test says the database returned, so
these tests are about the statements and the decisions around them. That the statements
are valid SQL which does what the protocol says is `tests/integration`, against a real
PostgreSQL, running the same conformance suites the in-process stores pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.sql_state import (
    EXPECTED_SCHEMA,
    SCHEMA_VERSION,
    PostgresStateStore,
    PostgresStoreSettings,
    PostgresWorkQueue,
    StateTables,
)
from tesserix_adk.core.errors import (
    ConfigurationError,
    LeaseLostError,
    PoolExhaustedError,
    QueueUnavailableError,
    StateConflictError,
    StateInUseError,
    StateNotFoundError,
    StatePersistenceError,
    WorkItemNotFoundError,
)
from tesserix_adk.core.primitives import ToolCall, Usage
from tesserix_adk.core.queue import QueuePolicy, WorkItem, WorkState
from tesserix_adk.core.run import RunState
from tesserix_adk.core.state import RunRecord, SessionRecord, StateDelta, StateKey, StateQuery
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence

NOW = 1_000.0
SETTINGS = PostgresStoreSettings(dsn=SecretStr("postgresql://adk:s3cret@db:5432/adk"))
KEY = StateKey(tenant="acme", id="run_1")
HEALTHY = [[[SCHEMA_VERSION]], [["5s"]]]


class FakeSql:
    """Answers with what the test says the database returned, and records what was asked."""

    def __init__(self, *replies: Any, fails: Sequence[Exception] = ()) -> None:
        self.replies = list(replies)
        self.fails = list(fails)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """Answer one statement."""
        self.calls.append((statement, args))
        if self.fails:
            raise self.fails.pop(0)
        return self.replies.pop(0) if self.replies else []

    @property
    def sent(self) -> str:
        """The last statement, tables and all."""
        return self.calls[-1][0]

    @property
    def bound(self) -> tuple[Any, ...]:
        """What the last statement was given."""
        return self.calls[-1][1]


class DriverError(Exception):
    """A driver error, which is where the SQLSTATE lives."""

    def __init__(self, sqlstate: str = "") -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


class PgcodeError(Exception):
    """The same thing under the other name drivers give it."""

    def __init__(self, pgcode: str) -> None:
        super().__init__(pgcode)
        self.pgcode = pgcode


class PoolTimeoutError(Exception):
    """A pool that ran out, named the way pool libraries name it."""


def store(sql: FakeSql, **kwargs: Any) -> PostgresStateStore:
    """A state store on a clock the test controls."""
    kwargs.setdefault("settings", SETTINGS)
    return PostgresStateStore(sql, clock=FakeClock(start=NOW), entropy=lambda: 0.5, **kwargs)


def queue(sql: FakeSql, **kwargs: Any) -> PostgresWorkQueue:
    """A work queue on a clock the test controls."""
    kwargs.setdefault("settings", SETTINGS)
    kwargs.setdefault("policy", QueuePolicy(max_attempts=3, backoff_seconds=2.0))
    return PostgresWorkQueue(sql, clock=FakeClock(start=NOW), entropy=lambda: 0.5, **kwargs)


def run(**fields: Any) -> RunRecord:
    """One run, filled in enough to be stored."""
    return RunRecord(run_id="run_1", tenant="acme", agent_name="planner", **fields)


def stored(record: RunRecord | None = None, *, seq: int = 7, version: int = 3) -> list[Any]:
    """A run row: the blob, then the counters the columns own."""
    return [seq, (record or run()).model_dump_json(), version, 40, 9, 500, 2, 6]


def work(**fields: Any) -> WorkItem:
    """One item, filled in enough to be enqueued."""
    fields.setdefault("id", "i1")
    fields.setdefault("tenant", "acme")
    return WorkItem(**fields)


def claimed(**fields: Any) -> WorkItem:
    """One item a worker holds, on a lease that has not lapsed."""
    fields.setdefault("worker", "w1")
    fields.setdefault("lease_expires_at", NOW + 30)
    fields.setdefault("first_claimed_at", NOW)
    return work(state=WorkState.CLAIMED, **fields)


class TestWhereTheRowsGo:
    """A table name cannot be a bound parameter, so it is checked instead."""

    def test_the_default_tables_are_the_documented_ones(self) -> None:
        assert StateTables().runs == "adk_runs"
        assert StateTables().work == "adk_work"

    def test_a_name_that_could_carry_sql_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="not a plain table identifier"):
            StateTables(runs="runs; DROP TABLE adk_runs")

    async def test_the_statement_names_the_tables_it_was_given(self) -> None:
        sql = FakeSql()
        await store(sql, tables=StateTables(runs="tenant_runs")).delete_run(KEY)
        assert "DELETE FROM tenant_runs" in sql.sent

    def test_the_expected_shape_is_published_for_the_migration_to_own(self) -> None:
        assert "CREATE TABLE adk_runs" in EXPECTED_SCHEMA
        assert "adk_schema" in EXPECTED_SCHEMA


class TestRefusingADatabaseThatIsNotTheRightShape:
    """A column that moved is a write into the wrong shape, and it is caught at startup."""

    async def test_the_schema_version_it_was_written_for_is_accepted(self) -> None:
        sql = FakeSql(*HEALTHY)
        await PostgresStateStore.open(sql, clock=FakeClock(), settings=SETTINGS)
        assert sql.calls[0][1] == ("state",)

    async def test_the_queue_checks_its_own_component(self) -> None:
        sql = FakeSql(*HEALTHY)
        await PostgresWorkQueue.open(sql, clock=FakeClock(), settings=SETTINGS)
        assert sql.calls[0][1] == ("queue",)

    async def test_a_schema_version_that_moved_is_refused(self) -> None:
        sql = FakeSql([[2]], [["5s"]])
        with pytest.raises(ConfigurationError, match="schema is version 2"):
            await PostgresStateStore.open(sql, clock=FakeClock(), settings=SETTINGS)

    async def test_a_database_with_no_schema_row_at_all_is_refused(self) -> None:
        sql = FakeSql([], [["5s"]])
        with pytest.raises(ConfigurationError, match="schema is version 0"):
            await PostgresStateStore.open(sql, clock=FakeClock(), settings=SETTINGS)

    async def test_a_connection_that_could_run_forever_is_refused(self) -> None:
        sql = FakeSql([[SCHEMA_VERSION]], [["0"]])
        with pytest.raises(ConfigurationError, match="statement_timeout"):
            await PostgresStateStore.open(sql, clock=FakeClock(), settings=SETTINGS)

    async def test_a_timeout_of_zero_milliseconds_is_the_same_refusal(self) -> None:
        sql = FakeSql([[SCHEMA_VERSION]], [["0ms"]])
        with pytest.raises(ConfigurationError, match="statement_timeout"):
            await PostgresWorkQueue.open(sql, clock=FakeClock(), settings=SETTINGS)


class TestReadingARun:
    """A run comes back as the blob it was written as, plus what was added to it since."""

    async def test_a_run_that_is_not_there_is_none(self) -> None:
        assert await store(FakeSql([])).get_run(KEY) is None

    async def test_the_columns_are_added_to_the_blob(self) -> None:
        found = await store(FakeSql([stored()])).get_run(KEY)
        assert found is not None
        assert found.version == 3
        assert found.usage.input_tokens == 40
        assert found.cost_micros == 500

    async def test_the_tenant_is_in_the_statement_rather_than_filtered_after(self) -> None:
        sql = FakeSql([stored()])
        await store(sql).get_run(KEY)
        assert "WHERE tenant = $1 AND run_id = $2" in sql.sent
        assert sql.bound == ("acme", "run_1")


class TestWritingARun:
    """The version compare is inside the write, so two workers cannot both think they won."""

    async def test_the_write_carries_the_version_it_was_read_at(self) -> None:
        sql = FakeSql([[3]])
        written = await store(sql).put_run(run(version=2))
        assert written.version == 3
        assert sql.bound[4] == 2
        assert "{runs}.version = $5" not in sql.sent

    async def test_the_counters_go_to_columns_and_not_into_the_blob(self) -> None:
        sql = FakeSql([[2]])
        await store(sql).put_run(run(usage=Usage(input_tokens=40, output_tokens=9)))
        assert sql.bound[6:8] == (40, 9)
        assert '"input_tokens":0' in sql.bound[11]

    async def test_a_write_whose_version_moved_is_refused_with_both_numbers(self) -> None:
        sql = FakeSql([], [[5]])
        with pytest.raises(StateConflictError) as refused:
            await store(sql).put_run(run(version=2))
        assert refused.value.expected_version == 2
        assert refused.value.actual_version == 5

    async def test_a_write_of_a_run_that_was_deleted_under_it_reads_as_version_zero(self) -> None:
        sql = FakeSql([], [])
        with pytest.raises(StateConflictError) as refused:
            await store(sql).put_run(run(version=2))
        assert refused.value.actual_version == 0

    async def test_a_tool_argument_is_scrubbed_before_the_row_is_built(self) -> None:
        call = ToolCall(id="c1", name="pay", arguments={"who": "ada@example.com"})
        sql = FakeSql([[1]])
        await store(sql).put_run(run(pending_tool_calls=(call,)))
        assert "ada@example.com" not in sql.bound[11]

    async def test_a_record_larger_than_the_store_writes_is_refused_rather_than_sent(self) -> None:
        sql = FakeSql([[1]])
        small = SETTINGS.model_copy(update={"max_value_bytes": 32})
        with pytest.raises(StatePersistenceError) as refused:
            await store(sql, settings=small).put_run(run())
        assert refused.value.reason == "too_large"
        assert refused.value.retryable is False
        assert not sql.calls


class TestPatchingARun:
    """A patch adds in SQL, so two patches arriving together both land."""

    async def test_the_amounts_go_to_the_server_rather_than_the_totals(self) -> None:
        sql = FakeSql([stored()])
        await store(sql).patch_run(
            KEY, StateDelta(usage=Usage(input_tokens=40, output_tokens=9), iterations=1)
        )
        assert "input_tokens = input_tokens + $3" in sql.sent
        assert sql.bound[2:] == (40, 9, 0, 1, 0)

    async def test_a_patch_with_no_usage_still_counts_what_it_carries(self) -> None:
        sql = FakeSql([stored()])
        await store(sql).patch_run(KEY, StateDelta(cost_micros=25))
        assert sql.bound[2:] == (0, 0, 25, 0, 0)

    async def test_a_patch_of_a_run_that_is_not_there_is_refused(self) -> None:
        with pytest.raises(StateNotFoundError, match="no run acme/run_1"):
            await store(FakeSql([])).patch_run(KEY, StateDelta(iterations=1))


class TestListingRuns:
    """Paging is by the sequence the store gave the row, which nothing rewrites."""

    async def test_a_full_page_hands_back_a_cursor(self) -> None:
        sql = FakeSql([stored(seq=1), stored(seq=2), stored(seq=3)])
        page = await store(sql).list_runs(StateQuery(tenant="acme", limit=2))
        assert len(page.records) == 2
        assert page.cursor == "2"

    async def test_a_short_page_is_the_end_of_the_listing(self) -> None:
        page = await store(FakeSql([stored()])).list_runs(StateQuery(tenant="acme", limit=2))
        assert page.cursor is None

    async def test_an_empty_listing_is_not_an_error(self) -> None:
        page = await store(FakeSql([])).list_runs(StateQuery(tenant="acme"))
        assert page.records == ()

    async def test_it_asks_for_one_more_than_the_page(self) -> None:
        sql = FakeSql([])
        await store(sql).list_runs(StateQuery(tenant="acme", limit=25))
        assert sql.bound[4] == 26

    async def test_every_narrowing_goes_into_the_statement(self) -> None:
        sql = FakeSql([])
        await store(sql).list_runs(
            StateQuery(
                tenant="acme", state=RunState.FAILED, updated_before=NOW, cursor="12", limit=5
            )
        )
        assert sql.bound[:4] == ("acme", 12, "failed", NOW)


class TestSessions:
    """A session is one row, and its runs are what stop it being deleted."""

    async def test_a_session_that_is_not_there_is_none(self) -> None:
        assert await store(FakeSql([])).get_session(StateKey(tenant="acme", id="s1")) is None

    async def test_a_written_session_comes_back_at_the_next_version(self) -> None:
        sql = FakeSql([[2]])
        written = await store(sql).put_session(SessionRecord(session_id="s1", tenant="acme"))
        assert written.version == 1
        assert written.updated_at == NOW

    async def test_a_stale_session_write_is_refused(self) -> None:
        sql = FakeSql([], [[4]])
        with pytest.raises(StateConflictError):
            await store(sql).put_session(SessionRecord(session_id="s1", tenant="acme", version=1))

    async def test_a_session_larger_than_the_store_writes_is_refused(self) -> None:
        small = SETTINGS.model_copy(update={"max_value_bytes": 8})
        with pytest.raises(StatePersistenceError):
            await store(FakeSql(), settings=small).put_session(
                SessionRecord(session_id="s1", tenant="acme")
            )

    async def test_a_session_with_no_live_runs_is_deleted(self) -> None:
        sql = FakeSql([])
        await store(sql).delete_session(StateKey(tenant="acme", id="s1"))
        assert sql.bound[3] is False

    async def test_a_session_with_a_live_run_is_not_deleted(self) -> None:
        sql = FakeSql([["run_1"], ["run_2"]])
        with pytest.raises(StateInUseError) as refused:
            await store(sql).delete_session(StateKey(tenant="acme", id="s1"))
        assert refused.value.details["live_runs"] == "run_1, run_2"

    async def test_a_cascade_takes_the_live_runs_with_it(self) -> None:
        sql = FakeSql([["run_1"]])
        await store(sql).delete_session(StateKey(tenant="acme", id="s1"), cascade=True)
        assert sql.bound[3] is True


class TestEnqueueing:
    """A second enqueue under a live dedupe key returns the first item, not a second one."""

    async def test_an_item_is_placed_at_the_clock_the_queue_keeps(self) -> None:
        sql = FakeSql([])
        placed = await queue(sql).enqueue(work())
        assert placed.enqueued_at == NOW
        assert placed.available_at == NOW
        assert placed.state is WorkState.QUEUED

    async def test_a_delay_the_caller_asked_for_is_kept(self) -> None:
        placed = await queue(FakeSql([])).enqueue(work(available_at=NOW + 60))
        assert placed.available_at == NOW + 60

    async def test_a_duplicate_is_collapsed_into_the_item_already_there(self) -> None:
        live = work(id="i0", dedupe_key="job-7", state=WorkState.QUEUED)
        placed = await queue(FakeSql([[live.model_dump_json()]])).enqueue(
            work(id="i9", dedupe_key="job-7")
        )
        assert placed.id == "i0"

    async def test_a_terminal_state_frees_the_key_for_the_same_job_again(self) -> None:
        sql = FakeSql([])
        await queue(sql).enqueue(work(dedupe_key="job-7"))
        assert sql.bound[8] == ["completed", "dead_lettered"]

    async def test_a_payload_larger_than_the_queue_writes_is_refused_before_the_write(self) -> None:
        small = SETTINGS.model_copy(update={"max_value_bytes": 16})
        sql = FakeSql([])
        with pytest.raises(StatePersistenceError) as refused:
            await queue(sql, settings=small).enqueue(work())
        assert refused.value.reason == "too_large"
        assert not sql.calls


class TestClaiming:
    """A claim locks one row and steps over the ones another worker holds."""

    async def test_an_empty_queue_hands_out_nothing(self) -> None:
        assert await queue(FakeSql([])).claim(worker="w1") is None

    async def test_the_claimed_item_names_its_worker_and_its_lease(self) -> None:
        sql = FakeSql([[work().model_dump_json()]], [[1]])
        item = await queue(sql).claim(worker="w1", lease_seconds=45)
        assert item is not None
        assert item.worker == "w1"
        assert item.lease_expires_at == NOW + 45

    async def test_the_lock_steps_over_what_another_worker_holds(self) -> None:
        sql = FakeSql([[work().model_dump_json()]], [[1]])
        await queue(sql).claim(worker="w1")
        assert "FOR UPDATE OF w SKIP LOCKED" in sql.calls[0][0]

    async def test_the_predicate_is_repeated_on_the_row_that_is_locked(self) -> None:
        sql = FakeSql([[work().model_dump_json()]], [[1]])
        await queue(sql).claim(worker="w1")
        assert "WHERE w.queue = $1 AND w.state = 'queued'" in sql.calls[0][0]

    async def test_a_served_tenant_goes_to_the_back_of_the_rotation(self) -> None:
        sql = FakeSql([[work().model_dump_json()]], [[1]])
        await queue(sql).claim(worker="w1")
        assert "ORDER BY COALESCE(t.turn, 0)" in sql.calls[0][0]
        assert "INSERT INTO adk_queue_turns" in sql.calls[0][0]

    async def test_the_default_lease_is_the_policy_s(self) -> None:
        sql = FakeSql([[work().model_dump_json()]], [[1]])
        item = await queue(sql, policy=QueuePolicy(lease_seconds=12)).claim(worker="w1")
        assert item is not None
        assert item.lease_expires_at == NOW + 12


class TestHeartbeats:
    """A renewal extends the claim, up to the bound a single claim may run for."""

    async def test_a_live_claim_is_extended(self) -> None:
        sql = FakeSql([[claimed().model_dump_json()]], [[1]])
        renewed = await queue(sql).heartbeat("i1", tenant="acme", worker="w1")
        assert renewed.lease_expires_at == NOW + 30

    async def test_the_renewal_is_fenced_on_the_lease_it_read(self) -> None:
        sql = FakeSql([[claimed().model_dump_json()]], [[1]])
        await queue(sql).heartbeat("i1", tenant="acme", worker="w1")
        assert sql.bound[9] == NOW + 30

    async def test_an_item_that_is_not_there_is_a_refusal(self) -> None:
        with pytest.raises(WorkItemNotFoundError):
            await queue(FakeSql([])).heartbeat("i1", tenant="acme", worker="w1")

    async def test_an_item_nobody_holds_is_no_longer_this_worker_s(self) -> None:
        sql = FakeSql([[work().model_dump_json()]])
        with pytest.raises(LeaseLostError) as lost:
            await queue(sql).heartbeat("i1", tenant="acme", worker="w1")
        assert lost.value.details["reason"] == "expired"

    async def test_an_item_another_worker_holds_says_who_holds_it(self) -> None:
        sql = FakeSql([[claimed(worker="w2").model_dump_json()]])
        with pytest.raises(LeaseLostError) as lost:
            await queue(sql).heartbeat("i1", tenant="acme", worker="w1")
        assert lost.value.details["reason"] == "taken"
        assert lost.value.details["holder"] == "w2"

    async def test_a_lease_that_lapsed_while_the_worker_was_away_is_gone(self) -> None:
        sql = FakeSql([[claimed(lease_expires_at=NOW - 1).model_dump_json()]])
        with pytest.raises(LeaseLostError) as lost:
            await queue(sql).heartbeat("i1", tenant="acme", worker="w1")
        assert lost.value.details["reason"] == "expired"

    async def test_a_claim_renewed_past_the_bound_is_capped(self) -> None:
        old = claimed(first_claimed_at=NOW - 4_000)
        sql = FakeSql([[old.model_dump_json()]])
        with pytest.raises(LeaseLostError) as lost:
            await queue(sql).heartbeat("i1", tenant="acme", worker="w1")
        assert lost.value.details["reason"] == "capped"


class TestFinishingWork:
    """What a worker says happened decides where the item goes next."""

    async def test_a_completed_item_holds_no_lease_and_no_worker(self) -> None:
        sql = FakeSql([[claimed().model_dump_json()]], [[1]])
        done = await queue(sql).complete("i1", tenant="acme", worker="w1")
        assert done.state is WorkState.COMPLETED
        assert done.worker is None
        assert done.lease_expires_at is None

    async def test_a_failure_comes_back_after_a_backoff(self) -> None:
        sql = FakeSql([[claimed().model_dump_json()]], [[1]])
        given = await queue(sql).fail("i1", tenant="acme", worker="w1", error="boom")
        assert given.state is WorkState.QUEUED
        assert given.attempts == 1
        assert given.available_at == NOW + 2.0
        assert given.failures == ("boom",)

    async def test_a_failure_that_cannot_succeed_skips_its_remaining_attempts(self) -> None:
        sql = FakeSql([[claimed().model_dump_json()]], [[1]])
        given = await queue(sql).fail(
            "i1", tenant="acme", worker="w1", error="bad payload", retryable=False
        )
        assert given.state is WorkState.DEAD_LETTERED

    async def test_the_last_attempt_goes_to_the_dead_letter(self) -> None:
        sql = FakeSql([[claimed(attempts=2).model_dump_json()]], [[1]])
        given = await queue(sql).fail("i1", tenant="acme", worker="w1", error="boom")
        assert given.state is WorkState.DEAD_LETTERED


class TestReaping:
    """A lapsed lease is work nobody is doing, and it goes back where a worker can see it."""

    async def test_a_lapsed_claim_is_requeued_one_attempt_worse_off(self) -> None:
        sql = FakeSql([[claimed().model_dump_json(), NOW + 30]], [[1]])
        given = await queue(sql).reap()
        assert given[0].state is WorkState.QUEUED
        assert given[0].attempts == 1
        assert "lease lapsed" in given[0].failures[0]

    async def test_the_write_is_fenced_on_the_lease_the_sweep_saw(self) -> None:
        sql = FakeSql([[claimed().model_dump_json(), NOW + 30]], [[1]])
        await queue(sql).reap()
        assert sql.bound[9] == NOW + 30
        assert sql.bound[10] is True

    async def test_an_item_a_heartbeat_saved_first_is_left_alone(self) -> None:
        sql = FakeSql([[claimed().model_dump_json(), NOW + 30]], [])
        assert await queue(sql).reap() == ()

    async def test_a_row_another_transaction_holds_is_skipped_rather_than_waited_on(self) -> None:
        sql = FakeSql([])
        await queue(sql).reap()
        assert "FOR UPDATE SKIP LOCKED" in sql.sent

    async def test_a_restarted_worker_gives_back_what_it_held_without_a_backoff(self) -> None:
        sql = FakeSql([[claimed().model_dump_json(), NOW + 30]], [[1]])
        given = await queue(sql).adopt(worker="w1")
        assert given[0].available_at == NOW
        assert "restarted" in given[0].failures[0]
        assert sql.bound[10] is False


class TestWhatTheQueueLooksLike:
    """The dead letter and the counters an operator alerts on."""

    async def test_the_dead_letter_is_read_in_the_order_it_filled(self) -> None:
        sql = FakeSql([[work(state=WorkState.DEAD_LETTERED).model_dump_json()]])
        dead = await queue(sql).dead_letters(tenant="acme", limit=10)
        assert dead[0].id == "i1"
        assert sql.bound == ("acme", 10)

    async def test_the_stats_carry_the_counters_no_row_could_hold(self) -> None:
        sql = FakeSql([[4, 2, 1, 12.5, 6, 3]])
        stats = await queue(sql).stats(queue="urgent")
        assert stats.depth == 4
        assert stats.claimed == 2
        assert stats.dead_lettered == 1
        assert stats.oldest_age_seconds == 12.5
        assert stats.reaped == 6
        assert stats.duplicates_suppressed == 3

    async def test_a_queue_nothing_has_touched_counts_zero_rather_than_none(self) -> None:
        stats = await queue(FakeSql([[0, 0, 0, None, None, None]])).stats()
        assert stats.reaped == 0

    def test_the_policy_is_readable_so_a_worker_can_pace_its_heartbeat(self) -> None:
        assert queue(FakeSql()).policy.max_attempts == 3


class TestOneTransactionForBoth:
    """A run and the event it caused commit together, or neither does."""

    async def test_a_bound_store_writes_through_the_caller_s_session(self) -> None:
        session = FakeSql([[1]])
        await store(FakeSql()).bound(session).put_run(run())
        assert session.calls

    async def test_a_bound_queue_writes_through_the_caller_s_session(self) -> None:
        session = FakeSql([])
        await queue(FakeSql()).bound(session).enqueue(work())
        assert session.calls

    async def test_a_bound_store_does_not_retry_a_transaction_it_cannot_repair(self) -> None:
        session = FakeSql(fails=[DriverError("40001"), DriverError("40001")])
        with pytest.raises(StatePersistenceError) as refused:
            await store(FakeSql()).bound(session).get_run(KEY)
        assert refused.value.reason == "contended"
        assert len(session.calls) == 1

    async def test_a_bound_queue_does_not_retry_either(self) -> None:
        session = FakeSql(fails=[DriverError("40P01")])
        with pytest.raises(QueueUnavailableError):
            await queue(FakeSql()).bound(session).stats()
        assert len(session.calls) == 1

    async def test_a_bound_store_keeps_the_policy_it_was_configured_with(self) -> None:
        bound = queue(FakeSql(), policy=QueuePolicy(max_attempts=9)).bound(FakeSql())
        assert bound.policy.max_attempts == 9


class TestWhenTheDatabaseSaysNo:
    """Every driver failure becomes something the caller can act on."""

    async def test_a_contended_write_is_retried_and_then_named(self) -> None:
        sql = FakeSql(fails=[DriverError("40001")] * 3)
        with pytest.raises(StatePersistenceError) as refused:
            await store(sql).get_run(KEY)
        assert refused.value.reason == "contended"
        assert refused.value.retryable is True
        assert len(sql.calls) == 3

    async def test_a_deadlock_is_the_same_answer_under_a_different_code(self) -> None:
        sql = FakeSql(fails=[PgcodeError("40P01")] * 3)
        with pytest.raises(StatePersistenceError) as refused:
            await store(sql).get_run(KEY)
        assert refused.value.reason == "contended"

    async def test_a_statement_that_ran_out_of_time_is_retried_as_unavailable(self) -> None:
        sql = FakeSql(fails=[DriverError("57014")] * 3)
        with pytest.raises(StatePersistenceError) as refused:
            await store(sql).get_run(KEY)
        assert refused.value.reason == "unavailable"

    async def test_a_retry_that_succeeds_is_not_a_failure(self) -> None:
        sql = FakeSql([stored()], fails=[DriverError("40001")])
        assert await store(sql).get_run(KEY) is not None
        assert len(sql.calls) == 2

    async def test_an_exhausted_pool_is_not_retried_into_the_ground(self) -> None:
        sql = FakeSql(fails=[DriverError("53300")])
        with pytest.raises(PoolExhaustedError):
            await store(sql).get_run(KEY)
        assert len(sql.calls) == 1

    async def test_a_pool_that_raises_its_own_error_is_recognised_by_name(self) -> None:
        sql = FakeSql(fails=[PoolTimeoutError("no connection free")])
        with pytest.raises(PoolExhaustedError):
            await queue(sql).stats()

    async def test_a_missing_table_is_a_deployment_problem_rather_than_a_retry(self) -> None:
        sql = FakeSql(fails=[DriverError("42P01")])
        with pytest.raises(ConfigurationError, match="apply the platform's migrations"):
            await store(sql).get_run(KEY)
        assert len(sql.calls) == 1

    async def test_a_column_that_is_not_there_is_the_same_problem(self) -> None:
        sql = FakeSql(fails=[DriverError("42703")])
        with pytest.raises(ConfigurationError):
            await queue(sql).stats()

    async def test_an_unreachable_queue_is_never_a_silent_drop(self) -> None:
        sql = FakeSql(fails=[DriverError()] * 3)
        with pytest.raises(QueueUnavailableError) as refused:
            await queue(sql).enqueue(work())
        assert refused.value.retryable is True

    async def test_the_wait_between_attempts_doubles(self) -> None:
        clock = FakeClock(start=NOW)
        sql = FakeSql(fails=[DriverError()] * 3)
        adapter = PostgresStateStore(sql, clock=clock, settings=SETTINGS, entropy=lambda: 0.5)
        with pytest.raises(StatePersistenceError):
            await adapter.get_run(KEY)
        assert clock.now() == NOW + SETTINGS.backoff_seconds * 3
