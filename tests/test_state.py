"""What a state store promises, and what the record types promise before one is involved."""

from __future__ import annotations

import pytest

from tesserix_adk.core.errors import (
    StateConflictError,
    StateNotFoundError,
    StatePersistenceError,
)
from tesserix_adk.core.primitives import ToolCall, Usage
from tesserix_adk.core.run import RunState
from tesserix_adk.core.state import (
    RunRecord,
    SessionRecord,
    StateDelta,
    StateKey,
    StatePage,
    StateQuery,
    StateStore,
)
from tesserix_adk.runtime import MemoryStateStore
from tesserix_adk.testing import FakeClock, StateStoreConformance

TENANT = "acme"
NOTHING_SPENT = Usage(input_tokens=0, output_tokens=0)


def a_run(
    run_id: str = "r1",
    *,
    session_id: str | None = None,
    state: RunState = RunState.PENDING,
    version: int = 0,
    message_cursor: int = 0,
    iterations: int = 0,
    cost_micros: int = 0,
    usage: Usage = NOTHING_SPENT,
    pending_tool_calls: tuple[ToolCall, ...] = (),
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        tenant=TENANT,
        agent_name="planner",
        session_id=session_id,
        state=state,
        version=version,
        message_cursor=message_cursor,
        iterations=iterations,
        cost_micros=cost_micros,
        usage=usage,
        pending_tool_calls=pending_tool_calls,
    )


class TestWhatARecordKnowsAboutItself:
    def test_a_record_names_where_it_is_stored(self) -> None:
        assert a_run().key == StateKey(tenant=TENANT, id="r1")

    def test_a_session_names_where_it_is_stored(self) -> None:
        session = SessionRecord(session_id="s1", tenant=TENANT)
        assert session.key == StateKey(tenant=TENANT, id="s1")

    def test_a_run_with_an_unanswered_call_says_so(self) -> None:
        call = ToolCall(id="c1", name="search", arguments={})
        assert a_run(pending_tool_calls=(call,)).mid_tool_call is True

    def test_a_run_with_nothing_outstanding_says_so(self) -> None:
        assert a_run().mid_tool_call is False

    def test_a_new_record_is_at_version_zero(self) -> None:
        """Which is what makes a first write a create rather than an overwrite."""
        assert a_run().version == 0

    def test_a_key_cannot_drop_its_tenant(self) -> None:
        with pytest.raises(ValueError, match="at least 1 character"):
            StateKey(tenant="", id="r1")


class TestWhatIsWrittenDown:
    def test_a_secret_in_a_tool_argument_is_masked(self) -> None:
        call = ToolCall(id="c1", name="pay", arguments={"token": "sk-live-0123456789abcd"})
        scrubbed = a_run(pending_tool_calls=(call,)).scrubbed()
        assert "sk-live" not in str(scrubbed.pending_tool_calls[0].arguments)

    def test_an_argument_that_is_not_text_is_left_alone(self) -> None:
        call = ToolCall(id="c1", name="page", arguments={"limit": 20})
        scrubbed = a_run(pending_tool_calls=(call,)).scrubbed()
        assert scrubbed.pending_tool_calls[0].arguments == {"limit": 20}

    def test_ordinary_arguments_survive(self) -> None:
        call = ToolCall(id="c1", name="search", arguments={"query": "kyoto in autumn"})
        scrubbed = a_run(pending_tool_calls=(call,)).scrubbed()
        assert scrubbed.pending_tool_calls[0].arguments == {"query": "kyoto in autumn"}


class TestWhatADeltaDoes:
    def test_amounts_are_added_rather_than_set(self) -> None:
        record = a_run(iterations=2, cost_micros=10, message_cursor=4)
        added = StateDelta(iterations=1, cost_micros=5, messages_read=2).applied_to(record)
        assert (added.iterations, added.cost_micros, added.message_cursor) == (3, 15, 6)

    def test_usage_accumulates(self) -> None:
        record = a_run(usage=Usage(input_tokens=10, output_tokens=2))
        added = StateDelta(usage=Usage(input_tokens=5, output_tokens=1)).applied_to(record)
        assert (added.usage.input_tokens, added.usage.output_tokens) == (15, 3)

    def test_a_delta_that_names_no_usage_leaves_it_alone(self) -> None:
        record = a_run(usage=Usage(input_tokens=10, output_tokens=2))
        assert StateDelta(iterations=1).applied_to(record).usage.input_tokens == 10

    def test_a_delta_with_nothing_in_it_says_so(self) -> None:
        assert StateDelta().empty is True

    def test_a_delta_carrying_usage_alone_is_not_empty(self) -> None:
        assert StateDelta(usage=Usage(input_tokens=1, output_tokens=0)).empty is False

    def test_a_delta_carrying_an_amount_is_not_empty(self) -> None:
        assert StateDelta(messages_read=1).empty is False

    def test_a_delta_cannot_take_spend_away(self) -> None:
        """A negative amount would let a worker unspend what another worker spent."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            StateDelta(cost_micros=-1)


class TestWhatAPageIs:
    def test_an_empty_page_may_not_promise_more(self) -> None:
        """A reaper handed one would ask for the next page forever."""
        with pytest.raises(ValueError, match="empty page"):
            StatePage(cursor="12")

    def test_an_empty_page_with_no_cursor_is_fine(self) -> None:
        assert StatePage().cursor is None

    def test_a_query_will_not_ask_for_the_whole_table(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 1000"):
            StateQuery(tenant=TENANT, limit=5000)


class TestWhatAFailureCarries:
    def test_a_conflict_names_both_versions(self) -> None:
        refused = StateConflictError("moved", key="acme/r1", expected_version=2, actual_version=5)
        assert (refused.expected_version, refused.actual_version) == (2, 5)
        assert refused.details["key"] == "acme/r1"

    def test_a_missing_record_names_what_was_looked_for(self) -> None:
        missing = StateNotFoundError("gone", key="acme/r1", kind="run")
        assert missing.details == {"key": "acme/r1", "kind": "run"}

    def test_a_store_that_could_not_be_reached_is_worth_retrying(self) -> None:
        assert StatePersistenceError("down", store="Redis").retryable is True

    def test_a_record_too_large_is_not(self) -> None:
        """It stays too large, so retrying is a loop rather than a recovery."""
        assert StatePersistenceError("big", store="Redis", reason="too_large").retryable is False


class TestTheMemoryStore:
    def test_it_is_a_state_store(self) -> None:
        assert isinstance(MemoryStateStore(), StateStore)

    async def test_a_write_records_when_it_happened(self) -> None:
        store = MemoryStateStore(FakeClock(start=1000.0))
        assert (await store.put_run(a_run())).updated_at == 1000.0

    async def test_a_store_with_no_clock_records_nothing_about_when(self) -> None:
        """Ordering never depends on it, so a store without one is still correct."""
        assert (await MemoryStateStore().put_run(a_run())).updated_at == 0.0

    async def test_a_patch_that_changes_nothing_is_not_a_write(self) -> None:
        store = MemoryStateStore(FakeClock(start=1000.0))
        stored = await store.put_run(a_run())
        assert await store.patch_run(stored.key, StateDelta()) == stored

    async def test_a_rewritten_run_keeps_its_place_in_the_listing(self) -> None:
        """A record that moved to the end would be handed to a reaper twice."""
        store = MemoryStateStore()
        first = await store.put_run(a_run("r1"))
        await store.put_run(a_run("r2"))
        await store.put_run(first.model_copy(update={"iterations": 1}))
        page = await store.list_runs(StateQuery(tenant=TENANT))
        assert [record.run_id for record in page.records] == ["r1", "r2"]

    async def test_a_cursor_page_reports_where_to_carry_on(self) -> None:
        store = MemoryStateStore()
        await store.put_run(a_run("r1"))
        await store.put_run(a_run("r2"))
        page = await store.list_runs(StateQuery(tenant=TENANT, limit=1))
        assert page.cursor is not None
        assert [record.run_id for record in page.records] == ["r1"]

    async def test_a_deleted_run_no_longer_holds_its_session_open(self) -> None:
        store = MemoryStateStore()
        session = await store.put_session(SessionRecord(session_id="s1", tenant=TENANT))
        run = await store.put_run(a_run(session_id="s1", state=RunState.RUNNING))
        await store.delete_run(run.key)
        await store.delete_session(session.key)
        assert await store.get_session(session.key) is None

    async def test_a_cascade_over_a_session_with_no_runs_is_fine(self) -> None:
        store = MemoryStateStore()
        session = await store.put_session(SessionRecord(session_id="s1", tenant=TENANT))
        await store.delete_session(session.key, cascade=True)
        assert await store.get_session(session.key) is None

    async def test_one_tenant_s_session_is_not_another_s(self) -> None:
        store = MemoryStateStore()
        await store.put_session(SessionRecord(session_id="s1", tenant="one"))
        assert await store.get_session(StateKey(tenant="two", id="s1")) is None


class TestTheMemoryStoreConforms(StateStoreConformance):
    def make_store(self) -> StateStore:
        return MemoryStateStore(FakeClock())
