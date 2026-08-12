"""What the Redis state store and work queue send, and what they make of what comes back.

The Lua is verified against a real server by the integration lane, which runs both
conformance suites unchanged. These are the translation: the right key, the right
comparison, a reply turned into the right record — and the refusals a deployment meets
before it has written anything, like an instance configured as a cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.state import (
    RedisStateStore,
    RedisStoreSettings,
    RedisWorkQueue,
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
from tesserix_adk.core.queue import QueuePolicy, WorkItem, WorkPriority, WorkState
from tesserix_adk.core.run import RunState
from tesserix_adk.core.state import (
    RunRecord,
    SessionRecord,
    StateDelta,
    StateKey,
    StateQuery,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Mapping

NOW = 1_000.0
SETTINGS = RedisStoreSettings(dsn=SecretStr("redis://:s3cret@cache:6379/0"))
KEY = StateKey(tenant="acme", id="run_1")

HEALTHY = {"maxmemory-policy": "noeviction", "appendonly": "yes", "save": ""}


class FakeRedis:
    """Answers with whatever the test says the server said, and records what was asked."""

    def __init__(self, *replies: Any, config: Mapping[str, str] | None = None) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.fails: list[Exception | None] = []
        self.config = dict(HEALTHY if config is None else config)

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        failure = self.fails.pop(0) if self.fails else None
        if failure is not None:
            raise failure
        self.calls.append((script, numkeys, args))
        return self.replies.pop(0) if self.replies else None

    async def config_get(self, pattern: str) -> Mapping[str, str]:
        failure = self.fails.pop(0) if self.fails else None
        if failure is not None:
            raise failure
        self.calls.append(("CONFIG GET", 0, (pattern,)))
        return self.config

    @property
    def keys(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[:numkeys]

    @property
    def argv(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[numkeys:]


class PoolTimeoutError(Exception):
    """What a driver raises when every pooled connection is already in use."""


def store(client: FakeRedis, **kwargs: Any) -> RedisStateStore:
    """A Redis state store on a clock a test can read back."""
    kwargs.setdefault("settings", SETTINGS)
    return RedisStateStore(client, clock=FakeClock(start=NOW), entropy=lambda: 0.0, **kwargs)


def queue(client: FakeRedis, **kwargs: Any) -> RedisWorkQueue:
    """A Redis work queue whose policy a test can pin."""
    kwargs.setdefault("settings", SETTINGS)
    kwargs.setdefault("policy", QueuePolicy(max_attempts=3, backoff_seconds=2.0))
    return RedisWorkQueue(client, clock=FakeClock(start=NOW), entropy=lambda: 0.0, **kwargs)


def run(**fields: Any) -> RunRecord:
    """A run as a caller hands it over."""
    return RunRecord(run_id="run_1", tenant="acme", agent_name="planner", **fields)


def stored(record: RunRecord, **counters: int) -> list[Any]:
    """A run as the server holds it: the blob, and the six numbers beside it."""
    numbers = {"version": 1, "input": 0, "output": 0, "cost": 0, "iterations": 0, "cursor": 0}
    numbers.update(counters)
    return [record.model_dump_json(), [str(value) for value in numbers.values()]]


def work(**fields: Any) -> WorkItem:
    """An item as a caller hands it over."""
    return WorkItem(id="i1", tenant="acme", **fields)


def held(item: WorkItem, worker: str = "w1") -> WorkItem:
    """The item as it looks while a worker holds it."""
    return item.model_copy(
        update={
            "state": WorkState.CLAIMED,
            "worker": worker,
            "lease_expires_at": NOW + 30.0,
            "first_claimed_at": NOW,
        }
    )


class TestRefusingAnInstanceThatCannotHoldState:
    async def test_a_server_that_evicts_is_refused_before_anything_is_written(self) -> None:
        """A cache answers a lost run and a run that never started identically."""
        client = FakeRedis(config={"maxmemory-policy": "allkeys-lru", "appendonly": "yes"})
        with pytest.raises(ConfigurationError, match="noeviction"):
            await RedisStateStore.open(client, clock=FakeClock(), settings=SETTINGS)

    @pytest.mark.parametrize(
        "policy", ["allkeys-lru", "allkeys-lfu", "allkeys-random", "volatile-ttl"]
    )
    async def test_every_eviction_policy_is_refused(self, policy: str) -> None:
        client = FakeRedis(config={"maxmemory-policy": policy, "appendonly": "yes"})
        with pytest.raises(ConfigurationError):
            await store(client).preflight()

    async def test_a_server_that_keeps_nothing_across_a_restart_is_refused(self) -> None:
        client = FakeRedis(config={"maxmemory-policy": "noeviction", "appendonly": "no"})
        with pytest.raises(ConfigurationError, match="persistence"):
            await store(client).preflight()

    async def test_save_points_count_as_persistence(self) -> None:
        client = FakeRedis(
            config={"maxmemory-policy": "noeviction", "appendonly": "no", "save": "900 1"}
        )
        await store(client).preflight()

    async def test_a_queue_that_does_not_need_durability_is_not_checked(self) -> None:
        """Work that can be recreated elsewhere may live on a cache. State may not."""
        client = FakeRedis(config={"maxmemory-policy": "allkeys-lru"})
        settings = SETTINGS.model_copy(update={"durable": False})
        await RedisWorkQueue.open(client, clock=FakeClock(), settings=settings)

        assert client.calls == []

    async def test_a_store_told_not_to_be_durable_is_not_checked_either(self) -> None:
        """Dev and tests run against whatever is on the machine; production sets durable."""
        client = FakeRedis(config={"maxmemory-policy": "allkeys-lru"})
        settings = SETTINGS.model_copy(update={"durable": False})
        await store(client, settings=settings).preflight()

        assert client.calls == []

    async def test_a_healthy_server_opens(self) -> None:
        opened = await RedisStateStore.open(FakeRedis(), clock=FakeClock(), settings=SETTINGS)

        assert isinstance(opened, RedisStateStore)

    async def test_the_queue_checks_the_same_thing(self) -> None:
        client = FakeRedis(config={"maxmemory-policy": "allkeys-lfu"})
        with pytest.raises(ConfigurationError, match="RedisWorkQueue"):
            await RedisWorkQueue.open(client, clock=FakeClock(), settings=SETTINGS)


class TestWhereTheKeysGo:
    def test_the_tenant_is_part_of_the_key_and_cannot_be_imitated(self) -> None:
        """A tenant called `a:b` and a tenant called `a` must not build the same key."""
        client = FakeRedis(None, None)
        first = store(client)
        assert first._run(StateKey(tenant="a:b", id="c")) != first._run(
            StateKey(tenant="a", id="b:c")
        )

    def test_the_tenant_is_measured_in_bytes_so_lua_agrees_with_python(self) -> None:
        """Lua's `#` counts bytes; a name outside ASCII would otherwise build two keys."""
        assert (
            store(FakeRedis())
            ._run(StateKey(tenant="भारत", id="r"))
            .startswith("adk:state:{12:भारत}")
        )

    def test_the_whole_namespace_is_one_cluster_hash_tag(self) -> None:
        """A claim reads several tenants' keys in one script, so they share a slot."""
        assert queue(FakeRedis())._item("2:ac:i1").startswith("{adk:queue}")


class TestReadingARun:
    async def test_a_run_that_was_never_written_is_none(self) -> None:
        assert await store(FakeRedis(None)).get_run(KEY) is None

    async def test_the_counters_are_added_to_what_the_blob_holds(self) -> None:
        """The blob is what a worker set; the hash is what everything since has added."""
        client = FakeRedis(stored(run(), version=4, input=30, output=7, cost=90, iterations=2))

        found = await store(client).get_run(KEY)
        assert found is not None
        assert (found.version, found.usage.input_tokens, found.cost_micros) == (4, 30, 90)
        assert found.iterations == 2

    async def test_a_run_whose_counters_were_never_written_reads_as_zero(self) -> None:
        client = FakeRedis([run().model_dump_json(), [None, None, None, None, None, None]])

        found = await store(client).get_run(KEY)
        assert found is not None
        assert found.version == 0

    async def test_the_reply_may_arrive_as_bytes(self) -> None:
        client = FakeRedis([run().model_dump_json().encode(), [b"2", b"0", b"0", b"0", b"0", b"0"]])

        found = await store(client).get_run(KEY)
        assert found is not None
        assert found.version == 2


class TestWritingARun:
    async def test_a_write_states_the_version_it_read(self) -> None:
        client = FakeRedis(["ok", "1"])
        await store(client).put_run(run(version=0))

        assert client.argv[0] == "0"

    async def test_the_write_lands_at_that_version_plus_one(self) -> None:
        client = FakeRedis(["ok", "3"])
        written = await store(client).put_run(run(version=2))

        assert written.version == 3

    async def test_a_version_that_moved_is_refused_with_both_numbers(self) -> None:
        """A failover to a stale replica, or another worker: either way, not a lost update."""
        client = FakeRedis(["conflict", "5"])
        with pytest.raises(StateConflictError) as refusal:
            await store(client).put_run(run(version=2))

        assert (refusal.value.expected_version, refusal.value.actual_version) == (2, 5)
        assert refusal.value.key == "acme/run_1"

    async def test_the_counters_are_written_beside_the_blob_and_not_in_it(self) -> None:
        """Two workers adding tokens both land; two workers writing a total do not."""
        client = FakeRedis(["ok", "1"])
        record = run(usage=Usage(input_tokens=11, output_tokens=3), cost_micros=42, iterations=1)
        await store(client).put_run(record)

        assert client.argv[2:7] == ("11", "3", "42", "1", "0")
        assert '"input_tokens":0' in client.argv[1]

    async def test_what_is_written_down_is_scrubbed(self) -> None:
        client = FakeRedis(["ok", "1"])
        call = ToolCall(id="c1", name="pay", arguments={"who": "ada@example.com"})
        await store(client).put_run(run(pending_tool_calls=(call,)))

        assert "ada@example.com" not in client.argv[1]

    async def test_a_run_in_a_session_joins_that_session(self) -> None:
        client = FakeRedis(["ok", "1"])
        await store(client).put_run(run(session_id="s1"))

        assert client.argv[8] == "s1"
        assert client.keys[4].endswith(":session:2:s1:runs")

    async def test_a_run_in_no_session_says_so(self) -> None:
        client = FakeRedis(["ok", "1"])
        await store(client).put_run(run())

        assert client.argv[8] == ""

    async def test_a_record_too_large_names_the_store_that_would_take_it(self) -> None:
        client = FakeRedis(["ok", "1"])
        settings = SETTINGS.model_copy(update={"max_value_bytes": 32})
        with pytest.raises(StatePersistenceError, match="PostgreSQL") as refusal:
            await store(client, settings=settings).put_run(run())

        assert refusal.value.reason == "too_large"
        assert not refusal.value.retryable
        assert client.calls == []


class TestPatchingARun:
    async def test_the_amounts_go_to_the_server_to_be_added_there(self) -> None:
        client = FakeRedis(stored(run(), version=2, input=5, iterations=1))
        delta = StateDelta(usage=Usage(input_tokens=5, output_tokens=0), iterations=1)
        patched = await store(client).patch_run(KEY, delta)

        assert client.argv == ("5", "0", "0", "1", "0")
        assert (patched.usage.input_tokens, patched.iterations, patched.version) == (5, 1, 2)

    async def test_an_empty_delta_still_asks_for_the_record(self) -> None:
        client = FakeRedis(stored(run(), version=2))
        assert (await store(client).patch_run(KEY, StateDelta())).version == 2

    async def test_patching_a_run_that_is_not_there_is_refused(self) -> None:
        """There is nothing to add to, and adding to nothing would create a phantom run."""
        with pytest.raises(StateNotFoundError, match="acme/run_1"):
            await store(FakeRedis(None)).patch_run(KEY, StateDelta(iterations=1))


class TestRemovingARun:
    async def test_the_blob_the_counters_and_the_listing_all_go(self) -> None:
        client = FakeRedis(1)
        await store(client).delete_run(KEY)

        assert len(client.keys) == 3
        assert client.argv == ("run_1", "adk:state:{4:acme}")


class TestListingRuns:
    async def test_a_full_page_carries_a_cursor_to_the_next_one(self) -> None:
        client = FakeRedis([["1", stored(run())], ["2", stored(run())]])
        page = await store(client).list_runs(StateQuery(tenant="acme", limit=1))

        assert len(page.records) == 1
        assert page.cursor == "1"

    async def test_the_last_page_carries_none(self) -> None:
        """A cursor onto nothing is a reaper that never stops."""
        client = FakeRedis([["1", stored(run())]])
        page = await store(client).list_runs(StateQuery(tenant="acme", limit=5))

        assert page.cursor is None

    async def test_an_empty_listing_is_a_page_of_nothing(self) -> None:
        page = await store(FakeRedis(None)).list_runs(StateQuery(tenant="acme"))

        assert (page.records, page.cursor) == ((), None)

    async def test_a_cursor_carries_on_after_what_it_named(self) -> None:
        client = FakeRedis([])
        await store(client).list_runs(StateQuery(tenant="acme", cursor="7"))

        assert client.argv[1] == "(7"

    async def test_the_first_page_starts_at_the_beginning(self) -> None:
        client = FakeRedis([])
        await store(client).list_runs(StateQuery(tenant="acme"))

        assert client.argv[1] == "-inf"

    async def test_the_filters_go_to_the_server(self) -> None:
        client = FakeRedis([])
        await store(client).list_runs(
            StateQuery(tenant="acme", state=RunState.RUNNING, updated_before=900.0)
        )

        assert client.argv[3:] == ("running", "900.0")

    async def test_an_unfiltered_listing_says_so(self) -> None:
        client = FakeRedis([])
        await store(client).list_runs(StateQuery(tenant="acme"))

        assert client.argv[3:] == ("", "")


class TestSessions:
    async def test_a_session_that_was_never_written_is_none(self) -> None:
        assert await store(FakeRedis(None)).get_session(KEY) is None

    async def test_a_written_session_comes_back(self) -> None:
        record = SessionRecord(session_id="s1", tenant="acme", runs=("r1",))
        client = FakeRedis(record.model_dump_json())

        found = await store(client).get_session(KEY)
        assert found is not None
        assert found.runs == ("r1",)

    async def test_a_session_write_lands_at_its_read_version_plus_one(self) -> None:
        client = FakeRedis(["ok", "2"])
        written = await store(client).put_session(
            SessionRecord(session_id="s1", tenant="acme", version=1)
        )

        assert (written.version, written.updated_at) == (2, NOW)

    async def test_a_session_version_that_moved_is_refused(self) -> None:
        client = FakeRedis(["conflict", "4"])
        with pytest.raises(StateConflictError):
            await store(client).put_session(SessionRecord(session_id="s1", tenant="acme"))

    async def test_a_session_expires_where_the_deployment_asked_it_to(self) -> None:
        client = FakeRedis(["ok", "1"])
        await store(client, session_ttl_seconds=60.0).put_session(
            SessionRecord(session_id="s1", tenant="acme")
        )

        assert client.argv[2] == "60000"

    async def test_a_session_without_a_ttl_stays_until_something_deletes_it(self) -> None:
        client = FakeRedis(["ok", "1"])
        await store(client).put_session(SessionRecord(session_id="s1", tenant="acme"))

        assert client.argv[2] == "0"

    async def test_deleting_a_session_that_still_has_live_runs_is_refused(self) -> None:
        """A live run under a deleted session is work nothing will ever reap."""
        client = FakeRedis(["live", "r1", "r2"])
        with pytest.raises(StateInUseError) as refusal:
            await store(client).delete_session(KEY)

        assert refusal.value.live_runs == ("r1", "r2")

    async def test_a_cascade_says_so_to_the_server(self) -> None:
        client = FakeRedis(["ok"])
        await store(client).delete_session(KEY, cascade=True)

        assert client.argv[0] == "1"

    async def test_the_terminal_states_are_the_ones_the_kit_knows(self) -> None:
        client = FakeRedis(["ok"])
        await store(client).delete_session(KEY)

        assert ",completed," in client.argv[2]
        assert ",running," not in client.argv[2]


class TestWhenTheServerIsNotThere:
    async def test_a_failed_write_is_never_a_silent_success(self) -> None:
        client = FakeRedis(["ok", "1"])
        client.fails = [ConnectionError("gone")] * 3
        with pytest.raises(StatePersistenceError) as refusal:
            await store(client).put_run(run())

        assert refusal.value.retryable

    async def test_a_transient_failure_is_waited_out(self) -> None:
        client = FakeRedis(["ok", "1"])
        client.fails = [ConnectionError("blip")]

        assert (await store(client).put_run(run())).version == 1

    async def test_the_wait_is_jittered_because_every_worker_restarts_together(self) -> None:
        client = FakeRedis(stored(run()))
        client.fails = [ConnectionError("blip")]
        clock = FakeClock(start=NOW)
        held_store = RedisStateStore(client, clock=clock, settings=SETTINGS, entropy=lambda: 1.0)
        await held_store.get_run(KEY)

        assert clock.slept == [SETTINGS.backoff_seconds * 1.5]

    async def test_an_exhausted_pool_is_reported_as_itself(self) -> None:
        """A pool with nothing left is a deployment that needs more, not a dead server."""
        client = FakeRedis(None)
        client.fails = [PoolTimeoutError("no connections")]
        with pytest.raises(PoolExhaustedError):
            await store(client).get_run(KEY)

    async def test_an_unreachable_queue_refuses_rather_than_dropping_the_work(self) -> None:
        client = FakeRedis(None)
        client.fails = [ConnectionError("gone")] * 3
        with pytest.raises(QueueUnavailableError) as refusal:
            await queue(client).enqueue(work())

        assert refusal.value.retryable


class TestEnqueueing:
    async def test_an_item_is_stamped_with_the_store_s_clock(self) -> None:
        client = FakeRedis(None)
        placed = await queue(client).enqueue(work())

        assert (placed.enqueued_at, placed.available_at) == (NOW, NOW)

    async def test_the_item_carries_its_own_times_where_it_has_them(self) -> None:
        client = FakeRedis(None)
        placed = await queue(client).enqueue(work(enqueued_at=1.0, available_at=2.0))

        assert (placed.enqueued_at, placed.available_at) == (1.0, 2.0)

    async def test_a_live_duplicate_comes_back_instead_of_a_second_item(self) -> None:
        first = work(dedupe_key="nightly").model_copy(update={"enqueued_at": 1.0})
        client = FakeRedis(first.model_dump_json())

        assert (await queue(client).enqueue(work(dedupe_key="nightly"))).enqueued_at == 1.0

    async def test_a_dedupe_key_is_named_to_the_server(self) -> None:
        client = FakeRedis(None)
        await queue(client).enqueue(work(dedupe_key="nightly"))

        assert client.argv[6] == "1"
        assert client.keys[5].endswith(":dedupe:4:acme:nightly")

    async def test_an_item_without_one_dedupes_on_nothing_but_itself(self) -> None:
        client = FakeRedis(None)
        await queue(client).enqueue(work())

        assert client.argv[6] == "0"

    async def test_a_payload_too_large_is_refused_before_it_blocks_the_server(self) -> None:
        settings = SETTINGS.model_copy(update={"max_value_bytes": 32})
        with pytest.raises(StatePersistenceError, match="claim check"):
            await queue(FakeRedis(None), settings=settings).enqueue(work())


class TestClaiming:
    async def test_an_empty_queue_hands_out_nothing(self) -> None:
        assert await queue(FakeRedis(None)).claim(worker="w1") is None

    async def test_a_claim_comes_back_held_by_the_worker_that_asked(self) -> None:
        client = FakeRedis(["4:acme:i1", work().model_dump_json()], "ok")
        claimed = await queue(client).claim(worker="w1")

        assert claimed is not None
        assert (claimed.worker, claimed.state) == ("w1", WorkState.CLAIMED)
        assert claimed.lease_expires_at == NOW + 30.0

    async def test_the_lease_the_caller_asked_for_is_the_one_written(self) -> None:
        client = FakeRedis(["4:acme:i1", work().model_dump_json()], "ok")
        claimed = await queue(client).claim(worker="w1", lease_seconds=5.0)

        assert claimed is not None
        assert claimed.lease_expires_at == NOW + 5.0

    async def test_the_deadline_the_server_enforces_is_the_one_it_was_given(self) -> None:
        client = FakeRedis(["4:acme:i1", work().model_dump_json()], "ok")
        await queue(client).claim(worker="w1", queue="slow")

        assert client.calls[0][2][8] == str(NOW + 30.0)

    async def test_a_named_queue_is_asked_for_by_name(self) -> None:
        client = FakeRedis(None)
        await queue(client).claim(worker="w1", queue="slow")

        assert client.keys[0].endswith(":tenants:4:slow")


class TestHeartbeats:
    async def test_a_live_claim_is_extended(self) -> None:
        client = FakeRedis(["ok", held(work()).model_dump_json()], "ok")
        renewed = await queue(client).heartbeat("i1", tenant="acme", worker="w1")

        assert renewed.lease_expires_at == NOW + 30.0

    async def test_a_claim_renewed_past_the_bound_is_taken_away(self) -> None:
        """A stuck run is indistinguishable from a busy one to everything but this."""
        stuck = held(work()).model_copy(update={"first_claimed_at": NOW - 4_000.0})
        client = FakeRedis(["ok", stuck.model_dump_json()])
        with pytest.raises(LeaseLostError) as refusal:
            await queue(client).heartbeat("i1", tenant="acme", worker="w1")

        assert refusal.value.reason == "capped"

    async def test_an_item_nobody_has_ever_heard_of_is_refused(self) -> None:
        with pytest.raises(WorkItemNotFoundError):
            await queue(FakeRedis(["missing"])).heartbeat("i1", tenant="acme", worker="w1")

    async def test_a_claim_another_worker_now_holds_names_the_holder(self) -> None:
        client = FakeRedis(["lost", "w2", "taken"])
        with pytest.raises(LeaseLostError) as refusal:
            await queue(client).heartbeat("i1", tenant="acme", worker="w1")

        assert (refusal.value.holder, refusal.value.reason) == ("w2", "taken")

    async def test_a_claim_that_lapsed_is_refused_with_nobody_holding_it(self) -> None:
        client = FakeRedis(["lost", None, "expired"])
        with pytest.raises(LeaseLostError) as refusal:
            await queue(client).heartbeat("i1", tenant="acme", worker="w1")

        assert refusal.value.holder is None


class TestFinishingWork:
    async def test_a_completed_item_is_held_by_nobody(self) -> None:
        client = FakeRedis(["ok", held(work()).model_dump_json()], "ok")
        done = await queue(client).complete("i1", tenant="acme", worker="w1")

        assert (done.state, done.worker) == (WorkState.COMPLETED, None)

    async def test_a_failure_comes_back_for_another_attempt_after_a_backoff(self) -> None:
        client = FakeRedis(["ok", held(work()).model_dump_json()], "ok")
        given = await queue(client).fail("i1", tenant="acme", worker="w1", error="timeout")

        assert (given.state, given.attempts) == (WorkState.QUEUED, 1)
        assert given.available_at == NOW + 2.0
        assert given.failures == ("timeout",)

    async def test_an_item_that_cannot_succeed_does_not_wait_to_fail_again(self) -> None:
        client = FakeRedis(["ok", held(work()).model_dump_json()], "ok")
        given = await queue(client).fail(
            "i1", tenant="acme", worker="w1", error="malformed", retryable=False
        )

        assert given.state == WorkState.DEAD_LETTERED

    async def test_the_last_attempt_goes_to_the_dead_letter(self) -> None:
        item = held(work(attempts=2))
        client = FakeRedis(["ok", item.model_dump_json()], "ok")
        given = await queue(client).fail("i1", tenant="acme", worker="w1", error="timeout")

        assert (given.state, given.attempts) == (WorkState.DEAD_LETTERED, 3)


class TestReaping:
    async def test_a_lapsed_claim_comes_back_with_its_attempt_counted(self) -> None:
        client = FakeRedis([[held(work()).model_dump_json(), "1030"]], "ok")
        moved = await queue(client).reap()

        assert [(item.state, item.attempts) for item in moved] == [(WorkState.QUEUED, 1)]
        assert moved[0].failures == ("lease lapsed with the item unfinished",)

    async def test_an_item_whose_lease_moved_since_the_scan_is_left_alone(self) -> None:
        """The worker was alive after all: settling it now would take live work away."""
        client = FakeRedis([[held(work()).model_dump_json(), "1030"]], "moved")

        assert await queue(client).reap() == ()

    async def test_the_scan_is_fenced_on_the_lease_it_saw(self) -> None:
        client = FakeRedis([[held(work()).model_dump_json(), "1030"]], "ok")
        await queue(client).reap()

        assert client.argv[7] == "1030"
        assert client.argv[8] == "1"

    async def test_an_empty_sweep_moves_nothing(self) -> None:
        assert await queue(FakeRedis(None)).reap() == ()

    async def test_a_restarted_worker_gives_its_work_back_without_a_backoff(self) -> None:
        """A rolled pod is not a poisonous item; the attempt counts, the wait does not."""
        client = FakeRedis([[held(work()).model_dump_json(), "1030"]], "ok")
        moved = await queue(client).adopt(worker="w1")

        assert [(item.available_at, item.attempts) for item in moved] == [(NOW, 1)]
        assert moved[0].failures == ("the worker holding it restarted",)

    async def test_adoption_is_not_counted_as_a_reap(self) -> None:
        client = FakeRedis([[held(work()).model_dump_json(), "1030"]], "ok")
        await queue(client).adopt(worker="w1")

        assert client.argv[8] == "0"


class TestTheDeadLetter:
    async def test_the_items_come_back_with_what_went_wrong(self) -> None:
        dead = work(state=WorkState.DEAD_LETTERED, attempts=3, failures=("timeout", "timeout"))
        client = FakeRedis([dead.model_dump_json()])

        found = await queue(client).dead_letters(tenant="acme")
        assert found[0].failures == ("timeout", "timeout")

    async def test_a_limit_is_a_range_the_server_understands(self) -> None:
        client = FakeRedis([])
        await queue(client).dead_letters(tenant="acme", limit=10)

        assert client.argv[1] == "9"

    async def test_an_item_deleted_between_the_index_and_the_read_is_skipped(self) -> None:
        client = FakeRedis([None])

        assert await queue(client).dead_letters(tenant="acme") == ()


class TestWhatTheQueueLooksLike:
    async def test_the_gauges_and_the_counters_come_back_together(self) -> None:
        client = FakeRedis([b"3", b"1", b"940.5", b"2", b"7", b"4"])
        stats = await queue(client).stats()

        assert (stats.depth, stats.claimed, stats.dead_lettered) == (3, 1, 2)
        assert (stats.reaped, stats.duplicates_suppressed) == (7, 4)
        assert stats.oldest_age_seconds == 59.5

    async def test_an_empty_queue_has_no_oldest_item(self) -> None:
        client = FakeRedis([b"0", b"0", b"0", b"0", b"0", b"0"])

        assert (await queue(client).stats()).oldest_age_seconds == 0.0

    async def test_the_dead_letter_is_counted_per_queue(self) -> None:
        client = FakeRedis([b"0", b"0", b"0", b"0", b"0", b"0"])
        await queue(client).stats(queue="slow")

        assert client.argv == ("dead:4:slow",)

    def test_the_queue_says_what_it_will_do_to_an_item(self) -> None:
        assert queue(FakeRedis()).policy.max_attempts == 3


class TestPriorityAndFairness:
    async def test_the_priority_travels_with_the_item(self) -> None:
        """Ordering is the server's, but it can only order what it was told."""
        client = FakeRedis(None)
        await queue(client).enqueue(work(priority=WorkPriority.URGENT))

        assert '"priority":40' in client.argv[3]
