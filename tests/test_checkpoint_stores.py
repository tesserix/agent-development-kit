"""What the durable checkpoint and lease stores send, and what they make of what comes back.

The Lua and the SQL are exercised against a real server by the durability suite in
`tests/integration/`. These are the translation: the right key, the right statement, and a
reply turned into the right frontier or the right refusal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.adapters import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_CHECKPOINT_SCHEMA,
    CheckpointTables,
    PostgresCheckpointStore,
    PostgresLeaseStore,
    RedisCheckpointStore,
    RedisLeaseStore,
)
from tesserix_adk.core import (
    Checkpoint,
    CheckpointBoundary,
    ConfigurationError,
    Message,
    RunLease,
    RunLeaseError,
    StatePersistenceError,
    TextPart,
)

if TYPE_CHECKING:
    from tesserix_adk.core import CheckpointStore, LeaseStore

pytestmark = pytest.mark.anyio


class FakeRedis:
    """Records what was evaluated and answers with whatever the test says the server said."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        self.calls.append((script, numkeys, args))
        return self.replies.pop(0) if self.replies else None

    @property
    def keys(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[:numkeys]

    @property
    def argv(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[numkeys:]


class FakeSql:
    """The same, for a database."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, statement: str, *args: Any) -> list[list[Any]]:
        self.statements.append((statement, args))
        return self.replies.pop(0) if self.replies else []

    @property
    def args(self) -> tuple[Any, ...]:
        return self.statements[-1][1]


def frontier(**overrides: object) -> Checkpoint:
    """A frontier with enough on it to be worth resuming."""
    fields: dict[str, object] = {
        "run_id": "r1",
        "tenant": "acme",
        "agent_name": "booking",
        "boundary": CheckpointBoundary.BEFORE_APPROVAL,
        "messages": (Message(role="user", content=[TextPart(text="book the 18:40")]),),
        "iterations": 3,
    }
    return Checkpoint(**(fields | overrides))  # type: ignore[arg-type]


class TestTheRedisCheckpointStore:
    async def test_the_frontier_is_keyed_by_tenant_and_run(self) -> None:
        client = FakeRedis(1)

        await RedisCheckpointStore(client).put(frontier())

        assert client.keys == ("adk:checkpoint:acme:r1",)

    async def test_nothing_written_reads_back_as_nothing(self) -> None:
        client = FakeRedis(None)

        assert await RedisCheckpointStore(client).latest("r1", tenant="acme") is None

    async def test_what_was_written_reads_back_as_the_frontier(self) -> None:
        client = FakeRedis(frontier().model_dump_json().encode())

        read = await RedisCheckpointStore(client).latest("r1", tenant="acme")

        assert read is not None
        assert read.iterations == 3

    async def test_the_key_carries_no_expiry(self) -> None:
        """A frontier that expires is a run that stops being resumable at a time nobody chose."""
        client = FakeRedis(1)

        await RedisCheckpointStore(client).put(frontier())

        assert "EX" not in client.calls[-1][0]

    async def test_forgetting_deletes_the_key(self) -> None:
        client = FakeRedis(1)

        await RedisCheckpointStore(client).forget("r1", tenant="acme")

        assert client.keys == ("adk:checkpoint:acme:r1",)

    async def test_a_frontier_too_large_is_refused_rather_than_truncated(self) -> None:
        client = FakeRedis()

        with pytest.raises(StatePersistenceError) as refused:
            await RedisCheckpointStore(client, max_value_bytes=64).put(frontier())

        assert refused.value.reason == "too_large"
        assert client.calls == []


class TestTheRedisLeaseStore:
    async def test_taking_a_free_run_returns_the_fence_the_server_gave(self) -> None:
        client = FakeRedis([b"ok", b"1", b"60"])

        lease = await RedisLeaseStore(client).acquire(
            "r1", tenant="acme", holder="w1", ttl_seconds=60.0
        )

        assert lease.fence == 1
        assert lease.expires_at == 60.0
        assert lease.holder == "w1"

    async def test_a_run_someone_else_holds_is_refused_with_their_name(self) -> None:
        client = FakeRedis([b"held", b"w1", b"90", b"4"])

        with pytest.raises(RunLeaseError) as refused:
            await RedisLeaseStore(client).acquire(
                "r1", tenant="acme", holder="w2", ttl_seconds=60.0
            )

        assert refused.value.holder == "w1"
        assert refused.value.requested_by == "w2"
        assert refused.value.fence == 4

    async def test_expiry_is_read_off_the_servers_own_clock(self) -> None:
        """A worker's clock is never sent, so a fast one cannot take a live lease."""
        client = FakeRedis([b"ok", b"1", b"60"])

        await RedisLeaseStore(client).acquire("r1", tenant="acme", holder="w1", ttl_seconds=60.0)

        assert "TIME" in client.calls[-1][0]
        assert client.argv == ("w1", "60.0")

    async def test_renewal_moves_the_expiry_and_keeps_the_fence(self) -> None:
        client = FakeRedis([b"ok", b"120"])
        held = RunLease(run_id="r1", tenant="acme", holder="w1", fence=2, expires_at=60.0)

        renewed = await RedisLeaseStore(client).renew(held, ttl_seconds=60.0)

        assert renewed.expires_at == 120.0
        assert renewed.fence == 2

    async def test_renewing_a_lease_that_has_moved_on_is_refused(self) -> None:
        client = FakeRedis([b"lost", b"w2"])
        held = RunLease(run_id="r1", tenant="acme", holder="w1", fence=2)

        with pytest.raises(RunLeaseError) as refused:
            await RedisLeaseStore(client).renew(held, ttl_seconds=60.0)

        assert refused.value.holder == "w2"

    async def test_releasing_states_the_fence_it_is_releasing(self) -> None:
        client = FakeRedis(1)
        held = RunLease(run_id="r1", tenant="acme", holder="w1", fence=2)

        await RedisLeaseStore(client).release(held)

        assert client.argv == ("w1", "2")

    async def test_a_run_nobody_holds_reads_back_as_nothing(self) -> None:
        client = FakeRedis(None)

        assert await RedisLeaseStore(client).held("r1", tenant="acme") is None

    async def test_a_held_run_reads_back_with_its_holder_and_fence(self) -> None:
        client = FakeRedis([b"w1", b"3", b"90"])

        lease = await RedisLeaseStore(client).held("r1", tenant="acme")

        assert lease is not None
        assert (lease.holder, lease.fence, lease.expires_at) == ("w1", 3, 90.0)


class TestThePostgresCheckpointStore:
    async def test_the_frontier_is_upserted_with_its_format_version(self) -> None:
        executor = FakeSql([[1]])

        await PostgresCheckpointStore(executor).put(frontier())

        assert executor.args[:2] == ("r1", "acme")
        assert executor.args[2] == frontier().format_version

    async def test_nothing_written_reads_back_as_nothing(self) -> None:
        executor = FakeSql([])

        assert await PostgresCheckpointStore(executor).latest("r1", tenant="acme") is None

    async def test_a_null_payload_reads_back_as_nothing(self) -> None:
        executor = FakeSql([[None]])

        assert await PostgresCheckpointStore(executor).latest("r1", tenant="acme") is None

    async def test_what_was_written_reads_back_as_the_frontier(self) -> None:
        executor = FakeSql([[frontier().model_dump_json()]])

        read = await PostgresCheckpointStore(executor).latest("r1", tenant="acme")

        assert read is not None
        assert read.iterations == 3

    async def test_forgetting_deletes_the_row(self) -> None:
        executor = FakeSql([[1]])

        await PostgresCheckpointStore(executor).forget("r1", tenant="acme")

        assert executor.args == ("r1", "acme")

    async def test_opening_against_the_shape_it_writes_is_fine(self) -> None:
        executor = FakeSql([[CHECKPOINT_SCHEMA_VERSION]])

        assert await PostgresCheckpointStore.open(executor) is not None

    async def test_a_schema_that_has_moved_is_refused_at_startup(self) -> None:
        executor = FakeSql([[CHECKPOINT_SCHEMA_VERSION + 1]])

        with pytest.raises(ConfigurationError, match="schema is version"):
            await PostgresCheckpointStore.open(executor)

    async def test_a_database_with_no_schema_row_is_refused(self) -> None:
        executor = FakeSql([])

        with pytest.raises(ConfigurationError):
            await PostgresCheckpointStore(executor).verify()

    def test_a_table_name_that_could_carry_sql_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="plain table identifier"):
            CheckpointTables(checkpoints="adk_checkpoints; DROP TABLE users")

    def test_the_expected_shape_is_published_for_the_migration_to_own(self) -> None:
        assert "CREATE TABLE adk_checkpoints" in EXPECTED_CHECKPOINT_SCHEMA
        assert "CREATE TABLE adk_run_leases" in EXPECTED_CHECKPOINT_SCHEMA


class TestThePostgresLeaseStore:
    async def test_taking_a_free_run_returns_the_row_the_insert_made(self) -> None:
        executor = FakeSql([["w1", 1, 60.0]])

        lease = await PostgresLeaseStore(executor).acquire(
            "r1", tenant="acme", holder="w1", ttl_seconds=60.0
        )

        assert (lease.holder, lease.fence, lease.expires_at) == ("w1", 1, 60.0)

    async def test_a_run_someone_else_holds_is_refused(self) -> None:
        executor = FakeSql([])

        with pytest.raises(RunLeaseError) as refused:
            await PostgresLeaseStore(executor).acquire(
                "r1", tenant="acme", holder="w2", ttl_seconds=60.0
            )

        assert refused.value.requested_by == "w2"

    async def test_a_row_returned_for_another_holder_is_refused_too(self) -> None:
        executor = FakeSql([["w1", 4, 90.0]])

        with pytest.raises(RunLeaseError) as refused:
            await PostgresLeaseStore(executor).acquire(
                "r1", tenant="acme", holder="w2", ttl_seconds=60.0
            )

        assert refused.value.holder == "w1"
        assert refused.value.fence == 4

    async def test_expiry_is_computed_in_the_database(self) -> None:
        executor = FakeSql([["w1", 1, 60.0]])

        await PostgresLeaseStore(executor).acquire(
            "r1", tenant="acme", holder="w1", ttl_seconds=60.0
        )

        assert "extract(epoch FROM now())" in executor.statements[-1][0]
        assert 60.0 in executor.args

    async def test_renewal_states_the_fence_it_holds(self) -> None:
        executor = FakeSql([["w1", 2, 120.0]])
        held = RunLease(run_id="r1", tenant="acme", holder="w1", fence=2, expires_at=60.0)

        renewed = await PostgresLeaseStore(executor).renew(held, ttl_seconds=60.0)

        assert renewed.expires_at == 120.0
        assert executor.args[2:4] == ("w1", 2)

    async def test_renewing_a_lease_that_has_moved_on_is_refused(self) -> None:
        executor = FakeSql([])
        held = RunLease(run_id="r1", tenant="acme", holder="w1", fence=2)

        with pytest.raises(RunLeaseError):
            await PostgresLeaseStore(executor).renew(held, ttl_seconds=60.0)

    async def test_releasing_expires_only_the_row_it_still_holds(self) -> None:
        executor = FakeSql([["w1", 2, 0.0]])
        held = RunLease(run_id="r1", tenant="acme", holder="w1", fence=2)

        await PostgresLeaseStore(executor).release(held)

        assert executor.args == ("r1", "acme", "w1", 2)

    async def test_a_run_nobody_holds_reads_back_as_nothing(self) -> None:
        executor = FakeSql([])

        assert await PostgresLeaseStore(executor).held("r1", tenant="acme") is None

    async def test_a_held_run_reads_back_with_its_holder_and_fence(self) -> None:
        executor = FakeSql([["w1", 3, 90.0]])

        lease = await PostgresLeaseStore(executor).held("r1", tenant="acme")

        assert lease is not None
        assert lease.fence == 3


class TestTheStoresAreWhatTheRuntimeAsksFor:
    def test_both_checkpoint_stores_satisfy_the_protocol(self) -> None:
        redis: CheckpointStore = RedisCheckpointStore(FakeRedis())
        postgres: CheckpointStore = PostgresCheckpointStore(FakeSql())

        assert redis is not None
        assert postgres is not None

    def test_both_lease_stores_satisfy_the_protocol(self) -> None:
        redis: LeaseStore = RedisLeaseStore(FakeRedis())
        postgres: LeaseStore = PostgresLeaseStore(FakeSql())

        assert isinstance(redis, type(redis))
        assert isinstance(postgres, type(postgres))
