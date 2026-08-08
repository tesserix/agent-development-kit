"""What the memory adapters send, and what they make of what comes back.

The Lua and the SQL are verified against a real server by the integration lane. These are
the translation: the right key, the right predicate, and a reply turned into the right
record — plus the things a real server does that a fake has to be asked to do, like
evicting a key or refusing a connection.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters import (
    MemoryStoreSettings,
    PgvectorMemoryStore,
    PostgresMemoryStore,
    RedisMemoryStore,
    RoutedMemoryStore,
)
from tesserix_adk.core import (
    EmbeddingDimensionError,
    MemoryConflictError,
    MemoryCorruptionError,
    MemoryScopeError,
    MemoryUnavailableError,
    PoolExhaustedError,
)
from tesserix_adk.memory import (
    Derivation,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Sequence

SCOPE = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1")
OTHER = MemoryScope(tenant_id="globex", user_id="u9")
NOW = 1_000.0
SETTINGS = MemoryStoreSettings(dsn=SecretStr("postgresql://adk:s3cret@db/adk"), max_attempts=3)


def record(
    kind: MemoryKind = MemoryKind.WORKING,
    key: str = "k",
    value: Any = "v",
    *,
    scope: MemoryScope = SCOPE,
    embedding: tuple[float, ...] | None = None,
    version: int = 1,
) -> MemoryRecord:
    """A record of any kind, filled in enough to be stored."""
    return MemoryRecord(
        id=f"{kind.value}:{key}",
        kind=kind,
        scope=scope,
        key=key,
        value=value,
        source="turn",
        valid_from=NOW,
        version=version,
        embedding=embedding,
    )


class FakeRedis:
    """Answers with whatever the test says the server said, and records what was asked."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.fails: list[Exception | None] = []

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        failure = self.fails.pop(0) if self.fails else None
        if failure is not None:
            raise failure
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
        self.fails: list[Exception | None] = []

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        failure = self.fails.pop(0) if self.fails else None
        if failure is not None:
            raise failure
        self.statements.append((statement, args))
        return self.replies.pop(0) if self.replies else []

    @property
    def sql(self) -> str:
        return self.statements[-1][0]

    @property
    def args(self) -> tuple[Any, ...]:
        return self.statements[-1][1]


class PoolTimeoutError(Exception):
    """What a driver raises when every pooled connection is already in use."""


def payload(held: MemoryRecord) -> list[Any]:
    """A row as the drivers hand it back: one jsonb column holding the whole record."""
    return [json.loads(held.model_dump_json())]


def redis_store(client: FakeRedis, **kwargs: Any) -> RedisMemoryStore:
    """A Redis working-memory store on a clock a test can read back."""
    return RedisMemoryStore(client, clock=FakeClock(start=NOW), settings=SETTINGS, **kwargs)


def sql_store(executor: FakeSql, **kwargs: Any) -> PostgresMemoryStore:
    """A PostgreSQL profile and episodic store."""
    return PostgresMemoryStore(executor, clock=FakeClock(start=NOW), settings=SETTINGS, **kwargs)


def vector_store(executor: FakeSql, **kwargs: Any) -> PgvectorMemoryStore:
    """A pgvector semantic store, three dimensions wide unless a test says otherwise."""
    kwargs.setdefault("dimensions", 3)
    return PgvectorMemoryStore(executor, clock=FakeClock(start=NOW), settings=SETTINGS, **kwargs)


class TestCredentialsComeFromSettingsAndNowhereElse:
    def test_a_blank_dsn_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dsn"):
            MemoryStoreSettings(dsn=SecretStr("  "))

    @pytest.mark.parametrize(
        "dsn",
        [
            "postgresql://postgres:postgres@db/adk",
            "postgresql://adk:password@db/adk",
            "redis://:changeme@cache:6379/0",
        ],
    )
    def test_a_shipped_default_password_is_refused(self, dsn: str) -> None:
        """An adapter that works out of the box works out of the box for everyone."""
        with pytest.raises(ValueError, match="default password"):
            MemoryStoreSettings(dsn=SecretStr(dsn))

    def test_the_dsn_does_not_appear_in_a_repr(self) -> None:
        assert "s3cret" not in repr(SETTINGS)

    def test_a_pool_of_no_connections_is_refused(self) -> None:
        with pytest.raises(ValueError, match="pool_size"):
            MemoryStoreSettings(dsn=SecretStr("redis://cache:6379/0"), pool_size=0)


class TestRedisHoldsTheWorkingSet:
    async def test_the_key_carries_the_whole_scope(self) -> None:
        """Two sessions under one tenant are two scratch spaces, not one."""
        client = FakeRedis(None)
        await redis_store(client).write(SCOPE, record())

        assert client.keys == ("adk:mem:acme:u1:s1::working:k",)

    async def test_a_write_carries_the_configured_ttl(self) -> None:
        client = FakeRedis(None)
        await redis_store(client, ttl_seconds=60.0).write(SCOPE, record())

        assert client.argv[1] == "60000"

    async def test_a_read_returns_the_stored_record(self) -> None:
        held = record(value={"seat": "aisle"})
        client = FakeRedis(held.model_dump_json())

        found = await redis_store(client).read(SCOPE, "k")
        assert found is not None
        assert found.value == {"seat": "aisle"}

    async def test_an_evicted_key_reads_as_absent_rather_than_raising(self) -> None:
        """maxmemory took it mid-run. Absent is the truth; an exception would not be."""
        assert await redis_store(FakeRedis(None)).read(SCOPE, "k") is None

    async def test_a_sliding_store_extends_the_key_it_just_read(self) -> None:
        client = FakeRedis(record().model_dump_json())
        await redis_store(client, ttl_seconds=60.0, sliding=True).read(SCOPE, "k")

        assert client.argv == ("60000",)

    async def test_a_store_that_does_not_slide_sends_no_expiry_on_a_read(self) -> None:
        client = FakeRedis(record().model_dump_json())
        await redis_store(client, ttl_seconds=60.0).read(SCOPE, "k")

        assert client.argv == ()

    async def test_an_append_is_one_server_side_call(self) -> None:
        """Read-modify-write from the client is two chances to lose the other append."""
        client = FakeRedis(3)
        position = await redis_store(client).append(SCOPE, "turns", "hello")

        assert position == 3
        assert len(client.calls) == 1

    async def test_an_append_after_an_eviction_says_it_is_the_first(self) -> None:
        """The caller can see the sequence was lost, which a silent 4 would hide."""
        assert await redis_store(FakeRedis(1)).append(SCOPE, "turns", "hello") == 1

    async def test_expiry_is_set_in_milliseconds(self) -> None:
        client = FakeRedis(1)
        await redis_store(client).expire(SCOPE, "k", ttl_seconds=1.5)

        assert client.argv == ("1500",)

    async def test_a_batch_recall_is_one_round_trip(self) -> None:
        first, second = record(key="a"), record(key="b")
        client = FakeRedis([first.model_dump_json(), None, second.model_dump_json()])

        found = await redis_store(client).recall(SCOPE, ("a", "gone", "b"))
        assert [held.key for held in found] == ["a", "b"]
        assert len(client.calls) == 1

    async def test_erasure_matches_the_scope_and_nothing_above_it(self) -> None:
        client = FakeRedis(4)
        receipt = await redis_store(client).erase(SCOPE)

        assert client.argv[0] == "adk:mem:acme:u1:s1::*"
        assert receipt.records == 4

    async def test_it_declares_what_it_cannot_do(self) -> None:
        """Semantic recall against a key-value store returns nothing, quietly, forever."""
        declared = redis_store(FakeRedis()).capabilities
        assert not declared.supports_semantic
        assert not declared.supports_supersession


class TestPostgresHoldsWhatOutlivesTheSession:
    async def test_the_tenant_is_in_the_predicate_and_not_applied_after(self) -> None:
        """A filter applied after the fetch has already read the other tenant's rows."""
        executor = FakeSql([])
        await sql_store(executor).profile(SCOPE, "seat")

        assert "tenant_id = $1" in executor.sql
        assert executor.args[0] == "acme"

    async def test_an_upsert_writes_the_whole_record_as_one_payload(self) -> None:
        executor = FakeSql([])
        await sql_store(executor).upsert(SCOPE, record(MemoryKind.PROFILE, "seat", "aisle"))

        assert "INSERT INTO adk_memory" in executor.sql
        assert json.loads(executor.args[-1])["value"] == "aisle"

    async def test_a_profile_read_comes_back_as_a_record(self) -> None:
        held = record(MemoryKind.PROFILE, "seat", "aisle")
        found = await sql_store(FakeSql([payload(held)])).profile(SCOPE, "seat")

        assert found is not None
        assert found.value == "aisle"

    async def test_an_as_of_read_asks_for_the_window_containing_the_instant(self) -> None:
        executor = FakeSql([])
        await sql_store(executor).profile(SCOPE, "seat", as_of=500.0)

        assert "valid_from <=" in executor.sql
        assert "valid_to" in executor.sql
        assert 500.0 in executor.args

    async def test_a_supersession_closes_the_old_version_and_writes_the_new(self) -> None:
        live = record(MemoryKind.PROFILE, "seat", "aisle")
        executor = FakeSql([payload(live)], [[1]], [])
        written = await sql_store(executor).supersede(
            SCOPE, record(MemoryKind.PROFILE, "seat", "window")
        )

        assert written.record.version == 2
        assert written.superseded is not None
        assert written.superseded.valid_to == NOW

    async def test_a_stale_expected_version_is_refused_by_the_predicate(self) -> None:
        """The check is the UPDATE's own WHERE, so two writers cannot both pass it."""
        live = record(MemoryKind.PROFILE, "seat", "aisle", version=2)
        executor = FakeSql([payload(live)], [])

        with pytest.raises(MemoryConflictError):
            await sql_store(executor).supersede(
                SCOPE, record(MemoryKind.PROFILE, "seat", "window"), expected_version=1
            )

    async def test_history_comes_back_oldest_first(self) -> None:
        first = record(MemoryKind.PROFILE, "seat", "aisle")
        second = record(MemoryKind.PROFILE, "seat", "window", version=2)
        executor = FakeSql([payload(first), payload(second)])

        trail = await sql_store(executor).history(SCOPE, "seat")
        assert [held.version for held in trail] == [1, 2]
        assert "ORDER BY version" in executor.sql

    async def test_an_episode_insert_is_idempotent_on_its_id(self) -> None:
        """A retry after a failover must not book the episode twice."""
        executor = FakeSql([])
        await sql_store(executor).log(SCOPE, record(MemoryKind.EPISODIC, "e"))

        assert "ON CONFLICT (id) DO NOTHING" in executor.sql

    async def test_a_wide_window_pages_rather_than_materialising(self) -> None:
        executor = FakeSql([])
        await sql_store(executor).episodes(
            SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, since=0.0, until=1e9, limit=50)
        )

        assert "LIMIT" in executor.sql
        assert 50 in executor.args

    async def test_a_page_hands_back_the_cursor_for_the_next_one(self) -> None:
        rows = [payload(record(MemoryKind.EPISODIC, f"e{n}")) for n in range(3)]
        executor = FakeSql(rows)
        page = await sql_store(executor).page(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, limit=2))

        assert len(page.hits) == 2
        assert page.cursor is not None

    async def test_the_last_page_has_no_cursor(self) -> None:
        rows = [payload(record(MemoryKind.EPISODIC, "e0"))]
        page = await sql_store(FakeSql(rows)).page(
            SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, limit=2)
        )

        assert page.cursor is None

    async def test_a_cursor_narrows_the_predicate_rather_than_skipping_rows(self) -> None:
        """OFFSET re-reads everything before it; a keyset cursor reads only what is left."""
        executor = FakeSql([])
        await sql_store(executor).page(
            SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, limit=2), cursor="500.0|episodic:e1"
        )

        assert "OFFSET" not in executor.sql
        assert 500.0 in executor.args

    async def test_erasure_reports_what_the_delete_removed_by_kind(self) -> None:
        receipt = await sql_store(FakeSql([["profile", 5], ["episodic", 2]])).erase(SCOPE)
        assert receipt.counts == {"profile": 5, "episodic": 2}
        assert receipt.records == 7
        assert receipt.complete


class TestPgvectorRanksBySql:
    async def test_the_scope_filter_is_pushed_into_the_predicate(self) -> None:
        executor = FakeSql([])
        await vector_store(executor).search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0, 0.0))
        )

        assert "tenant_id = $" in executor.sql
        assert "acme" in executor.args

    async def test_ranking_happens_in_the_database(self) -> None:
        executor = FakeSql([])
        await vector_store(executor).search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0, 0.0))
        )

        assert "ORDER BY embedding <=>" in executor.sql

    async def test_the_distance_operator_follows_the_configured_metric(self) -> None:
        executor = FakeSql([])
        await vector_store(executor, metric="l2").search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0, 0.0))
        )

        assert "<->" in executor.sql

    async def test_an_unknown_metric_is_refused_when_the_store_is_built(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            vector_store(FakeSql(), metric="hamming")

    async def test_a_hit_carries_the_record_and_its_distance_as_a_score(self) -> None:
        held = record(MemoryKind.SEMANTIC, "s", embedding=(1.0, 0.0, 0.0))
        rows = [[json.loads(held.model_dump_json()), 0.25]]
        found = await vector_store(FakeSql(rows)).search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0, 0.0))
        )

        assert found[0].record.key == "s"
        assert found[0].score == pytest.approx(0.75)

    async def test_an_embedding_of_the_wrong_width_is_refused_before_it_is_sent(self) -> None:
        executor = FakeSql([])
        with pytest.raises(EmbeddingDimensionError):
            await vector_store(executor).index(
                SCOPE, record(MemoryKind.SEMANTIC, "s", embedding=(1.0, 0.0))
            )
        assert executor.statements == []

    async def test_a_collection_narrower_than_the_embedder_is_caught_at_startup(self) -> None:
        """Not on the first recall a month later, when it silently ranks nothing."""
        executor = FakeSql([[2]])
        with pytest.raises(EmbeddingDimensionError, match="3"):
            await vector_store(executor).verify()

    async def test_a_matching_collection_passes_startup(self) -> None:
        await vector_store(FakeSql([[3]])).verify()

    async def test_a_collection_that_does_not_exist_yet_is_named_in_the_error(self) -> None:
        with pytest.raises(EmbeddingDimensionError, match="adk_semantic"):
            await vector_store(FakeSql([])).verify()

    async def test_the_index_type_is_what_the_store_was_configured_with(self) -> None:
        assert vector_store(FakeSql(), index_type="ivfflat").index_type == "ivfflat"


class TestWhenTheStoreIsUnreachable:
    async def test_it_retries_a_failover_and_commits_once(self) -> None:
        executor = FakeSql([])
        executor.fails = [ConnectionError("primary went away")]

        await sql_store(executor).log(SCOPE, record(MemoryKind.EPISODIC, "e"))
        assert len(executor.statements) == 1

    async def test_it_gives_up_after_the_configured_attempts(self) -> None:
        executor = FakeSql()
        executor.fails = [ConnectionError("down")] * 3

        with pytest.raises(MemoryUnavailableError) as failed:
            await sql_store(executor).log(SCOPE, record(MemoryKind.EPISODIC, "e"))
        assert failed.value.attempts == 3
        assert failed.value.store == "PostgresMemoryStore"

    async def test_it_backs_off_between_attempts_with_jitter(self) -> None:
        clock = FakeClock(start=NOW)
        executor = FakeSql([])
        executor.fails = [ConnectionError("down"), ConnectionError("down")]

        store = PostgresMemoryStore(executor, clock=clock, settings=SETTINGS, entropy=lambda: 1.0)
        await store.log(SCOPE, record(MemoryKind.EPISODIC, "e"))

        assert clock.slept == [pytest.approx(0.075), pytest.approx(0.15)]

    async def test_a_full_pool_is_reported_rather_than_retried(self) -> None:
        """Retrying into an exhausted pool is how a fan-out spike becomes an outage."""
        executor = FakeSql()
        executor.fails = [PoolTimeoutError("none available"), PoolTimeoutError("still none")]

        with pytest.raises(PoolExhaustedError):
            await sql_store(executor).log(SCOPE, record(MemoryKind.EPISODIC, "e"))
        assert len(executor.fails) == 1

    async def test_redis_fails_the_same_way(self) -> None:
        client = FakeRedis()
        client.fails = [ConnectionError("down")] * 3

        with pytest.raises(MemoryUnavailableError) as failed:
            await redis_store(client).read(SCOPE, "k")
        assert failed.value.store == "RedisMemoryStore"

    async def test_giving_up_now_does_not_mean_giving_up_for_good(self) -> None:
        """A store that came back is a run worth starting again, and callers ask."""
        client = FakeRedis()
        client.fails = [ConnectionError("down")] * 3

        with pytest.raises(MemoryUnavailableError) as failed:
            await redis_store(client).read(SCOPE, "k")
        assert failed.value.retryable

    async def test_the_error_says_nothing_about_the_credentials(self) -> None:
        executor = FakeSql()
        executor.fails = [ConnectionError("auth failed for adk:s3cret@db")] * 3

        with pytest.raises(MemoryUnavailableError) as failed:
            await sql_store(executor).log(SCOPE, record(MemoryKind.EPISODIC, "e"))
        assert "s3cret" not in str(failed.value)


class TestOneStoreOverThree:
    def routed(self, redis: FakeRedis, sql: FakeSql, vectors: FakeSql) -> RoutedMemoryStore:
        """Working memory in Redis, the durable kinds in PostgreSQL, vectors in pgvector."""
        return RoutedMemoryStore(
            working=redis_store(redis),
            durable=sql_store(sql),
            semantic=vector_store(vectors),
        )

    async def test_each_kind_goes_to_the_store_that_holds_it(self) -> None:
        redis, sql, vectors = FakeRedis(None), FakeSql([]), FakeSql([])
        store = self.routed(redis, sql, vectors)

        await store.write(SCOPE, record())
        await store.upsert(SCOPE, record(MemoryKind.PROFILE, "seat", "aisle"))
        await store.index(SCOPE, record(MemoryKind.SEMANTIC, "s", embedding=(1.0, 0.0, 0.0)))

        assert len(redis.calls) == 1
        assert len(sql.statements) == 1
        assert len(vectors.statements) == 1

    async def test_it_declares_everything_its_parts_can_do(self) -> None:
        declared = self.routed(FakeRedis(), FakeSql(), FakeSql()).capabilities
        assert declared.supports_semantic
        assert declared.supports_supersession
        assert declared.embedding_dimensions == 3

    async def test_an_erasure_reaches_all_three(self) -> None:
        redis = FakeRedis(2)
        sql, vectors = FakeSql([["profile", 3]]), FakeSql([["semantic", 1]])
        receipt = await self.routed(redis, sql, vectors).erase(SCOPE)

        assert receipt.counts == {"working": 2, "profile": 3, "semantic": 1}
        assert receipt.complete

    async def test_a_scope_it_never_wrote_to_erases_to_nothing(self) -> None:
        receipt = await self.routed(FakeRedis(0), FakeSql([]), FakeSql([])).erase(OTHER)
        assert receipt.records == 0


class TestTheThingsAFakeHasToBeAskedToDo:
    async def test_a_row_that_no_longer_validates_is_reported_as_corruption(self) -> None:
        executor = FakeSql([[{"id": "profile:seat"}]])
        with pytest.raises(MemoryCorruptionError):
            await sql_store(executor).profile(SCOPE, "seat")

    async def test_a_driver_that_hands_back_json_text_reads_the_same(self) -> None:
        """asyncpg leaves jsonb as text unless a codec is set; psycopg decodes it."""
        held = record(MemoryKind.PROFILE, "seat", "aisle")
        found = await sql_store(FakeSql([[held.model_dump_json()]])).profile(SCOPE, "seat")

        assert found is not None
        assert found.value == "aisle"

    async def test_a_corrupt_row_is_not_read_again(self) -> None:
        """Corruption does not heal on a second read, and the wait would be the caller's."""
        executor = FakeSql([[{"id": "profile:seat"}]], [[{"id": "profile:seat"}]])
        with pytest.raises(MemoryCorruptionError):
            await sql_store(executor).profile(SCOPE, "seat")
        assert len(executor.statements) == 1

    async def test_a_record_written_under_another_scope_is_refused(self) -> None:
        with pytest.raises(MemoryScopeError):
            await sql_store(FakeSql()).upsert(SCOPE, record(MemoryKind.PROFILE, "k", scope=OTHER))

    async def test_a_record_of_the_wrong_kind_is_refused(self) -> None:
        with pytest.raises(MemoryScopeError):
            await redis_store(FakeRedis()).write(SCOPE, record(MemoryKind.PROFILE, "k"))

    async def test_a_belief_carries_whatever_the_profile_read_found(self) -> None:
        held = record(MemoryKind.PROFILE, "seat", "aisle")
        found = await sql_store(FakeSql([payload(held)])).belief(SCOPE, "seat")

        assert found.record is not None
        assert found.record.value == "aisle"

    async def test_superseding_nothing_writes_the_first_version(self) -> None:
        executor = FakeSql([])
        written = await sql_store(executor).supersede(
            SCOPE, record(MemoryKind.PROFILE, "seat", "aisle")
        )

        assert written.superseded is None
        assert written.record.version == 1

    async def test_a_lost_race_is_refused_by_the_update_itself(self) -> None:
        """Both writers read version 1; only one UPDATE finds it still open."""
        live = record(MemoryKind.PROFILE, "seat", "aisle")
        executor = FakeSql([payload(live)], [])

        with pytest.raises(MemoryConflictError):
            await sql_store(executor).supersede(SCOPE, record(MemoryKind.PROFILE, "seat", "window"))

    async def test_what_was_derived_is_registered_against_its_source(self) -> None:
        executor = FakeSql([])
        await sql_store(executor).derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )

        assert "ON CONFLICT (artefact_id) DO NOTHING" in executor.sql
        assert "vec-1" in executor.args

    async def test_derivations_come_back_as_derivations(self) -> None:
        executor = FakeSql([["vec-1", "episodic:e", "vectors"]])
        found = await sql_store(executor).derivations(SCOPE, source_id="episodic:e")

        assert found[0].artefact_id == "vec-1"
        assert "episodic:e" in executor.args

    async def test_a_search_with_no_embedding_is_refused_before_it_is_sent(self) -> None:
        executor = FakeSql([])
        with pytest.raises(EmbeddingDimensionError):
            await vector_store(executor).search(SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC))
        assert executor.statements == []


class TestTheRouterDelegatesRatherThanReimplements:
    def routed(self, redis: FakeRedis, sql: FakeSql, vectors: FakeSql) -> RoutedMemoryStore:
        """The same composition the deployment binds."""
        return RoutedMemoryStore(
            working=redis_store(redis),
            durable=sql_store(sql),
            semantic=vector_store(vectors),
        )

    async def test_every_working_call_reaches_redis(self) -> None:
        redis = FakeRedis(record().model_dump_json(), 2, None)
        store = self.routed(redis, FakeSql(), FakeSql())

        assert (await store.read(SCOPE, "k")) is not None
        assert await store.append(SCOPE, "turns", "hello") == 2
        await store.expire(SCOPE, "k", ttl_seconds=1.0)
        assert len(redis.calls) == 3

    async def test_every_durable_call_reaches_postgres(self) -> None:
        held = record(MemoryKind.PROFILE, "seat", "aisle")
        sql = FakeSql([payload(held)], [payload(held)], [payload(held)], [], [], [["v", "s", "a"]])
        store = self.routed(FakeRedis(), sql, FakeSql())

        assert (await store.profile(SCOPE, "seat")) is not None
        assert (await store.belief(SCOPE, "seat")).record is not None
        assert len(await store.history(SCOPE, "seat")) == 1
        await store.log(SCOPE, record(MemoryKind.EPISODIC, "e"))
        assert await store.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC)) == ()
        assert (await store.derivations(SCOPE))[0].adapter == "a"

    async def test_a_supersession_through_the_router_still_closes_the_old_version(self) -> None:
        live = record(MemoryKind.PROFILE, "seat", "aisle")
        sql = FakeSql([payload(live)], [[1]], [])
        store = self.routed(FakeRedis(), sql, FakeSql())

        written = await store.supersede(SCOPE, record(MemoryKind.PROFILE, "seat", "window"))
        assert written.record.version == 2

    async def test_a_derivation_through_the_router_reaches_postgres(self) -> None:
        sql = FakeSql([])
        await self.routed(FakeRedis(), sql, FakeSql()).derived(
            SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:e", adapter="vectors")
        )
        assert len(sql.statements) == 1

    async def test_a_search_through_the_router_reaches_pgvector(self) -> None:
        vectors = FakeSql([])
        await self.routed(FakeRedis(), FakeSql(), vectors).search(
            SCOPE, MemoryQuery(kind=MemoryKind.SEMANTIC, embedding=(1.0, 0.0, 0.0))
        )
        assert len(vectors.statements) == 1

    @pytest.mark.parametrize("narrowing", [{"kinds": (MemoryKind.WORKING,)}, {"dry_run": True}])
    async def test_a_narrowed_or_rehearsed_erasure_is_refused_rather_than_faked(
        self, narrowing: dict[str, Any]
    ) -> None:
        """Three servers with no shared transaction cannot promise either one honestly."""
        store = self.routed(FakeRedis(), FakeSql(), FakeSql())
        with pytest.raises(NotImplementedError):
            await store.erase(SCOPE, **narrowing)


class TestABranchEndsBecauseSomebodyDecided:
    async def test_resolved_records_are_closed_by_the_same_write(self) -> None:
        """The decision is recorded rather than inferred from whichever version won."""
        live = record(MemoryKind.PROFILE, "seat", "aisle")
        executor = FakeSql([payload(live)], [["profile:other"]], [[1]], [])

        await sql_store(executor).supersede(
            SCOPE, record(MemoryKind.PROFILE, "seat", "window"), resolves=("profile:other",)
        )

        resolved = executor.statements[1]
        assert "id = ANY($5)" in resolved[0]
        assert resolved[1][4] == ["profile:other"]
