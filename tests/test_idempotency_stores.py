"""What the durable stores send, and what they make of what comes back.

The Lua and the SQL are verified against a real server by `IdempotencyStoreConformance`,
which is what `tests/integration/` runs. These are the translation: the right script, the
right key, and a reply turned into the right claim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.adapters import IN_FLIGHT, PostgresIdempotencyStore, RedisIdempotencyStore
from tesserix_adk.runtime import MemoryIdempotencyStore
from tesserix_adk.testing import FakeClock, IdempotencyStoreConformance

if TYPE_CHECKING:
    from tesserix_adk.core import IdempotencyStore

TTL = 900.0


class FakeRedis:
    """Records what was evaluated and answers with whatever the test says the server said."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.fail: Exception | None = None

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        if self.fail is not None:
            raise self.fail
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
        self.fail: Exception | None = None

    async def fetch(self, statement: str, *args: Any) -> list[list[Any]]:
        if self.fail is not None:
            raise self.fail
        self.statements.append((statement, args))
        return self.replies.pop(0) if self.replies else []

    @property
    def args(self) -> tuple[Any, ...]:
        return self.statements[-1][1]


def redis_store(client: FakeRedis, **kwargs: Any) -> RedisIdempotencyStore:
    return RedisIdempotencyStore(client, clock=FakeClock(start=7_200), **kwargs)


def postgres_store(executor: FakeSql, **kwargs: Any) -> PostgresIdempotencyStore:
    return PostgresIdempotencyStore(executor, clock=FakeClock(start=7_200), **kwargs)


class TestTheRedisStoreClaimsInOneRoundTrip:
    async def test_an_unclaimed_key_comes_back_free(self) -> None:
        client = FakeRedis(None)

        claimed = await redis_store(client).begin("k", tenant="acme", ttl_seconds=TTL)

        assert claimed.outcome is None
        assert claimed.in_flight is False

    async def test_the_key_is_namespaced_and_carries_the_tenant(self) -> None:
        """Two tenants deriving the same digest are two effects, not one."""
        client = FakeRedis(None)

        await redis_store(client).begin("k", tenant="acme", ttl_seconds=TTL)

        assert client.keys == ("adk:effect:acme:k",)

    async def test_the_retention_window_is_sent_in_milliseconds(self) -> None:
        client = FakeRedis(None)

        await redis_store(client).begin("k", tenant="acme", ttl_seconds=TTL)

        assert "900000" in client.argv

    async def test_a_key_another_caller_holds_comes_back_in_flight(self) -> None:
        client = FakeRedis(IN_FLIGHT)

        claimed = await redis_store(client).begin("k", tenant="acme", ttl_seconds=TTL)

        assert claimed.in_flight is True
        assert claimed.outcome is None

    async def test_a_key_that_has_an_answer_comes_back_with_it(self) -> None:
        client = FakeRedis(b"booked BA117")

        claimed = await redis_store(client).begin("k", tenant="acme", ttl_seconds=TTL)

        assert claimed.outcome == "booked BA117"
        assert claimed.in_flight is False

    async def test_recording_replaces_the_claim_with_the_answer(self) -> None:
        client = FakeRedis(1)

        await redis_store(client).record("k", tenant="acme", outcome="booked", ttl_seconds=TTL)

        assert "booked" in client.argv

    async def test_abandoning_deletes_the_key(self) -> None:
        client = FakeRedis(1)

        await redis_store(client).abandon("k", tenant="acme")

        assert client.keys == ("adk:effect:acme:k",)

    async def test_erasure_sweeps_one_tenant_s_prefix(self) -> None:
        client = FakeRedis(3)

        assert await redis_store(client).forget(tenant="acme") == 3
        assert "adk:effect:acme:*" in client.argv

    async def test_a_server_that_cannot_be_reached_says_so(self) -> None:
        """Failing closed is the dispatcher's job, so the store must not answer 'free'."""
        client = FakeRedis()
        client.fail = ConnectionError("down")

        with pytest.raises(ConnectionError):
            await redis_store(client).begin("k", tenant="acme", ttl_seconds=TTL)


class TestThePostgresStoreClaimsInOneStatement:
    async def test_an_unclaimed_key_comes_back_free(self) -> None:
        executor = FakeSql([[None]])

        claimed = await postgres_store(executor).begin("k", tenant="acme", ttl_seconds=TTL)

        assert claimed.outcome is None
        assert claimed.in_flight is False

    async def test_the_statement_carries_the_key_the_tenant_and_the_expiry(self) -> None:
        executor = FakeSql([[None]])

        await postgres_store(executor).begin("k", tenant="acme", ttl_seconds=TTL)

        assert executor.args[:2] == ("k", "acme")
        assert 8_100.0 in executor.args

    async def test_a_row_another_caller_holds_comes_back_in_flight(self) -> None:
        executor = FakeSql([[IN_FLIGHT]])

        claimed = await postgres_store(executor).begin("k", tenant="acme", ttl_seconds=TTL)

        assert claimed.in_flight is True

    async def test_a_row_that_has_an_answer_comes_back_with_it(self) -> None:
        executor = FakeSql([["booked BA117"]])

        claimed = await postgres_store(executor).begin("k", tenant="acme", ttl_seconds=TTL)

        assert claimed.outcome == "booked BA117"

    async def test_recording_writes_the_answer_against_the_claim(self) -> None:
        executor = FakeSql([[1]])

        await postgres_store(executor).record("k", tenant="acme", outcome="booked", ttl_seconds=TTL)

        assert "booked" in executor.args

    async def test_abandoning_deletes_the_row(self) -> None:
        executor = FakeSql([[1]])

        await postgres_store(executor).abandon("k", tenant="acme")

        assert executor.args == ("k", "acme")

    async def test_erasure_reports_how_many_rows_went(self) -> None:
        executor = FakeSql([[4]])

        assert await postgres_store(executor).forget(tenant="acme") == 4

    async def test_erasure_of_a_tenant_with_nothing_stored_is_none_rather_than_an_error(
        self,
    ) -> None:
        executor = FakeSql([])

        assert await postgres_store(executor).forget(tenant="acme") == 0

    async def test_the_table_is_a_deployment_s_to_create(self) -> None:
        """DDL on a request path fails at the worst moment; a deployment calls this."""
        executor = FakeSql([])

        await postgres_store(executor, table="effects").ensure_schema()

        assert "CREATE TABLE IF NOT EXISTS effects" in executor.statements[0][0]

    async def test_a_database_that_cannot_be_reached_says_so(self) -> None:
        executor = FakeSql()
        executor.fail = ConnectionError("down")

        with pytest.raises(ConnectionError):
            await postgres_store(executor).begin("k", tenant="acme", ttl_seconds=TTL)


class TestTheMemoryStoreConforms(IdempotencyStoreConformance):
    def make_store(self) -> IdempotencyStore:
        return MemoryIdempotencyStore(FakeClock())
