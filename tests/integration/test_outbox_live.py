"""The outbox against a real PostgreSQL, where the SQL has to be valid and the locks real.

Opted into with `-m integration` and an environment variable, because the default lane
reaches no network. A fake can prove which statement was sent; only a database can prove
that a rolled-back transaction publishes nothing, that two relays never take the same run,
and that `EXPECTED_OUTBOX_SCHEMA` is DDL PostgreSQL accepts.

    ADK_TEST_POSTGRES_DSN=postgresql://adk@localhost:5432/adk_test \\
    uv run pytest tests/integration -m integration
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.outbox import (
    EXPECTED_OUTBOX_SCHEMA,
    OutboxRelay,
    OutboxTables,
    PostgresOutbox,
    PostgresOutboxSettings,
)
from tesserix_adk.core.events import (
    Delivery,
    EventEnvelope,
    Eventing,
    EventType,
    RunCompleted,
)
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.testing import FakeClock
from tests.integration.test_sql_state_adapters_live import Psycopg

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = [pytest.mark.integration, pytest.mark.allow_network]

POSTGRES_DSN = os.environ.get("ADK_TEST_POSTGRES_DSN", "")
PREFIX = f"adk_{uuid.uuid4().hex[:8]}_"
TABLES = OutboxTables(rows=f"{PREFIX}outbox", schema=f"{PREFIX}schema")
SETTINGS = PostgresOutboxSettings(dsn=SecretStr(POSTGRES_DSN or "postgresql://localhost/adk_test"))
TIMED = "-c statement_timeout=5s"
NOW = 1_000.0

SCHEMA_TABLE = """
CREATE TABLE {schema} (component text PRIMARY KEY, version integer NOT NULL);
"""


@pytest.fixture(autouse=True)
async def schema() -> None:
    """Apply the shape the adapter expects. In production this is a migration."""
    if not POSTGRES_DSN:
        return
    psycopg = pytest.importorskip("psycopg")
    async with await psycopg.AsyncConnection.connect(POSTGRES_DSN, autocommit=True) as connection:
        await connection.execute(
            f"DROP TABLE IF EXISTS {TABLES.rows}, {TABLES.schema} CASCADE;"
            + SCHEMA_TABLE.format(schema=TABLES.schema)
            + EXPECTED_OUTBOX_SCHEMA.replace("adk_", PREFIX)
        )


class Transactor:
    """A `SqlTransactor` holding one connection open for the length of the transaction."""

    def __init__(self) -> None:
        self._psycopg = pytest.importorskip("psycopg")

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        return await Psycopg(POSTGRES_DSN).fetch(statement, *args)

    def transaction(self) -> Any:
        return _Transaction(self._psycopg)


class _Transaction:
    def __init__(self, psycopg: Any) -> None:
        self._psycopg = psycopg
        self._connection: Any = None

    async def __aenter__(self) -> Psycopg:
        self._connection = await self._psycopg.AsyncConnection.connect(POSTGRES_DSN, options=TIMED)
        await self._connection.__aenter__()
        return Psycopg(POSTGRES_DSN, self._connection)

    async def __aexit__(self, *exc: Any) -> None:
        await self._connection.__aexit__(*exc)


class Published:
    """Records what reached the transport."""

    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    async def publish(self, event: EventEnvelope) -> None:
        self.events.append(event)

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        self.events.extend(events)


def sql(connection: Any = None) -> Psycopg:
    pytest.importorskip("psycopg")
    return Psycopg(POSTGRES_DSN, connection)


def outbox(session: Psycopg | None = None) -> PostgresOutbox:
    return PostgresOutbox(session or sql(), clock=FakeClock(NOW), settings=SETTINGS, tables=TABLES)


def relay(
    publisher: Published, *, worker: str = "relay-1", now: float = NOW, **kwargs: Any
) -> OutboxRelay:
    return OutboxRelay(
        Transactor(), publisher, clock=FakeClock(now), worker=worker, tables=TABLES, **kwargs
    )


async def _event(run_id: str, tenant: str = "acme") -> EventEnvelope:
    """An envelope built the way a caller builds one, through `Eventing`."""
    eventing = Eventing(clock=FakeClock(NOW), delivery=Delivery.GUARANTEED)
    with tenant_scope(tenant, user="ada"):
        event = await eventing.emit(RunCompleted(run_id=run_id, iterations=2))
    assert event is not None
    return event


@pytest.mark.skipif(not POSTGRES_DSN, reason="no ephemeral PostgreSQL configured")
class TestWhatOnlyARealDatabaseShows:
    """Atomicity, contention and the DDL itself."""

    async def test_a_rolled_back_run_publishes_nothing(self) -> None:
        """The primary scenario: no downstream system sees a completion that did not happen."""
        psycopg = pytest.importorskip("psycopg")
        published = Published()
        async with await psycopg.AsyncConnection.connect(POSTGRES_DSN, options=TIMED) as held:
            with pytest.raises(RuntimeError, match="something else failed"):
                await _write_then_fail(held)

        assert await relay(published).deliver() == 0
        assert published.events == []

    async def test_a_committed_run_is_delivered_once(self) -> None:
        """Publish, mark, and the next poll finds nothing left to do."""
        published = Published()
        await outbox().publish(await _event("r1"))

        assert await relay(published).deliver() == 1
        assert await relay(published).deliver() == 0
        assert [event.type for event in published.events] == [EventType.RUN_COMPLETED]

    async def test_the_same_event_id_inserts_once(self) -> None:
        """A retried caller writing the same event twice leaves one row, not two."""
        published = Published()
        event = await _event("r1")
        await outbox().publish(event)
        await outbox().publish(event)

        assert await relay(published).deliver() == 1

    async def test_two_relays_never_take_the_same_run(self) -> None:
        """The advisory lock is the whole point: a run belongs to one relay at a time."""
        first, second = Published(), Published()
        for index in range(6):
            await outbox().publish(await _event(f"r{index}"))

        counts = await asyncio.gather(
            relay(first, worker="a").deliver(), relay(second, worker="b").deliver()
        )

        delivered = [event.run_id for event in (*first.events, *second.events)]
        assert sum(counts) == 6
        assert sorted(delivered) == [f"r{index}" for index in range(6)]

    async def test_lag_and_pruning_report_real_rows(self) -> None:
        """What an operator alerts on has to come from the table, not from arithmetic."""
        published = Published()
        await outbox().publish(await _event("r1"))
        worker = relay(published, retention_seconds=0.0)

        assert (await worker.lag()).unpublished == 1
        await worker.deliver()
        assert (await worker.lag()).unpublished == 0
        assert await relay(published, now=NOW + 1.0, retention_seconds=0.0).prune() == 1

    async def test_an_unpublished_row_is_never_pruned(self) -> None:
        """Retention removes history, never work that has not been done."""
        await outbox().publish(await _event("r1"))

        assert await relay(Published(), retention_seconds=0.0).prune() == 0

    async def test_a_schema_it_was_not_written_for_is_refused(self) -> None:
        """A column that moved is an event written into the wrong shape."""
        from tesserix_adk.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="schema is version"):
            await PostgresOutbox.open(
                sql(),
                clock=FakeClock(NOW),
                settings=SETTINGS.model_copy(update={"schema_version": 99}),
                tables=TABLES,
            )


async def _write_then_fail(held: Any) -> None:
    """Enqueue an event on the caller's transaction, then fail before it commits."""
    async with held.transaction():
        await outbox().bound(sql(held)).publish(await _event("r1"))
        raise RuntimeError("something else failed")
