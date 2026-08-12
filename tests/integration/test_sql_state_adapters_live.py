"""The state and queue conformance suites, unchanged, against a real PostgreSQL.

Opted into with `-m integration` and an environment variable, because the default lane
reaches no network. The unit tests check what the adapters send; this checks that what
they send is valid SQL that does what the protocol says — and that ten workers claiming
at once take ten different items, which no fake can tell you.

    ADK_TEST_POSTGRES_DSN=postgresql://adk@localhost:5432/adk_test \\
    uv run pytest tests/integration -m integration
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.sql_state import (
    EXPECTED_SCHEMA,
    PostgresStateStore,
    PostgresStoreSettings,
    PostgresWorkQueue,
    StateTables,
)
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.queue import QueuePolicy, WorkItem
from tesserix_adk.core.state import RunRecord, StateKey
from tesserix_adk.testing import FakeClock, StateStoreConformance, WorkQueueConformance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.queue import WorkQueue
    from tesserix_adk.core.state import StateStore

pytestmark = [pytest.mark.integration, pytest.mark.allow_network]

POSTGRES_DSN = os.environ.get("ADK_TEST_POSTGRES_DSN", "")
PREFIX = f"adk_{uuid.uuid4().hex[:8]}_"
TABLES = StateTables(
    runs=f"{PREFIX}runs",
    sessions=f"{PREFIX}sessions",
    work=f"{PREFIX}work",
    counters=f"{PREFIX}queue_counters",
    turns=f"{PREFIX}queue_turns",
    schema=f"{PREFIX}schema",
)
SETTINGS = PostgresStoreSettings(dsn=SecretStr(POSTGRES_DSN or "postgresql://localhost/adk_test"))
TIMED = "-c statement_timeout=5s"

_PLACEHOLDER = re.compile(r"\$(\d+)")


class Psycopg:
    """`SqlSession` over psycopg, a connection per call — pooling is the deployment's."""

    def __init__(self, dsn: str, connection: Any = None) -> None:
        self._dsn = dsn
        self._connection = connection

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """Run `statement` with positional arguments and return whatever it selected."""
        if self._connection is not None:
            return await _run(self._connection, statement, args)
        psycopg = pytest.importorskip("psycopg")
        async with await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True, options=TIMED
        ) as connection:
            return await _run(connection, statement, args)


async def _run(connection: Any, statement: str, args: Sequence[Any]) -> Sequence[Sequence[Any]]:
    """Rewrite `$n` to psycopg's `%s`, repeating the argument wherever `n` is reused."""
    ordered: list[Any] = []

    def swap(match: re.Match[str]) -> str:
        ordered.append(args[int(match.group(1)) - 1])
        return "%s"

    async with connection.cursor() as cursor:
        await cursor.execute(_PLACEHOLDER.sub(swap, statement), ordered)
        return list(await cursor.fetchall()) if cursor.description else []


DROP = "DROP TABLE IF EXISTS {} CASCADE;"


@pytest.fixture(autouse=True)
async def schema() -> None:
    """Apply the shape the adapters expect, fresh, so no test reads another's rows.

    In production this is a migration the platform owns, applied once.
    """
    if not POSTGRES_DSN:
        return
    psycopg = pytest.importorskip("psycopg")
    tables = (
        TABLES.runs,
        TABLES.sessions,
        TABLES.work,
        TABLES.counters,
        TABLES.turns,
        TABLES.schema,
    )
    async with await psycopg.AsyncConnection.connect(POSTGRES_DSN, autocommit=True) as connection:
        await connection.execute("".join(DROP.format(table) for table in tables))
        await connection.execute(EXPECTED_SCHEMA.replace("adk_", PREFIX))


def sql(connection: Any = None) -> Psycopg:
    """A session, on the caller's connection where it has one."""
    pytest.importorskip("psycopg")
    return Psycopg(POSTGRES_DSN, connection)


def store(**kwargs: Any) -> PostgresStateStore:
    """A state store on a clock the test controls."""
    return PostgresStateStore(sql(), clock=FakeClock(), settings=SETTINGS, tables=TABLES, **kwargs)


def queue(**kwargs: Any) -> PostgresWorkQueue:
    """A work queue on a clock the test controls."""
    return PostgresWorkQueue(
        sql(),
        clock=FakeClock(),
        settings=SETTINGS,
        policy=QueuePolicy(max_attempts=3),
        tables=TABLES,
        **kwargs,
    )


@pytest.mark.skipif(not POSTGRES_DSN, reason="no ephemeral PostgreSQL configured")
class TestRunStateInPostgres(StateStoreConformance):
    """Identical assertions to the in-process store's, against a real database."""

    def make_store(self) -> StateStore:
        return store()


@pytest.mark.skipif(not POSTGRES_DSN, reason="no ephemeral PostgreSQL configured")
class TestWorkQueuedInPostgres(WorkQueueConformance):
    """Identical assertions to the in-process queue's, against a real database."""

    def make_queue(self) -> WorkQueue:
        return queue()

    async def advance(self, work: WorkQueue, seconds: float) -> None:
        assert isinstance(work, PostgresWorkQueue)
        clock = work._clock
        assert isinstance(clock, FakeClock)
        clock.advance(seconds)


@pytest.mark.skipif(not POSTGRES_DSN, reason="no ephemeral PostgreSQL configured")
class TestWhatOnlyARealDatabaseShows:
    """Contention, atomicity and startup checks, which a scripted fake cannot answer."""

    async def test_ten_workers_take_ten_different_items(self) -> None:
        """`SKIP LOCKED` is the whole point: no duplicate claim, and nobody waits."""
        work = queue()
        for index in range(10):
            await work.enqueue(WorkItem(id=f"i{index}", tenant="acme"))

        claimed = await asyncio.gather(*(work.claim(worker=f"w{index}") for index in range(10)))

        taken = [item.id for item in claimed if item]
        assert sorted(taken) == [f"i{index}" for index in range(10)]

    async def test_a_state_change_and_its_event_commit_together(self) -> None:
        """One transaction: a run that was recorded and an item that was not is the bug."""
        psycopg = pytest.importorskip("psycopg")
        async with await psycopg.AsyncConnection.connect(POSTGRES_DSN, options=TIMED) as held:
            session = sql(held)
            with pytest.raises(RuntimeError, match="something else failed"):
                await _both_then_fail(held, session)

        assert await store().get_run(_key("r1")) is None
        assert await queue().stats() == (await queue().stats()).model_copy(update={"depth": 0})

    async def test_a_schema_it_was_not_written_for_is_refused(self) -> None:
        """A column that moved is a write into the wrong shape, caught at startup."""
        with pytest.raises(ConfigurationError, match="schema is version"):
            await PostgresStateStore.open(
                sql(),
                clock=FakeClock(),
                settings=SETTINGS.model_copy(update={"schema_version": 99}),
                tables=TABLES,
            )

    async def test_a_connection_that_could_run_forever_is_refused(self) -> None:
        """A statement with no timeout holds a pooled connection until the process dies."""
        psycopg = pytest.importorskip("psycopg")
        async with await psycopg.AsyncConnection.connect(POSTGRES_DSN, autocommit=True) as loose:
            await loose.execute("SET statement_timeout = 0")
            with pytest.raises(ConfigurationError, match="statement_timeout"):
                await PostgresWorkQueue.open(
                    sql(loose), clock=FakeClock(), settings=SETTINGS, tables=TABLES
                )


async def _both_then_fail(held: Any, session: Psycopg) -> None:
    """Write a run and the event it caused in one transaction, then fail before it commits."""
    async with held.transaction():
        run = RunRecord(run_id="r1", tenant="acme", agent_name="planner")
        await store().bound(session).put_run(run)
        await queue().bound(session).enqueue(WorkItem(id="i1", tenant="acme"))
        raise RuntimeError("something else failed")


def _key(run_id: str) -> StateKey:
    return StateKey(tenant="acme", id=run_id)
