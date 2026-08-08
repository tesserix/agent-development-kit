"""The same conformance suite the fakes pass, against real Redis and PostgreSQL.

Opted into with `-m integration` and two environment variables, because the default lane
reaches no network. The unit tests check what the adapters send; this checks that what
they send is valid Lua and valid SQL, which no fake can tell you.

    ADK_TEST_REDIS_URL=redis://localhost:6379/9 \\
    ADK_TEST_POSTGRES_DSN=postgresql://adk:...@localhost/adk_test \\
    uv run pytest tests/integration -m integration
"""

from __future__ import annotations

import os
import uuid
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
from tesserix_adk.testing import FakeClock, MemoryStoreConformance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.memory import MemoryScope

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("ADK_TEST_REDIS_URL", "")
POSTGRES_DSN = os.environ.get("ADK_TEST_POSTGRES_DSN", "")

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS adk_memory (
  id text PRIMARY KEY, tenant_id text NOT NULL, user_id text NOT NULL,
  session_id text NOT NULL, agent text NOT NULL, kind text NOT NULL, key text NOT NULL,
  version integer NOT NULL, valid_from double precision, valid_to double precision,
  superseded_by text, payload jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS adk_memory_scope
  ON adk_memory (tenant_id, user_id, session_id, agent, kind, key);
CREATE TABLE IF NOT EXISTS adk_memory_derivations (
  artefact_id text PRIMARY KEY, tenant_id text NOT NULL, user_id text NOT NULL,
  session_id text NOT NULL, agent text NOT NULL, source_id text NOT NULL, adapter text NOT NULL
);
CREATE TABLE IF NOT EXISTS adk_semantic (
  id text PRIMARY KEY, tenant_id text NOT NULL, user_id text NOT NULL,
  session_id text NOT NULL, agent text NOT NULL, key text NOT NULL,
  embedding vector(8), payload jsonb NOT NULL
);
"""


class Psycopg:
    """`SqlExecutor` over psycopg, a connection per call — pooling is the deployment's."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """Run `statement` with positional arguments and return whatever it selected."""
        psycopg = pytest.importorskip("psycopg")
        async with (
            await psycopg.AsyncConnection.connect(self._dsn, autocommit=True) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(_positional(statement), args)
            return list(await cursor.fetchall()) if cursor.description else []


def _positional(statement: str) -> str:
    """psycopg speaks %s; the adapters are written against PostgreSQL's own $n."""
    for index in range(12, 0, -1):
        statement = statement.replace(f"${index}", "%s")
    return statement


class Isolated:
    """Every store the suite makes gets its own tenant, so one run cannot see another's."""

    def __init__(self, store: RoutedMemoryStore) -> None:
        self._store = store
        self._run = uuid.uuid4().hex

    def __getattr__(self, name: str) -> Any:
        inner = getattr(self._store, name)
        if not callable(inner):
            return inner

        async def scoped(scope: MemoryScope, *args: Any, **kwargs: Any) -> Any:
            mine = scope.model_copy(update={"tenant_id": f"{scope.tenant_id}-{self._run}"})
            return await inner(mine, *args, **kwargs)

        return scoped


@pytest.fixture(scope="session", autouse=True)
async def schema() -> None:
    """Create the tables the adapters expect. In production these are migrations."""
    psycopg = pytest.importorskip("psycopg")
    async with await psycopg.AsyncConnection.connect(POSTGRES_DSN, autocommit=True) as connection:
        await connection.execute(SCHEMA)


@pytest.mark.skipif(not REDIS_URL or not POSTGRES_DSN, reason="no ephemeral servers configured")
class TestTheRealStores(MemoryStoreConformance):
    """Identical assertions to the in-memory store's, against three real servers."""

    def make_store(self) -> Any:
        redis = pytest.importorskip("redis.asyncio")
        settings = MemoryStoreSettings(dsn=SecretStr(POSTGRES_DSN))
        clock = FakeClock()
        sql = Psycopg(POSTGRES_DSN)
        return Isolated(
            RoutedMemoryStore(
                working=RedisMemoryStore(redis.from_url(REDIS_URL), clock=clock, settings=settings),
                durable=PostgresMemoryStore(sql, clock=clock, settings=settings),
                semantic=PgvectorMemoryStore(sql, clock=clock, settings=settings, dimensions=8),
            )
        )
