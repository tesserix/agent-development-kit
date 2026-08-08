"""What your scope becomes on the wire, without a server in the room.

The stand-ins below answer the way Redis and PostgreSQL would; what the adapters *send*
is the interesting part. Run it with `python examples/memory_adapters.py`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import JsonValue, SecretStr, ValidationError

from tesserix_adk.adapters import (
    MemoryStoreSettings,
    PgvectorMemoryStore,
    PostgresMemoryStore,
    RedisMemoryStore,
    RoutedMemoryStore,
)
from tesserix_adk.core import MemoryUnavailableError
from tesserix_adk.memory import MemoryKind, MemoryQuery, MemoryRecord, MemoryScope
from tesserix_adk.testing import FakeClock

SCOPE = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1")
NOW = 1_000.0
SETTINGS = MemoryStoreSettings(dsn=SecretStr("postgresql://adk:s3cret@db/adk"))


class ShoutingRedis:
    """Prints the key and returns whatever it was told to."""

    def __init__(self, *replies: Any) -> None:  # noqa: ANN401 — whatever the server would say
        self.replies = list(replies)
        self.down = 0

    async def eval(
        self,
        script: str,  # noqa: ARG002 — a real server would run it
        numkeys: int,  # noqa: ARG002 — a real server would split the keys off here
        *args: str,
    ) -> Any:  # noqa: ANN401 — the reply shape is the script's, not the client's
        """Answer one call, or fail the way an unreachable server fails."""
        if self.down:
            self.down -= 1
            raise ConnectionError("primary went away")
        print("  redis key:", args[0])  # noqa: T201
        return self.replies.pop(0) if self.replies else None


class ShoutingSql:
    """Prints the predicate the scope turned into."""

    def __init__(self, *replies: Any) -> None:  # noqa: ANN401 — whatever the rows would hold
        self.replies = list(replies)

    async def fetch(self, statement: str, *args: Any) -> list[list[Any]]:  # noqa: ANN401 — as above
        """Answer one statement, printing its first line and its scope arguments."""
        print("  sql:", statement.strip().splitlines()[0], "| scope:", args[:4])  # noqa: T201
        return self.replies.pop(0) if self.replies else []


def note(kind: MemoryKind, key: str, value: JsonValue) -> MemoryRecord:
    """One record, filled in enough to be stored."""
    return MemoryRecord(
        id=f"{kind.value}:{key}",
        kind=kind,
        scope=SCOPE,
        key=key,
        value=value,
        source="turn",
        valid_from=NOW,
        embedding=(1.0, 0.0, 0.0) if kind is MemoryKind.SEMANTIC else None,
    )


def routed(redis: ShoutingRedis, sql: ShoutingSql) -> RoutedMemoryStore:
    """Working memory in Redis, everything durable in PostgreSQL, vectors in pgvector."""
    clock = FakeClock(start=NOW)
    return RoutedMemoryStore(
        working=RedisMemoryStore(redis, clock=clock, settings=SETTINGS, ttl_seconds=3600),
        durable=PostgresMemoryStore(sql, clock=clock, settings=SETTINGS),
        semantic=PgvectorMemoryStore(sql, clock=clock, settings=SETTINGS, dimensions=3),
    )


async def each_kind_to_its_own_store() -> None:
    """One call per kind, and the key or predicate each one becomes."""
    print("each kind:")  # noqa: T201
    store = routed(ShoutingRedis(None), ShoutingSql([], []))
    await store.write(SCOPE, note(MemoryKind.WORKING, "draft", "half a sentence"))
    await store.log(SCOPE, note(MemoryKind.EPISODIC, "e", "booked seat 14C"))
    await store.index(SCOPE, note(MemoryKind.SEMANTIC, "s", "prefers the aisle"))


async def credentials_are_refused_not_defaulted() -> None:
    """A DSN nobody had to change is one nobody did change."""
    print("credentials:")  # noqa: T201
    try:
        MemoryStoreSettings(dsn=SecretStr("postgresql://adk:password@db/adk"))
    except ValidationError as refused:
        print("  refused:", refused.errors()[0]["msg"])  # noqa: T201
    print("  repr:", repr(SETTINGS.dsn))  # noqa: T201


async def a_wide_window_is_read_in_pages() -> None:
    """Three rows, two to a page, and where the next one starts."""
    print("paging:")  # noqa: T201
    rows = [[json.loads(note(MemoryKind.EPISODIC, f"e{n}", n).model_dump_json())] for n in range(3)]
    durable = PostgresMemoryStore(ShoutingSql(rows), clock=FakeClock(start=NOW), settings=SETTINGS)
    page = await durable.page(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC, limit=2))
    print("  held:", len(page.hits), "| next from:", page.cursor)  # noqa: T201


async def a_failover_is_waited_out_and_then_is_not() -> None:
    """Two failures are ordinary. Enough of them means the run fails closed."""
    print("failover:")  # noqa: T201
    redis = ShoutingRedis(None)
    redis.down = 3
    store = routed(redis, ShoutingSql())
    try:
        await store.read(SCOPE, "draft")
    except MemoryUnavailableError as gone:
        print("  gave up:", gone.store, "after", gone.attempts, "attempts")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await each_kind_to_its_own_store()
    await credentials_are_refused_not_defaulted()
    await a_wide_window_is_read_in_pages()
    await a_failover_is_waited_out_and_then_is_not()


if __name__ == "__main__":
    asyncio.run(main())
