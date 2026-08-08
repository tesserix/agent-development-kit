"""The four kinds of memory in the three stores that are actually good at holding them.

Working memory is a scratch pad that should expire on its own, which is Redis. Profiles and
episodes are a bitemporal history nobody may overwrite, which is PostgreSQL. Semantic recall
is a ranking, which is pgvector — and the ranking belongs in the SQL, because a scope filter
applied after the fetch has already read the rows it was supposed to exclude.

Each store is partial on purpose: `RedisMemoryStore` does not pretend to rank, and
`PostgresMemoryStore` does not pretend to expire. `RoutedMemoryStore` composes them into the
one `MemoryStore` a consumer binds.

Schema DDL is not here. Tables and indexes are owned by the platform's migration repo, so a
library import can never be the thing that alters a production table.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import Field, SecretStr, ValidationError, field_validator

from tesserix_adk.core.errors import (
    EmbeddingDimensionError,
    MemoryConflictError,
    MemoryCorruptionError,
    MemoryScopeError,
    MemoryUnavailableError,
    PoolExhaustedError,
)
from tesserix_adk.core.models import AdkModel
from tesserix_adk.memory.beliefs import Belief, Supersession
from tesserix_adk.memory.capabilities import MemoryCapabilities
from tesserix_adk.memory.erasure import Derivation, ErasureReceipt
from tesserix_adk.memory.records import MemoryHit, MemoryKind, MemoryRecord

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from pydantic import JsonValue

    from tesserix_adk.adapters.ledger import RedisClient, SqlExecutor
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.memory.records import MemoryQuery
    from tesserix_adk.memory.scope import MemoryScope

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_NAMESPACE",
    "DEFAULT_TABLE",
    "DISTANCE_OPERATORS",
    "MemoryPage",
    "MemoryStoreSettings",
    "PgvectorMemoryStore",
    "PostgresMemoryStore",
    "RedisMemoryStore",
    "RoutedMemoryStore",
]

DEFAULT_NAMESPACE = "adk:mem"
DEFAULT_TABLE = "adk_memory"
DEFAULT_COLLECTION = "adk_semantic"

DISTANCE_OPERATORS = {"cosine": "<=>", "l2": "<->", "inner": "<#>"}
"""pgvector's operator per metric. The index has to have been built for the same one."""

_SHIPPED_PASSWORDS = ("postgres", "password", "changeme", "redis", "admin", "secret")

T = TypeVar("T")


class MemoryStoreSettings(AdkModel):
    """Where a store connects and how hard it tries, from the deployment's secret manager.

    Nothing here is read from a database row or defaulted to something that works: an
    adapter that connects out of the box connects out of the box for everybody.

    Args:
        dsn: The connection string, held as a secret so it does not reach a log.
        pool_size: Connections to keep. Exhaustion is reported, not queued past a deadline.
        connect_timeout_seconds: How long a single connection attempt may take.
        max_attempts: Total attempts, including the first. A failover is waited out; a
            store that is still gone after this many tries fails the run closed.
        backoff_seconds: The base wait, doubled per attempt and jittered.
    """

    dsn: SecretStr
    pool_size: int = Field(default=10, ge=1)
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0)
    max_attempts: int = Field(default=3, ge=1)
    backoff_seconds: float = Field(default=0.05, gt=0.0)

    @field_validator("dsn")
    @classmethod
    def _dsn_is_a_dsn(cls, dsn: SecretStr) -> SecretStr:
        value = dsn.get_secret_value().strip()
        if not value:
            raise ValueError("dsn must name a server")
        if any(f":{shipped}@" in value for shipped in _SHIPPED_PASSWORDS):
            raise ValueError("dsn carries a default password")
        return dsn


class MemoryPage(AdkModel):
    """One page of episodes and where the next one starts.

    Args:
        hits: What this page holds, newest first.
        cursor: Where to resume, or None at the end. A keyset rather than an offset,
            because an offset re-reads every row before it to skip them.
    """

    hits: tuple[MemoryHit, ...] = ()
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class _Retrying:
    """Bounded, jittered retries around one driver call."""

    clock: Clock
    settings: MemoryStoreSettings
    entropy: Callable[[], float]
    store: str

    async def __call__(self, work: Callable[[], Awaitable[T]]) -> T:
        """Run `work`, retrying transport failures until the attempt budget is spent."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await work()
            except Exception as failure:
                self._reraise_if_terminal(failure, attempt)
                await self.clock.sleep(self._backoff(attempt))

    def _reraise_if_terminal(self, failure: Exception, attempt: int) -> None:
        if "pool" in type(failure).__name__.lower():
            raise PoolExhaustedError("memory connection pool exhausted") from failure
        if attempt >= self.settings.max_attempts:
            raise MemoryUnavailableError(
                f"{self.store} unreachable after {attempt} attempts",
                store=self.store,
                attempts=attempt,
            ) from failure

    def _backoff(self, attempt: int) -> float:
        doubling = float(2 ** (attempt - 1))
        return self.settings.backoff_seconds * doubling * (0.5 + self.entropy())


class RedisMemoryStore:
    """Working memory in Redis, where expiry is the server's job and not a sweeper's.

    Args:
        client: Anything that can `eval` Lua. The `redis` extra installs one.
        clock: Where time comes from.
        settings: Connection and retry policy.
        namespace: Key prefix, so this can share a Redis with everything else.
        ttl_seconds: How long a written key lives, or None to leave it to the deployment.
        sliding: Whether reading a key extends it. A conversation still in progress is
            not idle, but a store that slides on read cannot also be a cache with a budget.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        clock: Clock,
        settings: MemoryStoreSettings,
        namespace: str = DEFAULT_NAMESPACE,
        ttl_seconds: float | None = None,
        sliding: bool = False,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._client = client
        self._clock = clock
        self._namespace = namespace
        self._ttl = ttl_seconds
        self._sliding = sliding
        self._retry = _Retrying(clock, settings, entropy, type(self).__name__)

    @property
    def capabilities(self) -> MemoryCapabilities:
        """Working memory only: no ranking, no history, no versions."""
        return MemoryCapabilities(
            supports_semantic=False,
            supports_as_of=False,
            supports_erasure=True,
            supports_supersession=False,
        )

    async def write(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Replace working memory at `record.key`."""
        _belongs(scope, record, MemoryKind.WORKING)
        key = self._key(scope, record.key)
        await self._retry(
            lambda: self._client.eval(WRITE, 1, key, record.model_dump_json(), self._ttl_ms)
        )

    async def read(self, scope: MemoryScope, key: str) -> MemoryRecord | None:
        """Return the working record at `key`, or None where it is gone or expired."""
        argv = (self._ttl_ms,) if self._sliding and self._ttl else ()
        held = await self._retry(lambda: self._client.eval(READ, 1, self._key(scope, key), *argv))
        return _record(held) if held else None

    async def append(self, scope: MemoryScope, key: str, value: JsonValue) -> int:
        """Add `value` to the sequence at `key` and return its position, counting from 1."""
        stored = json.dumps(value)
        return int(
            await self._retry(
                lambda: self._client.eval(APPEND, 1, self._key(scope, key), stored, self._ttl_ms)
            )
        )

    async def expire(self, scope: MemoryScope, key: str, *, ttl_seconds: float) -> None:
        """Have `key` read as absent once `ttl_seconds` have passed."""
        millis = str(int(ttl_seconds * 1_000))
        await self._retry(lambda: self._client.eval(EXPIRE, 1, self._key(scope, key), millis))

    async def recall(self, scope: MemoryScope, keys: Sequence[str]) -> tuple[MemoryRecord, ...]:
        """Read several keys in one round trip, skipping those that are gone."""
        wanted = tuple(self._key(scope, key) for key in keys)
        held = await self._retry(lambda: self._client.eval(RECALL, len(wanted), *wanted))
        return tuple(_record(one) for one in (held or ()) if one)

    async def erase(self, scope: MemoryScope) -> ErasureReceipt:
        """Delete every key under `scope` and report how many went."""
        pattern = f"{self._namespace}:{':'.join(scope.path)}:*"
        gone = int(await self._retry(lambda: self._client.eval(ERASE, 0, pattern)) or 0)
        return ErasureReceipt(
            counts={MemoryKind.WORKING.value: gone} if gone else {},
            adapters=(type(self).__name__,),
            completed_at=self._clock.now(),
            complete=True,
        )

    @property
    def _ttl_ms(self) -> str:
        return str(int((self._ttl or 0) * 1_000))

    def _key(self, scope: MemoryScope, key: str) -> str:
        return f"{self._namespace}:{':'.join(scope.path)}:{MemoryKind.WORKING.value}:{key}"


WRITE = """
local ttl = tonumber(ARGV[2])
if ttl > 0 then return redis.call('SET', KEYS[1], ARGV[1], 'PX', ttl) end
return redis.call('SET', KEYS[1], ARGV[1])
"""

READ = """
local held = redis.call('GET', KEYS[1])
if held and ARGV[1] then redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[1])) end
return held
"""

APPEND = """
local at = redis.call('RPUSH', KEYS[1], ARGV[1])
local ttl = tonumber(ARGV[2])
if ttl > 0 then redis.call('PEXPIRE', KEYS[1], ttl) end
return at
"""

EXPIRE = "return redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[1]))"

RECALL = """
local held = {}
for i = 1, #KEYS do held[i] = redis.call('GET', KEYS[i]) or false end
return held
"""

ERASE = """
local gone, cursor = 0, '0'
repeat
  local page = redis.call('SCAN', cursor, 'MATCH', ARGV[1], 'COUNT', 500)
  cursor = page[1]
  if #page[2] > 0 then gone = gone + redis.call('DEL', unpack(page[2])) end
until cursor == '0'
return gone
"""


class PostgresMemoryStore:
    """Profiles and episodes in PostgreSQL, where nothing is overwritten.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: Where `valid_to` on a superseded record comes from.
        settings: Connection and retry policy.
        table: Where the records live. Created by the platform's migrations, not here.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        clock: Clock,
        settings: MemoryStoreSettings,
        table: str = DEFAULT_TABLE,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._sql = executor
        self._clock = clock
        self._table = table
        self._retry = _Retrying(clock, settings, entropy, type(self).__name__)

    @property
    def capabilities(self) -> MemoryCapabilities:
        """History and versions, but no vector index."""
        return MemoryCapabilities(supports_semantic=False)

    async def upsert(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Write or replace a profile record."""
        _belongs(scope, record, MemoryKind.PROFILE)
        await self._fetch(UPSERT_SQL, *_row(scope, record))

    async def log(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Record that something happened. Retried on its id, so a failover books it once."""
        _belongs(scope, record, MemoryKind.EPISODIC)
        await self._fetch(LOG_SQL, *_row(scope, record))

    async def profile(
        self, scope: MemoryScope, key: str, *, as_of: float | None = None
    ) -> MemoryRecord | None:
        """Return the profile record live at `as_of`, or now, or None."""
        statement = PROFILE_LIVE_SQL if as_of is None else PROFILE_AS_OF_SQL
        extra = () if as_of is None else (as_of,)
        rows = await self._fetch(statement, *scope.path, MemoryKind.PROFILE.value, key, *extra)
        return _record(rows[0][0]) if rows else None

    async def belief(self, scope: MemoryScope, key: str, *, as_of: float | None = None) -> Belief:
        """Return what the scope holds at `key`. Contradictions are a policy layer above."""
        return Belief(record=await self.profile(scope, key, as_of=as_of))

    async def history(self, scope: MemoryScope, key: str | None = None) -> tuple[MemoryRecord, ...]:
        """Return every version under `scope`, oldest first, for `key` or for all keys."""
        rows = await self._fetch(HISTORY_SQL, *scope.path, MemoryKind.PROFILE.value, key)
        return tuple(_record(row[0]) for row in rows)

    async def supersede(
        self,
        scope: MemoryScope,
        record: MemoryRecord,
        *,
        expected_version: int | None = None,
        resolves: tuple[str, ...] = (),
    ) -> Supersession:
        """Write a profile record as a new version, closing whatever it replaced."""
        _belongs(scope, record, MemoryKind.PROFILE)
        live = await self.profile(scope, record.key)
        if live is None:
            await self.upsert(scope, record)
            return Supersession(record=record)
        if expected_version is not None and expected_version != live.version:
            raise MemoryConflictError(
                "the version this write was based on is no longer live",
                key=record.key,
                expected_version=expected_version,
                actual_version=live.version,
            )
        return await self._close(scope, record, live, resolves)

    async def episodes(self, scope: MemoryScope, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        """Return episodes matching `query`, newest first."""
        rows = await self._fetch(
            EPISODES_SQL,
            *scope.path,
            query.kind.value,
            query.since,
            query.until,
            query.limit,
        )
        return tuple(MemoryHit(record=_record(row[0])) for row in rows)

    async def page(
        self, scope: MemoryScope, query: MemoryQuery, *, cursor: str | None = None
    ) -> MemoryPage:
        """Return one page of episodes and where the next one starts.

        A window wide enough to matter is wider than any one response, so it is read in
        keyset pages rather than materialised and truncated.
        """
        after, after_id = _cursor(cursor)
        rows = await self._fetch(
            PAGE_SQL,
            *scope.path,
            query.kind.value,
            query.since,
            query.until,
            after,
            after_id,
            query.limit + 1,
        )
        hits = tuple(MemoryHit(record=_record(row[0])) for row in rows[: query.limit])
        more = len(rows) > query.limit
        return MemoryPage(hits=hits, cursor=_next_cursor(hits) if more else None)

    async def derived(self, scope: MemoryScope, derivation: Derivation) -> None:
        """Record that an artefact was built from a record, so erasure can reach it."""
        await self._fetch(
            DERIVED_SQL,
            *scope.path,
            derivation.artefact_id,
            derivation.source_id,
            derivation.adapter,
        )

    async def derivations(
        self, scope: MemoryScope, *, source_id: str | None = None
    ) -> tuple[Derivation, ...]:
        """Return what has been derived under `scope`, or from one record within it."""
        rows = await self._fetch(DERIVATIONS_SQL, *scope.path, source_id)
        return tuple(
            Derivation(artefact_id=row[0], source_id=row[1], adapter=row[2]) for row in rows
        )

    async def erase(self, scope: MemoryScope) -> ErasureReceipt:
        """Delete every record under `scope` and report what went, by kind."""
        rows = await self._fetch(ERASE_SQL, *scope.path)
        return ErasureReceipt(
            counts={str(row[0]): int(row[1]) for row in rows},
            adapters=(type(self).__name__,),
            completed_at=self._clock.now(),
            complete=True,
        )

    async def _close(
        self,
        scope: MemoryScope,
        record: MemoryRecord,
        live: MemoryRecord,
        resolves: tuple[str, ...],
    ) -> Supersession:
        now = record.valid_from if record.valid_from is not None else self._clock.now()
        written = record.model_copy(update={"version": live.version + 1})
        if resolves:
            await self._fetch(RESOLVE_SQL, *scope.path, list(resolves), now, written.id)
        closed = await self._fetch(
            CLOSE_SQL, *scope.path, record.key, now, written.id, live.version
        )
        if not closed:
            raise MemoryConflictError(
                "another writer closed this version first",
                key=record.key,
                expected_version=live.version,
                actual_version=live.version + 1,
            )
        await self._fetch(UPSERT_SQL, *_row(scope, written))
        superseded = live.model_copy(update={"valid_to": now, "superseded_by": written.id})
        return Supersession(record=written, superseded=superseded, contradiction=None)

    async def _fetch(
        self,
        statement: str,
        *args: Any,  # noqa: ANN401 — SQL arguments are whatever the column holds
    ) -> Sequence[Sequence[Any]]:
        sql = statement.format(table=self._table)
        return await self._retry(lambda: self._sql.fetch(sql, *args))


UPSERT_SQL = """
INSERT INTO {table}
  (id, tenant_id, user_id, session_id, agent, kind, key, version,
   valid_from, valid_to, superseded_by, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
ON CONFLICT (id) DO UPDATE SET
  payload = EXCLUDED.payload, version = EXCLUDED.version,
  valid_from = EXCLUDED.valid_from, valid_to = EXCLUDED.valid_to,
  superseded_by = EXCLUDED.superseded_by
"""

LOG_SQL = """
INSERT INTO {table}
  (id, tenant_id, user_id, session_id, agent, kind, key, version,
   valid_from, valid_to, superseded_by, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
ON CONFLICT (id) DO NOTHING
"""

PROFILE_LIVE_SQL = """
SELECT payload FROM {table}
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  AND kind = $5 AND key = $6 AND valid_to IS NULL
ORDER BY version DESC LIMIT 1
"""

PROFILE_AS_OF_SQL = """
SELECT payload FROM {table}
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  AND kind = $5 AND key = $6
  AND valid_from <= $7 AND (valid_to IS NULL OR valid_to > $7)
ORDER BY version DESC LIMIT 1
"""

HISTORY_SQL = """
SELECT payload FROM {table}
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  AND kind = $5 AND ($6::text IS NULL OR key = $6)
ORDER BY version ASC, valid_from ASC
"""

CLOSE_SQL = """
UPDATE {table} SET valid_to = $6, superseded_by = $7
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  AND key = $5 AND valid_to IS NULL AND version = $8
RETURNING version
"""

RESOLVE_SQL = """
UPDATE {table} SET valid_to = $6, superseded_by = $7
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  AND id = ANY($5) AND valid_to IS NULL
RETURNING id
"""

EPISODES_SQL = """
SELECT payload FROM {table}
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4 AND kind = $5
  AND ($6::float8 IS NULL OR valid_from >= $6)
  AND ($7::float8 IS NULL OR valid_from <= $7)
ORDER BY valid_from DESC, id DESC LIMIT $8
"""

PAGE_SQL = """
SELECT payload FROM {table}
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4 AND kind = $5
  AND ($6::float8 IS NULL OR valid_from >= $6)
  AND ($7::float8 IS NULL OR valid_from <= $7)
  AND ($8::float8 IS NULL OR (valid_from, id) < ($8, $9))
ORDER BY valid_from DESC, id DESC LIMIT $10
"""

DERIVED_SQL = """
INSERT INTO {table}_derivations
  (tenant_id, user_id, session_id, agent, artefact_id, source_id, adapter)
VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (artefact_id) DO NOTHING
"""

DERIVATIONS_SQL = """
SELECT artefact_id, source_id, adapter FROM {table}_derivations
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  AND ($5::text IS NULL OR source_id = $5)
"""

ERASE_SQL = """
WITH gone AS (
  DELETE FROM {table}
  WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  RETURNING kind
)
SELECT kind, count(*) FROM gone GROUP BY kind
"""


class PgvectorMemoryStore:
    """Semantic recall in pgvector, ranked by the database rather than by the process.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: Where the erasure receipt's timestamp comes from.
        settings: Connection and retry policy.
        dimensions: The width this collection was built for. Declared rather than
            inferred, so `verify` can catch an embedder that disagrees at startup.
        collection: The table holding the vectors.
        metric: Which distance the index was built for. The operator follows it.
        index_type: `hnsw` or `ivfflat`, for a deployment that reports what it is using.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        clock: Clock,
        settings: MemoryStoreSettings,
        dimensions: int,
        collection: str = DEFAULT_COLLECTION,
        metric: str = "cosine",
        index_type: str = "hnsw",
        entropy: Callable[[], float] = random.random,
    ) -> None:
        if metric not in DISTANCE_OPERATORS:
            known = sorted(DISTANCE_OPERATORS)
            raise ValueError(f"unknown metric {metric!r}: expected one of {known}")
        self._sql = executor
        self._clock = clock
        self._dimensions = dimensions
        self._collection = collection
        self._metric = metric
        self._index_type = index_type
        self._retry = _Retrying(clock, settings, entropy, type(self).__name__)

    @property
    def capabilities(self) -> MemoryCapabilities:
        """Ranking, and nothing that needs a version."""
        return MemoryCapabilities(
            supports_as_of=False,
            supports_supersession=False,
            embedding_dimensions=self._dimensions,
        )

    @property
    def index_type(self) -> str:
        """Which index this collection was built with."""
        return self._index_type

    async def verify(self) -> None:
        """Check the collection is as wide as the embedder, at startup.

        A collection two dimensions narrower than the embedder does not fail: it ranks
        badly, on the first recall, a month later, with nobody watching.

        Raises:
            EmbeddingDimensionError: If the collection is missing or the wrong width.
        """
        rows = await self._retry(lambda: self._sql.fetch(DIMENSION_SQL, self._collection))
        if not rows:
            raise EmbeddingDimensionError(
                f"collection {self._collection!r} has no embedding column",
                expected=self._dimensions,
                received=0,
            )
        if int(rows[0][0]) != self._dimensions:
            raise EmbeddingDimensionError(
                f"collection {self._collection!r} is {rows[0][0]} wide, "
                f"but this store was configured for {self._dimensions}",
                expected=self._dimensions,
                received=int(rows[0][0]),
            )

    async def index(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Add a semantic record to the collection."""
        _belongs(scope, record, MemoryKind.SEMANTIC)
        vector = self._vector(record.embedding)
        await self._retry(
            lambda: self._sql.fetch(
                INDEX_SQL.format(collection=self._collection),
                record.id,
                *scope.path,
                record.key,
                vector,
                record.model_dump_json(),
            )
        )

    async def search(self, scope: MemoryScope, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        """Return semantic records ranked by resemblance, closest first."""
        vector = self._vector(query.embedding)
        statement = SEARCH_SQL.format(
            collection=self._collection, operator=DISTANCE_OPERATORS[self._metric]
        )
        rows = await self._retry(
            lambda: self._sql.fetch(statement, *scope.path, vector, query.limit)
        )
        return tuple(
            MemoryHit(record=_record(row[0]), score=max(0.0, 1.0 - float(row[1]))) for row in rows
        )

    async def erase(self, scope: MemoryScope) -> ErasureReceipt:
        """Delete every vector under `scope` and report how many went."""
        rows = await self._retry(
            lambda: self._sql.fetch(
                ERASE_VECTORS_SQL.format(collection=self._collection), *scope.path
            )
        )
        return ErasureReceipt(
            counts={str(row[0]): int(row[1]) for row in rows},
            adapters=(type(self).__name__,),
            completed_at=self._clock.now(),
            complete=True,
        )

    def _vector(self, embedding: tuple[float, ...] | None) -> str:
        if embedding is None or len(embedding) != self._dimensions:
            raise EmbeddingDimensionError(
                f"this collection holds {self._dimensions}-dimension vectors",
                expected=self._dimensions,
                received=len(embedding or ()),
            )
        return "[" + ",".join(str(one) for one in embedding) + "]"


DIMENSION_SQL = """
SELECT a.atttypmod FROM pg_attribute a
WHERE a.attrelid = $1::regclass AND a.attname = 'embedding'
"""

INDEX_SQL = """
INSERT INTO {collection}
  (id, tenant_id, user_id, session_id, agent, key, embedding, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload
"""

SEARCH_SQL = """
SELECT payload, embedding {operator} $5 AS distance FROM {collection}
WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
ORDER BY embedding {operator} $5 LIMIT $6
"""

ERASE_VECTORS_SQL = """
WITH gone AS (
  DELETE FROM {collection}
  WHERE tenant_id = $1 AND user_id = $2 AND session_id = $3 AND agent = $4
  RETURNING 'semantic' AS kind
)
SELECT kind, count(*) FROM gone GROUP BY kind
"""


class RoutedMemoryStore:
    """One `MemoryStore` over the three stores, routing by kind.

    Args:
        working: Where working memory lives.
        durable: Where profiles and episodes live.
        semantic: Where vectors live.
    """

    def __init__(
        self,
        *,
        working: RedisMemoryStore,
        durable: PostgresMemoryStore,
        semantic: PgvectorMemoryStore,
    ) -> None:
        self._working = working
        self._durable = durable
        self._semantic = semantic

    @property
    def capabilities(self) -> MemoryCapabilities:
        """Everything its parts can do between them."""
        return MemoryCapabilities(
            supports_semantic=self._semantic.capabilities.supports_semantic,
            supports_as_of=self._durable.capabilities.supports_as_of,
            supports_erasure=True,
            supports_supersession=self._durable.capabilities.supports_supersession,
            embedding_dimensions=self._semantic.capabilities.embedding_dimensions,
        )

    async def write(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Replace working memory at `record.key`."""
        await self._working.write(scope, record)

    async def read(self, scope: MemoryScope, key: str) -> MemoryRecord | None:
        """Return the working record at `key`, or None."""
        return await self._working.read(scope, key)

    async def append(self, scope: MemoryScope, key: str, value: JsonValue) -> int:
        """Add `value` to the sequence at `key` and return its position."""
        return await self._working.append(scope, key, value)

    async def expire(self, scope: MemoryScope, key: str, *, ttl_seconds: float) -> None:
        """Have `key` read as absent once `ttl_seconds` have passed."""
        await self._working.expire(scope, key, ttl_seconds=ttl_seconds)

    async def upsert(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Write or replace a profile record."""
        await self._durable.upsert(scope, record)

    async def profile(
        self, scope: MemoryScope, key: str, *, as_of: float | None = None
    ) -> MemoryRecord | None:
        """Return the profile record live at `as_of`, or now, or None."""
        return await self._durable.profile(scope, key, as_of=as_of)

    async def belief(self, scope: MemoryScope, key: str, *, as_of: float | None = None) -> Belief:
        """Return what the scope holds at `key`."""
        return await self._durable.belief(scope, key, as_of=as_of)

    async def history(self, scope: MemoryScope, key: str | None = None) -> tuple[MemoryRecord, ...]:
        """Return every version under `scope`, oldest first."""
        return await self._durable.history(scope, key)

    async def supersede(
        self,
        scope: MemoryScope,
        record: MemoryRecord,
        *,
        expected_version: int | None = None,
        resolves: tuple[str, ...] = (),
    ) -> Supersession:
        """Write a profile record as a new version, closing whatever it replaced."""
        return await self._durable.supersede(
            scope, record, expected_version=expected_version, resolves=resolves
        )

    async def log(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Record that something happened."""
        await self._durable.log(scope, record)

    async def episodes(self, scope: MemoryScope, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        """Return episodes matching `query`, newest first."""
        return await self._durable.episodes(scope, query)

    async def index(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Add a semantic record to the collection."""
        await self._semantic.index(scope, record)

    async def search(self, scope: MemoryScope, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        """Return semantic records ranked by resemblance, closest first."""
        return await self._semantic.search(scope, query)

    async def derived(self, scope: MemoryScope, derivation: Derivation) -> None:
        """Record that an artefact was built from a record."""
        await self._durable.derived(scope, derivation)

    async def derivations(
        self, scope: MemoryScope, *, source_id: str | None = None
    ) -> tuple[Derivation, ...]:
        """Return what has been derived under `scope`."""
        return await self._durable.derivations(scope, source_id=source_id)

    async def erase(
        self,
        scope: MemoryScope,
        *,
        kinds: tuple[MemoryKind, ...] = (),
        dry_run: bool = False,
    ) -> ErasureReceipt:
        """Delete everything under `scope` across all three stores.

        Raises:
            CapabilityError: Never. Every store here can erase.
            NotImplementedError: If `kinds` or `dry_run` is given. Neither is honest
                against three servers without a two-phase protocol they do not share.
        """
        if kinds or dry_run:
            raise NotImplementedError(
                "narrowed and dry-run erasure are not supported across three stores"
            )
        receipts = [
            await self._working.erase(scope),
            await self._durable.erase(scope),
            await self._semantic.erase(scope),
        ]
        counts: dict[str, int] = {}
        for receipt in receipts:
            for kind, gone in receipt.counts.items():
                counts[kind] = counts.get(kind, 0) + gone
        return ErasureReceipt(
            counts=counts,
            adapters=tuple(name for receipt in receipts for name in receipt.adapters),
            completed_at=receipts[-1].completed_at,
            complete=True,
        )


def _belongs(scope: MemoryScope, record: MemoryRecord, kind: MemoryKind) -> None:
    if record.scope.path != scope.path:
        raise MemoryScopeError(
            "the record does not belong to the scope it was written under",
            expected=str(record.scope.path),
            given=str(scope.path),
        )
    if record.kind is not kind:
        raise MemoryScopeError(
            f"this store holds {kind.value} records, not {record.kind.value}",
            expected=kind.value,
            given=record.kind.value,
        )


def _row(scope: MemoryScope, record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.id,
        *scope.path,
        record.kind.value,
        record.key,
        record.version,
        record.valid_from,
        record.valid_to,
        record.superseded_by,
        record.model_dump_json(),
    )


def _record(raw: Any) -> MemoryRecord:  # noqa: ANN401 — whatever the driver decoded
    """Turn a payload column into a record, whichever way the driver decoded it.

    Raises:
        MemoryCorruptionError: If what came back no longer validates as a record.
    """
    encoded = raw if isinstance(raw, str | bytes) else json.dumps(raw)
    try:
        return MemoryRecord.model_validate_json(encoded)
    except ValidationError as invalid:
        raise MemoryCorruptionError("a stored record no longer validates") from invalid


def _cursor(cursor: str | None) -> tuple[float | None, str | None]:
    if cursor is None:
        return (None, None)
    at, _, identifier = cursor.partition("|")
    return (float(at), identifier)


def _next_cursor(hits: tuple[MemoryHit, ...]) -> str:
    last = hits[-1].record
    return f"{last.valid_from}|{last.id}"
