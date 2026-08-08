"""Durable records of executed side effects, so a restart does not book a second seat.

The in-process store is enough for one replica and for tests. A deployment that runs more
than one worker, or that restarts, needs the record somewhere both workers can see it.
Both stores here claim a key in a single server-side operation: two round trips are two
chances for the other replica to slip between the read and the claim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from tesserix_adk.core.idempotency import Claim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "IN_FLIGHT",
    "PostgresIdempotencyStore",
    "RedisIdempotencyStore",
]

IN_FLIGHT = "\x00in-flight"
"""What a claim holds until an outcome replaces it. Not a value any tool can return."""


class RedisIdempotencyStore:
    """The record in Redis, with the claim and the read in one script.

    Args:
        client: Anything that can `eval` Lua. The `redis` extra installs one.
        clock: Where time comes from. Unused by the server, kept for parity with the
            in-process store and for a deployment that wants to log claim ages.
        namespace: Key prefix, so this can share a Redis with everything else.
    """

    def __init__(self, client: RedisClient, *, clock: Clock, namespace: str = "adk:effect") -> None:
        self._client = client
        self._clock = clock
        self._namespace = namespace

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Claim:
        """Claim `key`, or report what already holds it.

        Raises:
            Exception: Whatever the client raises. Unreachable is not free: the caller
                fails the call closed rather than treating silence as permission.
        """
        held = await self._client.eval(
            CLAIM, 1, self._key(key, tenant), IN_FLIGHT, str(int(ttl_seconds * 1_000))
        )
        return _claim_from(held)

    async def record(self, key: str, *, tenant: str, outcome: str, ttl_seconds: float) -> None:
        """Replace the claim with what the call returned."""
        await self._client.eval(
            RECORD, 1, self._key(key, tenant), outcome, str(int(ttl_seconds * 1_000))
        )

    async def abandon(self, key: str, *, tenant: str) -> None:
        """Release `key` without a record."""
        await self._client.eval(ABANDON, 1, self._key(key, tenant))

    async def forget(self, *, tenant: str) -> int:
        """Delete every record for `tenant`, and return how many. Erasure reaches here."""
        gone = await self._client.eval(FORGET, 0, f"{self._namespace}:{tenant}:*")
        return int(gone or 0)

    def _key(self, key: str, tenant: str) -> str:
        return f"{self._namespace}:{tenant}:{key}"


class PostgresIdempotencyStore:
    """The record in PostgreSQL, with the claim and the read in one statement.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: Where the expiry written into the row comes from.
        table: Where the records live.
    """

    def __init__(self, executor: SqlExecutor, *, clock: Clock, table: str = "adk_effects") -> None:
        self._sql = executor
        self._clock = clock
        self._table = table

    async def ensure_schema(self) -> None:
        """Create the table if it is not there. Called by a deployment, never on a request."""
        await self._sql.fetch(SCHEMA.format(table=self._table))

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Claim:
        """Claim `key`, or report what already holds it.

        Raises:
            Exception: Whatever the executor raises, for the same reason as the Redis store.
        """
        now = self._clock.now()
        rows = await self._sql.fetch(
            CLAIM_SQL.format(table=self._table), key, tenant, IN_FLIGHT, now, now + ttl_seconds
        )
        return _claim_from(rows[0][0] if rows and rows[0] else None)

    async def record(self, key: str, *, tenant: str, outcome: str, ttl_seconds: float) -> None:
        """Write what the call returned against the claim it holds."""
        now = self._clock.now()
        await self._sql.fetch(
            RECORD_SQL.format(table=self._table), key, tenant, outcome, now, now + ttl_seconds
        )

    async def abandon(self, key: str, *, tenant: str) -> None:
        """Release `key` without a record."""
        await self._sql.fetch(ABANDON_SQL.format(table=self._table), key, tenant)

    async def forget(self, *, tenant: str) -> int:
        """Delete every record for `tenant`, and return how many. Erasure reaches here."""
        rows = await self._sql.fetch(FORGET_SQL.format(table=self._table), tenant)
        return int(rows[0][0]) if rows and rows[0] else 0


def _claim_from(held: Any) -> Claim:
    """What the server said, as a claim. Nothing held means this caller may proceed."""
    if held is None:
        return Claim()
    outcome = held.decode() if isinstance(held, bytes | bytearray) else str(held)
    return Claim(in_flight=True) if outcome == IN_FLIGHT else Claim(outcome=outcome)


class RedisClient(Protocol):
    """The one call these stores make, and nothing else."""

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        """Run a Lua script server-side, which is where atomicity comes from."""
        ...


class SqlExecutor(Protocol):
    """The one call the PostgreSQL store makes."""

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """Run a statement and return its rows."""
        ...


CLAIM = """
local held = redis.call('GET', KEYS[1])
if held then return held end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
return false
"""

RECORD = """
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
return 1
"""

ABANDON = """
return redis.call('DEL', KEYS[1])
"""

FORGET = """
local gone, cursor = 0, '0'
repeat
  local scan = redis.call('SCAN', cursor, 'MATCH', ARGV[1], 'COUNT', 200)
  cursor = scan[1]
  for _, name in ipairs(scan[2]) do
    gone = gone + redis.call('DEL', name)
  end
until cursor == '0'
return gone
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    key        text NOT NULL,
    tenant     text NOT NULL,
    outcome    text NOT NULL,
    claimed_at double precision NOT NULL,
    expires_at double precision NOT NULL,
    PRIMARY KEY (tenant, key)
);
CREATE INDEX IF NOT EXISTS {table}_expiry ON {table} (expires_at);
"""

CLAIM_SQL = """
WITH swept AS (
    DELETE FROM {table} WHERE tenant = $2 AND key = $1 AND expires_at <= $4 RETURNING 1
), taken AS (
    INSERT INTO {table} (key, tenant, outcome, claimed_at, expires_at)
    VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (tenant, key) DO NOTHING
    RETURNING 1
)
SELECT CASE WHEN EXISTS (SELECT 1 FROM taken) THEN NULL
            ELSE (SELECT outcome FROM {table} WHERE tenant = $2 AND key = $1) END
"""

RECORD_SQL = """
INSERT INTO {table} (key, tenant, outcome, claimed_at, expires_at)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (tenant, key) DO UPDATE SET outcome = $3, expires_at = $5
RETURNING 1
"""

ABANDON_SQL = """
DELETE FROM {table} WHERE tenant = $2 AND key = $1 RETURNING 1
"""

FORGET_SQL = """
WITH gone AS (DELETE FROM {table} WHERE tenant = $1 RETURNING 1)
SELECT count(*) FROM gone
"""
