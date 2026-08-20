"""Frontiers and run leases in Redis or PostgreSQL, so a run survives the process.

Durability here is available without adopting a workflow engine: a checkpoint is one row or
one key, and the lease beside it is the reason only one worker advances the run. Both halves
are needed together — a frontier two workers can read is a frontier two workers resume.

Every acquisition decision is taken server-side and returns the fence with it, because the
alternative is a read followed by a write with another worker's acquisition in between. The
worker's own clock is never sent: expiry is `now()` on the server, on the one clock every
worker shares.

The DDL is not applied here. `EXPECTED_CHECKPOINT_SCHEMA` is what the migration repository
owns, and `verify()` refuses a shape this adapter was not written for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters.state import _text
from tesserix_adk.core.checkpoint import Checkpoint
from tesserix_adk.core.errors import ConfigurationError, RunLeaseError, StatePersistenceError
from tesserix_adk.core.lease import RunLease
from tesserix_adk.core.persistence import StateKind, packed, unpacked

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.adapters.idempotency import RedisClient, SqlExecutor

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_CHECKPOINT_NAMESPACE",
    "DEFAULT_CHECKPOINT_TABLES",
    "EXPECTED_CHECKPOINT_SCHEMA",
    "CheckpointTables",
    "PostgresCheckpointStore",
    "PostgresLeaseStore",
    "RedisCheckpointStore",
    "RedisLeaseStore",
]

CHECKPOINT_SCHEMA_VERSION = 1
"""The shape this adapter was written for, recorded by the migration that applied it."""

DEFAULT_CHECKPOINT_NAMESPACE = "adk:checkpoint"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class CheckpointTables:
    """Where the rows live. One database may hold more than one kit deployment.

    Args:
        checkpoints: One row per run, holding its latest frontier.
        leases: One row per run, holding who may advance it and until when.
        schema: Where the migration recorded which shape it applied.
    """

    checkpoints: str = "adk_checkpoints"
    leases: str = "adk_run_leases"
    schema: str = "adk_schema"

    def __post_init__(self) -> None:
        """Refuse a name that could carry SQL, since a table name cannot be a parameter."""
        for name in (self.checkpoints, self.leases, self.schema):
            if not _IDENTIFIER.fullmatch(name):
                raise ConfigurationError(f"{name!r} is not a plain table identifier")


DEFAULT_CHECKPOINT_TABLES = CheckpointTables()


class RedisCheckpointStore:
    """The frontier as one key per run, with no expiry on it.

    A checkpoint with a TTL is a run that stops being resumable at a time nobody chose. It
    is deleted when the run reaches a terminal state and not before, which is what `forget`
    is for.

    Args:
        client: Anything that can `eval` Lua. The `redis` extra installs one.
        namespace: Key prefix, so this can share an instance with everything else.
        max_value_bytes: The largest frontier this store will write. A value big enough to
            matter blocks the single thread every other worker is waiting on.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        namespace: str = DEFAULT_CHECKPOINT_NAMESPACE,
        max_value_bytes: int = 262_144,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._max_bytes = max_value_bytes

    async def put(self, checkpoint: Checkpoint) -> None:
        """Store the frontier, replacing any earlier one for the run.

        Raises:
            StatePersistenceError: If the frontier is larger than this store writes. Half a
                frontier resumes into a conversation that never happened.
        """
        blob = packed(checkpoint, kind=StateKind.CHECKPOINT)
        size = len(blob.encode())
        if size > self._max_bytes:
            raise StatePersistenceError(
                f"{checkpoint.run_id} is {size} bytes, over the {self._max_bytes} this store"
                " writes; hold a frontier this size in the PostgreSQL checkpoint store",
                store=type(self).__name__,
                reason="too_large",
            )
        await self._client.eval(PUT, 1, self._key(checkpoint.run_id, checkpoint.tenant), blob)

    async def latest(self, run_id: str, *, tenant: str) -> Checkpoint | None:
        """Return the frontier of `run_id`, or `None` where nothing was written."""
        blob = await self._client.eval(GET, 1, self._key(run_id, tenant))
        return None if not blob else unpacked(_text(blob), Checkpoint, kind=StateKind.CHECKPOINT)

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop the frontier, because the run reached a terminal state."""
        await self._client.eval(FORGET, 1, self._key(run_id, tenant))

    def _key(self, run_id: str, tenant: str) -> str:
        return f"{self._namespace}:{tenant}:{run_id}"


class RedisLeaseStore:
    """The lease as a hash per run, with the fence incremented inside the acquisition.

    Every decision is one script: take it if nobody live holds it, and return the fence the
    winner got. Expiry is judged against the server's `TIME`, so a worker whose own clock
    runs fast cannot take a lease that has not lapsed.

    Args:
        client: Anything that can `eval` Lua. The `redis` extra installs one.
        namespace: Key prefix, so this can share an instance with everything else.
    """

    def __init__(self, client: RedisClient, *, namespace: str = "adk:lease") -> None:
        self._client = client
        self._namespace = namespace

    async def acquire(
        self, run_id: str, *, tenant: str, holder: str, ttl_seconds: float
    ) -> RunLease:
        """Take the lease on `run_id`, or refuse because a live one is held elsewhere.

        Raises:
            RunLeaseError: If another holder's lease has not expired.
        """
        reply = await self._client.eval(
            ACQUIRE, 1, self._key(run_id, tenant), holder, str(ttl_seconds)
        )
        answer = [_text(part) for part in reply]
        if answer[0] == "held":
            raise RunLeaseError(
                f"{run_id} is held by {answer[1]} until {answer[2]}",
                run_id=run_id,
                holder=answer[1],
                requested_by=holder,
                fence=int(answer[3]),
            )
        return RunLease(
            run_id=run_id,
            tenant=tenant,
            holder=holder,
            fence=int(answer[1]),
            expires_at=float(answer[2]),
        )

    async def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        """Extend `lease`, or refuse because the fence has moved on.

        Raises:
            RunLeaseError: If the run is now held by somebody else, or by nobody.
        """
        reply = await self._client.eval(
            RENEW,
            1,
            self._key(lease.run_id, lease.tenant),
            lease.holder,
            str(lease.fence),
            str(ttl_seconds),
        )
        answer = [_text(part) for part in reply]
        if answer[0] == "lost":
            raise RunLeaseError(
                f"{lease.run_id} is no longer held by {lease.holder}",
                run_id=lease.run_id,
                holder=answer[1],
                requested_by=lease.holder,
                fence=lease.fence,
            )
        return lease.model_copy(update={"expires_at": float(answer[1])})

    async def release(self, lease: RunLease) -> None:
        """Give the lease up, unless it has already been taken by someone else."""
        await self._client.eval(
            RELEASE, 1, self._key(lease.run_id, lease.tenant), lease.holder, str(lease.fence)
        )

    async def held(self, run_id: str, *, tenant: str) -> RunLease | None:
        """Return the current lease on `run_id`, expired or not."""
        reply = await self._client.eval(HELD, 1, self._key(run_id, tenant))
        if not reply:
            return None
        holder, fence, expires_at = (_text(part) for part in reply)
        return RunLease(
            run_id=run_id,
            tenant=tenant,
            holder=holder,
            fence=int(fence),
            expires_at=float(expires_at),
        )

    def _key(self, run_id: str, tenant: str) -> str:
        return f"{self._namespace}:{tenant}:{run_id}"


class PostgresCheckpointStore:
    """The frontier as one row per run, upserted.

    Last write wins, which is what a frontier is: two workers writing one are two views of
    the same monotonic progress, and only one of them holds the lease anyway.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        tables: Where the rows live.
        schema_version: The shape this deployment expects, checked by `verify`.
    """

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        tables: CheckpointTables = DEFAULT_CHECKPOINT_TABLES,
        schema_version: int = CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        self._sql = executor
        self._tables = tables
        self._schema_version = schema_version

    @classmethod
    async def open(cls, executor: SqlExecutor, **kwargs: Any) -> PostgresCheckpointStore:  # noqa: ANN401 — the constructor's own keywords
        """Construct the store and refuse a database that is not the shape it expects.

        Raises:
            ConfigurationError: If the schema version differs from the adapter's.
        """
        store = cls(executor, **kwargs)
        await store.verify()
        return store

    async def verify(self) -> None:
        """Check the schema version. At startup, never per call.

        Raises:
            ConfigurationError: If the shape has moved. A column that moved is a frontier
                read back into the wrong shape, which resumes a run that never happened.
        """
        rows = await self._sql.fetch(SCHEMA_OF.format(schema=self._tables.schema), "checkpoints")
        found = int(rows[0][0]) if rows and rows[0] else 0
        if found != self._schema_version:
            raise ConfigurationError(
                f"checkpoints schema is version {found}, and this adapter writes version"
                f" {self._schema_version}"
            )

    async def put(self, checkpoint: Checkpoint) -> None:
        """Store the frontier, replacing any earlier one for the run."""
        await self._sql.fetch(
            PUT_SQL.format(checkpoints=self._tables.checkpoints),
            checkpoint.run_id,
            checkpoint.tenant,
            checkpoint.format_version,
            checkpoint.created_at,
            packed(checkpoint, kind=StateKind.CHECKPOINT),
        )

    async def latest(self, run_id: str, *, tenant: str) -> Checkpoint | None:
        """Return the frontier of `run_id`, or `None` where nothing was written."""
        rows = await self._sql.fetch(
            LATEST_SQL.format(checkpoints=self._tables.checkpoints), run_id, tenant
        )
        if not rows or not rows[0] or rows[0][0] is None:
            return None
        return unpacked(_text(rows[0][0]), Checkpoint, kind=StateKind.CHECKPOINT)

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop the frontier, because the run reached a terminal state."""
        await self._sql.fetch(
            FORGET_SQL.format(checkpoints=self._tables.checkpoints), run_id, tenant
        )


class PostgresLeaseStore:
    """The lease as one row per run, taken by an upsert that refuses a live holder.

    The insert either lands or does not, and the same statement returns the current row
    either way, so the loser is told who holds the run without a second read that could see
    a third state. `now()` is the database's, which is the clock every worker shares.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        tables: Where the rows live.
    """

    def __init__(
        self, executor: SqlExecutor, *, tables: CheckpointTables = DEFAULT_CHECKPOINT_TABLES
    ) -> None:
        self._sql = executor
        self._tables = tables

    async def acquire(
        self, run_id: str, *, tenant: str, holder: str, ttl_seconds: float
    ) -> RunLease:
        """Take the lease on `run_id`, or refuse because a live one is held elsewhere.

        Raises:
            RunLeaseError: If another holder's lease has not expired.
        """
        rows = await self._sql.fetch(
            ACQUIRE_SQL.format(leases=self._tables.leases), run_id, tenant, holder, ttl_seconds
        )
        current = _lease_row(rows, run_id, tenant)
        if current is None or current.holder != holder:
            held = "" if current is None else current.holder
            raise RunLeaseError(
                f"{run_id} is held by {held}",
                run_id=run_id,
                holder=held,
                requested_by=holder,
                fence=0 if current is None else current.fence,
            )
        return current

    async def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        """Extend `lease`, or refuse because the fence has moved on.

        Raises:
            RunLeaseError: If the run is now held by somebody else, or by nobody.
        """
        rows = await self._sql.fetch(
            RENEW_SQL.format(leases=self._tables.leases),
            lease.run_id,
            lease.tenant,
            lease.holder,
            lease.fence,
            ttl_seconds,
        )
        renewed = _lease_row(rows, lease.run_id, lease.tenant)
        if renewed is None:
            raise RunLeaseError(
                f"{lease.run_id} is no longer held by {lease.holder}",
                run_id=lease.run_id,
                holder="",
                requested_by=lease.holder,
                fence=lease.fence,
            )
        return renewed

    async def release(self, lease: RunLease) -> None:
        """Give the lease up, unless it has already been taken by someone else."""
        await self._sql.fetch(
            RELEASE_SQL.format(leases=self._tables.leases),
            lease.run_id,
            lease.tenant,
            lease.holder,
            lease.fence,
        )

    async def held(self, run_id: str, *, tenant: str) -> RunLease | None:
        """Return the current lease on `run_id`, expired or not."""
        rows = await self._sql.fetch(HELD_SQL.format(leases=self._tables.leases), run_id, tenant)
        return _lease_row(rows, run_id, tenant)


def _lease_row(rows: Sequence[Sequence[Any]], run_id: str, tenant: str) -> RunLease | None:
    """One row as a lease, or `None` where the statement matched nothing."""
    if not rows or not rows[0] or rows[0][0] is None:
        return None
    holder, fence, expires_at = rows[0][0], rows[0][1], rows[0][2]
    return RunLease(
        run_id=run_id,
        tenant=tenant,
        holder=_text(holder),
        fence=int(fence),
        expires_at=float(expires_at),
    )


PUT = """
redis.call('SET', KEYS[1], ARGV[1])
return 1
"""

GET = """
return redis.call('GET', KEYS[1])
"""

FORGET = """
return redis.call('DEL', KEYS[1])
"""

ACQUIRE = """
local now = tonumber(redis.call('TIME')[1])
local holder = redis.call('HGET', KEYS[1], 'holder')
local expires = tonumber(redis.call('HGET', KEYS[1], 'expires_at')) or 0
local fence = tonumber(redis.call('HGET', KEYS[1], 'fence')) or 0
if holder and holder ~= ARGV[1] and expires > now then
  return {'held', holder, tostring(expires), tostring(fence)}
end
local taken = fence + 1
local until_ = now + tonumber(ARGV[2])
redis.call('HSET', KEYS[1], 'holder', ARGV[1], 'fence', taken, 'expires_at', until_)
return {'ok', tostring(taken), tostring(until_)}
"""

RENEW = """
local holder = redis.call('HGET', KEYS[1], 'holder')
local fence = redis.call('HGET', KEYS[1], 'fence')
if not holder or holder ~= ARGV[1] or fence ~= ARGV[2] then
  return {'lost', holder or ''}
end
local until_ = tonumber(redis.call('TIME')[1]) + tonumber(ARGV[3])
redis.call('HSET', KEYS[1], 'expires_at', until_)
return {'ok', tostring(until_)}
"""

RELEASE = """
if redis.call('HGET', KEYS[1], 'holder') == ARGV[1]
   and redis.call('HGET', KEYS[1], 'fence') == ARGV[2] then
  redis.call('HSET', KEYS[1], 'expires_at', 0)
  redis.call('HDEL', KEYS[1], 'holder')
end
return 1
"""

HELD = """
local holder = redis.call('HGET', KEYS[1], 'holder')
if not holder then return false end
return {holder, redis.call('HGET', KEYS[1], 'fence'), redis.call('HGET', KEYS[1], 'expires_at')}
"""

PUT_SQL = """
INSERT INTO {checkpoints} (run_id, tenant, format_version, created_at, payload)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (tenant, run_id)
DO UPDATE SET format_version = $3, created_at = $4, payload = $5
RETURNING 1
"""

LATEST_SQL = """
SELECT payload FROM {checkpoints} WHERE run_id = $1 AND tenant = $2
"""

FORGET_SQL = """
DELETE FROM {checkpoints} WHERE run_id = $1 AND tenant = $2 RETURNING 1
"""

ACQUIRE_SQL = """
INSERT INTO {leases} (run_id, tenant, holder, fence, expires_at)
VALUES ($1, $2, $3, 1, extract(epoch FROM now()) + $4)
ON CONFLICT (tenant, run_id) DO UPDATE
SET holder = $3,
    fence = {leases}.fence + 1,
    expires_at = extract(epoch FROM now()) + $4
WHERE {leases}.holder = $3 OR {leases}.expires_at <= extract(epoch FROM now())
RETURNING holder, fence, expires_at
"""

RENEW_SQL = """
UPDATE {leases} SET expires_at = extract(epoch FROM now()) + $5
WHERE run_id = $1 AND tenant = $2 AND holder = $3 AND fence = $4
RETURNING holder, fence, expires_at
"""

RELEASE_SQL = """
UPDATE {leases} SET expires_at = 0
WHERE run_id = $1 AND tenant = $2 AND holder = $3 AND fence = $4
RETURNING holder, fence, expires_at
"""

HELD_SQL = """
SELECT holder, fence, expires_at FROM {leases} WHERE run_id = $1 AND tenant = $2
"""

SCHEMA_OF = """
SELECT version FROM {schema} WHERE component = $1
"""

EXPECTED_CHECKPOINT_SCHEMA = """
-- Owned by the platform's migration repository. The kit reads adk_schema and refuses a
-- version it was not written for; it never applies this itself.

INSERT INTO adk_schema (component, version) VALUES ('checkpoints', 1);

-- One row per run: the frontier, replaced as the run advances. No TTL, because a
-- checkpoint that expires is a run that stops being resumable at a time nobody chose.
CREATE TABLE adk_checkpoints (
    run_id         text NOT NULL,
    tenant         text NOT NULL,
    format_version integer NOT NULL,
    created_at     double precision NOT NULL,
    payload        jsonb NOT NULL,
    PRIMARY KEY (tenant, run_id)
);

-- One row per run: who may advance it, and until when. The fence only ever increases, so
-- a write from a holder whose lease was taken is refused on the number.
CREATE TABLE adk_run_leases (
    run_id     text NOT NULL,
    tenant     text NOT NULL,
    holder     text NOT NULL,
    fence      bigint NOT NULL,
    expires_at double precision NOT NULL,
    PRIMARY KEY (tenant, run_id)
);

CREATE INDEX adk_run_leases_expiry ON adk_run_leases (expires_at);
"""
