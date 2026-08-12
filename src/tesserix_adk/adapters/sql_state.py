"""Run state and work queues in PostgreSQL, in the database the product already has.

A small product should not need three pieces of infrastructure to run one agent. With the
state store and the queue in one database, a state change and the event it causes commit in
one transaction — so there is no window in which a run is recorded and its event is not, or
the other way about, which is the whole class of bug an outbox exists to close.

`bound(session)` is what makes that possible: it returns the same store running inside a
transaction the caller opened, and it does not retry, because a poisoned transaction cannot
be retried from the inside. The caller retries the transaction. The unbound store retries a
serialisation failure itself, since there is nothing else in flight to roll back.

The DDL is not here. `verify()` reads the schema version the platform's bootstrap applied
and refuses to start against a shape this adapter was not written for; the expected shape
is `EXPECTED_SCHEMA`, for the migration repository to own. Trade-offs, indexes and the
autovacuum guidance are in `docs/state-adapters.md`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import Field

from tesserix_adk.adapters.memory import MemoryStoreSettings
from tesserix_adk.adapters.state import (
    LEASE_LAPSED,
    WORKER_RESTARTED,
    _merged,
    _named,
    _text,
    _zeroed,
)
from tesserix_adk.core.errors import (
    ConfigurationError,
    LeaseLostError,
    PoolExhaustedError,
    QueueUnavailableError,
    StateConflictError,
    StateInUseError,
    StateNotFoundError,
    StatePersistenceError,
    WorkItemNotFoundError,
)
from tesserix_adk.core.primitives import Usage
from tesserix_adk.core.queue import QueuePolicy, QueueStats, WorkItem, WorkState
from tesserix_adk.core.run import RunState
from tesserix_adk.core.state import RunRecord, SessionRecord, StateDelta, StateKey, StatePage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from contextlib import AbstractAsyncContextManager

    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.core.state import StateQuery

__all__ = [
    "DEFAULT_TABLES",
    "EXPECTED_SCHEMA",
    "SCHEMA_VERSION",
    "PostgresStateStore",
    "PostgresStoreSettings",
    "PostgresWorkQueue",
    "SqlSession",
    "SqlTransactor",
    "StateTables",
]

SCHEMA_VERSION = 1
"""The shape these adapters were written for, recorded by the migration that applied it."""

T = TypeVar("T")

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class StateTables:
    """Where the rows live. One database may hold more than one kit deployment.

    Args:
        runs: Run state, one row per run.
        sessions: Session state, one row per session.
        work: Queue items, one row per item, live and dead alike.
        counters: Per-queue totals that no single row could carry.
        turns: Whose turn it is next, one row per tenant per queue.
        schema: Where the migration recorded which shape it applied.
    """

    runs: str = "adk_runs"
    sessions: str = "adk_sessions"
    work: str = "adk_work"
    counters: str = "adk_queue_counters"
    turns: str = "adk_queue_turns"
    schema: str = "adk_schema"

    def __post_init__(self) -> None:
        """Refuse a name that could carry SQL, since a table name cannot be a parameter."""
        for name in (
            self.runs,
            self.sessions,
            self.work,
            self.counters,
            self.turns,
            self.schema,
        ):
            # Table names are interpolated, not bound: a placeholder cannot name a relation.
            if not _IDENTIFIER.fullmatch(name):
                raise ConfigurationError(f"{name!r} is not a plain table identifier")


DEFAULT_TABLES = StateTables()


class PostgresStoreSettings(MemoryStoreSettings):
    """How to reach PostgreSQL, and what this deployment needs it to promise.

    Args:
        schema_version: The shape the adapter expects. `verify()` refuses anything else,
            rather than writing into a table whose columns have moved.
        statement_timeout_ms: The timeout the role or pool must already set. The adapter
            checks it and refuses zero; it never sets it, because a library that
            reconfigures a production server is a library nobody can reason about.
        max_value_bytes: The largest record this store will write. PostgreSQL will take
            far more, but a row that size is a toast write on every heartbeat.
    """

    schema_version: int = SCHEMA_VERSION
    statement_timeout_ms: int = Field(default=5_000, ge=1)
    max_value_bytes: int = Field(default=1_048_576, ge=1)


class SqlSession(Protocol):
    """The one call these stores make."""

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        """Run a statement and return its rows."""
        ...


class SqlTransactor(SqlSession, Protocol):
    """A session that can also open a transaction, which is where atomicity comes from."""

    def transaction(self) -> AbstractAsyncContextManager[SqlSession]:
        """Begin a transaction, committing on exit and rolling back on any exception."""
        ...


class PostgresStateStore:
    """Session and run state in PostgreSQL, with the version compare inside the write.

    A run is one row: a JSONB payload for what a worker set, and columns for what SQL has
    to filter, order or add on. `put_run` commits at the version it read plus one in the
    `UPDATE`'s own predicate, so two workers holding the same run cannot both think they
    wrote it. `patch_run` adds in SQL — `SET input_tokens = input_tokens + $1` — so two
    patches arriving together both land.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: Where `updated_at` comes from.
        settings: Connection, retry, timeout and size policy.
        tables: Where the rows live.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlSession,
        *,
        clock: Clock,
        settings: PostgresStoreSettings,
        tables: StateTables = DEFAULT_TABLES,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._sql = executor
        self._clock = clock
        self._settings = settings
        self._tables = tables
        self._run_sql = _Running(clock, settings, entropy, _state_failure(type(self).__name__))

    @classmethod
    async def open(cls, executor: SqlSession, **kwargs: Any) -> PostgresStateStore:  # noqa: ANN401 — the constructor's own keywords
        """Construct the store and refuse a database that is not the shape it expects.

        Raises:
            ConfigurationError: If the schema version differs from the adapter's, or the
                connection has no statement timeout.
        """
        store = cls(executor, **kwargs)
        await store.verify()
        return store

    def bound(self, session: SqlSession) -> PostgresStateStore:
        """The same store, running inside the caller's transaction and retrying nothing.

        A retry inside a transaction that has already failed runs against a connection
        that will refuse everything until it is rolled back. The caller retries the
        transaction, which is the only thing that can be retried honestly.
        """
        store = PostgresStateStore(
            session,
            clock=self._clock,
            settings=self._settings,
            tables=self._tables,
        )
        store._run_sql = replace(self._run_sql, attempts=1)
        return store

    async def verify(self) -> None:
        """Check the schema version and the statement timeout. At startup, never per call."""
        await _verify(self._run_sql, self._sql, self._settings, self._tables, "state")

    async def get_session(self, key: StateKey) -> SessionRecord | None:
        """Return the session, or None where there is not one."""
        rows = await self._fetch(GET_SESSION, key.tenant, key.id)
        return None if not rows else SessionRecord.model_validate_json(_text(rows[0][0]))

    async def put_session(self, record: SessionRecord) -> SessionRecord:
        """Store the session at its read version plus one, or refuse."""
        written = record.model_copy(
            update={"version": record.version + 1, "updated_at": self._clock.now()}
        )
        blob = self._sized(written.model_dump_json(), record.key)
        rows = await self._fetch(
            PUT_SESSION, record.tenant, record.session_id, record.version, written.updated_at, blob
        )
        if not rows:
            await self._refuse_a_stale_write(GET_SESSION_VERSION, record.key, record.version)
        return written

    async def delete_session(self, key: StateKey, *, cascade: bool = False) -> None:
        """Remove the session, and its runs where `cascade` says so."""
        rows = await self._fetch(DELETE_SESSION, key.tenant, key.id, _TERMINAL_RUNS, cascade)
        if rows and not cascade:
            raise StateInUseError(
                f"{_named(key)} still has runs that have not finished",
                key=_named(key),
                live_runs=tuple(_text(row[0]) for row in rows),
            )

    async def get_run(self, key: StateKey) -> RunRecord | None:
        """Return the run, with everything patched into it since it was written."""
        rows = await self._fetch(GET_RUN, key.tenant, key.id)
        return None if not rows else _run_of(rows[0])

    async def put_run(self, record: RunRecord) -> RunRecord:
        """Store the run at its read version plus one, or refuse."""
        written = record.scrubbed().model_copy(update={"updated_at": self._clock.now()})
        key = record.key
        blob = self._sized(_zeroed(written).model_dump_json(), key)
        rows = await self._fetch(
            PUT_RUN,
            key.tenant,
            key.id,
            record.session_id,
            record.state.value,
            record.version,
            written.updated_at,
            record.usage.input_tokens,
            record.usage.output_tokens,
            record.cost_micros,
            record.iterations,
            record.message_cursor,
            blob,
        )
        if not rows:
            await self._refuse_a_stale_write(GET_RUN_VERSION, key, record.version)
        return written.model_copy(update={"version": record.version + 1})

    async def patch_run(self, key: StateKey, delta: StateDelta) -> RunRecord:
        """Add `delta` to the run in SQL, so two patches at once both land."""
        usage = delta.usage or Usage(input_tokens=0, output_tokens=0)
        rows = await self._fetch(
            PATCH_RUN,
            key.tenant,
            key.id,
            usage.input_tokens,
            usage.output_tokens,
            delta.cost_micros,
            delta.iterations,
            delta.messages_read,
        )
        if not rows:
            raise StateNotFoundError(f"no run {_named(key)}", key=_named(key), kind="run")
        return _run_of(rows[0])

    async def delete_run(self, key: StateKey) -> None:
        """Remove the run."""
        await self._fetch(DELETE_RUN, key.tenant, key.id)

    async def list_runs(self, query: StateQuery) -> StatePage:
        """Return one page of the tenant's runs, in the order the store first saw them."""
        rows = await self._fetch(
            LIST_RUNS,
            query.tenant,
            int(query.cursor) if query.cursor else None,
            query.state.value if query.state else None,
            query.updated_before,
            # One more than the page was asked for, because a cursor that leads nowhere
            # loops a reaper for as long as it keeps asking.
            query.limit + 1,
        )
        more = len(rows) > query.limit
        page = rows[: query.limit]
        return StatePage(
            records=tuple(_run_of(row) for row in page),
            cursor=_text(page[-1][0]) if more else None,
        )

    async def _refuse_a_stale_write(self, statement: str, key: StateKey, expected: int) -> None:
        """A write against a version that moved is refused, with both numbers."""
        rows = await self._fetch(statement, key.tenant, key.id)
        actual = int(rows[0][0]) if rows else 0
        raise StateConflictError(
            f"{_named(key)} moved to version {actual} while it was held at {expected}",
            key=_named(key),
            expected_version=expected,
            actual_version=actual,
        )

    def _sized(self, blob: str, key: StateKey) -> str:
        """The blob, if the store will take it — and a refusal naming what will."""
        size = len(blob.encode())
        if size > self._settings.max_value_bytes:
            raise StatePersistenceError(
                f"{_named(key)} is {size} bytes, over the {self._settings.max_value_bytes} this"
                " store writes; hold a record this size behind a claim check",
                store=type(self).__name__,
                reason="too_large",
            )
        return blob

    async def _fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        return await self._run_sql(lambda: self._sql.fetch(self._filled(statement), *args))

    def _filled(self, statement: str) -> str:
        return statement.format(
            runs=self._tables.runs, sessions=self._tables.sessions, schema=self._tables.schema
        )


class PostgresWorkQueue:
    """A work queue in PostgreSQL, claimed with `FOR UPDATE ... SKIP LOCKED`.

    A claim is one statement: the row is locked, and a worker that arrives while it is
    locked steps over it to the next one rather than waiting behind it or taking it twice.
    Tenants are served in rotation: the claim orders by the turn each tenant was last given
    and sends the served tenant to the back, so a backlog cannot starve the tenant beside it.

    Every settle is fenced on the lease the caller last saw, so a reaper that lost a race
    to a live heartbeat writes nothing. Rows the reaper cannot lock are skipped rather than
    waited on: a lease held inside a caller's long transaction is work still in progress.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: Where lease deadlines and backoff are measured from.
        settings: Connection, retry, timeout and size policy.
        policy: How long a claim lasts, how many attempts an item gets, how it backs off.
        tables: Where the rows live.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlSession,
        *,
        clock: Clock,
        settings: PostgresStoreSettings,
        policy: QueuePolicy | None = None,
        tables: StateTables = DEFAULT_TABLES,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._sql = executor
        self._clock = clock
        self._settings = settings
        self._policy = policy or QueuePolicy()
        self._tables = tables
        self._run_sql = _Running(clock, settings, entropy, _queue_failure(type(self).__name__))

    @classmethod
    async def open(cls, executor: SqlSession, **kwargs: Any) -> PostgresWorkQueue:  # noqa: ANN401 — the constructor's own keywords
        """Construct the queue and refuse a database that is not the shape it expects."""
        queue = cls(executor, **kwargs)
        await queue.verify()
        return queue

    def bound(self, session: SqlSession) -> PostgresWorkQueue:
        """The same queue, running inside the caller's transaction and retrying nothing.

        This is how an enqueue commits with the state change that caused it: neither is
        visible unless both are.
        """
        queue = PostgresWorkQueue(
            session,
            clock=self._clock,
            settings=self._settings,
            policy=self._policy,
            tables=self._tables,
        )
        queue._run_sql = replace(self._run_sql, attempts=1)
        return queue

    async def verify(self) -> None:
        """Check the schema version and the statement timeout. At startup, never per call."""
        await _verify(self._run_sql, self._sql, self._settings, self._tables, "queue")

    @property
    def policy(self) -> QueuePolicy:
        """How long a claim lasts here, and how many attempts an item gets."""
        return self._policy

    async def enqueue(self, item: WorkItem) -> WorkItem:
        """Add `item`, or return the live item its dedupe key already names."""
        now = self._clock.now()
        placed = item.model_copy(
            update={
                "enqueued_at": item.enqueued_at or now,
                "available_at": item.available_at or now,
                "state": WorkState.QUEUED,
                "worker": None,
                "lease_expires_at": None,
            }
        )
        rows = await self._fetch(
            ENQUEUE,
            placed.tenant,
            placed.id,
            placed.queue,
            placed.dedupe_key,
            int(placed.priority),
            placed.enqueued_at,
            placed.available_at,
            self._sized(placed),
            _TERMINAL_WORK,
        )
        return placed if not rows else WorkItem.model_validate_json(_text(rows[0][0]))

    async def claim(
        self, *, worker: str, queue: str = "default", lease_seconds: float | None = None
    ) -> WorkItem | None:
        """Take the next due item, stepping over rows another worker has locked."""
        now = self._clock.now()
        lease = lease_seconds or self._policy.lease_seconds
        rows = await self._fetch(CLAIM, queue, now, worker, now + lease)
        if not rows:
            return None
        item = WorkItem.model_validate_json(_text(rows[0][0]))
        claimed = self._policy.claimed(item, worker=worker, now=now, lease_seconds=lease_seconds)
        await self._write(claimed)
        return claimed

    async def heartbeat(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Extend the claim, up to the total the policy allows a single claim to run for."""
        item = await self._holding(item_id, tenant=tenant, worker=worker)
        now = self._clock.now()
        if self._policy.overheld(item, now):
            raise LeaseLostError(
                f"{item_id!r} has been held for longer than {self._policy.max_lease_seconds}s",
                item_id=item_id,
                worker=worker,
                holder=worker,
                reason="capped",
            )
        renewed = self._policy.renewed(item, now=now)
        await self._write(renewed, fence=item.lease_expires_at)
        return renewed

    async def complete(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Finish the item. Its dedupe key is free again; the same job may be enqueued anew."""
        item = await self._holding(item_id, tenant=tenant, worker=worker)
        done = self._policy.completed(item)
        await self._write(done, fence=item.lease_expires_at)
        return done

    async def fail(
        self, item_id: str, *, tenant: str, worker: str, error: str, retryable: bool = True
    ) -> WorkItem:
        """Give the item back for a retry after a backoff, or to the dead letter."""
        item = await self._holding(item_id, tenant=tenant, worker=worker)
        given = self._policy.returned(item, error=error, now=self._clock.now(), retryable=retryable)
        await self._write(given, fence=item.lease_expires_at)
        return given

    async def reap(self) -> tuple[WorkItem, ...]:
        """Requeue everything whose lease has lapsed, by this queue's clock."""
        rows = await self._fetch(LAPSED, self._clock.now(), _SCAN)
        return await self._give_back(rows, LEASE_LAPSED, backoff=True, reaped=True)

    async def adopt(self, *, worker: str) -> tuple[WorkItem, ...]:
        """Give back what `worker` held before it restarted, without waiting out the lease."""
        rows = await self._fetch(OWNED, worker, _SCAN)
        return await self._give_back(rows, WORKER_RESTARTED, backoff=False, reaped=False)

    async def dead_letters(self, *, tenant: str, limit: int = 50) -> tuple[WorkItem, ...]:
        """Return items that ran out of attempts, in the order they were enqueued."""
        rows = await self._fetch(DEAD_LETTERS, tenant, limit)
        return tuple(WorkItem.model_validate_json(_text(row[0])) for row in rows)

    async def stats(self, *, queue: str = "default") -> QueueStats:
        """Return the depth, ages and counters a deployment alerts on."""
        rows = await self._fetch(STATS, queue, self._clock.now())
        depth, claimed, dead, oldest, reaped, duplicates = (float(value or 0) for value in rows[0])
        return QueueStats(
            depth=int(depth),
            claimed=int(claimed),
            dead_lettered=int(dead),
            oldest_age_seconds=oldest,
            reaped=int(reaped),
            duplicates_suppressed=int(duplicates),
        )

    async def _give_back(
        self,
        rows: Sequence[Sequence[Any]],
        error: str,
        *,
        backoff: bool,
        reaped: bool,
    ) -> tuple[WorkItem, ...]:
        """Return every item in `rows` to the queue, where its lease has not moved since."""
        moved = []
        for row in rows:
            item = WorkItem.model_validate_json(_text(row[0]))
            given = self._policy.returned(item, error=error, now=self._clock.now(), backoff=backoff)
            if await self._write(given, fence=float(row[1]), reaped=reaped):
                moved.append(given)
        return tuple(moved)

    async def _holding(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """The item, if this worker still holds it — and a refusal if it does not."""
        rows = await self._fetch(GET_ITEM, tenant, item_id)
        if not rows:
            raise WorkItemNotFoundError(
                f"no item {item_id!r} in {tenant!r}", item_id=item_id, tenant=tenant
            )
        item = WorkItem.model_validate_json(_text(rows[0][0]))
        if not item.held or item.worker != worker:
            raise LeaseLostError(
                f"{item_id!r} is no longer held by {worker!r}",
                item_id=item_id,
                worker=worker,
                holder=item.worker,
                reason="taken" if item.held else "expired",
            )
        if item.lapsed_at(self._clock.now()):
            raise LeaseLostError(
                f"the claim {worker!r} holds on {item_id!r} has expired",
                item_id=item_id,
                worker=worker,
                holder=item.worker,
                reason="expired",
            )
        return item

    async def _write(
        self, item: WorkItem, *, fence: float | None = None, reaped: bool = False
    ) -> bool:
        """Write the item where its new state says it belongs, if its lease has not moved."""
        rows = await self._fetch(
            SETTLE,
            item.tenant,
            item.id,
            item.state.value,
            item.worker,
            item.attempts,
            item.available_at,
            item.lease_expires_at,
            item.first_claimed_at,
            self._sized(item),
            fence,
            reaped,
            item.queue,
        )
        return bool(rows)

    def _sized(self, item: WorkItem) -> str:
        """The item as it will be written, if the queue will take it."""
        blob = item.model_dump_json()
        size = len(blob.encode())
        if size > self._settings.max_value_bytes:
            raise StatePersistenceError(
                f"{item.id!r} is {size} bytes, over the {self._settings.max_value_bytes} this"
                " queue writes; put a payload this size behind a claim check",
                store=type(self).__name__,
                reason="too_large",
            )
        return blob

    async def _fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        return await self._run_sql(lambda: self._sql.fetch(self._filled(statement), *args))

    def _filled(self, statement: str) -> str:
        return statement.format(
            work=self._tables.work,
            counters=self._tables.counters,
            turns=self._tables.turns,
            schema=self._tables.schema,
        )


_SCAN = 200
"""How many rows one sweep looks at. A reaper that scanned everything would hold the table."""

_TERMINAL_RUNS = [state.value for state in RunState if state.is_terminal]
_TERMINAL_WORK = [WorkState.COMPLETED.value, WorkState.DEAD_LETTERED.value]
"""An item in one of these holds no dedupe key: the same job may be sent again."""


@dataclass(frozen=True, slots=True)
class _Running:
    """Bounded, jittered retries around one statement, and the SQLSTATE it failed with."""

    clock: Clock
    settings: PostgresStoreSettings
    entropy: Callable[[], float]
    failed: Callable[[int, str], Exception]
    attempts: int = 0

    async def __call__(self, work: Callable[[], Awaitable[T]]) -> T:
        """Run `work`, retrying a contended or unreachable database until the budget is spent."""
        budget = self.attempts or self.settings.max_attempts
        attempt = 0
        while True:
            attempt += 1
            try:
                return await work()
            except Exception as failure:
                self._reraise_if_terminal(failure, attempt, budget)
                await self.clock.sleep(self._backoff(attempt))

    def _reraise_if_terminal(self, failure: Exception, attempt: int, budget: int) -> None:
        state = _sqlstate(failure)
        if state in _POOL_STATES or "pool" in type(failure).__name__.lower():
            raise PoolExhaustedError("postgres connection pool exhausted") from failure
        if state in _SCHEMA_STATES:
            raise ConfigurationError(
                f"the database is missing what this adapter reads ({state}); apply the"
                " platform's migrations before starting"
            ) from failure
        if attempt >= budget:
            raise self.failed(attempt, _WHY.get(state, "unavailable")) from failure

    def _backoff(self, attempt: int) -> float:
        """Jittered, because every worker that started together retries together."""
        return self.settings.backoff_seconds * float(2 ** (attempt - 1)) * (0.5 + self.entropy())


_POOL_STATES = frozenset({"53300", "53400"})
_SCHEMA_STATES = frozenset({"42P01", "42703"})
_WHY = {"40001": "contended", "40P01": "contended"}
"""Serialisation failure and deadlock: the transaction lost, and losing is retryable."""


def _sqlstate(failure: Exception) -> str:
    """The SQLSTATE a driver recorded, under whichever name it records it."""
    code = getattr(failure, "sqlstate", None) or getattr(failure, "pgcode", None)
    return str(code) if code else ""


def _state_failure(store: str) -> Callable[[int, str], Exception]:
    """What an unreachable or contended store raises. A write that failed is not a write."""

    def failed(attempt: int, reason: str) -> Exception:
        return StatePersistenceError(
            f"{store} could not commit after {attempt} attempts ({reason})",
            store=store,
            reason=reason,
        )

    return failed


def _queue_failure(store: str) -> Callable[[int, str], Exception]:
    """What an unreachable queue raises, so an enqueue is never a silent drop."""

    def failed(attempt: int, reason: str) -> Exception:
        return QueueUnavailableError(
            f"{store} could not commit after {attempt} attempts ({reason})",
            queue=store,
            operation="commit",
        )

    return failed


async def _verify(
    run_sql: _Running,
    sql: SqlSession,
    settings: PostgresStoreSettings,
    tables: StateTables,
    component: str,
) -> None:
    """Refuse a database whose shape has moved, or one that will let a statement run forever."""
    rows = await run_sql(lambda: sql.fetch(SCHEMA_OF.format(schema=tables.schema), component))
    found = int(rows[0][0]) if rows else 0
    if found != settings.schema_version:
        raise ConfigurationError(
            f"{component} schema is version {found}, and this adapter writes version"
            f" {settings.schema_version}; a column that moved is a write into the wrong shape"
        )
    timeout = await run_sql(lambda: sql.fetch(TIMEOUT))
    if _text(timeout[0][0]) in {"0", "0ms"}:
        raise ConfigurationError(
            "this connection has no statement_timeout; one held statement would hold a"
            f" pooled connection until the process restarts. Set it to about"
            f" {settings.statement_timeout_ms}ms on the role or the pool"
        )


def _run_of(row: Sequence[Any]) -> RunRecord:
    """A run as the payload it was written as, plus everything added to it since."""
    return _merged([row[1], row[2:]])


EXPECTED_SCHEMA = """
-- Owned by the platform's migration repository. The kit reads adk_schema and refuses a
-- version it was not written for; it never applies this itself.

CREATE TABLE adk_schema (component text PRIMARY KEY, version integer NOT NULL);
INSERT INTO adk_schema (component, version) VALUES ('state', 1), ('queue', 1);

CREATE TABLE adk_sessions (
    tenant     text NOT NULL,
    session_id text NOT NULL,
    version    integer NOT NULL,
    updated_at double precision NOT NULL,
    payload    jsonb NOT NULL,
    PRIMARY KEY (tenant, session_id)
);

CREATE TABLE adk_runs (
    tenant         text NOT NULL,
    run_id         text NOT NULL,
    session_id     text,
    state          text NOT NULL,
    version        integer NOT NULL,
    seq            bigserial NOT NULL,
    updated_at     double precision NOT NULL,
    input_tokens   bigint NOT NULL DEFAULT 0,
    output_tokens  bigint NOT NULL DEFAULT 0,
    cost_micros    bigint NOT NULL DEFAULT 0,
    iterations     integer NOT NULL DEFAULT 0,
    message_cursor integer NOT NULL DEFAULT 0,
    payload        jsonb NOT NULL,
    PRIMARY KEY (tenant, run_id)
);
CREATE INDEX adk_runs_listing ON adk_runs (tenant, seq);
CREATE INDEX adk_runs_state ON adk_runs (tenant, state, updated_at);
CREATE INDEX adk_runs_session ON adk_runs (tenant, session_id);

CREATE TABLE adk_work (
    tenant           text NOT NULL,
    item_id          text NOT NULL,
    queue            text NOT NULL,
    state            text NOT NULL,
    priority         integer NOT NULL,
    attempts         integer NOT NULL DEFAULT 0,
    worker           text,
    dedupe_key       text,
    seq              bigserial NOT NULL,
    enqueued_at      double precision NOT NULL,
    available_at     double precision NOT NULL,
    lease_expires_at double precision,
    first_claimed_at double precision,
    payload          jsonb NOT NULL,
    PRIMARY KEY (tenant, item_id)
);
CREATE INDEX adk_work_due ON adk_work (queue, state, available_at, priority DESC, seq);
CREATE INDEX adk_work_leases ON adk_work (state, lease_expires_at);
CREATE INDEX adk_work_owner ON adk_work (worker, state);
CREATE INDEX adk_work_dead ON adk_work (tenant, state, seq);
CREATE UNIQUE INDEX adk_work_dedupe ON adk_work (tenant, queue, dedupe_key)
    WHERE dedupe_key IS NOT NULL AND state NOT IN ('completed', 'dead_lettered');

CREATE TABLE adk_queue_counters (
    queue      text PRIMARY KEY,
    reaped     bigint NOT NULL DEFAULT 0,
    duplicates bigint NOT NULL DEFAULT 0
);

-- Whose turn it is. A tenant goes to the back of the queue's rotation when it is served,
-- so a tenant with a backlog of ten thousand cannot hold the one beside it behind them.
CREATE TABLE adk_queue_turns (
    queue  text NOT NULL,
    tenant text NOT NULL,
    turn   bigint NOT NULL,
    PRIMARY KEY (queue, tenant)
);

-- A queue table is rewritten on every claim, heartbeat and settle. The defaults vacuum it
-- far too rarely to keep the due-index tight; see docs/state-adapters.md.
ALTER TABLE adk_work SET (autovacuum_vacuum_scale_factor = 0.01, autovacuum_vacuum_cost_delay = 0);
"""

SCHEMA_OF = """
SELECT version FROM {schema} WHERE component = $1
"""

TIMEOUT = """
SHOW statement_timeout
"""

GET_SESSION = """
SELECT payload::text FROM {sessions} WHERE tenant = $1 AND session_id = $2
"""

GET_SESSION_VERSION = """
SELECT version FROM {sessions} WHERE tenant = $1 AND session_id = $2
"""

PUT_SESSION = """
INSERT INTO {sessions} (tenant, session_id, version, updated_at, payload)
VALUES ($1, $2, $3::integer + 1, $4, $5::jsonb)
ON CONFLICT (tenant, session_id) DO UPDATE
   SET version = {sessions}.version + 1, updated_at = EXCLUDED.updated_at,
       payload = EXCLUDED.payload
 WHERE {sessions}.version = $3::integer
RETURNING version
"""

DELETE_SESSION = """
WITH live AS (
    SELECT run_id FROM {runs}
     WHERE tenant = $1 AND session_id = $2 AND state <> ALL($3::text[])
), purged AS (
    DELETE FROM {runs}
     WHERE tenant = $1 AND session_id = $2
       AND ($4::boolean OR NOT EXISTS (SELECT 1 FROM live))
    RETURNING 1
), dropped AS (
    DELETE FROM {sessions}
     WHERE tenant = $1 AND session_id = $2
       AND ($4::boolean OR NOT EXISTS (SELECT 1 FROM live))
    RETURNING 1
)
SELECT run_id FROM live
"""

GET_RUN = """
SELECT seq, payload::text, version, input_tokens, output_tokens, cost_micros,
       iterations, message_cursor
  FROM {runs} WHERE tenant = $1 AND run_id = $2
"""

GET_RUN_VERSION = """
SELECT version FROM {runs} WHERE tenant = $1 AND run_id = $2
"""

PUT_RUN = """
INSERT INTO {runs} (tenant, run_id, session_id, state, version, updated_at,
                    input_tokens, output_tokens, cost_micros, iterations,
                    message_cursor, payload)
VALUES ($1, $2, $3, $4, $5::integer + 1, $6, $7, $8, $9, $10, $11, $12::jsonb)
ON CONFLICT (tenant, run_id) DO UPDATE
   SET session_id = EXCLUDED.session_id, state = EXCLUDED.state,
       version = {runs}.version + 1, updated_at = EXCLUDED.updated_at,
       input_tokens = EXCLUDED.input_tokens, output_tokens = EXCLUDED.output_tokens,
       cost_micros = EXCLUDED.cost_micros, iterations = EXCLUDED.iterations,
       message_cursor = EXCLUDED.message_cursor, payload = EXCLUDED.payload
 WHERE {runs}.version = $5::integer
RETURNING version
"""

PATCH_RUN = """
UPDATE {runs}
   SET input_tokens = input_tokens + $3, output_tokens = output_tokens + $4,
       cost_micros = cost_micros + $5, iterations = iterations + $6,
       message_cursor = message_cursor + $7, version = version + 1
 WHERE tenant = $1 AND run_id = $2
RETURNING seq, payload::text, version, input_tokens, output_tokens, cost_micros,
          iterations, message_cursor
"""

DELETE_RUN = """
DELETE FROM {runs} WHERE tenant = $1 AND run_id = $2
"""

LIST_RUNS = """
SELECT seq, payload::text, version, input_tokens, output_tokens, cost_micros,
       iterations, message_cursor
  FROM {runs}
 WHERE tenant = $1
   AND ($2::bigint IS NULL OR seq > $2::bigint)
   AND ($3::text IS NULL OR state = $3::text)
   AND ($4::double precision IS NULL OR updated_at < $4::double precision)
 ORDER BY seq
 LIMIT $5::integer
"""

GET_ITEM = """
SELECT payload::text FROM {work} WHERE tenant = $1 AND item_id = $2
"""

ENQUEUE = """
WITH live AS (
    SELECT payload::text AS blob FROM {work}
     WHERE tenant = $1 AND state <> ALL($9::text[])
       AND (item_id = $2 OR (dedupe_key IS NOT NULL AND dedupe_key = $4 AND queue = $3))
     LIMIT 1
), placed AS (
    INSERT INTO {work} (tenant, item_id, queue, state, priority, attempts, dedupe_key,
                        enqueued_at, available_at, payload)
    SELECT $1, $2, $3, 'queued', $5, 0, $4, $6, $7, $8::jsonb
     WHERE NOT EXISTS (SELECT 1 FROM live)
    ON CONFLICT (tenant, item_id) DO UPDATE
       SET queue = EXCLUDED.queue, state = 'queued', priority = EXCLUDED.priority,
           attempts = 0, worker = NULL, lease_expires_at = NULL, first_claimed_at = NULL,
           dedupe_key = EXCLUDED.dedupe_key, enqueued_at = EXCLUDED.enqueued_at,
           available_at = EXCLUDED.available_at, payload = EXCLUDED.payload
    RETURNING 1
), counted AS (
    INSERT INTO {counters} (queue, duplicates)
    SELECT $3, 1 WHERE EXISTS (SELECT 1 FROM live)
    ON CONFLICT (queue) DO UPDATE SET duplicates = {counters}.duplicates + 1
    RETURNING 1
)
SELECT blob FROM live
"""

CLAIM = """
WITH picked AS (
    SELECT w.tenant, w.item_id
      FROM {work} w
      LEFT JOIN {turns} t ON t.queue = w.queue AND t.tenant = w.tenant
     -- Repeated on the locked row rather than left in a subquery: this is the predicate
     -- PostgreSQL re-checks when a concurrent claim committed under us, and a predicate it
     -- cannot re-check is a row two workers both take.
     WHERE w.queue = $1 AND w.state = 'queued' AND w.available_at <= $2
     ORDER BY COALESCE(t.turn, 0), w.priority DESC, w.seq
       FOR UPDATE OF w SKIP LOCKED
     LIMIT 1
), served AS (
    INSERT INTO {turns} (queue, tenant, turn)
    SELECT $1, tenant, COALESCE((SELECT max(turn) FROM {turns} WHERE queue = $1), 0) + 1
      FROM picked
    ON CONFLICT (queue, tenant) DO UPDATE SET turn = EXCLUDED.turn
    RETURNING 1
), taken AS (
    UPDATE {work} w
       SET state = 'claimed', worker = $3, lease_expires_at = $4, first_claimed_at = $2
      FROM picked p
     WHERE w.tenant = p.tenant AND w.item_id = p.item_id
    RETURNING w.payload::text AS blob
)
SELECT blob FROM taken
"""

SETTLE = """
WITH settled AS (
    UPDATE {work} w
       SET state = $3, worker = $4, attempts = $5, available_at = $6,
           lease_expires_at = $7, first_claimed_at = $8, payload = $9::jsonb
     WHERE w.tenant = $1 AND w.item_id = $2
       AND ($10::double precision IS NULL OR w.lease_expires_at = $10::double precision)
    RETURNING 1
), counted AS (
    INSERT INTO {counters} (queue, reaped)
    SELECT $12, 1 WHERE $11::boolean AND EXISTS (SELECT 1 FROM settled)
    ON CONFLICT (queue) DO UPDATE SET reaped = {counters}.reaped + 1
    RETURNING 1
)
SELECT 1 FROM settled
"""

LAPSED = """
SELECT payload::text, lease_expires_at FROM {work}
 WHERE state = 'claimed' AND lease_expires_at <= $1
 ORDER BY lease_expires_at
   FOR UPDATE SKIP LOCKED
 LIMIT $2::integer
"""

OWNED = """
SELECT payload::text, lease_expires_at FROM {work}
 WHERE worker = $1 AND state = 'claimed'
 ORDER BY seq
   FOR UPDATE SKIP LOCKED
 LIMIT $2::integer
"""

DEAD_LETTERS = """
SELECT payload::text FROM {work}
 WHERE tenant = $1 AND state = 'dead_lettered'
 ORDER BY seq
 LIMIT $2::integer
"""

STATS = """
SELECT count(*) FILTER (WHERE state = 'queued'),
       count(*) FILTER (WHERE state = 'claimed'),
       count(*) FILTER (WHERE state = 'dead_lettered'),
       COALESCE(max($2::double precision - enqueued_at)
                FILTER (WHERE state = 'queued'), 0),
       COALESCE((SELECT reaped FROM {counters} WHERE queue = $1), 0),
       COALESCE((SELECT duplicates FROM {counters} WHERE queue = $1), 0)
  FROM {work} WHERE queue = $1
"""
