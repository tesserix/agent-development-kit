"""State and its event, committed or rolled back as one thing.

Persisting a run result and publishing its event as two operations lets one succeed alone:
a completed run nothing downstream hears about, or an event for a state change that rolled
back. Neither is visible afterwards — there is no error, just a disagreement that surfaces
weeks later as a count nobody can explain.

So the event is written where the state is written, inside the caller's transaction, and a
relay moves it onto the transport afterwards. The relay is at-least-once by construction:
it publishes before it marks, so a crash between the two republishes, which the transport's
own dedupe on the event id collapses.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters.sql_state import (
    PostgresStoreSettings,
    SqlSession,
    SqlTransactor,
    _Running,
    _state_failure,
)
from tesserix_adk.adapters.state import _text
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.events import DEFAULT_MAX_EVENT_BYTES, EventEnvelope

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tesserix_adk.adapters.jetstream import DeadLetter
    from tesserix_adk.core.events import EventPublisher
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "DEFAULT_OUTBOX_TABLES",
    "DEFAULT_RETENTION_SECONDS",
    "EXPECTED_OUTBOX_SCHEMA",
    "OUTBOX_SCHEMA_VERSION",
    "OutboxLag",
    "OutboxRelay",
    "OutboxTables",
    "PostgresOutbox",
    "PostgresOutboxSettings",
]

OUTBOX_SCHEMA_VERSION = 1
"""The shape this adapter was written for, recorded by the migration that applied it."""

DEFAULT_RETENTION_SECONDS = 604_800.0
"""How long a published row is kept. Long enough to answer "did we send it?"."""

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class OutboxTables:
    """Where the rows live. One database may hold more than one kit deployment.

    Args:
        rows: One row per event, written by the caller and cleared by the relay.
        schema: Where the migration recorded which shape it applied.
    """

    rows: str = "adk_outbox"
    schema: str = "adk_schema"

    def __post_init__(self) -> None:
        """Refuse a name that could carry SQL, since a table name cannot be a parameter."""
        for name in (self.rows, self.schema):
            if not _IDENTIFIER.fullmatch(name):
                raise ConfigurationError(f"{name!r} is not a plain table identifier")


DEFAULT_OUTBOX_TABLES = OutboxTables()


class PostgresOutboxSettings(PostgresStoreSettings):
    """How to reach PostgreSQL, and what this deployment needs it to promise."""

    schema_version: int = OUTBOX_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class OutboxLag:
    """How far behind the relay is, which is the only number worth alerting on.

    Args:
        unpublished: Rows the relay has not delivered.
        oldest_seconds: How long the oldest of them has been waiting.
    """

    unpublished: int = 0
    oldest_seconds: float = 0.0


class PostgresOutbox:
    """An `EventPublisher` that inserts rather than publishes.

    Use it through `bound`, on the session the caller's state change is running on: an
    insert on any other connection is a second operation again, and the whole point is
    that there is only one.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: Where `enqueued_at` comes from.
        settings: Connection, retry and timeout policy.
        tables: Where the rows live.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlSession,
        *,
        clock: Clock,
        settings: PostgresOutboxSettings,
        tables: OutboxTables = DEFAULT_OUTBOX_TABLES,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._sql = executor
        self._clock = clock
        self._settings = settings
        self._tables = tables
        self._entropy = entropy
        self._run_sql = _Running(clock, settings, entropy, _state_failure(type(self).__name__))

    @classmethod
    async def open(cls, executor: SqlSession, **kwargs: Any) -> PostgresOutbox:  # noqa: ANN401 — the constructor's own keywords
        """Construct the outbox and refuse a database that is not the shape it expects.

        Raises:
            ConfigurationError: If the schema version differs from the adapter's, or the
                connection has no statement timeout.
        """
        outbox = cls(executor, **kwargs)
        await outbox.verify()
        return outbox

    def bound(self, session: SqlSession) -> PostgresOutbox:
        """The same outbox, inserting inside the caller's transaction and retrying nothing.

        A retry inside a transaction that has already failed runs against a connection
        that will refuse everything until it is rolled back.
        """
        outbox = PostgresOutbox(
            session,
            clock=self._clock,
            settings=self._settings,
            tables=self._tables,
            entropy=self._entropy,
        )
        outbox._run_sql = replace(self._run_sql, attempts=1)
        return outbox

    async def verify(self) -> None:
        """Check the schema version and the statement timeout. At startup, never per write."""
        rows = await self._fetch(SCHEMA_OF.format(schema=self._tables.schema), "outbox")
        found = int(rows[0][0]) if rows else 0
        if found != self._settings.schema_version:
            raise ConfigurationError(
                f"outbox schema is version {found}, and this adapter writes version"
                f" {self._settings.schema_version}; a column that moved is an event"
                f" written into the wrong shape"
            )
        timeout = await self._fetch(TIMEOUT)
        if timeout and _text(timeout[0][0]) in {"0", "0ms"}:
            raise ConfigurationError(
                "this connection has no statement timeout, so an insert holding the"
                " caller's transaction open could hold it for ever"
            )

    async def publish(self, event: EventEnvelope) -> None:
        """Write `event` to the outbox. The relay is what publishes it.

        The envelope is already redacted — attributes are scrubbed where they are built —
        so the outbox never stores content that should not be persisted.
        """
        await self._fetch(
            ENQUEUE.format(rows=self._tables.rows),
            event.event_id,
            event.tenant,
            event.run_id,
            str(event.type),
            event.occurred_at,
            self._clock.now(),
            event.to_json(),
        )

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        """Write a batch, in order. One statement each, so one conflict is not all of them."""
        for event in events:
            await self.publish(event)

    async def _fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        return await self._run_sql(lambda: self._sql.fetch(statement, *args))


class OutboxRelay:
    """Moves committed rows onto the transport, in order per run, without losing any.

    Safe in several replicas: a run is claimed whole under a transaction-scoped advisory
    lock, so two relays never hold events of the same run at once and its events cannot
    overtake each other. Across runs there is no global order, and there is no way to have
    one without serialising every relay in the fleet.

    Args:
        executor: A session that can open a transaction.
        publisher: The real transport.
        clock: What lag and retention are measured against.
        worker: Who claimed the rows, so an operator can see which replica is stuck.
        tables: Where the rows live.
        batch: How many runs one `deliver` takes.
        max_event_bytes: The largest event the transport will carry. A bigger one is
            buried rather than retried for ever at the head of its run.
        dead_letter: Where an undeliverable row goes.
        retention_seconds: How long a published row is kept before `prune` removes it.
    """

    __slots__ = (
        "_batch",
        "_clock",
        "_dead_letter",
        "_max_bytes",
        "_publisher",
        "_retention",
        "_sql",
        "_tables",
        "_worker",
        "buried",
        "delivered",
        "failed",
    )

    def __init__(
        self,
        executor: SqlTransactor,
        publisher: EventPublisher,
        *,
        clock: Clock,
        worker: str,
        tables: OutboxTables = DEFAULT_OUTBOX_TABLES,
        batch: int = 32,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        dead_letter: DeadLetter | None = None,
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
    ) -> None:
        self._sql = executor
        self._publisher = publisher
        self._clock = clock
        self._worker = worker
        self._tables = tables
        self._batch = batch
        self._max_bytes = max_event_bytes
        self._dead_letter = dead_letter
        self._retention = retention_seconds
        self.delivered = 0
        self.buried = 0
        self.failed = 0

    async def deliver(self) -> int:
        """Publish one batch of claimed rows, and return how many were delivered.

        Publishing happens before the rows are marked, inside the claiming transaction: a
        crash in between republishes on the next poll, which the transport's dedupe on the
        event id collapses. The reverse order would lose events instead.

        Raises:
            Exception: Whatever the transport raised. Nothing is marked, the transaction
                rolls back, and the rows are claimed again next time.
        """
        try:
            async with self._sql.transaction() as session:
                return await self._batch_of(session)
        except Exception:
            self.failed += 1
            raise

    async def _batch_of(self, session: SqlSession) -> int:
        """One claim, its publishes, and the marks that follow them."""
        rows = await session.fetch(
            CLAIM.format(rows=self._tables.rows), self._worker, self._batch, self._clock.now()
        )
        if not rows:
            return 0
        sent: list[int] = []
        for identifier, run_id, payload in ((row[0], row[1], _text(row[2])) for row in rows):
            if await self._deliverable(payload, run_id):
                await self._publisher.publish(EventEnvelope.model_validate_json(payload))
                self.delivered += 1
            sent.append(int(identifier))
        await session.fetch(MARK_PUBLISHED.format(rows=self._tables.rows), sent, self._clock.now())
        return len(sent)

    async def _deliverable(self, payload: str, run_id: str) -> bool:
        """Whether this row goes to the transport, or to the dead letter instead."""
        if len(payload.encode()) > self._max_bytes:
            await self._bury(payload, "too_large", run_id)
            return False
        try:
            EventEnvelope.model_validate_json(payload)
        except Exception:  # anything that is not our envelope
            await self._bury(payload, "undecodable", run_id)
            return False
        return True

    async def _bury(self, payload: str, reason: str, run_id: str) -> None:
        """Keep it where somebody can look, so it stops blocking the head of its run."""
        if self._dead_letter is not None:
            await self._dead_letter.bury(payload.encode(), reason=reason, history=(run_id,))
        self.buried += 1

    async def lag(self) -> OutboxLag:
        """How many rows are waiting and how long the oldest has waited. Alert on both."""
        rows = await self._sql.fetch(LAG.format(rows=self._tables.rows), self._clock.now())
        if not rows:
            return OutboxLag()
        return OutboxLag(unpublished=int(rows[0][0]), oldest_seconds=float(rows[0][1]))

    async def prune(self) -> int:
        """Remove published rows past their retention, and return how many. Never unpublished."""
        rows = await self._sql.fetch(
            PRUNE.format(rows=self._tables.rows), self._clock.now() - self._retention
        )
        return int(rows[0][0]) if rows else 0


SCHEMA_OF = """
SELECT version FROM {schema} WHERE component = $1
"""

TIMEOUT = """
SHOW statement_timeout
"""

ENQUEUE = """
INSERT INTO {rows} (event_id, tenant, run_id, event_type, occurred_at, enqueued_at, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
ON CONFLICT (event_id) DO NOTHING
RETURNING id
"""

# A run is claimed whole under an advisory lock held for the transaction, which is what
# stops a second relay taking the rest of a run this one is part way through.
CLAIM = """
WITH candidates AS (
    SELECT run_id FROM {rows}
    WHERE published_at IS NULL
    GROUP BY run_id
    ORDER BY min(id)
    LIMIT $2
), held AS (
    SELECT run_id FROM candidates WHERE pg_try_advisory_xact_lock(hashtext(run_id))
)
UPDATE {rows} SET claimed_by = $1, claimed_at = $3, attempts = attempts + 1
WHERE published_at IS NULL AND run_id IN (SELECT run_id FROM held)
RETURNING id, run_id, payload::text
"""

MARK_PUBLISHED = """
UPDATE {rows} SET published_at = $2 WHERE id = ANY($1::bigint[])
"""

LAG = """
SELECT count(*), coalesce(max($1 - enqueued_at), 0.0)
FROM {rows} WHERE published_at IS NULL
"""

PRUNE = """
WITH removed AS (
    DELETE FROM {rows} WHERE published_at IS NOT NULL AND published_at < $1 RETURNING 1
)
SELECT count(*) FROM removed
"""

EXPECTED_OUTBOX_SCHEMA = """
-- Owned by the platform's migration repository. The kit reads adk_schema and refuses a
-- version it was not written for; it never applies this itself.

INSERT INTO adk_schema (component, version) VALUES ('outbox', 1);

CREATE TABLE adk_outbox (
    id bigserial PRIMARY KEY,
    event_id text NOT NULL UNIQUE,
    tenant text NOT NULL,
    run_id text NOT NULL,
    event_type text NOT NULL,
    occurred_at double precision NOT NULL,
    enqueued_at double precision NOT NULL,
    payload jsonb NOT NULL,
    attempts integer NOT NULL DEFAULT 0,
    claimed_by text,
    claimed_at double precision,
    published_at double precision
);

-- The relay's own read: unpublished rows, oldest run first.
CREATE INDEX adk_outbox_unpublished ON adk_outbox (run_id, id) WHERE published_at IS NULL;
-- Pruning published rows without scanning the unpublished ones.
CREATE INDEX adk_outbox_published ON adk_outbox (published_at) WHERE published_at IS NOT NULL;
"""
