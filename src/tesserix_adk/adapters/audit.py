"""Where the record of an unattended action is actually kept.

Two stores, on purpose. PostgreSQL is what the question is asked of afterwards — one
tenant, one period, declines included — and it is append-only, so a decision recorded
cannot later be edited into a decision somebody would rather have taken. JetStream is the
same record on a stream, for a deployment that wants audit off the transaction path or in a
second administrative domain, where the write is a publish nobody with database access can
quietly undo.

Neither is the telemetry pipeline, and that is the whole design. Sampling that is correct
for spans is a missing record here, and the missing one is always the one being asked about.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tesserix_adk.adapters.sql_state import (
    PostgresStoreSettings,
    SqlSession,
    _Running,
    _state_failure,
)
from tesserix_adk.adapters.state import _text
from tesserix_adk.core.audit import AuditDecision, AuditEvent, pseudonym
from tesserix_adk.core.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DEFAULT_AUDIT_SUBJECT",
    "DEFAULT_AUDIT_TABLES",
    "EXPECTED_AUDIT_SCHEMA",
    "AuditTables",
    "JetStreamAudit",
    "JetStreamPublisher",
    "PostgresAuditSettings",
    "PostgresAuditSink",
]

AUDIT_SCHEMA_VERSION = 1
"""The shape this adapter was written for, recorded by the migration that applied it."""

DEFAULT_AUDIT_SUBJECT = "adk.audit"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SUBJECT_TOKEN = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class AuditTables:
    """Where the rows live. One database may hold more than one kit deployment.

    Args:
        events: One row per decision, appended and never removed.
        schema: Where the migration recorded which shape it applied.
    """

    events: str = "adk_audit"
    schema: str = "adk_schema"

    def __post_init__(self) -> None:
        """Refuse a name that could carry SQL, since a table name cannot be a parameter."""
        for name in (self.events, self.schema):
            if not _IDENTIFIER.fullmatch(name):
                raise ConfigurationError(f"{name!r} is not a plain table identifier")


DEFAULT_AUDIT_TABLES = AuditTables()


class PostgresAuditSettings(PostgresStoreSettings):
    """How to reach PostgreSQL, and what this deployment needs it to promise."""

    schema_version: int = AUDIT_SCHEMA_VERSION


class PostgresAuditSink:
    """Decisions in one append-only table, read back by whoever is accountable for them.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        settings: Connection, retry and timeout policy.
        clock: What the retry backoff is measured against.
        tables: Where the rows live.
        pseudonym_salt: What an erasure's stand-in name is taken under, so two deployments
            cannot join their audit stores on the same person.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlSession,
        *,
        settings: PostgresAuditSettings,
        clock: Clock,
        tables: AuditTables = DEFAULT_AUDIT_TABLES,
        pseudonym_salt: str = "",
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._sql = executor
        self._settings = settings
        self._tables = tables
        self._salt = pseudonym_salt
        self._run_sql = _Running(clock, settings, entropy, _state_failure(type(self).__name__))

    @classmethod
    async def open(cls, executor: SqlSession, **kwargs: Any) -> PostgresAuditSink:  # noqa: ANN401 — the constructor's own keywords
        """Construct the sink and refuse a database that is not the shape it expects.

        Raises:
            ConfigurationError: If the schema version differs from the adapter's, or the
                connection has no statement timeout.
        """
        sink = cls(executor, **kwargs)
        await sink.verify()
        return sink

    async def verify(self) -> None:
        """Check the schema version and the statement timeout. At startup, never per write.

        Raises:
            ConfigurationError: If the shape has moved, or a statement could run forever.
        """
        rows = await self._fetch(SCHEMA_OF.format(schema=self._tables.schema), "audit")
        found = int(rows[0][0]) if rows else 0
        if found != self._settings.schema_version:
            raise ConfigurationError(
                f"audit schema is version {found}, and this adapter writes version"
                f" {self._settings.schema_version}; a column that moved is a decision"
                f" recorded into the wrong shape"
            )
        timeout = await self._fetch(TIMEOUT)
        if _text(timeout[0][0]) in {"0", "0ms"}:
            raise ConfigurationError(
                "this connection has no statement_timeout; one held statement would hold a"
                f" pooled connection until the process restarts. Set it to about"
                f" {self._settings.statement_timeout_ms}ms on the role or the pool"
            )

    async def append(self, event: AuditEvent) -> AuditEvent:
        """Record `event`, or return what was already recorded for that decision.

        Retried, unlike a grant: the write is idempotent on the decision, so an activity
        that retries adds nothing, and a decision lost to one timeout is the failure this
        store exists to prevent.

        Raises:
            StatePersistenceError: If the database could not take the record. The caller
                fails the action closed rather than acting unaudited.
        """
        rows = await self._fetch(
            APPEND.format(events=self._tables.events),
            event.run_id,
            event.idempotency_key,
            str(event.decision),
            event.tenant,
            event.sequence,
            event.recorded_at,
            event.model_dump_json(),
        )
        if rows:
            return event
        stored = await self._fetch(
            RECORDED.format(events=self._tables.events),
            event.run_id,
            event.idempotency_key,
            str(event.decision),
        )
        return AuditEvent.model_validate_json(_text(stored[0][0])) if stored else event

    async def records(
        self,
        *,
        tenant: str,
        since: float = 0.0,
        until: float | None = None,
        decision: AuditDecision | None = None,
    ) -> tuple[AuditEvent, ...]:
        """Every decision for `tenant` in the period, oldest first, declines included.

        Raises:
            StatePersistenceError: If the database could not answer. A read that failed is
                not a read of no decisions.
        """
        rows = await self._fetch(
            RECORDS.format(events=self._tables.events),
            tenant,
            since,
            until,
            str(decision) if decision is not None else None,
        )
        return tuple(AuditEvent.model_validate_json(_text(row[0])) for row in rows)

    async def pseudonymise(self, *, tenant: str, subject: str) -> int:
        """Replace `subject` wherever it named a person, and say how many rows changed.

        The one statement here that is not an insert, which is why the schema grants it to
        a role of its own: the decision survives an erasure, the person does not, and a
        deletion would take the evidence that the action was permitted with it.

        Raises:
            StatePersistenceError: If the database could not answer.
        """
        rows = await self._fetch(
            PSEUDONYMISE.format(events=self._tables.events),
            tenant,
            subject,
            pseudonym(subject, salt=self._salt),
        )
        return len(rows)

    async def _fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        return await self._run_sql(lambda: self._sql.fetch(statement, *args))


@runtime_checkable
class JetStreamPublisher(Protocol):
    """The one method this adapter uses from a JetStream client."""

    async def publish(self, subject: str, payload: bytes) -> None:
        """Publish `payload` on `subject`, acknowledged by the stream."""
        ...


class JetStreamAudit:
    """Publishes each decision on a per-tenant subject, for a stream that keeps it.

    The subject carries the tenant so a consumer can be authorised for its own and no
    other, which is the isolation the stream can enforce and the payload cannot. Reading
    back is the stream's own job, not this adapter's: use it alongside a queryable sink, or
    behind a consumer that writes one.

    Args:
        publisher: A connected JetStream client. Its publish must be acknowledged — a
            fire-and-forget NATS publish is not a durable audit record.
        subject: The subject root. The tenant is appended to it.

    Raises:
        ConfigurationError: If the root is not a plain subject token.
    """

    def __init__(
        self, publisher: JetStreamPublisher, *, subject: str = DEFAULT_AUDIT_SUBJECT
    ) -> None:
        self._publisher = publisher
        self._subject = _token(subject, allow_dots=True)

    async def append(self, event: AuditEvent) -> AuditEvent:
        """Publish `event`, and return it as recorded.

        The stream deduplicates on the message id a consumer derives from the decision, so
        a retried publish is one record there for the same reason it is one row in
        PostgreSQL.

        Raises:
            ConfigurationError: If the tenant could not be used as a subject token, which
                would publish the decision to a wider audience than the tenant.
            Exception: Whatever the client raises when the stream did not acknowledge.
        """
        await self._publisher.publish(
            f"{self._subject}.{_token(event.tenant)}", event.model_dump_json().encode()
        )
        return event


def _token(value: str, *, allow_dots: bool = False) -> str:
    """A subject token, or a refusal — a wildcard here widens who hears the decision."""
    parts = value.split(".") if allow_dots else [value]
    if not value or not all(_SUBJECT_TOKEN.fullmatch(part) for part in parts):
        raise ConfigurationError(f"{value!r} is not a plain subject token")
    return value


APPEND = """
INSERT INTO {events} (run_id, idempotency_key, decision, tenant, sequence, recorded_at, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (run_id, idempotency_key, decision) DO NOTHING
RETURNING 1
"""

RECORDED = """
SELECT payload FROM {events}
WHERE run_id = $1 AND idempotency_key = $2 AND decision = $3
"""

RECORDS = """
SELECT payload FROM {events}
WHERE tenant = $1 AND recorded_at >= $2
  AND ($3::double precision IS NULL OR recorded_at < $3)
  AND ($4::text IS NULL OR decision = $4)
ORDER BY recorded_at, run_id, sequence
"""

PSEUDONYMISE = """
UPDATE {events}
SET payload = jsonb_set(
        jsonb_set(payload, '{{user}}',
            CASE WHEN payload->>'user' = $2 THEN to_jsonb($3::text) ELSE payload->'user' END),
        '{{approver}}',
        CASE WHEN payload->>'approver' = $2 THEN to_jsonb($3::text) ELSE payload->'approver' END)
WHERE tenant = $1 AND (payload->>'user' = $2 OR payload->>'approver' = $2)
RETURNING 1
"""

SCHEMA_OF = """
SELECT version FROM {schema} WHERE component = $1
"""

TIMEOUT = "SHOW statement_timeout"

EXPECTED_AUDIT_SCHEMA = """
-- Owned by the platform's migration repository. The kit reads adk_schema and refuses a
-- version it was not written for; it never applies this itself.

INSERT INTO adk_schema (component, version) VALUES ('audit', 1);

-- Append-only. One row per decision about one attempted action, executions and refusals
-- alike: a ceiling nobody recorded holding is a ceiling nobody can show held.
CREATE TABLE adk_audit (
    audit_id        bigserial PRIMARY KEY,
    run_id          text NOT NULL,
    idempotency_key text NOT NULL,
    decision        text NOT NULL,
    tenant          text NOT NULL,
    sequence        integer NOT NULL,
    recorded_at     double precision NOT NULL,
    payload         jsonb NOT NULL,
    UNIQUE (run_id, idempotency_key, decision)
);

-- The question this store exists to answer: one tenant, one period, declines included.
CREATE INDEX adk_audit_asked ON adk_audit (tenant, recorded_at);
CREATE INDEX adk_audit_by_run ON adk_audit (run_id, sequence);

-- Nothing here is ever deleted and nothing rewrites a decision. An erasure request
-- pseudonymises the person and keeps the decision, which is the one statement that is not
-- an insert, so it is granted to a role of its own and audited where that role is used.
REVOKE UPDATE, DELETE ON adk_audit FROM PUBLIC;
GRANT UPDATE (payload) ON adk_audit TO adk_erasure;

-- Retention: decisions are kept for as long as the deployment is accountable for the
-- actions they permitted — seven years is the usual floor where money moved. Expiry is a
-- scheduled job in the migration repository, never a statement the kit can issue.
"""
