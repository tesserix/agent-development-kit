"""Autonomy grants in PostgreSQL, written by people and read by the runtime.

A grant is the record of a decision somebody made, so this store appends and never
rewrites: an id in use is refused rather than updated, because a decision already recorded
against that id has to stay readable as what it permitted. Revocation is a separate story
and lands as its own column, not as an overwrite of the row.

The read is deliberately narrow — one tenant, one action class, unexpired at one moment —
so that the only thing reachable from inside a run is what is live right now. There is no
write path on the reader the ladder holds.

The DDL is not here. `verify()` reads the schema version the platform's bootstrap applied
and refuses a shape this adapter was not written for; the expected shape is
`EXPECTED_GRANT_SCHEMA`, for the migration repository to own.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters.sql_state import (
    PostgresStoreSettings,
    SqlSession,
    _Running,
    _sqlstate,
    _state_failure,
)
from tesserix_adk.adapters.state import _text
from tesserix_adk.core.autonomy import AutonomyGrant
from tesserix_adk.core.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "DEFAULT_GRANT_TABLES",
    "EXPECTED_GRANT_SCHEMA",
    "GRANT_SCHEMA_VERSION",
    "GrantTables",
    "PostgresGrantSettings",
    "PostgresGrantStore",
]

GRANT_SCHEMA_VERSION = 1
"""The shape this adapter was written for, recorded by the migration that applied it."""

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALREADY_ISSUED = "23505"


@dataclass(frozen=True, slots=True)
class GrantTables:
    """Where the rows live. One database may hold more than one kit deployment.

    Args:
        grants: Issued grants, one row per grant, expired and live alike.
        schema: Where the migration recorded which shape it applied.
    """

    grants: str = "adk_grants"
    schema: str = "adk_schema"

    def __post_init__(self) -> None:
        """Refuse a name that could carry SQL, since a table name cannot be a parameter."""
        for name in (self.grants, self.schema):
            if not _IDENTIFIER.fullmatch(name):
                raise ConfigurationError(f"{name!r} is not a plain table identifier")


DEFAULT_GRANT_TABLES = GrantTables()


class PostgresGrantSettings(PostgresStoreSettings):
    """How to reach PostgreSQL, and what this deployment needs it to promise."""

    schema_version: int = GRANT_SCHEMA_VERSION


class PostgresGrantStore:
    """Grants in one table, read live and written once.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: What "unexpired" is measured against, so a test can move it.
        settings: Connection, retry and timeout policy.
        tables: Where the rows live.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlSession,
        *,
        clock: Clock,
        settings: PostgresGrantSettings,
        tables: GrantTables = DEFAULT_GRANT_TABLES,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._sql = executor
        self._clock = clock
        self._settings = settings
        self._tables = tables
        self._run_sql = _Running(clock, settings, entropy, _state_failure(type(self).__name__))

    @classmethod
    async def open(cls, executor: SqlSession, **kwargs: Any) -> PostgresGrantStore:  # noqa: ANN401 — the constructor's own keywords
        """Construct the store and refuse a database that is not the shape it expects.

        Raises:
            ConfigurationError: If the schema version differs from the adapter's, or the
                connection has no statement timeout.
        """
        store = cls(executor, **kwargs)
        await store.verify()
        return store

    async def verify(self) -> None:
        """Check the schema version and the statement timeout. At startup, never per call.

        Raises:
            ConfigurationError: If the shape has moved, or a statement could run forever.
        """
        rows = await self._fetch(SCHEMA_OF.format(schema=self._tables.schema), "grants")
        found = int(rows[0][0]) if rows else 0
        if found != self._settings.schema_version:
            raise ConfigurationError(
                f"grants schema is version {found}, and this adapter writes version"
                f" {self._settings.schema_version}; a column that moved is a write into the"
                f" wrong shape"
            )
        timeout = await self._fetch(TIMEOUT)
        if _text(timeout[0][0]) in {"0", "0ms"}:
            raise ConfigurationError(
                "this connection has no statement_timeout; one held statement would hold a"
                f" pooled connection until the process restarts. Set it to about"
                f" {self._settings.statement_timeout_ms}ms on the role or the pool"
            )

    async def grants_for(self, *, tenant: str, action_class: str) -> Sequence[AutonomyGrant]:
        """Every unexpired grant that could speak for `tenant` about `action_class`.

        Grants on the tenants above this one come back too; whether they reach down is the
        grant's own `includes_subtenants`, decided where the rest of the grant is read.

        Raises:
            StatePersistenceError: If the database could not answer. A read that failed is
                not a read of no grants.
        """
        rows = await self._fetch(
            GRANTS_FOR.format(grants=self._tables.grants),
            tenant,
            action_class,
            self._clock.now(),
        )
        return [AutonomyGrant.model_validate_json(_text(row[0])) for row in rows]

    async def issue(self, grant: AutonomyGrant) -> AutonomyGrant:
        """Record `grant`, refusing an id that already exists.

        Not retried: issuance is a person's deliberate write, and a retry of an insert
        that may already have landed is how an id ends up meaning two things.

        Raises:
            ConfigurationError: If the id is in use. A grant is never rewritten, because a
                decision recorded against an id must stay readable as what it permitted.
            StatePersistenceError: If the database could not answer.
        """
        try:
            rows = await self._sql.fetch(
                ISSUE.format(grants=self._tables.grants),
                grant.id,
                grant.tenant,
                grant.action_class,
                grant.expires_at,
                grant.model_dump_json(),
            )
        except Exception as failure:
            if _sqlstate(failure) == _ALREADY_ISSUED:
                raise self._taken(grant) from failure
            raise self._unreachable(failure) from failure
        if not rows:
            raise self._taken(grant)
        return grant

    def _unreachable(self, failure: Exception) -> Exception:
        """What a database that could not take the write raises."""
        return _state_failure(type(self).__name__)(1, _sqlstate(failure) or "unavailable")

    def _taken(self, grant: AutonomyGrant) -> ConfigurationError:
        """What an id already in use raises, however the database said so."""
        return ConfigurationError(
            f"grant {grant.id!r} already exists; re-granting mints a new id so that a "
            f"decision recorded against an id stays readable as what it permitted"
        )

    async def _fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        return await self._run_sql(lambda: self._sql.fetch(statement, *args))


GRANTS_FOR = """
SELECT payload FROM {grants}
WHERE action_class = $2 AND expires_at > $3
  AND ($1 = tenant OR $1 LIKE tenant || '/%')
"""

ISSUE = """
INSERT INTO {grants} (grant_id, tenant, action_class, expires_at, payload)
VALUES ($1, $2, $3, $4, $5)
RETURNING 1
"""

SCHEMA_OF = """
SELECT version FROM {schema} WHERE component = $1
"""

TIMEOUT = "SHOW statement_timeout"

EXPECTED_GRANT_SCHEMA = """
-- Owned by the platform's migration repository. The kit reads adk_schema and refuses a
-- version it was not written for; it never applies this itself.

INSERT INTO adk_schema (component, version) VALUES ('grants', 1);

-- Append-only. A grant is the record of a decision, so nothing here is ever updated:
-- re-granting inserts a new id, and an id in use is refused by the primary key.
CREATE TABLE adk_grants (
    grant_id     text PRIMARY KEY,
    tenant       text NOT NULL,
    action_class text NOT NULL,
    expires_at   double precision NOT NULL,
    payload      jsonb NOT NULL
);

-- The dispatch-path read: one tenant, one class, unexpired.
CREATE INDEX adk_grants_live ON adk_grants (tenant, action_class, expires_at);

REVOKE UPDATE, DELETE ON adk_grants FROM PUBLIC;
"""
