"""A ceiling ledger in PostgreSQL, where the check and the take are one statement.

Two processes reading headroom and then writing against it is the leak this exists to
close: the reserve here is a single `INSERT ... SELECT` whose `WHERE` is the ceiling test,
so a row either lands under the limit or does not land. Nothing decides in Python.

The idempotency key is the unique index rather than a lookup, which is what makes a retry
of a call that may already have gone out ask about the same action instead of taking fresh
headroom. Amounts are `numeric` throughout: a ceiling in floating point is one that
reconciles against a bank statement a hundredth at a time.

The DDL is not here. `verify()` reads the schema version the platform's bootstrap applied
and refuses a shape this adapter was not written for; the expected shape is
`EXPECTED_CEILING_SCHEMA`, for the migration repository to own.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters.sql_state import (
    PostgresStoreSettings,
    SqlSession,
    _Running,
    _sqlstate,
    _state_failure,
)
from tesserix_adk.adapters.state import _text
from tesserix_adk.core.ceiling import DEFAULT_HOLD_SECONDS, Hold, HoldState, exact
from tesserix_adk.core.errors import CeilingExceededError, ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tesserix_adk.core.autonomy import Ceiling
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "CEILING_SCHEMA_VERSION",
    "DEFAULT_CEILING_TABLES",
    "EXPECTED_CEILING_SCHEMA",
    "CeilingTables",
    "PostgresCeilingLedger",
    "PostgresCeilingSettings",
]

CEILING_SCHEMA_VERSION = 1
"""The shape this adapter was written for, recorded by the migration that applied it."""

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALREADY_HELD = "23505"


@dataclass(frozen=True, slots=True)
class CeilingTables:
    """Where the rows live. One database may hold more than one kit deployment.

    Args:
        holds: Reservations, one row per action, held and settled alike.
        schema: Where the migration recorded which shape it applied.
    """

    holds: str = "adk_ceiling_holds"
    schema: str = "adk_schema"

    def __post_init__(self) -> None:
        """Refuse a name that could carry SQL, since a table name cannot be a parameter."""
        for name in (self.holds, self.schema):
            if not _IDENTIFIER.fullmatch(name):
                raise ConfigurationError(f"{name!r} is not a plain table identifier")


DEFAULT_CEILING_TABLES = CeilingTables()


class PostgresCeilingSettings(PostgresStoreSettings):
    """How to reach PostgreSQL, and what this deployment needs it to promise."""

    schema_version: int = CEILING_SCHEMA_VERSION


class PostgresCeilingLedger:
    """Headroom taken and settled across processes, on one row per action.

    Args:
        executor: Anything that can `fetch`. The `postgres` extra installs one.
        clock: What windows and expiry are measured against.
        settings: Connection, retry and timeout policy.
        tables: Where the rows live.
        hold_seconds: How long a reservation nobody settled keeps counting. A process that
            dies mid-call must not hold headroom until somebody notices.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        executor: SqlSession,
        *,
        clock: Clock,
        settings: PostgresCeilingSettings,
        tables: CeilingTables = DEFAULT_CEILING_TABLES,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._sql = executor
        self._clock = clock
        self._settings = settings
        self._tables = tables
        self._hold = hold_seconds
        self._run_sql = _Running(clock, settings, entropy, _state_failure(type(self).__name__))

    @classmethod
    async def open(cls, executor: SqlSession, **kwargs: Any) -> PostgresCeilingLedger:  # noqa: ANN401 — the constructor's own keywords
        """Construct the ledger and refuse a database that is not the shape it expects.

        Raises:
            ConfigurationError: If the schema version differs from the adapter's, or the
                connection has no statement timeout.
        """
        ledger = cls(executor, **kwargs)
        await ledger.verify()
        return ledger

    async def verify(self) -> None:
        """Check the schema version and the statement timeout. At startup, never per call.

        Raises:
            ConfigurationError: If the shape has moved, or a statement could run forever.
        """
        rows = await self._fetch(SCHEMA_OF.format(schema=self._tables.schema), "ceiling")
        found = int(rows[0][0]) if rows else 0
        if found != self._settings.schema_version:
            raise ConfigurationError(
                f"ceiling schema is version {found}, and this adapter writes version"
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

    async def reserve(
        self,
        *,
        tenant: str,
        action_class: str,
        ceiling: Ceiling,
        amount: Decimal,
        idempotency_key: str,
    ) -> Hold:
        """Take `amount` of headroom in one statement, or refuse because there is none.

        Not retried on a unique violation: a key that is already held is the same action,
        and it comes back as the reservation it already took.

        Raises:
            CeilingExceededError: If what is held and committed leaves less than `amount`.
            InexactAmountError: If `amount` is a float, is not a number, or is negative.
            StatePersistenceError: If the database could not answer. A reserve that failed
                is not a reserve of nothing.
        """
        asked = exact(amount)
        now = self._clock.now()
        try:
            rows = await self._sql.fetch(
                RESERVE.format(holds=self._tables.holds, columns=_COLUMNS),
                idempotency_key,
                tenant,
                action_class,
                ceiling.currency,
                asked,
                now,
                now + self._hold,
                ceiling.window_seconds,
                ceiling.amount,
            )
        except Exception as failure:
            if _sqlstate(failure) == _ALREADY_HELD:
                return await self._standing(idempotency_key)
            raise self._unreachable(failure) from failure
        if not rows:
            raise await self._refused(tenant, action_class, ceiling, asked)
        return self._read(rows[0])

    async def commit(self, idempotency_key: str) -> Hold | None:
        """Record that the action happened. Nothing where no live reservation is held.

        Raises:
            StatePersistenceError: If the database could not answer.
        """
        rows = await self._fetch(
            SETTLE.format(holds=self._tables.holds, columns=_COLUMNS),
            idempotency_key,
            HoldState.COMMITTED.value,
            self._clock.now(),
        )
        return self._read(rows[0]) if rows else None

    async def release(self, idempotency_key: str) -> None:
        """Give back headroom for an action that did not happen.

        A committed action is never released: the `WHERE` only matches a row still held,
        so a late retry of a settled call cannot hand its money back.

        Raises:
            StatePersistenceError: If the database could not answer.
        """
        await self._fetch(
            SETTLE.format(holds=self._tables.holds, columns=_COLUMNS),
            idempotency_key,
            HoldState.RELEASED.value,
            self._clock.now(),
        )

    async def committed(self, *, tenant: str, action_class: str, window_seconds: float) -> Decimal:
        """What this tenant has spent on this class in the window, exactly.

        Live holds count, because a ladder reading this is deciding whether to authorise
        one more action and headroom already promised to an action in flight is gone.

        Raises:
            StatePersistenceError: If the database could not answer.
        """
        rows = await self._fetch(
            SPENT.format(holds=self._tables.holds),
            tenant,
            action_class,
            self._clock.now(),
            window_seconds,
        )
        return exact(rows[0][0]) if rows and rows[0][0] is not None else Decimal(0)

    async def reap(self) -> int:
        """Release every hold past its TTL, and return how many. Run on a timer.

        Raises:
            StatePersistenceError: If the database could not answer.
        """
        rows = await self._fetch(REAP.format(holds=self._tables.holds), self._clock.now())
        return len(rows)

    async def _standing(self, idempotency_key: str) -> Hold:
        """The reservation a key already took, for a retry that asks about it again."""
        rows = await self._fetch(
            HELD_UNDER.format(holds=self._tables.holds, columns=_COLUMNS), idempotency_key
        )
        if not rows:
            raise self._unreachable(RuntimeError("the held row vanished between statements"))
        return self._read(rows[0])

    async def _refused(
        self, tenant: str, action_class: str, ceiling: Ceiling, asked: Decimal
    ) -> CeilingExceededError:
        """What no headroom raises, with what was actually left when it was asked."""
        spent = await self.committed(
            tenant=tenant, action_class=action_class, window_seconds=ceiling.window_seconds
        )
        headroom = ceiling.amount - spent
        return CeilingExceededError(
            f"{asked} is over the {headroom} {ceiling.currency} left under the ceiling",
            action_class=action_class,
            headroom=str(headroom),
            requested=str(asked),
        )

    def _read(self, row: Sequence[Any]) -> Hold:
        """One row as the reservation it records."""
        return Hold(
            id=str(row[0]),
            tenant=_text(row[1]),
            action_class=_text(row[2]),
            currency=_text(row[3]),
            amount=exact(row[4]),
            idempotency_key=_text(row[5]),
            reserved_at=float(row[6]),
            expires_at=float(row[7]),
            state=HoldState(_text(row[8])),
        )

    def _unreachable(self, failure: Exception) -> Exception:
        """What a database that could not take the write raises."""
        return _state_failure(type(self).__name__)(1, _sqlstate(failure) or "unavailable")

    async def _fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        return await self._run_sql(lambda: self._sql.fetch(statement, *args))


_COLUMNS = (
    "hold_id, tenant, action_class, currency, amount, idempotency_key, "
    "reserved_at, expires_at, state"
)

# The WHERE is the ceiling test, so the row lands under the limit or does not land at all.
RESERVE = """
INSERT INTO {holds}
    (idempotency_key, tenant, action_class, currency, amount, reserved_at, expires_at, state)
SELECT $1, $2, $3, $4, $5, $6, $7, 'held'
WHERE (
    SELECT COALESCE(SUM(h.amount), 0) FROM {holds} h
    WHERE h.tenant = $2 AND h.action_class = $3 AND h.currency = $4
      AND h.reserved_at > $6 - $8
      AND (h.state = 'committed' OR (h.state = 'held' AND h.expires_at > $6))
) + $5 <= $9
RETURNING {columns}
"""

SETTLE = """
UPDATE {{holds}} SET state = $2, settled_at = $3
WHERE idempotency_key = $1 AND state = 'held' AND expires_at > $3
RETURNING {columns}
"""

HELD_UNDER = """
SELECT {columns} FROM {{holds}} WHERE idempotency_key = $1
"""

SPENT = """
SELECT COALESCE(SUM(amount), 0) FROM {holds}
WHERE tenant = $1 AND action_class = $2 AND reserved_at > $3 - $4
  AND (state = 'committed' OR (state = 'held' AND expires_at > $3))
"""

REAP = """
UPDATE {holds} SET state = 'released', settled_at = $1
WHERE state = 'held' AND expires_at <= $1
RETURNING hold_id
"""

SCHEMA_OF = """
SELECT version FROM {schema} WHERE component = $1
"""

TIMEOUT = "SHOW statement_timeout"

EXPECTED_CEILING_SCHEMA = """
-- Owned by the platform's migration repository. The kit reads adk_schema and refuses a
-- version it was not written for; it never applies this itself.

INSERT INTO adk_schema (component, version) VALUES ('ceiling', 1);

-- One row per action, never one per attempt: the unique key on idempotency_key is what
-- makes a retry of a call that may already have gone out ask about the same reservation.
-- numeric, not double precision: a ceiling in floating point does not reconcile.
CREATE TABLE adk_ceiling_holds (
    hold_id         bigserial PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    tenant          text NOT NULL,
    action_class    text NOT NULL,
    currency        char(3) NOT NULL,
    amount          numeric(20, 4) NOT NULL CHECK (amount >= 0),
    reserved_at     double precision NOT NULL,
    expires_at      double precision NOT NULL,
    settled_at      double precision,
    state           text NOT NULL CHECK (state IN ('held', 'committed', 'released'))
);

-- The window sum, which is on the dispatch path of every action that carries an amount.
CREATE INDEX adk_ceiling_holds_window
    ON adk_ceiling_holds (tenant, action_class, currency, reserved_at)
    WHERE state <> 'released';

REVOKE DELETE ON adk_ceiling_holds FROM PUBLIC;
"""
