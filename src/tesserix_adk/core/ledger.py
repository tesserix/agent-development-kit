"""The vocabulary of a spend ledger shared between replicas.

A per-process budget cannot hold a per-tenant ceiling: eight replicas each believe they
have the whole allowance, and the tenant spends eight times it. The ledger is where the
window actually lives, so the ceiling means the same thing however many pods are serving.

Only the vocabulary is here. The stores that implement it are in `tesserix_adk.adapters`,
because which store a deployment runs is a deployment's decision.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, field_validator

from tesserix_adk.core.models import AdkModel

__all__ = [
    "SEPARATOR",
    "LedgerKey",
    "Reservation",
    "SpendLedger",
    "Window",
    "WindowKind",
    "WindowSpend",
]

# Keys are built by joining fields, so a field carrying the separator could name another
# tenant's window. Rejected at construction rather than escaped.
SEPARATOR = ":"


class WindowKind(StrEnum):
    """How a window's boundary is decided."""

    ROLLING = "rolling"
    """The last `seconds`, moving continuously. No cliff, no thundering herd at the hour."""

    CALENDAR = "calendar"
    """Fixed buckets aligned to the epoch, which is what an invoice period looks like."""


class Window(AdkModel):
    """The period a ceiling applies over.

    Args:
        kind: Rolling or calendar.
        seconds: Its length. An hourly ceiling is 3600 either way; what differs is when it
            resets.
    """

    kind: WindowKind
    seconds: int = Field(gt=0)

    def bucket(self, now: float) -> int:
        """Which calendar bucket `now` falls in. Always 0 for a rolling window."""
        return 0 if self.kind is WindowKind.ROLLING else int(now // self.seconds)

    def opened_at(self, now: float) -> float:
        """The earliest moment whose spend still counts against this window."""
        if self.kind is WindowKind.ROLLING:
            return now - self.seconds
        return float(self.bucket(now) * self.seconds)

    def resets_at(self, now: float) -> float:
        """When the allowance returns in full, at the latest.

        A rolling window has no boundary, so the answer is a window-length away: the worst
        case, which is what somebody deciding whether to wait needs.
        """
        if self.kind is WindowKind.ROLLING:
            return now + self.seconds
        return self.opened_at(now) + self.seconds


class LedgerKey(AdkModel):
    """Whose window this is.

    Identifiers and nothing else. Ledger contents are read by operators who were never
    cleared to read prompts, and a key carrying a user's question would put it in front of
    them by accident.

    Args:
        tenant: Who is billed.
        agent: A per-agent sub-ceiling, or `None` for the tenant's whole allowance.
        window: The period.
    """

    tenant: str = Field(min_length=1)
    agent: str | None = None
    window: Window

    @field_validator("tenant", "agent")
    @classmethod
    def _no_separator(cls, value: str | None) -> str | None:
        if value is not None and SEPARATOR in value:
            raise ValueError(
                f"a tenant or agent name may not contain the key separator {SEPARATOR!r}: "
                f"one that does can name another tenant's window"
            )
        return value

    @property
    def name(self) -> str:
        """The key as a store writes it, without the calendar bucket."""
        return SEPARATOR.join(
            (self.tenant, self.agent or "", self.window.kind, str(self.window.seconds))
        )

    def at(self, now: float) -> str:
        """The key including the bucket `now` falls in, which is what a store counts under."""
        return f"{self.name}{SEPARATOR}{self.window.bucket(now)}"


class Reservation(AdkModel):
    """An allowance held while a run spends it.

    Args:
        id: Unique to this hold. What settle and release name.
        key: The window it was taken from, as `LedgerKey.at` wrote it.
        amount: What is held.
        currency: What it is held in.
        taken_at: When the hold started.
        expires_at: When the lease lapses. A replica that dies must not hold a tenant's
            allowance until the window rolls.
        degraded: Whether the ledger was unreachable and a configured degraded mode let
            this through unchecked. Never inferred; recorded so a bill can be explained.
    """

    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    amount: Decimal
    currency: str = "USD"
    taken_at: float = 0.0
    expires_at: float = 0.0
    degraded: bool = False


class WindowSpend(AdkModel):
    """What a window holds right now.

    Args:
        settled: Spend that has happened.
        reserved: Spend held against in-flight runs.
        currency: What both are in.
        resets_at: When the allowance next returns in full.
    """

    settled: Decimal = Decimal(0)
    reserved: Decimal = Decimal(0)
    currency: str = "USD"
    resets_at: float = 0.0

    @property
    def committed(self) -> Decimal:
        """Settled plus held, which is what a ceiling is checked against."""
        return self.settled + self.reserved

    def remaining(self, ceiling: Decimal) -> Decimal:
        """What is left under `ceiling`, never negative."""
        return max(ceiling - self.committed, Decimal(0))


@runtime_checkable
class SpendLedger(Protocol):
    """Where a shared ceiling is actually enforced.

    Reserve before the spend, settle after it. Checking a ceiling after the money is gone
    is reporting, not enforcement.

    Every method fails closed: a store that cannot be reached raises rather than returning
    a permissive answer, because carrying on without the ledger is how one outage becomes
    an unbounded bill.
    """

    async def reserve(
        self,
        key: LedgerKey,
        amount: Decimal,
        *,
        ceiling: Decimal,
        lease_seconds: float = ...,
    ) -> Reservation:
        """Hold `amount` against `key`'s window, atomically with the ceiling check.

        Raises:
            BudgetExceededError: The hold would take the window past `ceiling`.
            BudgetUnavailableError: The store could not be reached.
        """
        ...

    async def settle(self, reservation: Reservation, actual: Decimal) -> None:
        """Convert a hold into spend of `actual`, releasing whatever was over-held.

        Raises:
            BudgetUnavailableError: The reservation is unknown, already closed, or the
                store could not be reached.
        """
        ...

    async def release(self, reservation: Reservation) -> None:
        """Give a hold back unspent.

        Raises:
            BudgetUnavailableError: As `settle`.
        """
        ...

    async def record_progress(self, reservation: Reservation, spent: Decimal) -> None:
        """Say what a run has spent so far, so a lease that lapses can settle for it.

        Without it, reconciliation can only release: nothing recorded what the replica
        that died had already been charged for.

        Raises:
            BudgetUnavailableError: As `settle`.
        """
        ...

    async def read_window(self, key: LedgerKey) -> WindowSpend:
        """What `key`'s window holds now.

        Raises:
            BudgetUnavailableError: The store could not be reached.
        """
        ...

    async def reconcile(self) -> int:
        """Close out holds whose lease has lapsed, and say how many.

        A hold with recorded progress settles against it; one with none is released. The
        window is then neither permanently reduced by a dead replica nor credited with
        spend that did happen.
        """
        ...

    async def forget(self, tenant: str) -> WindowSpend:
        """Erase a tenant's records, returning the aggregate that was dropped.

        What remains after erasure is a number, not a history: enough to answer what the
        window held, carrying nothing that names anybody.
        """
        ...
