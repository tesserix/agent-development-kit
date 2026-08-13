"""The ledger a ceiling is enforced by: reserve, commit, release, in exact decimals.

A ceiling is only a ceiling if it cannot be walked around, and it leaks in three standard
ways. Two actions each read the same headroom and both fit. One action arrives as ten
small ones, each under the limit. A timed-out action is retried and takes fresh headroom
on top of spend that already happened. Each is answered here by the same thing: one
ledger, where the reservation is taken before the action and released or committed after
it, keyed by what a ceiling is actually about — tenant, action class, currency and window.

Arithmetic is `Decimal` and amounts arrive through `exact`. A float amount is refused
rather than rounded, because a ceiling that is a hundredth out is a ceiling nobody can
reconcile against a bank statement.

Credits are recorded, never netted off. Money coming back is a fact worth auditing, but
subtracting it from the window would hand an agent fresh headroom that nobody granted.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.errors import CeilingExceededError, InexactAmountError
from tesserix_adk.core.ledger import Window, WindowKind
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.autonomy import Ceiling
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "CeilingLedger",
    "Credit",
    "Hold",
    "HoldState",
    "InMemoryCeilingLedger",
    "exact",
]

DEFAULT_HOLD_SECONDS = 300.0
"""How long a reservation nobody came back for is honoured before it is reaped."""


def exact(amount: object) -> Decimal:
    """`amount` as a `Decimal`, or a typed refusal.

    Integers, decimals and numeric strings are exact and are taken as they are. A float is
    refused rather than converted: `0.1 + 0.2` is not `0.3`, and a limit built on that
    arithmetic is off by whatever the hardware felt like.

    Raises:
        InexactAmountError: If `amount` is a float, is not a number, or is negative.
    """
    if isinstance(amount, float):
        raise InexactAmountError(
            f"{amount!r} is a float; a ceiling counted in floats is a ceiling that drifts",
            amount=repr(amount),
        )
    if isinstance(amount, Decimal | int) and not isinstance(amount, bool):
        return _not_negative(Decimal(amount))
    try:
        return _not_negative(Decimal(str(amount)))
    except (InvalidOperation, ValueError, TypeError) as failure:
        raise InexactAmountError(
            f"{amount!r} is not an amount this ledger can add up", amount=repr(amount)
        ) from failure


def _not_negative(amount: Decimal) -> Decimal:
    """The amount, provided it is one a ceiling can be counted against."""
    if amount < 0:
        raise InexactAmountError(
            f"{amount} is negative; a negative action is a credit, which is signed separately",
            amount=str(amount),
        )
    return amount


class HoldState(StrEnum):
    """Where a reservation got to. Held counts against the ceiling exactly as committed does."""

    HELD = "held"
    COMMITTED = "committed"
    RELEASED = "released"


class Hold(AdkModel):
    """Headroom taken for one action, before the action happens.

    Args:
        id: Identity of this reservation. A new one is minted whenever headroom is taken
            afresh, so a released key that is reserved again is a different reservation.
        tenant: Whose ceiling it comes out of.
        action_class: Which class it is counted against.
        currency: ISO 4217. A ceiling in one currency counts nothing in another.
        amount: How much, exactly.
        idempotency_key: What a retry asks with. The same key never takes headroom twice.
        reserved_at: Unix seconds. Also what decides which window it lands in.
        expires_at: Unix seconds. Past this, a hold nobody settled stops counting.
        window_start: The start of the calendar bucket it belongs to, where windows are
            calendar. Rolling windows have none: the reservation's own age decides.
        state: Held, committed, or released.
    """

    id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    action_class: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    amount: Decimal = Field(ge=0)
    idempotency_key: str = Field(min_length=1)
    reserved_at: float
    expires_at: float
    window_start: float | None = None
    state: HoldState = HoldState.HELD

    def live(self, at: float) -> bool:
        """Whether this reservation still counts against a ceiling at `at`."""
        if self.state is HoldState.COMMITTED:
            return True
        return self.state is HoldState.HELD and self.expires_at > at


class Credit(AdkModel):
    """Money coming back, recorded against the class it came back from.

    A credit never creates headroom. It is here so that a reconciliation can see what was
    refunded without an agent being handed the same limit twice in one window.

    Args:
        tenant: Whose class it lands against.
        action_class: Which class.
        currency: ISO 4217.
        amount: How much came back, exactly.
        authorised_by: The human or system identity that signed it off.
        reason: Why, for whoever reads the reconciliation.
        at: Unix seconds.
    """

    tenant: str = Field(min_length=1)
    action_class: str = Field(min_length=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    amount: Decimal = Field(gt=0)
    authorised_by: str = Field(min_length=1)
    reason: str = ""
    at: float


@runtime_checkable
class CeilingLedger(Protocol):
    """Reserve before the action, settle after it. Also what a ladder reads headroom from."""

    async def reserve(
        self,
        *,
        tenant: str,
        action_class: str,
        ceiling: Ceiling,
        amount: Decimal,
        idempotency_key: str,
    ) -> Hold:
        """Take headroom for one action, or refuse because there is not enough."""
        ...

    async def commit(self, idempotency_key: str) -> Hold | None:
        """Record that the action happened. Nothing where the reservation is gone."""
        ...

    async def release(self, idempotency_key: str) -> None:
        """Give back headroom for an action that did not happen."""
        ...

    async def committed(self, *, tenant: str, action_class: str, window_seconds: float) -> Decimal:
        """The total committed in the window, exactly."""
        ...


class InMemoryCeilingLedger:
    """One process's ledger, correct under concurrency within that process.

    Reserve, commit and release are serialised on one lock, which is what stops two
    concurrent actions each reading the same headroom and both fitting under it. Across
    processes, that guarantee is the database's — see `PostgresCeilingLedger`.

    Args:
        clock: What windows and expiry are measured against.
        hold_seconds: How long a reservation nobody settled keeps counting. A process that
            dies mid-call must not hold headroom until somebody notices.
        windows: Rolling or calendar, as `WindowKind` reads them elsewhere in the kit.
            Rolling is the default: it is what a ceiling like "5000 a day" usually means.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        windows: WindowKind = WindowKind.ROLLING,
    ) -> None:
        self._clock = clock
        self._hold = hold_seconds
        self._windows = windows
        self._lock = asyncio.Lock()
        self._reservations: dict[str, Hold] = {}
        self._credits: list[Credit] = []
        self._minted = 0

    async def reserve(
        self,
        *,
        tenant: str,
        action_class: str,
        ceiling: Ceiling,
        amount: Decimal,
        idempotency_key: str,
    ) -> Hold:
        """Take `amount` of headroom, or refuse.

        The same `idempotency_key` returns the reservation it already took: a retry of an
        action asks about the action, never for more headroom.

        Raises:
            CeilingExceededError: If what is held and committed leaves less than `amount`.
            InexactAmountError: If `amount` is a float, is not a number, or is negative.
        """
        asked = exact(amount)
        async with self._lock:
            now = self._clock.now()
            standing = self._reservations.get(idempotency_key)
            if standing is not None and standing.live(now):
                return standing
            spent = self._spent(tenant, action_class, ceiling, now)
            headroom = ceiling.amount - spent
            if asked > headroom:
                raise CeilingExceededError(
                    f"{asked} {ceiling.currency} is over the {headroom} left of the "
                    f"{ceiling.amount} {ceiling.currency} ceiling on {action_class}",
                    action_class=action_class,
                    headroom=str(headroom),
                    requested=str(asked),
                )
            taken = Hold(
                id=self._id(),
                tenant=tenant,
                action_class=action_class,
                currency=ceiling.currency,
                amount=asked,
                idempotency_key=idempotency_key,
                reserved_at=now,
                expires_at=now + self._hold,
                window_start=self._window_start(now, ceiling.window_seconds),
            )
            self._reservations[idempotency_key] = taken
            return taken

    async def commit(self, idempotency_key: str) -> Hold | None:
        """Turn a live hold into spend, once however many times it is asked."""
        async with self._lock:
            standing = self._reservations.get(idempotency_key)
            if standing is None or not standing.live(self._clock.now()):
                return None
            settled = standing.model_copy(update={"state": HoldState.COMMITTED})
            self._reservations[idempotency_key] = settled
            return settled

    async def release(self, idempotency_key: str) -> None:
        """Give back a hold that never became spend. A committed action is never released."""
        async with self._lock:
            standing = self._reservations.get(idempotency_key)
            if standing is None or standing.state is not HoldState.HELD:
                return
            self._reservations[idempotency_key] = standing.model_copy(
                update={"state": HoldState.RELEASED}
            )

    async def reap(self) -> int:
        """Release every hold past its TTL, and return how many. Run on a timer."""
        async with self._lock:
            now = self._clock.now()
            stale = [
                key
                for key, held in self._reservations.items()
                if held.state is HoldState.HELD and held.expires_at <= now
            ]
            for key in stale:
                self._reservations[key] = self._reservations[key].model_copy(
                    update={"state": HoldState.RELEASED}
                )
            return len(stale)

    async def committed(self, *, tenant: str, action_class: str, window_seconds: float) -> Decimal:
        """What this tenant has spent on this class in the window, exactly.

        Live holds count, because a ladder reading this is deciding whether to authorise
        one more action and headroom already promised to an action in flight is gone.
        """
        now = self._clock.now()
        start = self._window_start(now, window_seconds)
        return sum(
            (
                held.amount
                for held in self._reservations.values()
                if held.tenant == tenant
                and held.action_class == action_class
                and held.live(now)
                and self._inside(held, now, window_seconds, start)
            ),
            Decimal(0),
        )

    async def credit(
        self,
        *,
        tenant: str,
        action_class: str,
        currency: str,
        amount: Decimal,
        authorised_by: str,
        reason: str = "",
    ) -> Credit:
        """Record money coming back. It is audited, and it does not become headroom.

        Raises:
            ValueError: If nobody is named as having authorised it.
        """
        if not authorised_by:
            raise ValueError(
                "a credit names who authorised it; an unsigned credit is an adjustment "
                "nobody can be asked about"
            )
        recorded = Credit(
            tenant=tenant,
            action_class=action_class,
            currency=currency,
            amount=exact(amount),
            authorised_by=authorised_by,
            reason=reason,
            at=self._clock.now(),
        )
        self._credits.append(recorded)
        return recorded

    def credits(self) -> Sequence[Credit]:
        """Every credit recorded, for reconciliation."""
        return tuple(self._credits)

    def _spent(self, tenant: str, action_class: str, ceiling: Ceiling, now: float) -> Decimal:
        """Everything held or committed against this ceiling inside its window."""
        start = self._window_start(now, ceiling.window_seconds)
        return sum(
            (
                held.amount
                for held in self._reservations.values()
                if held.tenant == tenant
                and held.action_class == action_class
                and held.currency == ceiling.currency
                and held.live(now)
                and self._inside(held, now, ceiling.window_seconds, start)
            ),
            Decimal(0),
        )

    def _inside(self, held: Hold, now: float, window_seconds: float, start: float | None) -> bool:
        """Whether `held` falls in the window that is current at `now`.

        A calendar window compares the bucket the hold was taken in, not the one it settled
        in: an action that crosses midnight belongs to the day it was authorised.
        """
        if start is not None:
            return held.window_start == start
        return held.reserved_at > now - window_seconds

    def _window_start(self, now: float, window_seconds: float) -> float | None:
        """The start of the calendar bucket `now` is in, or nothing for a rolling window."""
        if self._windows is WindowKind.ROLLING:
            return None
        return Window(kind=self._windows, seconds=int(window_seconds)).opened_at(now)

    def _id(self) -> str:
        """The next reservation id. Sequential, so a test can read the order back."""
        self._minted += 1
        return f"res-{self._minted}"
