"""Stopping work nobody is waiting for any more.

Cancellation in an agent loop is not politeness. A caller that abandons a request and an
agent that keeps calling a provider are the same bug seen from two ends, and the bill
arrives regardless. A `CancellationToken` is the switch the caller flips; a `Deadline` is
the same switch on a timer.

Both are cooperative. The runtime cancels the task doing the work and waits a grace
window for it to unwind; work that ignores the abort is dropped and reported orphaned,
never waited on and never assumed not to have happened.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tesserix_adk.core import CancelledError

__all__ = ["CancellationToken", "Deadline"]

_DEFAULT_REASON = "cancelled by the caller"


class CancellationToken:
    """A one-way switch that asks a run to stop.

    Cancelling is idempotent, and the first reason is the one that stands: two callers
    racing to cancel the same run must not leave it disagreeing with itself about why it
    stopped.

    Example:
        >>> token = CancellationToken()
        >>> token.cancelled
        False
        >>> token.cancel("caller went away")
        >>> (token.cancelled, token.reason)
        (True, 'caller went away')
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested. Never goes back to False."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """Why, as given by whoever cancelled first. None until then."""
        return self._reason

    def cancel(self, reason: str = _DEFAULT_REASON) -> None:
        """Request cancellation. A second call is a no-op and does not rewrite `reason`."""
        if self._event.is_set():
            return
        self._reason = reason
        self._event.set()

    def raise_if_cancelled(self) -> None:
        """Raise if cancellation has been requested, otherwise do nothing.

        For code that checks between steps rather than racing an await.

        Raises:
            CancelledError: The kit's own, not `asyncio.CancelledError` — this is a
                decision the run records, not a task teardown that must propagate.
        """
        if self._event.is_set():
            raise CancelledError(self._reason or _DEFAULT_REASON)

    async def wait(self) -> str:
        """Suspend until cancelled and return the reason, so it can be raced against work."""
        await self._event.wait()
        return self._reason or _DEFAULT_REASON


@dataclass(frozen=True, slots=True)
class Deadline:
    """An instant after which a run must stop, as absolute seconds on the injected clock.

    An instant rather than a duration, because a duration passed down a call chain
    restarts at every hop and a run made of four thirty-second steps quietly becomes a
    two-minute run.

    Args:
        at: The instant, in the clock's own units.

    Example:
        >>> deadline = Deadline.in_seconds(30, now=100.0)
        >>> (deadline.at, deadline.remaining(now=110.0), deadline.expired(now=130.0))
        (130.0, 20.0, True)
    """

    at: float

    @classmethod
    def in_seconds(cls, seconds: float, *, now: float) -> Deadline:
        """Return the deadline `seconds` after `now`.

        Raises:
            ValueError: If `seconds` is not positive. A deadline of zero cancels the run
                before it starts, which is a mistake rather than a policy.
        """
        if seconds <= 0:
            raise ValueError(f"a deadline must be a positive number of seconds, got {seconds}")
        return cls(at=now + seconds)

    def remaining(self, now: float) -> float:
        """Seconds left, floored at zero so an overrun does not read as unlimited time."""
        return max(0.0, self.at - now)

    def expired(self, now: float) -> bool:
        """Whether the instant has arrived."""
        return now >= self.at

    def narrowed_to(self, other: Deadline | None) -> Deadline:
        """Return the earlier of this deadline and `other`.

        A deadline is inherited, never extended: a sub-agent handed a longer deadline than
        its parent would outlive the run that is waiting for it.
        """
        if other is None:
            return self
        return self if self.at <= other.at else other
