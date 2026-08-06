"""Spacing calls out before the vendor does it for us.

A key's allowance belongs to the key, not to the process holding it: twenty runs sharing
one key each get a twentieth of the limit, find that out as 429s, and retry into the same
wall together. One limiter in front of the provider knows the whole allowance and spends
it in order, so the calls are shaped before they are sent.

Two buckets, because every vendor meters both: requests and tokens. A call waits for
whichever of the two is short, and a bucket refills continuously rather than in steps — a
window that resets on the minute is a stampede on the minute.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.runtime.loop import SystemClock

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import Clock

__all__ = ["RateLimiter"]

_A_MINUTE = 60.0


class _Bucket:
    """One continuously-refilling allowance."""

    def __init__(self, per_minute: float, burst: float, now: float) -> None:
        self.rate = per_minute / _A_MINUTE
        self.capacity = per_minute * burst
        self._left = self.capacity
        self._filled_at = now

    def wait_for(self, cost: float, now: float) -> float:
        """Seconds until `cost` is affordable, zero where it already is."""
        self._refill(now)
        return 0.0 if cost <= self._left else (cost - self._left) / self.rate

    def take(self, cost: float, now: float) -> None:
        """Spend `cost`, which the caller has already waited to be able to afford."""
        self._refill(now)
        self._left -= cost

    def _refill(self, now: float) -> None:
        # Capped at the capacity: an idle hour does not bank an hour's worth of burst.
        self._left = min(self.capacity, self._left + (now - self._filled_at) * self.rate)
        self._filled_at = now


class RateLimiter:
    """A shared token bucket over requests and tokens, spending in arrival order.

    Args:
        requests_per_minute: Calls allowed per minute. `None` leaves calls unmetered.
        tokens_per_minute: Tokens allowed per minute. `None` leaves tokens unmetered.
        burst: How much of a minute's allowance may be spent at once, as a fraction. One
            lets a minute's worth go out in a moment, which is what a vendor's own window
            allows; a smaller number spreads a burst that would otherwise arrive as one.
        clock: Source of time, injected so a test asserts a wait rather than taking one.

    Raises:
        ConfigurationError: If a limit is given but is not above zero — a limit of nothing
            is a deadlock rather than a limit.

    One limiter is shared by every caller on a key, and each awaits `acquire` with its
    estimated token cost before sending.

    Example:
        >>> shared = RateLimiter(requests_per_minute=600, tokens_per_minute=90_000)
        >>> RateLimiter(requests_per_minute=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        tesserix_adk.core.errors.ConfigurationError: requests_per_minute must be above...
    """

    def __init__(
        self,
        *,
        requests_per_minute: float | None = None,
        tokens_per_minute: float | None = None,
        burst: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or SystemClock()
        self._lock = asyncio.Lock()
        self._requests = _limit("requests_per_minute", requests_per_minute, burst, self._clock)
        self._tokens = _limit("tokens_per_minute", tokens_per_minute, burst, self._clock)

    async def acquire(self, tokens: int = 0) -> None:
        """Wait until this call fits inside the allowance, then spend it.

        Args:
            tokens: What the call is estimated to cost. Zero where nothing meters tokens.

        Raises:
            ConfigurationError: If `tokens` exceeds the whole token allowance, which no
                amount of waiting would make room for.
            CancelledError: If the caller is cancelled while waiting. Nothing is spent —
                a cancelled call never reaches the vendor, so it never used the allowance.
        """
        if self._requests is None and self._tokens is None:
            return
        self._refuse_the_impossible(tokens)
        async with self._lock:
            while (wait := self._wait_for(tokens)) > 0:
                await self._clock.sleep(wait)
            now = self._clock.now()
            for bucket, cost in self._costs(tokens):
                bucket.take(cost, now)

    def _wait_for(self, tokens: int) -> float:
        now = self._clock.now()
        return max((bucket.wait_for(cost, now) for bucket, cost in self._costs(tokens)), default=0)

    def _costs(self, tokens: int) -> list[tuple[_Bucket, float]]:
        costs = [(self._requests, 1.0), (self._tokens, float(tokens))]
        return [(bucket, cost) for bucket, cost in costs if bucket is not None and cost]

    def _refuse_the_impossible(self, tokens: int) -> None:
        if self._tokens is not None and tokens > self._tokens.capacity:
            raise ConfigurationError(
                f"a request of {tokens} tokens is larger than the whole allowance of "
                f"{self._tokens.capacity:.0f}; waiting would never make room for it"
            )


def _limit(name: str, per_minute: float | None, burst: float, clock: Clock) -> _Bucket | None:
    if per_minute is None:
        return None
    if per_minute <= 0 or burst <= 0:
        raise ConfigurationError(f"{name} must be above zero to be a limit at all")
    return _Bucket(per_minute, burst, clock.now())
