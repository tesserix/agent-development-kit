"""An in-process record of what has already run, for a single replica and for tests.

Nothing here outlives the process, which is exactly what a restart-survival guarantee is
not. A deployment that runs more than one worker wants a shared store; this one exists so
the guarantee can be exercised without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tesserix_adk.core.idempotency import Claim

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import Clock

__all__ = ["MemoryIdempotencyStore"]


@dataclass(frozen=True, slots=True)
class _Held:
    outcome: str | None
    expires_at: float


class MemoryIdempotencyStore:
    """An `IdempotencyStore` in a dict, keyed by tenant and key.

    A record that has passed its retention window is dropped on the way past rather than
    honoured: an expired record is not a record, and treating it as one is how a very
    late retry gets a success nobody stored.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._held: dict[tuple[str, str], _Held] = {}
        self._clock = clock

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Claim:
        """Claim `key`, or report what already holds it."""
        held = self._current(tenant, key)
        if held is not None:
            return Claim(outcome=held.outcome, in_flight=held.outcome is None)
        self._held[tenant, key] = _Held(outcome=None, expires_at=self._now() + ttl_seconds)
        return Claim()

    async def record(self, key: str, *, tenant: str, outcome: str, ttl_seconds: float) -> None:
        """Record what the call returned."""
        self._held[tenant, key] = _Held(outcome=outcome, expires_at=self._now() + ttl_seconds)

    async def abandon(self, key: str, *, tenant: str) -> None:
        """Release `key` without a record."""
        self._held.pop((tenant, key), None)

    async def forget(self, *, tenant: str) -> int:
        """Remove every record for `tenant`, and return how many."""
        doomed = [held for held in self._held if held[0] == tenant]
        for held in doomed:
            del self._held[held]
        return len(doomed)

    def _current(self, tenant: str, key: str) -> _Held | None:
        held = self._held.get((tenant, key))
        if held is None:
            return None
        if held.expires_at <= self._now():
            del self._held[tenant, key]
            return None
        return held

    def _now(self) -> float:
        return 0.0 if self._clock is None else self._clock.now()
