"""In-memory implementations of the core protocols, for tests that must not touch a network.

These are the reference implementations the conformance suite is written against.
They are deliberately simple: a fake that acquires its own behaviour stops being a
control and becomes a second thing to debug.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.errors import AdkError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "BudgetExceededError",
    "FakeBudgetPolicy",
    "FakeClock",
    "FakeMemoryStore",
    "FakeTracer",
    "RecordedEvent",
]


class BudgetExceededError(AdkError):
    """Raised by `FakeBudgetPolicy` when a reservation would breach its limit."""


class FakeClock:
    """A clock that only moves when a test moves it.

    `sleep` advances the clock instead of suspending, so a test for a timeout runs
    in microseconds and never depends on wall-clock scheduling.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self.slept: list[float] = []

    def now(self) -> float:
        """Return the current fake time."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Advance the clock by `seconds` without suspending."""
        self.slept.append(seconds)
        self._now += seconds

    def advance(self, seconds: float) -> None:
        """Move the clock forward without recording a sleep."""
        self._now += seconds


class FakeMemoryStore:
    """A dictionary-backed `MemoryStore`."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        """Return the stored value, or None."""
        return self._items.get(key)

    async def put(self, key: str, value: Any) -> None:
        """Store `value` under `key`."""
        self._items[key] = value

    async def delete(self, key: str) -> None:
        """Remove `key` if present."""
        self._items.pop(key, None)


@dataclass(frozen=True, slots=True)
class RecordedEvent:
    """A span or event captured by `FakeTracer`."""

    kind: str
    name: str
    attributes: dict[str, object]


class FakeTracer:
    """A tracer that records instead of exporting, and never raises."""

    def __init__(self) -> None:
        self.recorded: list[RecordedEvent] = []

    @contextlib.contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[None]:
        """Record a span around the wrapped block."""
        self.recorded.append(RecordedEvent("span", name, attributes))
        yield

    def event(self, name: str, **attributes: object) -> None:
        """Record a point-in-time event."""
        self.recorded.append(RecordedEvent("event", name, attributes))

    def names(self) -> list[str]:
        """Return the recorded names in order, for concise assertions."""
        return [r.name for r in self.recorded]


class FakeBudgetPolicy:
    """A counting budget with a hard limit.

    Args:
        limit: Total units permitted across the lifetime of the policy.
    """

    def __init__(self, limit: int = 1_000_000) -> None:
        self.limit = limit
        self.reserved = 0
        self.spent = 0

    async def reserve(self, estimate: int) -> None:
        """Reserve `estimate` units.

        Raises:
            BudgetExceededError: If the reservation would breach `limit`.
        """
        if self.spent + self.reserved + estimate > self.limit:
            raise BudgetExceededError(
                f"reserving {estimate} would exceed limit {self.limit} "
                f"(spent {self.spent}, reserved {self.reserved})"
            )
        self.reserved += estimate

    async def record(self, actual: int) -> None:
        """Record `actual` consumption and release the outstanding reservation."""
        self.spent += actual
        self.reserved = 0
