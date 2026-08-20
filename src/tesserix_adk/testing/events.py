"""A publisher a test can read back, and one assertion for the sequence a run emitted."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.events import EventEnvelope, EventType

__all__ = ["InMemoryEventPublisher", "assert_events"]


class InMemoryEventPublisher:
    """Everything published, in the order it was published, with no broker anywhere."""

    __slots__ = ("_events", "batches")

    def __init__(self) -> None:
        self._events: list[EventEnvelope] = []
        self.batches = 0

    @property
    def events(self) -> tuple[EventEnvelope, ...]:
        """What was published, oldest first."""
        return tuple(self._events)

    async def publish(self, event: EventEnvelope) -> None:
        """Keep one event."""
        self._events.append(event)

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        """Keep a batch, counting it as one delivery."""
        self._events.extend(events)
        self.batches += 1

    def of_type(self, kind: EventType) -> tuple[EventEnvelope, ...]:
        """Everything of one type, in order."""
        return tuple(event for event in self._events if event.type is kind)

    def clear(self) -> None:
        """Forget everything, for a test with more than one phase."""
        self._events.clear()
        self.batches = 0


def assert_events(published: Sequence[EventEnvelope], *expected: EventType) -> None:
    """Assert a run emitted exactly `expected`, in order.

    Raises:
        AssertionError: Naming the first position that differs, and both types at it.
    """
    actual = [event.type for event in published]
    for position, (was, wanted) in enumerate(zip(actual, expected, strict=False)):
        if was is not wanted:
            raise AssertionError(
                f"event {position} is {was.value}, expected {wanted.value}; the sequence was "
                f"{[kind.value for kind in actual]}"
            )
    if len(actual) != len(expected):
        raise AssertionError(
            f"expected {[kind.value for kind in expected]}, published "
            f"{[kind.value for kind in actual]}"
        )
