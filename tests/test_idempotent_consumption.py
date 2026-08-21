"""At-least-once delivery, handled once.

The same event arrives again after an ack timeout, a reaper requeue or a rolling deploy.
These are the assertions that a redelivery cannot produce a second side effect, and that a
message nothing can handle stops being redelivered.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.adapters.idempotent_events import (
    DEFAULT_DEDUPE_TTL_SECONDS,
    IdempotentConsumer,
    dedupe_key,
)
from tesserix_adk.core import (
    ConfigurationError,
    Delivery,
    DuplicateInFlightError,
    EventIdReuseError,
    Eventing,
    ToolCallCompleted,
    tenant_scope,
)
from tesserix_adk.runtime import MemoryIdempotencyStore
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Callable

    from tesserix_adk.core import EventEnvelope

TENANT = "acme"
GROUP = "billing"


async def _event(run_id: str = "run_1", *, tenant: str = TENANT) -> EventEnvelope:
    eventing = Eventing(clock=FakeClock(), delivery=Delivery.GUARANTEED)
    with tenant_scope(tenant):
        event = await eventing.emit(
            ToolCallCompleted(run_id=run_id, tool="charge", tool_call_id="c1", state="ok")
        )
    assert event is not None
    return event


@dataclass(slots=True)
class _Charges:
    """A side effect somebody would notice happening twice."""

    charged: list[str] = field(default_factory=list)

    async def charge(self, event: EventEnvelope) -> None:
        await asyncio.sleep(0)
        self.charged.append(event.event_id)


@dataclass(slots=True)
class _DeadLetter:
    letters: list[tuple[bytes, str, tuple[str, ...]]] = field(default_factory=list)

    async def bury(self, payload: bytes, *, reason: str, history: tuple[str, ...] = ()) -> None:
        self.letters.append((payload, reason, history))


@dataclass(slots=True)
class _Unreachable:
    """A dedupe store that is down, which must stop the handler running."""

    async def begin(self, key: str, *, tenant: str, ttl_seconds: float) -> Any:  # noqa: ARG002 — the protocol's own signature
        raise ConnectionError("dedupe store is down")

    async def record(self, key: str, *, tenant: str, outcome: str, ttl_seconds: float) -> None:  # noqa: ARG002 — the protocol's own signature
        raise ConnectionError("dedupe store is down")

    async def abandon(self, key: str, *, tenant: str) -> None:  # noqa: ARG002 — the protocol's own signature
        raise ConnectionError("dedupe store is down")

    async def forget(self, *, tenant: str) -> int:  # noqa: ARG002 — the protocol's own signature
        raise ConnectionError("dedupe store is down")


def _ignored(event: EventEnvelope) -> None:
    del event


def _noted(seen: list[str], group: str) -> Callable[[EventEnvelope], None]:
    def note(event: EventEnvelope) -> None:
        del event
        seen.append(group)

    return note


def _consumer(
    handler: Callable[[EventEnvelope], Any], **options: Any
) -> tuple[IdempotentConsumer, MemoryIdempotencyStore]:
    store = options.pop("store", None) or MemoryIdempotencyStore(FakeClock())
    consumer = IdempotentConsumer(handler, store=store, group=GROUP, **options)
    return consumer, store


class TestTheSameEventTwice:
    async def test_the_effect_happens_once(self) -> None:
        charges = _Charges()
        consumer, _ = _consumer(charges.charge)
        event = await _event()
        for _ in range(3):
            await consumer.handle(event)
        assert charges.charged == [event.event_id]

    async def test_the_later_deliveries_are_counted_as_suppressed(self) -> None:
        charges = _Charges()
        consumer, _ = _consumer(charges.charge)
        event = await _event()
        for _ in range(3):
            await consumer.handle(event)
        assert (consumer.handled, consumer.suppressed) == (1, 2)

    async def test_two_workers_racing_the_same_redelivery_run_it_once(self) -> None:
        charges = _Charges()
        store = MemoryIdempotencyStore(FakeClock())
        first, _ = _consumer(charges.charge, store=store)
        second, _ = _consumer(charges.charge, store=store)
        event = await _event()
        outcomes = await asyncio.gather(
            first.handle(event), second.handle(event), return_exceptions=True
        )
        assert charges.charged == [event.event_id]
        assert any(isinstance(outcome, DuplicateInFlightError) for outcome in outcomes)

    async def test_a_redelivery_after_a_worker_died_mid_ack_is_still_suppressed(self) -> None:
        charges = _Charges()
        consumer, store = _consumer(charges.charge)
        event = await _event()
        await consumer.handle(event)
        again, _ = _consumer(charges.charge, store=store)
        await again.handle(event)
        assert (charges.charged, again.suppressed) == ([event.event_id], 1)

    async def test_a_different_event_is_not_suppressed(self) -> None:
        charges = _Charges()
        consumer, _ = _consumer(charges.charge)
        await consumer.handle(await _event("run_1"))
        await consumer.handle(await _event("run_2"))
        assert len(charges.charged) == 2


class TestWhatTheKeyIsScopedTo:
    async def test_two_groups_each_get_to_handle_the_event(self) -> None:
        store = MemoryIdempotencyStore(FakeClock())
        seen: list[str] = []
        event = await _event()
        for group in ("billing", "notifications"):
            consumer = IdempotentConsumer(_noted(seen, group), store=store, group=group)
            await consumer.handle(event)
        assert len(seen) == 2

    async def test_an_id_clash_across_tenants_cannot_collide(self) -> None:
        charges = _Charges()
        consumer, _ = _consumer(charges.charge)
        first = await _event(tenant="acme")
        second = first.model_copy(update={"tenant": "globex"})
        await consumer.handle(first)
        await consumer.handle(second)
        assert consumer.handled == 2

    async def test_the_key_names_the_group_and_the_event(self) -> None:
        event = await _event()
        assert dedupe_key(group=GROUP, event=event) == f"{GROUP}:{event.event_id}"


class TestAPublisherThatReusedAnId:
    async def test_a_second_unrelated_event_under_the_same_id_is_rejected(self) -> None:
        charges = _Charges()
        consumer, _ = _consumer(charges.charge)
        first = await _event("run_1")
        forged = first.model_copy(update={"run_id": "run_9"})
        await consumer.handle(first)
        with pytest.raises(EventIdReuseError):
            await consumer.handle(forged)

    async def test_the_rejection_names_the_event_and_not_its_body(self) -> None:
        charges = _Charges()
        consumer, _ = _consumer(charges.charge)
        first = await _event("run_1")
        await consumer.handle(first)
        with pytest.raises(EventIdReuseError) as refused:
            await consumer.handle(first.model_copy(update={"run_id": "run_9"}))
        assert first.event_id in str(refused.value)
        assert "run_9" not in str(refused.value)


class TestWhenTheDedupeStoreIsUnreachable:
    async def test_the_handler_does_not_run_at_all(self) -> None:
        charges = _Charges()
        consumer, _ = _consumer(charges.charge, store=_Unreachable())
        with pytest.raises(ConnectionError):
            await consumer.handle(await _event())
        assert charges.charged == []


class TestRetentionAgainstTheRedeliveryHorizon:
    async def test_a_window_shorter_than_the_horizon_is_a_misconfiguration(self) -> None:
        with pytest.raises(ConfigurationError, match="redelivery"):
            IdempotentConsumer(
                _ignored,
                store=MemoryIdempotencyStore(FakeClock()),
                group=GROUP,
                ttl_seconds=60.0,
                redelivery_horizon_seconds=3_600.0,
            )

    async def test_the_default_window_covers_a_day(self) -> None:
        assert DEFAULT_DEDUPE_TTL_SECONDS == 86_400.0


class TestAHandlerThatKeepsFailing:
    async def test_a_failure_leaves_the_event_for_redelivery(self) -> None:
        def explode(event: EventEnvelope) -> None:
            del event
            raise RuntimeError("downstream is down")

        consumer, _ = _consumer(explode)
        with pytest.raises(RuntimeError):
            await consumer.handle(await _event())

    async def test_a_failed_event_is_not_marked_as_processed(self) -> None:
        charges = _Charges()
        attempts: list[int] = []

        async def flaky(event: EventEnvelope) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("downstream is down")
            await charges.charge(event)

        consumer, _ = _consumer(flaky)
        event = await _event()
        with pytest.raises(RuntimeError):
            await consumer.handle(event)
        await consumer.handle(event)
        assert charges.charged == [event.event_id]

    async def test_the_last_attempt_is_buried_with_what_went_wrong(self) -> None:
        letters = _DeadLetter()

        def explode(event: EventEnvelope) -> None:
            del event
            raise RuntimeError("downstream is down")

        consumer, _ = _consumer(explode, max_attempts=3, dead_letter=letters)
        event = await _event()
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await consumer.handle(event)
        await consumer.handle(event)
        payload, reason, history = letters.letters[0]
        assert (reason, len(history)) == ("poison", 3)
        assert payload == event.to_json().encode()

    async def test_a_buried_event_is_not_redelivered_into_the_handler_again(self) -> None:
        letters = _DeadLetter()
        attempts: list[int] = []

        def explode(event: EventEnvelope) -> None:
            del event
            attempts.append(1)
            raise RuntimeError("downstream is down")

        consumer, _ = _consumer(explode, max_attempts=2, dead_letter=letters)
        event = await _event()
        with pytest.raises(RuntimeError):
            await consumer.handle(event)
        await consumer.handle(event)
        await consumer.handle(event)
        assert (len(attempts), consumer.buried) == (2, 1)

    async def test_the_history_carries_the_error_type_and_not_the_message(self) -> None:
        letters = _DeadLetter()

        def explode(event: EventEnvelope) -> None:
            del event
            raise RuntimeError("customer ada@example.gov could not be charged")

        consumer, _ = _consumer(explode, max_attempts=1, dead_letter=letters)
        await consumer.handle(await _event())
        _, _, history = letters.letters[0]
        assert "RuntimeError" in history[0]
        assert "example.gov" not in " ".join(history)

    async def test_without_a_dead_letter_the_poison_event_still_stops_being_retried(self) -> None:
        attempts: list[int] = []

        def explode(event: EventEnvelope) -> None:
            del event
            attempts.append(1)
            raise RuntimeError("downstream is down")

        consumer, _ = _consumer(explode, max_attempts=1)
        event = await _event()
        await consumer.handle(event)
        await consumer.handle(event)
        assert len(attempts) == 1


class TestTheHandlerAndItsStateChangeTogether:
    async def test_the_marker_is_written_inside_the_handler_s_transaction(self) -> None:
        order: list[str] = []

        class _Transaction:
            async def __aenter__(self) -> None:
                order.append("begin")

            async def __aexit__(self, *exception: object) -> None:
                order.append("commit")

        async def charge(event: EventEnvelope) -> None:
            del event
            order.append("handler")

        consumer, _ = _consumer(charge, transaction=_Transaction)
        await consumer.handle(await _event())
        assert order == ["begin", "handler", "commit"]

    async def test_a_transaction_that_rolls_back_leaves_nothing_processed(self) -> None:
        commits: list[int] = []

        class _Transaction:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *exception: object) -> None:
                commits.append(1)
                if len(commits) == 1:
                    raise RuntimeError("the transaction could not commit")

        charges = _Charges()
        consumer, _ = _consumer(charges.charge, transaction=_Transaction)
        event = await _event()
        with pytest.raises(RuntimeError):
            await consumer.handle(event)
        await consumer.handle(event)
        assert len(charges.charged) == 2
