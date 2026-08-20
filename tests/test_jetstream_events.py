"""Publishing the event spine onto JetStream, and consuming it back durably.

What a real broker does with these subjects and headers is `tests/integration/`'s
question. These are the translation: the right subject, the right dedupe header, the
right thing done with an ack that never arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.adapters.jetstream import (
    DEFAULT_EVENT_SUBJECT,
    DurableConsumer,
    JetStreamEventPublisher,
    StreamRequirement,
    subject_for,
)
from tesserix_adk.core import (
    ConfigurationError,
    Delivery,
    Eventing,
    EventPublishError,
    EventTooLargeError,
    EventType,
    RunStarted,
    tenant_scope,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.core import EventEnvelope

TENANT = "acme"
RUN = "run_1"


@dataclass(slots=True)
class _Ack:
    stream: str = "ADK_EVENTS"
    seq: int = 1
    duplicate: bool = False


@dataclass(slots=True)
class _Config:
    subjects: tuple[str, ...] = ("adk.events.>",)
    retention: str = "limits"
    max_age: float = 604_800.0
    max_msg_size: int = -1


@dataclass(slots=True)
class _Info:
    config: _Config = field(default_factory=_Config)


@dataclass(slots=True)
class _Broker:
    """A JetStream that remembers what it was told, and can be told to misbehave."""

    info: _Info | None = field(default_factory=_Info)
    published: list[tuple[str, bytes, Mapping[str, str]]] = field(default_factory=list)
    fail_times: int = 0
    seen: set[str] = field(default_factory=set)

    async def stream_info(self, name: str) -> _Info:
        if self.info is None:
            raise ConnectionError(f"no stream {name}")
        return self.info

    async def publish(
        self,
        subject: str,
        payload: bytes,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — the client's own signature
    ) -> _Ack:
        del timeout
        if self.fail_times:
            self.fail_times -= 1
            raise TimeoutError("no ack")
        identifier = (headers or {}).get("Nats-Msg-Id", "")
        duplicate = identifier in self.seen
        self.seen.add(identifier)
        if not duplicate:
            self.published.append((subject, payload, dict(headers or {})))
        return _Ack(duplicate=duplicate)


async def _publisher(broker: _Broker, **options: Any) -> JetStreamEventPublisher:
    return await JetStreamEventPublisher.open(broker, clock=FakeClock(auto_advance=True), **options)


def _envelope(eventing: Eventing) -> Eventing:
    return eventing


async def _emitted(publisher: JetStreamEventPublisher | None = None) -> EventEnvelope:
    eventing = Eventing(publisher, clock=FakeClock(), delivery=Delivery.GUARANTEED)
    with tenant_scope(TENANT):
        event = await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
    assert event is not None
    return event


class TestWhereAnEventIsPublished:
    async def test_the_subject_carries_the_tenant_and_the_kind(self) -> None:
        broker = _Broker()
        await _emitted(await _publisher(broker))
        subject, _, _ = broker.published[0]
        assert subject == f"{DEFAULT_EVENT_SUBJECT}.{TENANT}.{EventType.RUN_STARTED}"

    async def test_a_tenant_that_is_not_a_subject_token_is_refused(self) -> None:
        broker = _Broker()
        publisher = await _publisher(broker)
        forged = (await _emitted()).model_copy(update={"tenant": "acme.>"})
        with pytest.raises(ConfigurationError, match="subject token"):
            await publisher.publish(forged)

    async def test_no_subject_this_kit_builds_can_reach_another_tenant(self) -> None:
        for wildcard in ("*", ">", "acme.*", "acme>"):
            with pytest.raises(ConfigurationError):
                subject_for(DEFAULT_EVENT_SUBJECT, tenant=wildcard, kind=EventType.RUN_STARTED)

    async def test_the_body_is_the_envelope_a_consumer_parses_back(self) -> None:
        broker = _Broker()
        event = await _emitted(await _publisher(broker))
        _, payload, _ = broker.published[0]
        assert payload.decode() == event.to_json()


class TestAnAckThatNeverArrived:
    async def test_the_retry_is_deduplicated_by_the_event_id(self) -> None:
        broker = _Broker()
        publisher = await _publisher(broker)
        event = await _emitted(publisher)
        await publisher.publish(event)
        assert len(broker.published) == 1

    async def test_the_dedupe_header_is_the_event_id(self) -> None:
        broker = _Broker()
        event = await _emitted(await _publisher(broker))
        _, _, headers = broker.published[0]
        assert headers["Nats-Msg-Id"] == event.event_id

    async def test_an_ambiguous_ack_is_retried_and_counted(self) -> None:
        broker = _Broker(fail_times=1)
        publisher = await _publisher(broker)
        await _emitted(publisher)
        assert (publisher.published, publisher.ambiguous) == (1, 1)

    async def test_a_duplicate_the_broker_recognised_is_counted_as_one(self) -> None:
        broker = _Broker()
        publisher = await _publisher(broker)
        event = await _emitted(publisher)
        await publisher.publish(event)
        assert publisher.duplicates == 1

    async def test_a_batch_is_published_one_acknowledged_message_at_a_time(self) -> None:
        broker = _Broker()
        publisher = await _publisher(broker)
        first, second = await _emitted(), await _emitted()
        await publisher.publish_batch((first, second))
        assert len(broker.published) == 2

    async def test_it_gives_up_after_the_configured_attempts(self) -> None:
        broker = _Broker(fail_times=99)
        publisher = await _publisher(broker, attempts=3, delivery=Delivery.GUARANTEED)
        with pytest.raises(EventPublishError):
            await _emitted(publisher)
        assert publisher.attempted == 3


class TestTheStreamThisAdapterNeeds:
    async def test_a_missing_stream_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="ADK_EVENTS"):
            await _publisher(_Broker(info=None))

    async def test_a_retention_that_does_not_match_names_the_mismatch(self) -> None:
        broker = _Broker(info=_Info(_Config(retention="workqueue")))
        with pytest.raises(ConfigurationError, match="workqueue"):
            await _publisher(broker)

    async def test_a_stream_that_forgets_sooner_than_documented_is_refused(self) -> None:
        broker = _Broker(info=_Info(_Config(max_age=60.0)))
        with pytest.raises(ConfigurationError, match="retains"):
            await _publisher(broker, requirement=StreamRequirement(min_age_seconds=86_400.0))

    async def test_a_stream_that_does_not_carry_the_subject_is_refused(self) -> None:
        broker = _Broker(info=_Info(_Config(subjects=("other.>",))))
        with pytest.raises(ConfigurationError, match=r"adk\.events"):
            await _publisher(broker)

    async def test_a_stream_whose_messages_are_smaller_than_our_ceiling_is_refused(self) -> None:
        broker = _Broker(info=_Info(_Config(max_msg_size=64)))
        with pytest.raises(ConfigurationError, match="64"):
            await _publisher(broker)

    async def test_a_broker_that_is_not_there_fails_closed(self) -> None:
        with pytest.raises(ConfigurationError):
            await _publisher(_Broker(info=None), delivery=Delivery.GUARANTEED)


class TestWhenTheBrokerCannotKeepUp:
    async def test_best_effort_buffers_what_it_could_not_send(self) -> None:
        broker = _Broker(fail_times=99)
        publisher = await _publisher(broker, attempts=1, max_pending=4)
        await _emitted(publisher)
        assert publisher.pending == 1

    async def test_the_buffer_does_not_grow_without_limit(self) -> None:
        broker = _Broker(fail_times=99)
        publisher = await _publisher(broker, attempts=1, max_pending=2)
        for _ in range(5):
            await _emitted(publisher)
        assert (publisher.pending, publisher.dropped) == (2, 3)

    async def test_what_was_buffered_goes_out_once_the_broker_returns(self) -> None:
        broker = _Broker(fail_times=1)
        publisher = await _publisher(broker, attempts=1, max_pending=4)
        await _emitted(publisher)
        await publisher.flush()
        assert (publisher.pending, len(broker.published)) == (0, 1)

    async def test_guaranteed_delivery_never_buffers_silently(self) -> None:
        broker = _Broker(fail_times=99)
        publisher = await _publisher(broker, attempts=1, delivery=Delivery.GUARANTEED)
        with pytest.raises(EventPublishError):
            await _emitted(publisher)
        assert publisher.pending == 0

    async def test_an_event_over_the_stream_s_ceiling_is_not_truncated(self) -> None:
        broker = _Broker()
        publisher = await _publisher(broker, requirement=StreamRequirement(max_payload_bytes=8))
        with pytest.raises(EventTooLargeError):
            await _emitted(publisher)
        assert broker.published == []


@dataclass(slots=True)
class _Metadata:
    num_delivered: int = 1


@dataclass(slots=True)
class _Message:
    data: bytes
    metadata: _Metadata = field(default_factory=_Metadata)
    acked: bool = False
    naked: bool = False
    terminated: bool = False

    async def ack(self) -> None:
        self.acked = True

    async def nak(self) -> None:
        self.naked = True

    async def term(self) -> None:
        self.terminated = True


@dataclass(slots=True)
class _Subscription:
    messages: list[_Message] = field(default_factory=list)

    async def fetch(
        self,
        batch: int,
        timeout: float | None = None,  # noqa: ASYNC109 — the client's own signature
    ) -> Sequence[_Message]:
        del timeout
        taken, self.messages = self.messages[:batch], self.messages[batch:]
        return taken


@dataclass(slots=True)
class _DeadLetter:
    letters: list[tuple[bytes, str]] = field(default_factory=list)

    async def bury(self, payload: bytes, *, reason: str) -> None:
        self.letters.append((payload, reason))


def _ignored(event: EventEnvelope) -> None:
    del event


async def _message(delivered: int = 1) -> _Message:
    event = await _emitted()
    return _Message(data=event.to_json().encode(), metadata=_Metadata(delivered))


class TestConsumingDurably:
    async def test_a_handled_event_is_acknowledged(self) -> None:
        message = await _message()
        handled: list[EventEnvelope] = []
        consumer = DurableConsumer(_Subscription([message]), handler=handled.append)
        await consumer.consume()
        assert (message.acked, len(handled)) == (True, 1)

    async def test_the_handler_receives_the_envelope_not_the_bytes(self) -> None:
        message = await _message()
        seen: list[EventEnvelope] = []
        await DurableConsumer(_Subscription([message]), handler=seen.append).consume()
        assert (seen[0].tenant, seen[0].type) == (TENANT, EventType.RUN_STARTED)

    async def test_a_failed_handler_leaves_it_for_redelivery(self) -> None:
        message = await _message()

        def explode(event: EventEnvelope) -> None:
            del event
            raise RuntimeError("downstream is down")

        consumer = DurableConsumer(_Subscription([message]), handler=explode)
        await consumer.consume()
        assert (message.naked, message.acked) == (True, False)

    async def test_the_last_delivery_is_buried_rather_than_redelivered_forever(self) -> None:
        message = await _message(delivered=5)
        letters = _DeadLetter()

        def explode(event: EventEnvelope) -> None:
            del event
            raise RuntimeError("downstream is down")

        consumer = DurableConsumer(
            _Subscription([message]), handler=explode, max_deliver=5, dead_letter=letters
        )
        await consumer.consume()
        assert message.terminated
        assert letters.letters[0][1] == "handler_failed"

    async def test_a_payload_that_is_not_an_envelope_is_buried_at_once(self) -> None:
        message = _Message(data=b"{not json")
        letters = _DeadLetter()
        consumer = DurableConsumer(_Subscription([message]), handler=_ignored, dead_letter=letters)
        await consumer.consume()
        assert (message.terminated, letters.letters[0][1]) == (True, "undecodable")

    async def test_an_undecodable_payload_without_a_dead_letter_is_still_not_replayed(
        self,
    ) -> None:
        message = _Message(data=b"{not json")
        await DurableConsumer(_Subscription([message]), handler=_ignored).consume()
        assert message.terminated

    async def test_an_async_handler_is_awaited_before_the_ack(self) -> None:
        message = await _message()
        seen: list[EventEnvelope] = []

        async def record(event: EventEnvelope) -> None:
            seen.append(event)

        await DurableConsumer(_Subscription([message]), handler=record).consume()
        assert (len(seen), message.acked) == (1, True)

    async def test_it_counts_what_it_did(self) -> None:
        messages = [await _message(), await _message()]
        consumer = DurableConsumer(_Subscription(messages), handler=_ignored)
        await consumer.consume()
        assert (consumer.handled, consumer.buried) == (2, 0)

    async def test_an_empty_fetch_is_not_an_error(self) -> None:
        consumer = DurableConsumer(_Subscription([]), handler=_ignored)
        await consumer.consume()
        assert consumer.handled == 0
