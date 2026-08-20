"""The event spine on the message bus the org already runs.

Publication and consumption are one story here on purpose. A publisher that retries
without a dedupe header and a consumer that acks before it has finished both look correct
in isolation, and together they lose or double every event the broker ever hiccups on.

The stream itself is not this adapter's to create — the platform charts own that. What
this adapter owns is refusing to start against a stream that will not keep what it is
about to be told, which is the failure that otherwise looks healthy for a week.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tesserix_adk.core.errors import ConfigurationError, EventPublishError, EventTooLargeError
from tesserix_adk.core.events import (
    DEFAULT_MAX_EVENT_BYTES,
    Delivery,
    EventEnvelope,
    EventType,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "DEFAULT_EVENT_SUBJECT",
    "DEFAULT_STREAM_REQUIREMENT",
    "DeadLetter",
    "DurableConsumer",
    "JetStreamContext",
    "JetStreamEventPublisher",
    "JetStreamMessage",
    "PublishAck",
    "StreamRequirement",
    "subject_for",
]

DEFAULT_EVENT_SUBJECT = "adk.events"
"""The subject root. The tenant and the kind are appended to it."""

DEDUPE_HEADER = "Nats-Msg-Id"

_TOKEN = re.compile(r"[A-Za-z0-9_-]+")


@runtime_checkable
class PublishAck(Protocol):
    """What the stream says it did with a message."""

    stream: str
    seq: int
    duplicate: bool


@runtime_checkable
class JetStreamContext(Protocol):
    """The two methods this adapter uses from a JetStream client."""

    async def publish(
        self,
        subject: str,
        payload: bytes,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — the client's own signature
    ) -> PublishAck:
        """Publish `payload`, and wait for the stream to acknowledge it."""
        ...

    async def stream_info(self, name: str) -> Any:  # noqa: ANN401 — the client's own info object
        """Describe the named stream, or raise where there is none."""
        ...


@runtime_checkable
class JetStreamMessage(Protocol):
    """One delivery, and the three things a consumer may do about it."""

    data: bytes

    async def ack(self) -> None:
        """Confirm it, so it is not delivered again."""
        ...

    async def nak(self) -> None:
        """Decline it, so it is delivered again."""
        ...

    async def term(self) -> None:
        """Refuse it for good, so it is never delivered again."""
        ...


@runtime_checkable
class PullSubscription(Protocol):
    """A durable pull consumer, which is the only kind whose ack discipline is ours."""

    async def fetch(
        self,
        batch: int,
        timeout: float | None = None,  # noqa: ASYNC109 — the client's own signature
    ) -> Sequence[Any]:
        """Take up to `batch` messages."""
        ...


@runtime_checkable
class DeadLetter(Protocol):
    """Where a message that will never be handled goes, instead of round again."""

    async def bury(self, payload: bytes, *, reason: str) -> None:
        """Keep `payload` where somebody can look at it."""
        ...


@dataclass(frozen=True, slots=True)
class StreamRequirement:
    """What the stream must already be, for publishing onto it to mean anything.

    Args:
        name: The stream the platform provisioned.
        retention: The policy this adapter was written against. `interest` or
            `workqueue` would let a consumer's own state decide what other consumers
            can still read.
        min_age_seconds: The shortest retention a consumer can be offline for and still
            catch up. Zero on the stream means unlimited, which satisfies any minimum.
        max_payload_bytes: The largest event this adapter will hand to the broker.
    """

    name: str = "ADK_EVENTS"
    retention: str = "limits"
    min_age_seconds: float = 86_400.0
    max_payload_bytes: int = DEFAULT_MAX_EVENT_BYTES


DEFAULT_STREAM_REQUIREMENT = StreamRequirement()


def subject_for(root: str, *, tenant: str, kind: EventType) -> str:
    """The subject one event is published on.

    The tenant is a token in the subject so a consumer can be authorised for its own and
    no other. A wildcard smuggled in as a tenant would be authorised for everybody's.

    Raises:
        ConfigurationError: If the root, the tenant or the kind is not a plain token.
    """
    parts = [_token(part) for part in root.split(".")]
    return ".".join([*parts, _token(tenant), _token(str(kind))])


def _token(value: str) -> str:
    """A subject token, or a refusal — a wildcard here widens who hears the event."""
    if not _TOKEN.fullmatch(value):
        raise ConfigurationError(f"{value!r} is not a plain subject token")
    return value


class JetStreamEventPublisher:
    """Publishes envelopes onto JetStream, deduplicated by the event id.

    Construct it with `open`, which checks the stream before the first event rather than
    discovering the mismatch when somebody asks for a week that was never kept.

    Args:
        context: A connected JetStream client.
        clock: What the retry backoff is measured against.
        requirement: What the stream must already be.
        subject_root: The subject root the tenant and kind are appended to.
        delivery: Whether a publish that could not be acknowledged stops the caller.
        attempts: How many times an unacknowledged publish is retried.
        ack_timeout: How long one publish waits for its ack.
        max_pending: How many unsent events best-effort delivery holds before dropping
            the oldest. A buffer without a ceiling is a memory leak with a schedule.
    """

    __slots__ = (
        "_attempts",
        "_clock",
        "_context",
        "_delivery",
        "_pending",
        "_requirement",
        "_root",
        "_timeout",
        "ambiguous",
        "attempted",
        "dropped",
        "duplicates",
        "published",
    )

    def __init__(
        self,
        context: JetStreamContext,
        *,
        clock: Clock,
        requirement: StreamRequirement = DEFAULT_STREAM_REQUIREMENT,
        subject_root: str = DEFAULT_EVENT_SUBJECT,
        delivery: Delivery = Delivery.BEST_EFFORT,
        attempts: int = 3,
        ack_timeout: float = 2.0,
        max_pending: int = 256,
    ) -> None:
        self._context = context
        self._clock = clock
        self._requirement = requirement
        self._root = ".".join(_token(part) for part in subject_root.split("."))
        self._delivery = delivery
        self._attempts = max(1, attempts)
        self._timeout = ack_timeout
        self._pending: deque[EventEnvelope] = deque(maxlen=max_pending)
        self.published = 0
        self.duplicates = 0
        self.ambiguous = 0
        self.attempted = 0
        self.dropped = 0

    @classmethod
    async def open(cls, context: JetStreamContext, **kwargs: Any) -> JetStreamEventPublisher:  # noqa: ANN401 — the constructor's own keywords
        """Construct the publisher and refuse a stream that is not what it needs.

        Raises:
            ConfigurationError: If the stream is missing, unreachable, or configured to
                keep less than this deployment documented.
        """
        publisher = cls(context, **kwargs)
        await publisher.verify()
        return publisher

    @property
    def pending(self) -> int:
        """How many events are buffered, waiting for a broker that would not take them."""
        return len(self._pending)

    async def verify(self) -> None:
        """Check the stream exists and keeps what this adapter is about to publish."""
        wanted = self._requirement
        try:
            info = await self._context.stream_info(wanted.name)
        except Exception as failure:
            raise ConfigurationError(
                f"stream {wanted.name} could not be described, so publishing onto it would "
                f"be publishing into a void that reports success"
            ) from failure
        config = getattr(info, "config", info)
        self._matching_subjects(config)
        self._matching_retention(config)

    def _matching_subjects(self, config: Any) -> None:  # noqa: ANN401 — the client's own config object
        """Refuse a stream that does not carry the subjects we are about to publish on."""
        subjects = tuple(getattr(config, "subjects", ()) or ())
        if not any(_covers(subject, self._root) for subject in subjects):
            raise ConfigurationError(
                f"stream {self._requirement.name} carries {subjects} and this publisher "
                f"publishes on {self._root}.*, which nothing on it would store"
            )

    def _matching_retention(self, config: Any) -> None:  # noqa: ANN401 — the client's own config object
        """Refuse a retention, age or size that would quietly lose an event."""
        wanted = self._requirement
        retention = str(getattr(config, "retention", wanted.retention))
        if retention.rsplit(".", 1)[-1].lower() != wanted.retention:
            raise ConfigurationError(
                f"stream {wanted.name} retains by {retention} and this adapter needs "
                f"{wanted.retention}; one consumer's acks would decide what another can read"
            )
        age = float(getattr(config, "max_age", 0.0) or 0.0)
        if 0.0 < age < wanted.min_age_seconds:
            raise ConfigurationError(
                f"stream {wanted.name} retains events for {age}s and this deployment "
                f"documented {wanted.min_age_seconds}s"
            )
        size = int(getattr(config, "max_msg_size", -1) or -1)
        if 0 < size < wanted.max_payload_bytes:
            raise ConfigurationError(
                f"stream {wanted.name} takes messages of {size} bytes and this adapter "
                f"publishes up to {wanted.max_payload_bytes}"
            )

    async def publish(self, event: EventEnvelope) -> None:
        """Publish one event, retrying an unacknowledged send under its own id.

        Raises:
            ConfigurationError: If the tenant could not be used as a subject token.
            EventTooLargeError: If the event is over the stream's ceiling.
            EventPublishError: If the stream never acknowledged it, under guaranteed
                delivery. Under best effort it is buffered instead, and the oldest
                buffered event is dropped once the buffer is full.
        """
        await self._drain()
        await self._delivered(event)

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        """Publish a batch. JetStream acknowledges one message at a time, so this does."""
        for event in events:
            await self.publish(event)

    async def flush(self) -> None:
        """Try the buffered events again, oldest first."""
        await self._drain()

    async def _drain(self) -> None:
        """Send what the broker would not take last time, while it will take it now."""
        while self._pending:
            event = self._pending[0]
            try:
                await self._acknowledged(event)
            except EventPublishError:
                return
            self._pending.popleft()

    async def _delivered(self, event: EventEnvelope) -> None:
        """One event, published or absorbed the way the delivery mode says."""
        payload = event.to_json().encode()
        self._within_the_ceiling(event, len(payload))
        try:
            await self._acknowledged(event, payload)
        except EventPublishError:
            if self._delivery is Delivery.GUARANTEED:
                raise
            if len(self._pending) == self._pending.maxlen:
                self.dropped += 1
            self._pending.append(event)

    async def _acknowledged(self, event: EventEnvelope, payload: bytes | None = None) -> None:
        """Publish until the stream acknowledges, or until the attempts run out."""
        subject = subject_for(self._root, tenant=event.tenant, kind=event.type)
        body = payload if payload is not None else event.to_json().encode()
        for attempt in range(1, self._attempts + 1):
            self.attempted += 1
            try:
                ack = await self._context.publish(
                    subject,
                    body,
                    headers={DEDUPE_HEADER: event.event_id},
                    timeout=self._timeout,
                )
            except Exception as failure:
                self.ambiguous += 1
                if attempt == self._attempts:
                    raise EventPublishError(
                        f"{event.type.value} was not acknowledged after {attempt} attempts",
                        event_type=event.type,
                        run_id=event.run_id,
                        tenant=event.tenant,
                    ) from failure
                await self._clock.sleep(self._timeout * attempt)
                continue
            if getattr(ack, "duplicate", False):
                self.duplicates += 1
            else:
                self.published += 1
            return

    def _within_the_ceiling(self, event: EventEnvelope, size: int) -> None:
        """Refuse an event the stream would reject, rather than shortening it into one."""
        limit = self._requirement.max_payload_bytes
        if size > limit:
            raise EventTooLargeError(
                f"{event.type.value} is {size} bytes and stream "
                f"{self._requirement.name} takes {limit}",
                event_type=event.type,
                size_bytes=size,
                limit_bytes=limit,
                run_id=event.run_id,
                tenant=event.tenant,
            )


def _covers(subject: str, root: str) -> bool:
    """Whether a configured stream subject would store what we publish under `root`."""
    if subject in {">", f"{root}.>"}:
        return True
    return subject.startswith(f"{root}.")


class DurableConsumer:
    """Reads events back with explicit acks, and buries what will never be handled.

    Acking after the handler returns is the whole point: a handler that raises leaves the
    message for redelivery, and the delivery the consumer's `max_deliver` calls the last
    one is buried rather than replayed for ever.

    Args:
        subscription: A durable pull consumer.
        handler: What to do with each envelope. Anything it raises is a redelivery.
        max_deliver: The consumer's own max-deliver, so the last attempt is recognised
            as the last one.
        dead_letter: Where an unhandleable message goes. Without one it is still
            terminated, because replaying it for ever helps nobody.
        batch: How many messages one `consume` takes.
        fetch_timeout: How long a fetch waits for them.
    """

    __slots__ = (
        "_batch",
        "_dead_letter",
        "_handler",
        "_max_deliver",
        "_subscription",
        "_timeout",
        "buried",
        "handled",
        "redelivered",
    )

    def __init__(
        self,
        subscription: PullSubscription,
        *,
        handler: Callable[[EventEnvelope], Any],
        max_deliver: int = 5,
        dead_letter: DeadLetter | None = None,
        batch: int = 32,
        fetch_timeout: float = 5.0,
    ) -> None:
        self._subscription = subscription
        self._handler = handler
        self._max_deliver = max_deliver
        self._dead_letter = dead_letter
        self._batch = batch
        self._timeout = fetch_timeout
        self.handled = 0
        self.buried = 0
        self.redelivered = 0

    async def consume(self) -> int:
        """Take one batch and settle every message in it. Returns how many were handled."""
        messages = await self._subscription.fetch(self._batch, timeout=self._timeout)
        handled = 0
        for message in messages:
            handled += await self._settled(message)
        return handled

    async def _settled(self, message: Any) -> int:  # noqa: ANN401 — the client's own message
        """One message, acknowledged, left for redelivery, or buried."""
        try:
            envelope = EventEnvelope.model_validate_json(message.data)
        except Exception:
            await self._buried(message, "undecodable")
            return 0
        try:
            outcome = self._handler(envelope)
            if hasattr(outcome, "__await__"):
                await outcome
        except Exception:
            if _delivered(message) >= self._max_deliver:
                await self._buried(message, "handler_failed")
                return 0
            self.redelivered += 1
            await message.nak()
            return 0
        await message.ack()
        self.handled += 1
        return 1

    async def _buried(self, message: Any, reason: str) -> None:  # noqa: ANN401 — the client's own message
        """Keep it where somebody can look, and stop the redelivery loop either way."""
        if self._dead_letter is not None:
            await self._dead_letter.bury(message.data, reason=reason)
        self.buried += 1
        await message.term()


def _delivered(message: Any) -> int:  # noqa: ANN401 — the client's own message
    """How many times this message has been delivered, or once where it does not say."""
    metadata = getattr(message, "metadata", None)
    return int(getattr(metadata, "num_delivered", 1) or 1)
