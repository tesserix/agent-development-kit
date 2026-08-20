"""Agent activity as events, on one contract, carrying no content.

Each product otherwise wires its own notifications — a log line here, an ad-hoc publish
there — so nothing downstream can consume agent activity uniformly, and the payloads that
do exist have already carried message content into stores it should never reach.

An event here is an envelope: a stable identifier, what happened, when, and the scope it
happened in, with a body drawn from a fixed allowlist of identifiers, counts and spend.
There is no field a prompt, a tool argument or a retrieved passage could travel in, which
is the point — a redactor that runs after the publish is a redactor that runs too late.

Publishing is optional. The default publisher publishes nowhere, and the delivery mode says
in one place whether a broker being down stops the run or is counted and stepped over.
"""

from __future__ import annotations

from enum import StrEnum
from secrets import token_bytes
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.errors import ConfigurationError, EventPublishError, EventTooLargeError
from tesserix_adk.core.instrumentation import span_here
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.pii import redact
from tesserix_adk.core.tenancy import tenant_here

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "ALLOWED_ATTRIBUTES",
    "DEFAULT_MAX_EVENT_BYTES",
    "EVENT_SCHEMA_VERSION",
    "ApprovalDecided",
    "ApprovalRequested",
    "BudgetExceeded",
    "Delivery",
    "EventEnvelope",
    "EventPayload",
    "EventPublisher",
    "EventType",
    "Eventing",
    "MemoryErased",
    "NullEventPublisher",
    "PublishReport",
    "RunCancelled",
    "RunCompleted",
    "RunFailed",
    "RunStarted",
    "ToolCallCompleted",
    "ToolCallRequested",
]

EVENT_SCHEMA_VERSION = 1
"""The version of the envelope contract. Consumers branch on this, not on a field's absence."""

DEFAULT_MAX_EVENT_BYTES = 8_192
"""What one event may weigh. Under every broker's default, and far over any legitimate event."""

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RANDOM_BITS = 80
_TIME_BITS = 48


class EventType(StrEnum):
    """Everything the kit reports. A consumer switches on this and nothing else."""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    BUDGET_EXCEEDED = "budget_exceeded"
    MEMORY_ERASED = "memory_erased"


ALLOWED_ATTRIBUTES = frozenset(
    {
        "agent",
        "agent_version",
        "approval_id",
        "approver",
        "attempt",
        "cost_micros",
        "decision",
        "duration_ms",
        "error_code",
        "input_tokens",
        "iterations",
        "limit",
        "model",
        "model_calls",
        "output_tokens",
        "provider",
        "reason_code",
        "records_erased",
        "run_id",
        "scope",
        "session_id",
        "state",
        "subject",
        "tool",
        "tool_call_id",
        "tool_calls",
    }
)
"""Every name an event body may carry: identifiers, counts, the model and what it cost.

There is deliberately no entry for a message, a tool's arguments or a retrieved passage. A
payload that cannot express content cannot leak it into a store somebody later queries.
"""


class EventPayload(AdkModel):
    """What happened, as attributes drawn from the allowlist.

    Subclasses declare their own fields; every one of them must be on
    `ALLOWED_ATTRIBUTES`, which `attributes()` holds them to on the way out.
    """

    type: ClassVar[EventType]

    def attributes(self) -> dict[str, str]:
        """The body as the envelope carries it: strings, allowlisted, no empties.

        Raises:
            ConfigurationError: A field is not on the allowlist.
        """
        declared = set(self.__class__.model_fields)
        outside = declared - ALLOWED_ATTRIBUTES
        if outside:
            raise ConfigurationError(
                f"{type(self).__name__} declares {sorted(outside)}, which an event may not "
                f"carry; extend ALLOWED_ATTRIBUTES as a review decision"
            )
        return {
            name: str(value)
            for name, value in self.model_dump(mode="json").items()
            if value not in ("", None)
        }


class RunStarted(EventPayload):
    """A run began.

    Args:
        run_id: The run.
        agent: Which agent is running.
        agent_version: At which version.
        model: The model it opened with, where one was resolved by then.
        session_id: The session it belongs to, where it belongs to one.
    """

    type: ClassVar[EventType] = EventType.RUN_STARTED
    run_id: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    agent_version: str = ""
    model: str = ""
    session_id: str = ""


class RunCompleted(EventPayload):
    """A run finished on its own terms.

    Args:
        run_id: The run.
        iterations: How many turns it took.
        tool_calls: How many tool calls it made.
        model_calls: How many model calls it made.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.
        cost_micros: What it cost, in millionths of the accounting currency.
        duration_ms: How long it took, by the run's clock.
    """

    type: ClassVar[EventType] = EventType.RUN_COMPLETED
    run_id: str = Field(min_length=1)
    iterations: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_micros: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)


class RunFailed(EventPayload):
    """A run ended in a failure.

    Args:
        run_id: The run.
        error_code: Which failure, as a code a consumer can group by. Never a message: an
            exception's text is where content leaks into a dashboard.
        attempt: Which attempt this was.
    """

    type: ClassVar[EventType] = EventType.RUN_FAILED
    run_id: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)


class RunCancelled(EventPayload):
    """A run stopped because something asked it to.

    Args:
        run_id: The run.
        reason_code: Who or what stopped it, as a code.
    """

    type: ClassVar[EventType] = EventType.RUN_CANCELLED
    run_id: str = Field(min_length=1)
    reason_code: str = ""


class ToolCallRequested(EventPayload):
    """A tool call was about to go out.

    Args:
        run_id: The run.
        tool: Which tool.
        tool_call_id: The call, so the request and its completion pair up.
        attempt: Which try this is.
    """

    type: ClassVar[EventType] = EventType.TOOL_CALL_REQUESTED
    run_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)


class ToolCallCompleted(EventPayload):
    """A tool call came back, one way or another.

    Args:
        run_id: The run.
        tool: Which tool.
        tool_call_id: The call this answers.
        state: How it ended, as a code.
        duration_ms: How long it took.
        error_code: Which failure, where it failed.
    """

    type: ClassVar[EventType] = EventType.TOOL_CALL_COMPLETED
    run_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    state: str = Field(min_length=1)
    duration_ms: int = Field(default=0, ge=0)
    error_code: str = ""


class ApprovalRequested(EventPayload):
    """A human was asked before something ran.

    Args:
        run_id: The run.
        approval_id: The request, so a decision pairs with it.
        tool: What was waiting on the answer.
    """

    type: ClassVar[EventType] = EventType.APPROVAL_REQUESTED
    run_id: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)


class ApprovalDecided(EventPayload):
    """A human answered.

    Args:
        run_id: The run.
        approval_id: Which request.
        decision: What they decided, as a code.
        approver: Who decided, redacted on the way out like every other attribute.
    """

    type: ClassVar[EventType] = EventType.APPROVAL_DECIDED
    run_id: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    approver: str = ""


class BudgetExceeded(EventPayload):
    """A ceiling stopped the work.

    Args:
        run_id: The run.
        scope: Which ceiling — the run, the tenant, the window.
        limit: Which limit within it.
        cost_micros: What had been spent when it bit.
    """

    type: ClassVar[EventType] = EventType.BUDGET_EXCEEDED
    run_id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    limit: str = Field(min_length=1)
    cost_micros: int = Field(default=0, ge=0)


class MemoryErased(EventPayload):
    """A subject's records were erased, which downstream stores have to hear about.

    Args:
        subject: Whose. Redacted on the way out, so what travels is a pseudonym.
        records_erased: How many.
        run_id: The run that asked, where one did.
    """

    type: ClassVar[EventType] = EventType.MEMORY_ERASED
    subject: str = Field(min_length=1)
    records_erased: int = Field(default=0, ge=0)
    run_id: str = ""


class EventEnvelope(AdkModel):
    """One event as it travels.

    Args:
        event_id: A ULID. Sorts by the moment it was created, so a consumer can order a
            run's events without trusting a broker to.
        type: What happened.
        schema_version: Which version of this contract it speaks.
        occurred_at: When, by the emitting run's clock.
        tenant: The isolation boundary the work was under.
        user: On whose behalf, where the run had a principal.
        run_id: The run.
        trace_id: The trace it belongs to, which for kit-emitted events is the run.
        span_id: The span that was open when it was emitted, where one was.
        correlation_id: The caller's own identifier for this work.
        causation_id: The event that caused this one, where one did.
        attributes: The body, allowlisted and redacted.
    """

    event_id: str = Field(min_length=1)
    type: EventType
    schema_version: int = Field(default=EVENT_SCHEMA_VERSION, ge=1)
    occurred_at: float
    tenant: str = Field(min_length=1)
    user: str = ""
    run_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    correlation_id: str = ""
    causation_id: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)

    def to_json(self) -> str:
        """The envelope as a broker carries it."""
        return self.model_dump_json()


@runtime_checkable
class EventPublisher(Protocol):
    """Where events go. Implemented by a broker adapter, faked in tests, or absent."""

    async def publish(self, event: EventEnvelope) -> None:
        """Deliver one event, or raise.

        Ordering is per run: a publisher may reorder events from different runs and must
        not reorder one run's. Raising is how a transport says it could not deliver, and
        `Delivery` decides what that means for the run.
        """
        ...

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        """Deliver a batch, or raise. Partial delivery is the publisher's own business."""
        ...


class NullEventPublisher:
    """The default. Publishes nowhere, so eventing is optional rather than mandatory."""

    async def publish(self, event: EventEnvelope) -> None:
        """Accept the event and do nothing with it."""
        del event

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        """Accept the batch and do nothing with it."""
        del events


class Delivery(StrEnum):
    """What a publisher that cannot deliver means for the run.

    Never implicit: a dashboard missing an event and a state change without its event are
    different incidents, and which one a deployment prefers is the deployment's call.
    """

    BEST_EFFORT = "best_effort"
    GUARANTEED = "guaranteed"


class PublishReport(AdkModel):
    """What became of a batch.

    Args:
        published: The events that went out.
        rejected: The ones that did not, each with the reason as a code.
    """

    published: tuple[EventEnvelope, ...] = ()
    rejected: tuple[tuple[EventType, str], ...] = ()


class Eventing:
    """Builds envelopes from payloads and publishes them under one delivery mode.

    Scope comes from the ambient tenant context rather than from each publish site, so no
    caller can emit an event that quietly belongs to nobody, and the redaction runs here —
    once, before the publisher — rather than in each adapter.

    Args:
        publisher: Where events go. `NullEventPublisher` by default.
        clock: What stamps `occurred_at`. Injected, so tests are deterministic.
        delivery: What a failed publish means for the run.
        max_event_bytes: The largest event this transport takes.
        entropy: Where a ULID's randomness comes from. Injected for reproducible tests.
    """

    __slots__ = ("_clock", "_delivery", "_entropy", "_last", "_max_bytes", "_publisher", "dropped")

    def __init__(
        self,
        publisher: EventPublisher | None = None,
        *,
        clock: Clock,
        delivery: Delivery = Delivery.BEST_EFFORT,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        entropy: Callable[[int], bytes] = token_bytes,
    ) -> None:
        self._publisher = publisher or NullEventPublisher()
        self._clock = clock
        self._delivery = delivery
        self._max_bytes = max_event_bytes
        self._entropy = entropy
        self._last: tuple[int, int] | None = None
        self.dropped = 0

    async def emit(
        self, payload: EventPayload, *, caused_by: EventEnvelope | None = None
    ) -> EventEnvelope | None:
        """Publish one event.

        Args:
            payload: What happened.
            caused_by: The event this one follows from, where there is one.

        Returns:
            The envelope that went out, or None where best-effort delivery dropped it.

        Raises:
            ConfigurationError: No tenant is bound, or the payload carries a field the
                allowlist does not permit.
            EventTooLargeError: The event is over the transport's ceiling, under
                guaranteed delivery.
            EventPublishError: The publisher could not deliver, under guaranteed delivery.
        """
        envelope = self._envelope(payload, caused_by)
        try:
            self._within_the_ceiling(envelope)
            await self._publisher.publish(envelope)
        except Exception as failure:
            self._absorbed(envelope, failure)
            return None
        return envelope

    async def emit_all(
        self, payloads: Iterable[EventPayload], *, caused_by: EventEnvelope | None = None
    ) -> PublishReport:
        """Publish a batch as one call, keeping what can be sent.

        An event that is over the ceiling is left out of the batch rather than taking the
        rest of it down; under guaranteed delivery the ones that fit still go out and the
        refusal is raised afterwards, so a consumer is never silently short of events it
        was told nothing about.

        Args:
            payloads: What happened, in order.
            caused_by: The event they all follow from, where there is one.

        Returns:
            What went out and what did not.

        Raises:
            EventTooLargeError: An event was over the ceiling, under guaranteed delivery.
            EventPublishError: The publisher could not deliver, under guaranteed delivery.
        """
        sendable: list[EventEnvelope] = []
        rejected: list[tuple[EventType, str]] = []
        refused: EventTooLargeError | None = None
        for payload in payloads:
            envelope = self._envelope(payload, caused_by)
            try:
                self._within_the_ceiling(envelope)
            except EventTooLargeError as failure:
                self.dropped += 1
                rejected.append((envelope.type, "too_large"))
                refused = refused or failure
                continue
            sendable.append(envelope)
        published = await self._batch(tuple(sendable), rejected)
        if refused is not None and self._delivery is Delivery.GUARANTEED:
            raise refused
        return PublishReport(published=published, rejected=tuple(rejected))

    async def _batch(
        self, sendable: tuple[EventEnvelope, ...], rejected: list[tuple[EventType, str]]
    ) -> tuple[EventEnvelope, ...]:
        """The batch, delivered, or absorbed the way the delivery mode says."""
        if not sendable:
            return ()
        try:
            await self._publisher.publish_batch(sendable)
        except Exception as failure:
            if self._delivery is Delivery.GUARANTEED:
                raise EventPublishError(
                    f"{len(sendable)} events could not be published, so the run stops rather "
                    f"than diverging from what was reported",
                    event_type=sendable[0].type,
                    run_id=sendable[0].run_id,
                    tenant=sendable[0].tenant,
                ) from failure
            self.dropped += len(sendable)
            rejected.extend((envelope.type, "publish_failed") for envelope in sendable)
            return ()
        return sendable

    def _absorbed(self, envelope: EventEnvelope, failure: BaseException) -> None:
        """What a failed publish means here, which is what the delivery mode says."""
        if self._delivery is Delivery.GUARANTEED:
            if isinstance(failure, EventTooLargeError):
                raise failure
            raise EventPublishError(
                f"{envelope.type.value} could not be published, so the run stops rather than "
                f"diverging from what was reported",
                event_type=envelope.type,
                run_id=envelope.run_id,
                tenant=envelope.tenant,
            ) from failure
        self.dropped += 1
        return

    def _envelope(self, payload: EventPayload, caused_by: EventEnvelope | None) -> EventEnvelope:
        """The payload, scoped from the ambient context and redacted."""
        context = tenant_here()
        if context is None:
            raise ConfigurationError(
                f"{payload.type.value} was emitted outside any tenant scope; an event nobody "
                f"can attribute is an event no consumer may read"
            )
        run_id = str(payload.model_dump(mode="json").get("run_id", "") or "")
        return EventEnvelope(
            event_id=self._identifier(),
            type=payload.type,
            occurred_at=self._clock.now(),
            tenant=context.tenant,
            user=context.user or "",
            run_id=run_id,
            trace_id=run_id,
            span_id=span_here() or "",
            correlation_id=context.correlation_id or "",
            causation_id=caused_by.event_id if caused_by else "",
            attributes=_scrubbed(payload.attributes(), tenant=context.tenant),
        )

    def _within_the_ceiling(self, envelope: EventEnvelope) -> None:
        """Refuse an event no transport will take, rather than truncating it into one."""
        size = len(envelope.to_json().encode())
        if size > self._max_bytes:
            raise EventTooLargeError(
                f"{envelope.type.value} is {size} bytes, over the {self._max_bytes} this "
                f"transport takes; publish an identifier and let the consumer fetch the rest",
                event_type=envelope.type,
                size_bytes=size,
                limit_bytes=self._max_bytes,
                run_id=envelope.run_id,
                tenant=envelope.tenant,
            )

    def _identifier(self) -> str:
        """A ULID, monotonic within a millisecond so two events never tie."""
        moment = int(self._clock.now() * 1000)
        previous, randomness = self._last or (-1, 0)
        if moment == previous:
            randomness += 1
        else:
            randomness = int.from_bytes(self._entropy(_RANDOM_BITS // 8))
        self._last = (moment, randomness)
        return _base32((moment << _RANDOM_BITS) | randomness)


def _base32(value: int) -> str:
    """A 128-bit value in Crockford base32, the form a ULID sorts by as text."""
    digits = []
    for _ in range((_TIME_BITS + _RANDOM_BITS) // 5):
        digits.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(digits))


def _scrubbed(attributes: Mapping[str, str], *, tenant: str) -> dict[str, str]:
    """Every attribute with identifiers taken out, before anything has seen them."""
    return {
        name: redact(value, tenant=tenant).text if not value.isdigit() else value
        for name, value in attributes.items()
    }
