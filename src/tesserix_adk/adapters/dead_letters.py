"""Seeing what failed, and putting it back through the path that would have handled it.

When a consumer bug corrupts a day of processing, the recovery is where the second incident
comes from: somebody reads the stream by hand and reprocesses with a script that bypasses the
idempotency the live path has. So replay here goes through the *same* handler as live
traffic — the dedupe marker, the transaction and the ordering all still apply — and every
guardrail is about keeping it that way.

A replay is always scoped to one tenant, capped to a batch, marked on every envelope it
redelivers, and audited as one event naming who ran it and how many records moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from secrets import token_bytes
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field, ValidationInfo, field_validator

from tesserix_adk.core.errors import (
    ScopeViolationError,
    UnknownEventTypeError,
    UnsupportedEventVersionError,
)
from tesserix_adk.core.event_schema import EVENT_SCHEMAS
from tesserix_adk.core.events import EventEnvelope, EventsReplayed, EventType
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.tenancy import tenant_scope

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from tesserix_adk.core.events import Eventing
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "MAX_REPLAY_BATCH",
    "UNDECODABLE",
    "UNKNOWN_TENANT",
    "DeadLetterQuery",
    "DeadLetterRecord",
    "DeadLetterStats",
    "DeadLetterStore",
    "ErasureCheck",
    "InMemoryDeadLetters",
    "ReplayHandler",
    "ReplayPlan",
    "ReplayReport",
    "Replayer",
]

MAX_REPLAY_BATCH = 500
"""The most one replay may move. A backlog is paged; one enormous replay is an outage."""

UNDECODABLE = "undecodable"
"""The record is not an envelope this kit can read, so nothing may act on it."""

UNKNOWN_TENANT = "unknown"
"""Whose an unreadable record is cannot be known, and guessing would misfile it."""

type ReplayHandler = Callable[[EventEnvelope], Awaitable[object]]
"""The live consumer path. Idempotency and ordering come from it, not from the replay.

Returning `False` reports that the delivery was recognised as already applied, which is
what `IdempotentConsumer.handle` does; any other return counts as a redelivery.
"""

type ErasureCheck = Callable[[DeadLetterRecord], Awaitable[bool]]
"""Whether the scope this record refers to has since been erased."""


class DeadLetterRecord(AdkModel):
    """One event that will not be handled, and what is known about why.

    Args:
        envelope: The event as it was published.
        group: The consumer group that could not handle it.
        reason: What the consumer called the failure.
        last_error: The type of the last exception. Never its message, which carries content.
        attempts: How many times it has been buried.
        first_seen: When it first arrived here.
        last_seen: When it last did.
    """

    envelope: EventEnvelope
    group: str = ""
    reason: str = ""
    last_error: str = ""
    attempts: int = Field(default=1, ge=1)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def inspected(self) -> dict[str, str]:
        """What an operator may see: identifiers, counts and timings, never the body.

        The attribute *names* are shown so an operator can tell one failure shape from
        another; their values are the body, and a listing of a day's failures on a terminal
        is a second copy of it.
        """
        return {
            "event_id": self.envelope.event_id,
            "type": str(self.envelope.type),
            "schema_version": str(self.envelope.schema_version),
            "tenant": self.envelope.tenant,
            "run_id": self.envelope.run_id,
            "group": self.group,
            "reason": self.reason,
            "last_error": self.last_error,
            "attempts": str(self.attempts),
            "first_seen": str(self.first_seen),
            "last_seen": str(self.last_seen),
            "attributes": ", ".join(sorted(self.envelope.attributes)),
        }


class DeadLetterQuery(AdkModel):
    """Which records to look at, or to move. Never more than one tenant's.

    Args:
        tenant: The isolation boundary. Mandatory: a listing that crossed one would show an
            operator another tenant's identifiers, and a replay would act on them.
        event_type: One type, where the failure was a type's.
        group: One consumer group, where the failure was a group's.
        since: The earliest arrival to include.
        until: The latest arrival to include, where the window has an end.
        limit: How many, capped so a backlog is paged rather than replayed at once.
    """

    tenant: str = Field(min_length=1)
    event_type: EventType | None = None
    group: str = ""
    since: float = 0.0
    until: float = 0.0
    limit: int = Field(default=100, ge=1, le=MAX_REPLAY_BATCH)

    @field_validator("until")
    @classmethod
    def _ordered(cls, until: float, info: ValidationInfo) -> float:
        """A window that ends before it starts selects nothing, which no caller means."""
        since = float(info.data.get("since", 0.0))
        if until and until < since:
            raise ValueError(f"until {until} is before since {since}")
        return until

    def matches(self, record: DeadLetterRecord) -> bool:
        """Whether this record is in the selection, for a store that cannot filter in SQL."""
        if record.envelope.tenant != self.tenant:
            return False
        if self.event_type is not None and record.envelope.type is not self.event_type:
            return False
        if self.group and record.group != self.group:
            return False
        if record.first_seen < self.since:
            return False
        return not (self.until and record.first_seen > self.until)


@dataclass(frozen=True, slots=True)
class DeadLetterStats:
    """The backlog, as the two numbers worth alerting on.

    Args:
        buried: How many records are waiting for somebody to look at them.
        oldest_seconds: How long the oldest has waited.
        arrivals: How many arrived since the process started, for a rate.
    """

    buried: int = 0
    oldest_seconds: float = 0.0
    arrivals: int = 0


@runtime_checkable
class DeadLetterStore(Protocol):
    """Where undeliverable events are kept, and how an operator reads them back.

    `bury` is the `DeadLetter` shape every consumer in the kit already writes to, so an
    existing consumer becomes inspectable by being pointed at one of these.
    """

    async def bury(
        self, payload: bytes, *, reason: str, history: tuple[str, ...] = (), group: str = ""
    ) -> None:
        """Record one undeliverable event."""
        ...

    async def list(self, query: DeadLetterQuery) -> Sequence[DeadLetterRecord]:
        """The records in the selection, oldest first."""
        ...

    async def forget(self, event_ids: Sequence[str]) -> None:
        """Drop records that have been dealt with."""
        ...

    async def stats(self, tenant: str) -> DeadLetterStats:
        """The backlog for one tenant."""
        ...


@dataclass(slots=True)
class InMemoryDeadLetters:
    """A store in this process. The reference implementation, and what tests and the CLI use.

    A deployment that must survive a restart points the same consumers at a durable store
    with this shape; nothing above the protocol changes.
    """

    clock: Clock
    records: dict[str, DeadLetterRecord] = field(default_factory=dict)
    arrivals: int = 0
    leak: bool = False
    _undecodable: int = 0

    async def bury(
        self, payload: bytes, *, reason: str, history: tuple[str, ...] = (), group: str = ""
    ) -> None:
        """Record one undeliverable event, counting a repeat rather than duplicating it."""
        self.arrivals += 1
        now = self.clock.now()
        envelope = _decoded(payload)
        if envelope is None:
            self._undecodable += 1
            self.records[f"{UNDECODABLE}:{self.arrivals}"] = _unreadable(payload, now, group)
            return
        existing = self.records.get(envelope.event_id)
        self.records[envelope.event_id] = DeadLetterRecord(
            envelope=envelope,
            group=group or (existing.group if existing else ""),
            reason=reason,
            last_error=_type_of(history),
            attempts=(existing.attempts + 1) if existing else 1,
            first_seen=existing.first_seen if existing else now,
            last_seen=now,
        )

    async def list(self, query: DeadLetterQuery) -> Sequence[DeadLetterRecord]:
        """The records in the selection, oldest first and capped by the query's limit."""
        found = [
            record
            for record in self.records.values()
            if self.leak or query.matches(record) or _is_unreadable(record)
        ]
        found.sort(key=lambda record: (record.first_seen, record.envelope.event_id))
        return found[: query.limit]

    async def forget(self, event_ids: Sequence[str]) -> None:
        """Drop records that have been replayed or written off."""
        for event_id in event_ids:
            self.records.pop(event_id, None)

    async def stats(self, tenant: str) -> DeadLetterStats:
        """The backlog for one tenant: how much is waiting, and how long the oldest has."""
        mine = [
            record
            for record in self.records.values()
            if record.envelope.tenant == tenant or _is_unreadable(record)
        ]
        if not mine:
            return DeadLetterStats(arrivals=self.arrivals)
        oldest = min(record.first_seen for record in mine)
        return DeadLetterStats(
            buried=len(mine), oldest_seconds=self.clock.now() - oldest, arrivals=self.arrivals
        )

    def for_group(self, group: str) -> _GroupBound:
        """This store as one consumer group's dead letter.

        The `DeadLetter` shape every consumer writes to has no group in it, so the group
        comes from which view of the store the consumer was given.
        """
        return _GroupBound(self, group)

    async def undecodable(self) -> int:
        """How many arrivals were not events this kit can read."""
        return self._undecodable


@dataclass(frozen=True, slots=True)
class _GroupBound:
    """One consumer group's view of a store, which is what a consumer is handed."""

    letters: InMemoryDeadLetters
    group: str

    async def bury(self, payload: bytes, *, reason: str, history: tuple[str, ...] = ()) -> None:
        """Record one undeliverable event against this group."""
        await self.letters.bury(payload, reason=reason, history=history, group=self.group)


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """What a replay would do, without having done any of it.

    Args:
        replayable: How many records would be redelivered.
        refusals: The records that would not be, and why.
        remaining: How many are left in the selection beyond this batch.
    """

    replayable: int = 0
    refusals: tuple[tuple[str, str], ...] = ()
    remaining: int = 0


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """What a replay did.

    Args:
        replay_id: The batch, which every redelivered envelope carries.
        replayed: How many reached the handler and were accepted.
        suppressed: How many the consumer's own idempotency recognised as already done.
        failed: How many the handler raised on, which stay where they were.
        refused: How many this could not safely redeliver.
        refusals: Those records and the reason, for the operator to act on.
        remaining: How many are left in the selection beyond this batch.
    """

    replay_id: str
    replayed: int = 0
    suppressed: int = 0
    failed: int = 0
    refused: int = 0
    refusals: tuple[tuple[str, str], ...] = ()
    remaining: int = 0


class Replayer:
    """Puts dead-lettered events back through the live consumer path, under guardrails.

    Args:
        store: Where the records are.
        handler: The consumer path. Wrap an `IdempotentConsumer.handle` here and a replay
            of an event that was already applied is suppressed rather than applied twice.
        clock: What timings are measured against.
        eventing: Where the audit event goes. Absent, nothing is audited, which is only
            appropriate in a test.
        erased: Asked before each record whether its scope has since been erased.
        entropy: Where a replay id's randomness comes from. Injected for reproducible tests.
    """

    __slots__ = ("_clock", "_entropy", "_erased", "_eventing", "_handler", "_store")

    def __init__(
        self,
        store: DeadLetterStore,
        *,
        handler: ReplayHandler,
        clock: Clock,
        eventing: Eventing | None = None,
        erased: ErasureCheck | None = None,
        entropy: Callable[[int], bytes] = token_bytes,
    ) -> None:
        self._store = store
        self._handler = handler
        self._clock = clock
        self._eventing = eventing
        self._erased = erased
        self._entropy = entropy

    async def records(self, query: DeadLetterQuery) -> Sequence[DeadLetterRecord]:
        """The selection, for an operator to look at before deciding anything.

        Raises:
            ScopeViolationError: The selection reaches beyond the tenant it is scoped to.
        """
        return await self._selected(query)

    async def plan(self, query: DeadLetterQuery) -> ReplayPlan:
        """What `replay` would do with this selection, without sending anything.

        Raises:
            ScopeViolationError: The selection reaches beyond the tenant it is scoped to.
        """
        records = await self._selected(query)
        refusals = tuple(
            (record.envelope.event_id, reason)
            for record, reason in [(record, await self._refusal(record)) for record in records]
            if reason
        )
        return ReplayPlan(
            replayable=len(records) - len(refusals),
            refusals=refusals,
            remaining=await self._remaining(query, len(records)),
        )

    async def replay(
        self, query: DeadLetterQuery, *, operator: str, reason: str = ""
    ) -> ReplayReport:
        """Redeliver the selection through the consumer path, and audit that it happened.

        Records are replayed oldest first, which is the order their run emitted them, so a
        run's events cannot overtake each other even while live traffic flows.

        Args:
            query: What to replay. Scoped to one tenant, capped to one batch.
            operator: Who is running it, for the audit record.
            reason: Why, in a form an audit reader can act on.

        Raises:
            ScopeViolationError: The selection reaches beyond the tenant it is scoped to.
        """
        records = await self._selected(query)
        remaining = await self._remaining(query, len(records))
        replay_id = self._identifier()
        replayed: list[str] = []
        refusals: list[tuple[str, str]] = []
        suppressed = failed = 0
        for record in records:
            refused = await self._refusal(record)
            if refused:
                refusals.append((record.envelope.event_id, refused))
                continue
            outcome = await self._delivered(record, replay_id)
            if outcome == "replayed":
                replayed.append(record.envelope.event_id)
            elif outcome == "suppressed":
                suppressed += 1
                replayed.append(record.envelope.event_id)
            else:
                failed += 1
        await self._store.forget(replayed)
        report = ReplayReport(
            replay_id=replay_id,
            replayed=len(replayed) - suppressed,
            suppressed=suppressed,
            failed=failed,
            refused=len(refusals),
            refusals=tuple(refusals),
            remaining=remaining,
        )
        await self._audited(query, report, operator=operator, reason=reason)
        return report

    async def _selected(self, query: DeadLetterQuery) -> Sequence[DeadLetterRecord]:
        """The records to work on, refusing a selection that reaches past its tenant."""
        records = await self._store.list(query)
        outside = {
            record.envelope.tenant
            for record in records
            if record.envelope.tenant not in {query.tenant, UNKNOWN_TENANT}
        }
        if outside:
            raise ScopeViolationError(
                f"this selection also holds events of {sorted(outside)}, and an operator's "
                f"authority does not reach another tenant's consumers",
                expected=query.tenant,
                found=tuple(sorted(outside)),
            )
        return records

    async def _refusal(self, record: DeadLetterRecord) -> str:
        """Why this record may not be redelivered, or an empty string where it may."""
        if _is_unreadable(record):
            return UNDECODABLE
        try:
            EVENT_SCHEMAS.upcast(record.envelope)
        except (UnsupportedEventVersionError, UnknownEventTypeError):
            return "unsupported_version"
        if self._erased is not None and await self._erased(record):
            return "erased"
        return ""

    async def _delivered(self, record: DeadLetterRecord, replay_id: str) -> str:
        """One record through the consumer path, marked as a replay."""
        envelope = EVENT_SCHEMAS.upcast(record.envelope).model_copy(update={"replay_id": replay_id})
        try:
            with tenant_scope(envelope.tenant, user=envelope.user or None):
                outcome = await self._handler(envelope)
        except Exception:  # the consumer path's own failure, which leaves the record where it is
            return "failed"
        return "suppressed" if outcome is False else "replayed"

    async def _remaining(self, query: DeadLetterQuery, taken: int) -> int:
        """How much of the selection this batch did not reach."""
        if taken < query.limit:
            return 0
        wider = query.model_copy(update={"limit": MAX_REPLAY_BATCH})
        return max(len(await self._store.list(wider)) - taken, 0)

    async def _audited(
        self, query: DeadLetterQuery, report: ReplayReport, *, operator: str, reason: str
    ) -> None:
        """Record that a recovery happened, naming who and how many, never what."""
        if self._eventing is None:
            return
        with tenant_scope(query.tenant, user=operator or None):
            await self._eventing.emit(
                EventsReplayed(
                    replay_id=report.replay_id,
                    records=report.replayed + report.suppressed,
                    source_group=query.group,
                    approver=operator,
                    reason_code=reason,
                )
            )

    def _identifier(self) -> str:
        """A replay id, unique enough that two operators' batches never share one."""
        return f"rp_{int(self._clock.now() * 1000):x}{self._entropy(6).hex()}"


def _decoded(payload: bytes) -> EventEnvelope | None:
    """The event, keeping one this kit cannot yet read so a replay can refuse it by name."""
    try:
        return EVENT_SCHEMAS.read(payload)
    except (UnknownEventTypeError, UnsupportedEventVersionError):
        pass
    except ValueError:
        return None
    try:
        return EventEnvelope.model_validate_json(payload)
    except ValueError:
        return None


def _unreadable(payload: bytes, now: float, group: str) -> DeadLetterRecord:
    """A record for something that is not an envelope, so an operator can still see it."""
    return DeadLetterRecord(
        envelope=EventEnvelope(
            event_id=f"{UNDECODABLE}:{int(now * 1000)}:{len(payload)}",
            type=EventType.RUN_FAILED,
            occurred_at=now,
            tenant=UNKNOWN_TENANT,
        ),
        group=group,
        reason=UNDECODABLE,
        attempts=1,
        first_seen=now,
        last_seen=now,
    )


def _is_unreadable(record: DeadLetterRecord) -> bool:
    """An unreadable record belongs to no known tenant, so no tenant filter would show it."""
    return record.envelope.tenant == UNKNOWN_TENANT


def _type_of(history: tuple[str, ...]) -> str:
    """The exception type from the failure history, without whatever the message carried."""
    return history[-1].split(":", 1)[0].strip() if history else ""
