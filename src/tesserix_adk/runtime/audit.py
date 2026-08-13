"""Writing a decision down before acting on it.

The trail is where the parts of an audit record come together: the ladder's decision, the
run it was taken in, the headroom either side of it, and a digest of a payload that is
never stored. It is deliberately the last thing between a decision and the call going out,
because a record written afterwards is the record an outage eats.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.audit import AuditDecision, AuditEvent, digest_of_arguments, pseudonym
from tesserix_adk.core.autonomy import AutonomyLevel
from tesserix_adk.core.errors import AuditUnavailableError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from decimal import Decimal

    from tesserix_adk.core.audit import AuditSink
    from tesserix_adk.core.autonomy import AutonomyDecision
    from tesserix_adk.core.protocols import Clock

__all__ = ["AuditTrail", "MemoryAuditSink"]

UNCLASSIFIED = "unknown"


class MemoryAuditSink:
    """Decisions in a list, for tests and single-process demos.

    Append-only in the sense that matters — there is no method here that removes a record,
    and `pseudonymise` replaces the person while keeping the decision. Nothing in it
    outlives the process, which is what a durable audit trail is not: use it to see the
    shape of the records, and a real store to keep them.
    """

    def __init__(self) -> None:
        self._records: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEvent:
        """Record `event`, or return the record already there for that decision."""
        already = self._already(event)
        if already is not None:
            return already
        self._records.append(event)
        return event

    async def records(
        self,
        *,
        tenant: str,
        since: float = 0.0,
        until: float | None = None,
        decision: AuditDecision | None = None,
    ) -> tuple[AuditEvent, ...]:
        """Every decision for `tenant` in the period, in the order they were taken."""
        return tuple(
            held
            for held in self._records
            if held.tenant == tenant
            and held.recorded_at >= since
            and (until is None or held.recorded_at < until)
            and (decision is None or held.decision is decision)
        )

    async def pseudonymise(self, *, tenant: str, subject: str) -> int:
        """Replace `subject` wherever it named a person, and say how many records changed."""
        stood_in = pseudonym(subject)
        changed = 0
        for index, held in enumerate(self._records):
            if held.tenant != tenant or subject not in {held.user, held.approver}:
                continue
            self._records[index] = held.model_copy(
                update={
                    "user": stood_in if held.user == subject else held.user,
                    "approver": stood_in if held.approver == subject else held.approver,
                }
            )
            changed += 1
        return changed

    def _already(self, event: AuditEvent) -> AuditEvent | None:
        """What is stored for this decision about this call, where anything is."""
        return next(
            (
                held
                for held in self._records
                if held.run_id == event.run_id
                and held.idempotency_key == event.idempotency_key
                and held.decision is event.decision
            ),
            None,
        )


class AuditTrail:
    """Turns one decision into one durable record, and refuses to lose it quietly.

    Args:
        sink: Where records are appended.
        clock: What `recorded_at` is taken from.
        redact_patterns: Shapes this deployment knows about, applied before the payload is
            digested — a local case or account reference the built-in shapes cannot know.

    Example:
        >>> import asyncio
        >>> from tesserix_adk.core.audit import AuditDecision
        >>> from tesserix_adk.testing import FakeClock
        >>> sink = MemoryAuditSink()
        >>> trail = AuditTrail(sink, clock=FakeClock(start=0.0))
        >>> recorded = asyncio.run(
        ...     trail.record(
        ...         None,
        ...         AuditDecision.REFUSED,
        ...         run_id="run_1",
        ...         tenant="acme",
        ...         tool="refund",
        ...         arguments={"amount": 900},
        ...         reason="beyond the ceiling",
        ...     )
        ... )
        >>> recorded.decision
        <AuditDecision.REFUSED: 'refused'>
    """

    def __init__(
        self, sink: AuditSink, *, clock: Clock, redact_patterns: Sequence[str] = ()
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._patterns = tuple(redact_patterns)
        self._sequence: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def record(
        self,
        decided: AutonomyDecision | None,
        decision: AuditDecision,
        *,
        run_id: str,
        tenant: str,
        tool: str,
        arguments: Mapping[str, Any],
        key: str | None = None,
        user: str | None = None,
        agent: str = "",
        agent_version: str = "",
        approver: str | None = None,
        amount: Decimal | None = None,
        reason: str = "",
    ) -> AuditEvent:
        """Write down what was decided about this call, and return what is now stored.

        Args:
            decided: What the ladder said, where a ladder was consulted. Absent for a call
                that never reached one, such as an approval a hook demanded.
            decision: Acted, escalated, refused, or withdrawn under the run.
            run_id: The run the decision was taken in.
            tenant: The isolation boundary it was taken in.
            tool: The call the decision is about.
            arguments: What it would be made with. Digested, never stored.
            key: What identifies the call rather than the attempt, so a retry writes once.
                Absent, the tool and the payload's digest stand in, which makes one call
                idempotent but two identical calls in a run indistinguishable — pass one.
            user: On whose behalf the run is going.
            agent: Which agent attempted the call, and `agent_version` at what version.
            agent_version: The agent's version.
            approver: Who decided, where a human did.
            amount: What this action commits, for the headroom it would leave.
            reason: Why, where the decision carries a reason the ladder's does not.

        Returns:
            The record now in the store, which is the earlier one where this decision about
            this call had already been recorded.

        Raises:
            AuditUnavailableError: If the store could not take the record. The caller must
                not proceed: an action nobody could record is an action nobody can defend.
        """
        digest = digest_of_arguments(arguments, extra_patterns=self._patterns)
        event = AuditEvent(
            run_id=run_id,
            sequence=await self._next(run_id),
            tenant=tenant,
            user=user,
            agent_name=agent,
            agent_version=agent_version,
            tool=tool,
            action_class=decided.action_class if decided is not None else UNCLASSIFIED,
            level=decided.level if decided is not None else AutonomyLevel.ASK_ALWAYS,
            decision=decision,
            reason=reason or (decided.reason if decided is not None else ""),
            grant_id=decided.grant_id if decided is not None else None,
            headroom_before=decided.headroom if decided is not None else None,
            headroom_after=self._left(decided, decision, amount),
            approver=approver,
            arguments_digest=digest,
            idempotency_key=key or f"{tool}:{digest}",
            recorded_at=self._clock.now(),
        )
        try:
            return await self._sink.append(event)
        except Exception as unreachable:
            raise AuditUnavailableError(
                f"the decision to {decision} {tool!r} could not be recorded, so the call "
                f"does not go out",
                tool=tool,
                decision=str(decision),
                run_id=run_id,
                tenant=tenant,
            ) from unreachable

    async def _next(self, run_id: str) -> int:
        """The next position in this run's order, taken under a lock so fan-out cannot share one."""
        async with self._lock:
            position = self._sequence.get(run_id, 0)
            self._sequence[run_id] = position + 1
            return position

    def _left(
        self, decided: AutonomyDecision | None, decision: AuditDecision, amount: Decimal | None
    ) -> Decimal | None:
        """What the ceiling has left after this, where anything was committed against one."""
        if decided is None or decided.headroom is None or amount is None:
            return None
        if decision is not AuditDecision.EXECUTED:
            return decided.headroom
        return decided.headroom - amount
