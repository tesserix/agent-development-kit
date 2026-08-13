"""The ladder as the loop consults it, at the point a tool call would go out.

`AutonomyLadder` decides about one action from what it is told. This is what tells it:
what the tenant has already committed against the class in the grant's window, and
whether a report an earlier action owed was ever delivered. Both are seams — the
arithmetic that makes commitments hard to game by splitting or retrying is its own
concern, and so is the shape of the audit record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core.audit import AuditDecision
from tesserix_adk.core.autonomy import (
    ActionRequest,
    AutonomyDecision,
    AutonomyOutcome,
    InFlightPolicy,
)
from tesserix_adk.core.errors import CeilingExceededError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from typing import Any

    from tesserix_adk.core.autonomy import AutonomyGrant, AutonomyLadder, Revocation
    from tesserix_adk.core.ceiling import CeilingLedger
    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.runtime.audit import AuditTrail

__all__ = ["AutonomyGate", "InMemoryReports", "ReportLog", "RevocationBroadcast", "RevocationWatch"]

_AUDITED = {
    AutonomyOutcome.ESCALATE: AuditDecision.ESCALATED,
    AutonomyOutcome.REFUSE: AuditDecision.REFUSED,
}


@runtime_checkable
class RevocationBroadcast(Protocol):
    """How a revocation reaches the processes that are already running.

    A grant read once at run start is not withdrawable, so the store is re-read per action
    and this is what makes that read prompt rather than eventual. It is an accelerator,
    never the authority: a missed message costs latency, and the store still refuses.
    """

    async def publish(self, revocation: Revocation) -> None:
        """Tell every listener that this authority is withdrawn."""
        ...

    def listen(self) -> AsyncIterator[Revocation]:
        """Every revocation as it is published."""
        ...


class RevocationWatch:
    """What this process has heard about withdrawn authority, and how lately.

    Args:
        clock: What freshness is measured against.
        stale_after_seconds: How long this view may go unconfirmed before it is no longer
            trusted. Past it, an action that would have been taken unattended is refused
            instead — a view nobody has confirmed cannot say that a grant is still live.
    """

    def __init__(self, *, clock: Clock, stale_after_seconds: float = 30.0) -> None:
        self._clock = clock
        self._stale_after = stale_after_seconds
        self._revocations: list[Revocation] = []
        self._confirmed_at = clock.now()

    async def heard(self, revocation: Revocation) -> None:
        """Take a revocation from the bus, which also confirms the view is current."""
        self._revocations.append(revocation)
        self._confirmed_at = self._clock.now()

    async def follow(self, broadcast: RevocationBroadcast) -> None:
        """Consume the bus until it ends. Run as a task for the life of the process."""
        async for revocation in broadcast.listen():
            await self.heard(revocation)

    def confirm(self) -> None:
        """Record that this view is current, for a caller that just re-read the store."""
        self._confirmed_at = self._clock.now()

    def fresh(self) -> bool:
        """Whether this view has been confirmed lately enough to be acted on."""
        return self._clock.now() - self._confirmed_at <= self._stale_after

    def revoked(self, grant: AutonomyGrant) -> bool:
        """Whether anything this view heard withdrew `grant`."""
        return any(held.covers(grant) for held in self._revocations)

    def withdrew(self, grant_id: str) -> Revocation | None:
        """The revocation that named `grant_id`, where the bus carried one."""
        return next((held for held in self._revocations if held.grant_id == grant_id), None)


@runtime_checkable
class ReportLog(Protocol):
    """Which `act_and_report` reports are owed, and whether they arrived."""

    async def outstanding(self, *, tenant: str, action_class: str) -> bool:
        """Whether a report for an earlier action of this class is still undelivered."""
        ...

    async def owed(self, *, tenant: str, action_class: str, run_id: str) -> None:
        """Record that acting unattended has just obliged a report."""
        ...


class InMemoryReports:
    """Owed reports in a set, for tests and single-process deployments.

    An owed report degrades the next action of the same class to asking a human, which is
    what stops `act_and_report` becoming `act` the moment nobody is reading the reports.
    """

    def __init__(self) -> None:
        self._owed: dict[tuple[str, str], set[str]] = {}

    async def outstanding(self, *, tenant: str, action_class: str) -> bool:
        """Whether anything is owed for this tenant and class."""
        return bool(self._owed.get((tenant, action_class)))

    async def owed(self, *, tenant: str, action_class: str, run_id: str) -> None:
        """Record that `run_id` acted and owes a report."""
        self._owed.setdefault((tenant, action_class), set()).add(run_id)

    async def delivered(self, *, tenant: str, action_class: str, run_id: str) -> None:
        """Record that the report `run_id` owed was delivered."""
        self._owed.get((tenant, action_class), set()).discard(run_id)


class AutonomyGate:
    """Answers, for one attempted tool call, whether a human has to be asked.

    Args:
        ladder: What the grants say. The ladder holds the commitment ledger, because what
            counts as already committed depends on which grant answered.
        reports: Where an owed report is recorded. Absent, `act_and_report` acts and the
            obligation is the consumer's to honour elsewhere.
        revocations: What this process has heard withdrawn. Absent, the store's own read
            is the only check, which is correct but no faster than the next action.
        revoked_runs: What a run already under way does when the grant behind the call a
            human just approved turns out to have been withdrawn.
        commitments: Where headroom is taken and settled. Absent, the ladder's read of the
            ceiling is the only check, which two concurrent processes can both pass.
        audit: Where decisions are written down. Absent, nothing is recorded and the gate
            behaves as it always did — a deployment that wants unattended action to be
            defensible afterwards wires one.
    """

    def __init__(
        self,
        ladder: AutonomyLadder,
        *,
        reports: ReportLog | None = None,
        revocations: RevocationWatch | None = None,
        revoked_runs: InFlightPolicy = InFlightPolicy.CANCEL,
        commitments: CeilingLedger | None = None,
        audit: AuditTrail | None = None,
    ) -> None:
        self._ladder = ladder
        self._reports = reports
        self._revocations = revocations
        self._commitments = commitments
        self._audit = audit
        self.revoked_runs = revoked_runs

    async def decide(
        self,
        *,
        tool: str,
        tenant: str,
        arguments: Mapping[str, Any],
        run_id: str,
        user: str | None = None,
        key: str | None = None,
        agent: str = "",
        agent_version: str = "",
    ) -> AutonomyDecision:
        """What may happen about this call, and record the report acting on it owes.

        `key` identifies the call rather than the attempt, so a retry of a call that may
        already have gone out asks about the same action instead of taking headroom twice.

        An escalation or a refusal is written to the audit trail here, because this is
        where it was decided. An action the ladder permits is not: only the loop knows
        whether every other gate cleared it too, and a record of an action that never went
        out is as misleading as no record of one that did.

        Raises:
            AuditUnavailableError: If a trail is wired and a refusal or escalation could
                not be recorded.
        """
        action_class = self._ladder.classify(tool)
        decided = await self._ladder.decide(
            ActionRequest(
                tool=tool,
                tenant=tenant,
                user=user,
                arguments=dict(arguments),
                reports_outstanding=await self._owing(tenant, action_class),
            )
        )
        decided = self._still_granted(decided)
        decided = await self._held(decided, tool=tool, tenant=tenant, arguments=arguments, key=key)
        if decided.unattended and decided.reports and self._reports is not None:
            await self._reports.owed(
                tenant=tenant, action_class=decided.action_class, run_id=run_id
            )
        if decided.outcome is not AutonomyOutcome.ACT:
            await self.record(
                decided,
                _AUDITED[decided.outcome],
                run_id=run_id,
                tenant=tenant,
                tool=tool,
                arguments=arguments,
                key=key,
                user=user,
                agent=agent,
                agent_version=agent_version,
            )
        return decided

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
        reason: str = "",
    ) -> None:
        """Write one decision to the audit trail, or do nothing where none is wired.

        The loop calls this for the two decisions it owns rather than the ladder: a call
        cleared to go out, and a grant withdrawn under a run that was waiting on a human.

        Raises:
            AuditUnavailableError: If the trail could not take the record.
        """
        if self._audit is None:
            return
        await self._audit.record(
            decided,
            decision,
            run_id=run_id,
            tenant=tenant,
            tool=tool,
            arguments=arguments,
            key=key,
            user=user,
            agent=agent,
            agent_version=agent_version,
            approver=approver,
            amount=self._ladder.amount_in(tool, arguments),
            reason=reason,
        )

    async def _held(
        self,
        decided: AutonomyDecision,
        *,
        tool: str,
        tenant: str,
        arguments: Mapping[str, Any],
        key: str | None,
    ) -> AutonomyDecision:
        """Take the headroom this call would use, before it is anybody's to use twice.

        Headroom is taken for a pending escalation too: the money a human is being asked
        about is not available to a parallel action while they are deciding. Where the
        ledger refuses, the ladder's answer was made against a window somebody else has
        since filled, and the call goes to a human rather than out.
        """
        amount = self._ladder.amount_in(tool, arguments)
        if self._commitments is None or key is None or decided.ceiling is None or amount is None:
            return decided
        if decided.outcome is AutonomyOutcome.REFUSE:
            return decided
        try:
            await self._commitments.reserve(
                tenant=tenant,
                action_class=decided.action_class,
                ceiling=decided.ceiling,
                amount=amount,
                idempotency_key=key,
            )
        except CeilingExceededError as refused:
            return decided.model_copy(
                update={"outcome": AutonomyOutcome.ESCALATE, "reason": str(refused)}
            )
        return decided

    async def settle(self, key: str, *, happened: bool) -> None:
        """Commit the headroom a call used, or give it back where the call did not happen."""
        if self._commitments is None:
            return
        if happened:
            await self._commitments.commit(key)
            return
        await self._commitments.release(key)

    def _still_granted(self, decided: AutonomyDecision) -> AutonomyDecision:
        """Refuse an unattended action whose authority is withdrawn, or unverifiable."""
        if self._revocations is None or not decided.unattended:
            return decided
        if not self._revocations.fresh():
            return self._refuse(decided, "the revocation view is stale and cannot be trusted")
        if decided.grant_id is None or self._revocations.withdrew(decided.grant_id) is None:
            return decided
        return self._refuse(decided, f"grant {decided.grant_id} was withdrawn")

    def withdrawn(self, decided: AutonomyDecision) -> Revocation | None:
        """The revocation that has since taken back the grant behind `decided`.

        Asked again after a human decides, because a run suspended on an approval was
        asleep while the authority behind it was withdrawn.
        """
        if self._revocations is None or decided.grant_id is None:
            return None
        return self._revocations.withdrew(decided.grant_id)

    def _refuse(self, decided: AutonomyDecision, reason: str) -> AutonomyDecision:
        """The same decision, turned into a refusal that names why."""
        return decided.model_copy(update={"outcome": AutonomyOutcome.REFUSE, "reason": reason})

    async def _owing(self, tenant: str, action_class: str | None) -> bool:
        """Whether a report is owed, or nothing where no log tracks them."""
        if self._reports is None or action_class is None:
            return False
        return await self._reports.outstanding(tenant=tenant, action_class=action_class)
