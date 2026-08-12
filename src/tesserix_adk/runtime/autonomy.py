"""The ladder as the loop consults it, at the point a tool call would go out.

`AutonomyLadder` decides about one action from what it is told. This is what tells it:
what the tenant has already committed against the class in the grant's window, and
whether a report an earlier action owed was ever delivered. Both are seams — the
arithmetic that makes commitments hard to game by splitting or retrying is its own
concern, and so is the shape of the audit record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core.autonomy import ActionRequest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from tesserix_adk.core.autonomy import AutonomyDecision, AutonomyLadder

__all__ = ["AutonomyGate", "InMemoryReports", "ReportLog"]


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
    """

    def __init__(self, ladder: AutonomyLadder, *, reports: ReportLog | None = None) -> None:
        self._ladder = ladder
        self._reports = reports

    async def decide(
        self,
        *,
        tool: str,
        tenant: str,
        arguments: Mapping[str, Any],
        run_id: str,
        user: str | None = None,
    ) -> AutonomyDecision:
        """What may happen about this call, and record the report acting on it owes."""
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
        if decided.unattended and decided.reports and self._reports is not None:
            await self._reports.owed(
                tenant=tenant, action_class=decided.action_class, run_id=run_id
            )
        return decided

    async def _owing(self, tenant: str, action_class: str | None) -> bool:
        """Whether a report is owed, or nothing where no log tracks them."""
        if self._reports is None or action_class is None:
            return False
        return await self._reports.outstanding(tenant=tenant, action_class=action_class)
