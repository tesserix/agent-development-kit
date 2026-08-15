"""Driving the unwind: what was applied, in what order it comes back off.

The engine is deliberately dull. It reads the ledger, walks it newest first, and asks each
paired reversal to take one action back. Nothing here consults a model, and nothing here
decides that an action probably ran — an action whose result never came back is queried
before it is reversed or retried, and where nothing can say, the run ends incomplete rather
than guessing in either direction.

Cancellation arriving mid-unwind does not abandon it. Half a rollback is worse than none,
because what is left applied is no longer written down anywhere a person is looking.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tesserix_adk.core.compensation import (
    Applied,
    CompensatedFailure,
    CompensationIncomplete,
    Reversal,
    ReversalOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.core.compensation import Compensating, CompensationLedger
    from tesserix_adk.core.idempotency import IdempotencyStore
    from tesserix_adk.core.protocols import Clock

__all__ = ["Compensator", "MemoryCompensationLedger"]


class MemoryCompensationLedger:
    """A `CompensationLedger` that lives and dies with the process.

    For tests and for a single-process consumer. A deployment whose worker can be replaced
    mid-run needs a durable one: what is not written down survives nothing, and the applied
    hotel booking is exactly what has to survive.
    """

    def __init__(self) -> None:
        self._applied: dict[tuple[str, str, str], Applied] = {}
        self._reversals: list[Reversal] = []

    async def record(self, action: Applied) -> None:
        """Write down that `action` was applied, keyed so a retried write records one."""
        self._applied[(action.tenant, action.run_id, action.idempotency_key)] = action

    async def mark(self, reversal: Reversal) -> None:
        """Record the attempt, and drop the action only where the world let it go."""
        self._reversals.append(reversal)
        if reversal.settled:
            action = reversal.applied
            self._applied.pop((action.tenant, action.run_id, action.idempotency_key), None)

    async def outstanding(
        self, run_id: str, *, tenant: str, branch: str | None = None
    ) -> Sequence[Applied]:
        """Everything still applied for this run, newest first, narrowed to a branch."""
        found = [
            action
            for (held_tenant, held_run, _), action in self._applied.items()
            if held_tenant == tenant
            and held_run == run_id
            and (branch is None or action.branch == branch)
        ]
        return tuple(sorted(found, key=lambda one: one.step, reverse=True))

    async def forget(self, *, tenant: str) -> int:
        """Remove every record for `tenant`, and return how many went."""
        going = [key for key in self._applied if key[0] == tenant]
        for key in going:
            del self._applied[key]
        self._reversals = [one for one in self._reversals if one.applied.tenant != tenant]
        return len(going)

    @property
    def attempts(self) -> Sequence[Reversal]:
        """Every reversal attempted, in the order they were made. For assertions and reports."""
        return tuple(self._reversals)


class Compensator:
    """Takes a run's applied work back, in reverse order, without asking a model anything.

    Args:
        ledger: Where the applied actions are.
        reversals: The reversal paired with each forward tool, by tool name. A tool with no
            entry cannot be taken back by this runtime, which is reported rather than
            assumed away.
        idempotency: Where an unknown outcome is queried before anything is reversed. Absent,
            an action whose result never came back is left outstanding rather than guessed at.
        clock: What stamps the attempts.
        attempts: How many times one reversal is tried before it is called failed.

    Example:
        >>> compensator = Compensator(MemoryCompensationLedger(), reversals={})
        >>> compensator.attempts
        3
    """

    def __init__(
        self,
        ledger: CompensationLedger,
        *,
        reversals: Mapping[str, Compensating],
        idempotency: IdempotencyStore | None = None,
        clock: Clock | None = None,
        attempts: int = 3,
    ) -> None:
        self._ledger = ledger
        self._reversals = dict(reversals)
        self._idempotency = idempotency
        self._clock = clock
        self.attempts = attempts

    async def unwind(
        self, run_id: str, *, tenant: str, cause: str, branch: str | None = None
    ) -> CompensatedFailure | CompensationIncomplete:
        """Take back everything this run applied, and say plainly whether it all came back.

        The walk is newest first and never stops at the first refusal: a supplier being
        down is not a reason to leave the other three bookings standing. Cancellation is
        held off until the walk finishes, because a half-finished unwind is work nobody is
        left holding a list of.

        Args:
            run_id: The run to unwind.
            tenant: Its isolation boundary.
            cause: What failed first, as the consumer rendered it.
            branch: One fan-out branch, or every branch where absent.

        Returns:
            A `CompensatedFailure` where the world holds nothing of this run any more, and
            a `CompensationIncomplete` — the alert-grade state — where it still does.
        """
        return await asyncio.shield(self._walk(run_id, tenant=tenant, cause=cause, branch=branch))

    async def _walk(
        self, run_id: str, *, tenant: str, cause: str, branch: str | None
    ) -> CompensatedFailure | CompensationIncomplete:
        """The unwind itself, run under a shield so cancellation cannot cut it in half."""
        made: list[Reversal] = []
        for action in await self._ledger.outstanding(run_id, tenant=tenant, branch=branch):
            reversal = await self._reverse(action)
            await self._ledger.mark(reversal)
            made.append(reversal)
        left = tuple(await self._ledger.outstanding(run_id, tenant=tenant, branch=branch))
        if left:
            return CompensationIncomplete(
                run_id=run_id,
                tenant=tenant,
                cause=cause,
                outstanding=left,
                reversals=tuple(made),
            )
        return CompensatedFailure(run_id=run_id, tenant=tenant, cause=cause, reversals=tuple(made))

    async def _reverse(self, action: Applied) -> Reversal:
        """One action, taken back or accounted for. Never raises: the walk has to finish."""
        if action.irreversible:
            return Reversal(
                applied=action,
                outcome=ReversalOutcome.REFUSED_IRREVERSIBLE,
                detail=f"{action.tool} cannot be taken back; reconcile by hand",
                attempts=0,
            )
        decided = await self._decided(action)
        if decided is None:
            return Reversal(
                applied=action,
                outcome=ReversalOutcome.UNKNOWN,
                detail=f"nothing can say whether {action.tool} ran",
                attempts=0,
            )
        reversal = self._reversals.get(action.tool)
        if reversal is None:
            return Reversal(
                applied=action,
                outcome=ReversalOutcome.FAILED,
                detail=f"no reversal is paired with {action.tool}",
                attempts=0,
            )
        return await self._tried(reversal, decided)

    async def _decided(self, action: Applied) -> Applied | None:
        """The action with its outcome settled, or `None` where nothing can settle it."""
        if action.outcome_known:
            return action
        if self._idempotency is None:
            return None
        claim = await self._idempotency.begin(
            action.idempotency_key, tenant=action.tenant, ttl_seconds=0.0
        )
        if claim.outcome is not None:
            return action.model_copy(update={"result_ref": claim.outcome})
        return None

    async def _tried(self, reversal: Compensating, action: Applied) -> Reversal:
        """Ask one reversal to take one action back, up to `attempts` times."""
        failure = ""
        for attempt in range(1, self.attempts + 1):
            try:
                await reversal.compensate(action)
            except Exception as refused:
                failure = str(refused) or type(refused).__name__
                continue
            return Reversal(applied=action, outcome=ReversalOutcome.REVERSED, attempts=attempt)
        return Reversal(
            applied=action,
            outcome=ReversalOutcome.FAILED,
            detail=failure,
            attempts=self.attempts,
        )
