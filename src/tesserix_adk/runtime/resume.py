"""Carrying a run on days later, under a lease, with the transcript it actually had.

`Resumer` is the whole read path in one place: take the lease so only one worker advances
the run, read the frontier and refuse one this kit cannot read, resolve the transcript the
checkpoint points at and fail closed if it has gone, then plan what each outstanding call
needs. Nothing is dispatched until every one of those has held.

The masking a checkpoint gets before it is persisted lives with the writer, in
`tesserix_adk.runtime.checkpoint`.

The decisions behind this are in `docs/checkpointing.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core.errors import HistoryUnavailableError
from tesserix_adk.core.lease import DEFAULT_LEASE
from tesserix_adk.runtime.checkpoint import plan_resume
from tesserix_adk.runtime.lease import Leaseholder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.checkpoint import Checkpoint, ResumePlan
    from tesserix_adk.core.idempotency import IdempotencyStore
    from tesserix_adk.core.lease import LeasePolicy, LeaseStore, RunLease
    from tesserix_adk.core.primitives import Message
    from tesserix_adk.runtime.checkpoint import Checkpointer

__all__ = ["HistoryStore", "ResumedRun", "Resumer"]


@runtime_checkable
class HistoryStore(Protocol):
    """Where a transcript too large to carry inline is kept.

    A handle is only worth what the store behind it still holds, so the one call here may
    answer `None` — and a resume treats that as fatal rather than as an empty conversation.
    """

    async def fetch(self, handle: str, *, tenant: str) -> Sequence[Message] | None:
        """Return the transcript under `handle`, or `None` where it has gone."""
        ...


class ResumedRun:
    """A run taken over: the frontier, the transcript, the plan and the lease held on it.

    Args:
        checkpoint: The frontier, with its transcript restored.
        plan: What each outstanding call needs. Check `plan.safe` before acting on it.
        lease: The hold on the run. Release it when the turn ends.
    """

    def __init__(self, *, checkpoint: Checkpoint, plan: ResumePlan, lease: RunLease) -> None:
        self.checkpoint = checkpoint
        self.plan = plan
        self.lease = lease

    @property
    def iterations(self) -> int:
        """How many model calls the run had made before it stopped."""
        return self.checkpoint.iterations

    @property
    def awaiting_approval(self) -> str:
        """The approval request the run is waiting on, or empty where it is not waiting."""
        return self.checkpoint.pending_approval


class Resumer:
    """Takes a run over, in the order that makes taking it over safe.

    Args:
        checkpoints: The frontier store, already refusing formats this kit cannot read.
        leases: Where the one-holder-at-a-time decision is made.
        history: Where a transcript carried by handle is kept. Absent, a checkpoint with a
            handle cannot be resumed at all, which is the honest answer rather than a
            truncated one.
        idempotency: Where executed side effects are recorded, used to decide each
            outstanding call.
        policy: How long the hold lasts and when it is renewed.
    """

    def __init__(
        self,
        *,
        checkpoints: Checkpointer,
        leases: LeaseStore,
        history: HistoryStore | None = None,
        idempotency: IdempotencyStore | None = None,
        policy: LeasePolicy = DEFAULT_LEASE,
    ) -> None:
        self._checkpoints = checkpoints
        self._leases = leases
        self._history = history
        self._idempotency = idempotency
        self._policy = policy

    def holder(self, worker: str) -> Leaseholder:
        """A hold this worker can take, renew and release around a turn."""
        return Leaseholder(self._leases, holder=worker, policy=self._policy)

    async def resume(self, run_id: str, *, tenant: str, worker: str) -> ResumedRun | None:
        """Take `run_id` over, or return `None` because nothing was ever checkpointed.

        The lease is taken first. A run whose frontier is unreadable, or whose transcript
        has been evicted, has already cost the caller a lease it must release — which is
        why `holder` exists and why the failures below leave the run held rather than
        silently open to the next worker.

        Raises:
            RunLeaseError: If another worker holds the run.
            CheckpointFormatError: If the frontier was written by a newer kit.
            HistoryUnavailableError: If the transcript the frontier points at has gone.
        """
        holder = self.holder(worker)
        lease = await holder.acquire(run_id, tenant=tenant)
        checkpoint = await self._checkpoints.latest(run_id, tenant=tenant)
        if checkpoint is None:
            await holder.release()
            return None
        restored = await self._restored(checkpoint)
        return ResumedRun(
            checkpoint=restored,
            plan=await plan_resume(restored, self._idempotency),
            lease=lease,
        )

    async def _restored(self, checkpoint: Checkpoint) -> Checkpoint:
        """The checkpoint with its transcript back, or a refusal to carry on without it."""
        if not checkpoint.history_handle:
            return checkpoint
        messages = (
            None
            if self._history is None
            else await self._history.fetch(checkpoint.history_handle, tenant=checkpoint.tenant)
        )
        if messages is None:
            raise HistoryUnavailableError(
                f"{checkpoint.run_id} points at transcript {checkpoint.history_handle}, "
                f"which no longer resolves",
                run_id=checkpoint.run_id,
                handle=checkpoint.history_handle,
            )
        return checkpoint.model_copy(update={"messages": tuple(messages)})
