"""Writing a run's frontier down, and working out what carrying it on would involve.

`Checkpointer` is the write side: it applies the policy, refuses a payload over the cap
rather than truncating one, and keeps the latest frontier per run. `plan_resume` is the
read side: it asks the idempotency store, per outstanding call, whether that call ran, and
returns a plan that either is safe to act on or names what nobody can decide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.checkpoint import (
    CHECKPOINT_FORMAT,
    Checkpoint,
    CheckpointPolicy,
    ResumePlan,
    Resumption,
    ToolDisposition,
)
from tesserix_adk.core.errors import (
    CheckpointFormatError,
    CheckpointTooLargeError,
    IndeterminateToolCallError,
    ResumeConflictError,
)

if TYPE_CHECKING:
    from tesserix_adk.core.checkpoint import CheckpointStore, PendingCall
    from tesserix_adk.core.idempotency import IdempotencyStore
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "Checkpointer",
    "MemoryCheckpointStore",
    "claim_resume",
    "plan_resume",
    "refuse_if_undecidable",
]

RESUME_TTL_SECONDS = 300.0
"""How long a resume claim is held. Long enough for a turn, short enough to free a dead worker's."""


class MemoryCheckpointStore:
    """A `CheckpointStore` in a dict, keyed by tenant and run.

    Nothing here outlives the process, which is what surviving a restart is not. It exists
    so resume can be exercised without a database.
    """

    def __init__(self) -> None:
        self._held: dict[tuple[str, str], Checkpoint] = {}

    async def put(self, checkpoint: Checkpoint) -> None:
        """Store the frontier, replacing any earlier one for the run."""
        self._held[checkpoint.tenant, checkpoint.run_id] = checkpoint

    async def latest(self, run_id: str, *, tenant: str) -> Checkpoint | None:
        """Return the frontier of `run_id`, or `None` where nothing was written."""
        return self._held.get((tenant, run_id))

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop the frontier. A completed run has nothing left to resume."""
        self._held.pop((tenant, run_id), None)


class Checkpointer:
    """Writes a run's frontier at the boundaries a policy names.

    A checkpoint the loop could not write is not a failure of the run: the run carries on
    uncheckpointed and the caller finds out from `last_error`, because losing the ability
    to resume is worth knowing about and is not worth failing a live run over.

    Args:
        store: Where frontiers are kept.
        policy: When one is written, and how large it may be.
        clock: What stamps it. Absent, a checkpoint says nothing about when — which is
            fine, since nothing orders by it.
    """

    def __init__(
        self,
        store: CheckpointStore,
        policy: CheckpointPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._store = store
        self._policy = policy or CheckpointPolicy()
        self._clock = clock
        self.last_error: Exception | None = None

    @property
    def policy(self) -> CheckpointPolicy:
        """When this writes, and how much it will write."""
        return self._policy

    async def record(self, checkpoint: Checkpoint) -> bool:
        """Write `checkpoint` where the policy calls for it, and say whether it was.

        A payload over the cap is refused rather than truncated: half a frontier resumes
        into a conversation that never happened, and nothing downstream could tell.
        """
        self.last_error = None
        if not self._policy.writes_at(checkpoint.boundary):
            return False
        stamped = checkpoint.model_copy(update={"created_at": self._now()})
        size = stamped.size_bytes
        if size > self._policy.max_bytes:
            self.last_error = CheckpointTooLargeError(
                f"{checkpoint.run_id} would take {size} bytes, "
                f"over the {self._policy.max_bytes} cap",
                run_id=checkpoint.run_id,
                size_bytes=size,
                max_bytes=self._policy.max_bytes,
            )
            return False
        try:
            await self._store.put(stamped)
        except Exception as failed:  # an unwritable checkpoint never fails a live run
            self.last_error = failed
            return False
        return True

    async def latest(self, run_id: str, *, tenant: str) -> Checkpoint | None:
        """Return the frontier of `run_id`, refusing one this version cannot read.

        Raises:
            CheckpointFormatError: If it was written by a newer kit. Reading fields it
                would have to guess at is how a resume replays a call that already ran.
        """
        checkpoint = await self._store.latest(run_id, tenant=tenant)
        if checkpoint is not None and not checkpoint.resumable_by(CHECKPOINT_FORMAT):
            raise CheckpointFormatError(
                f"{run_id} was checkpointed at format {checkpoint.format_version}, "
                f"and this kit reads {CHECKPOINT_FORMAT}",
                run_id=run_id,
                format_version=checkpoint.format_version,
                readable_version=CHECKPOINT_FORMAT,
            )
        return checkpoint

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop the frontier, because the run reached a terminal state."""
        await self._store.forget(run_id, tenant=tenant)

    def _now(self) -> float:
        return 0.0 if self._clock is None else self._clock.now()


async def plan_resume(
    checkpoint: Checkpoint, idempotency: IdempotencyStore | None = None
) -> ResumePlan:
    """Work out what carrying `checkpoint` on would involve, without doing any of it.

    Each outstanding call is put to the idempotency store: a recorded outcome means it
    finished and is replayed rather than re-executed, a key nobody holds means it never
    started, and a key held in flight by a process that is gone means nobody can say.

    A call the loop dispatched with no idempotency key is indeterminate too. The absence
    of a key is the absence of a guarantee, not permission to run it again.

    Args:
        checkpoint: The frontier to plan from.
        idempotency: Where executed side effects are recorded. Absent, every dispatched
            call is indeterminate, because nothing was ever in a position to know.

    Returns:
        The plan. Check `safe` before acting on it.
    """
    resumptions = [
        await _resolved(pending, checkpoint.tenant, idempotency) for pending in checkpoint.pending
    ]
    return ResumePlan(checkpoint=checkpoint, resumptions=tuple(resumptions))


async def claim_resume(
    run_id: str, *, tenant: str, idempotency: IdempotencyStore, ttl_seconds: float | None = None
) -> None:
    """Take exclusive charge of resuming `run_id`, or refuse because someone else has.

    Two workers resuming one run is two runs spending one budget and dispatching each
    outstanding call twice. The claim is the same at-most-once machinery a tool call uses,
    held for long enough to finish a turn and no longer, so a worker that dies does not
    strand the run forever.

    Raises:
        ResumeConflictError: If another worker holds the run.
    """
    claim = await idempotency.begin(
        f"resume:{run_id}", tenant=tenant, ttl_seconds=ttl_seconds or RESUME_TTL_SECONDS
    )
    if claim.in_flight or claim.outcome is not None:
        raise ResumeConflictError(f"{run_id} is already being carried on", run_id=run_id)


def refuse_if_undecidable(plan: ResumePlan) -> None:
    """Stop before a resume that would have to guess whether a call already happened.

    Raises:
        IndeterminateToolCallError: Naming every call nobody can decide.
    """
    if plan.safe:
        return
    calls = tuple(one.call.name for one in plan.indeterminate)
    raise IndeterminateToolCallError(
        f"{plan.checkpoint.run_id} cannot be carried on: "
        f"nothing can say whether {', '.join(calls)} ran",
        run_id=plan.checkpoint.run_id,
        calls=calls,
    )


async def _resolved(
    pending: PendingCall, tenant: str, idempotency: IdempotencyStore | None
) -> Resumption:
    """What the record says about one outstanding call."""
    if not pending.dispatched:
        return Resumption(call=pending.call, disposition=ToolDisposition.NEVER_RAN)
    if idempotency is None or pending.idempotency_key is None:
        return Resumption(call=pending.call, disposition=ToolDisposition.INDETERMINATE)
    claim = await idempotency.begin(
        pending.idempotency_key, tenant=tenant, ttl_seconds=RESUME_TTL_SECONDS
    )
    if claim.outcome is not None:
        return Resumption(
            call=pending.call, disposition=ToolDisposition.COMPLETED, outcome=claim.outcome
        )
    if claim.in_flight:
        return Resumption(call=pending.call, disposition=ToolDisposition.INDETERMINATE)
    await idempotency.abandon(pending.idempotency_key, tenant=tenant)
    return Resumption(call=pending.call, disposition=ToolDisposition.NEVER_RAN)
