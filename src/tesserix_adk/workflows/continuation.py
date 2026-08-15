"""Starting the workflow again with what it had, before its history grows too large to replay.

A run that iterates for an hour accumulates an event history the engine has to replay on every
worker that picks it up, and past a point that replay is slower than the run. Continue-as-new
ends the current execution and starts a fresh one carrying only what the next execution needs:
the transcript handle, the ledger, the iteration count, and — the part that is easy to drop —
the approval the run is waiting on and the grant it is acting under.

The frontier that goes across is a `Checkpoint`, the same shape the in-process resumer and
`adk inspect` read, so a run that continued as new is not a run that became unreadable.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a change
to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from tesserix_adk.core.checkpoint import Checkpoint, CheckpointBoundary
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.workflows.durable import (  # noqa: TC001 — pydantic needs the runtime types
    Journal,
    WorkflowState,
)

__all__ = ["DEFAULT_CONTINUATION", "Continuation", "ContinuationPolicy", "continued"]


class ContinuationPolicy(AdkModel):
    """When an execution has run long enough to be worth starting again.

    Args:
        max_steps: How many recorded activities one execution may accumulate. The journal
            is replayed in full on every worker that picks the run up.
        max_history_bytes: How large the engine's own event history may grow, where the
            deployment can measure it. Zero means only `max_steps` decides.

    Example:
        >>> DEFAULT_CONTINUATION.max_steps
        200
    """

    max_steps: int = Field(default=200, gt=0)
    max_history_bytes: int = Field(default=1_500_000, ge=0)

    def due(self, journal: Journal, *, history_bytes: int = 0) -> bool:
        """Whether this execution should end and carry its frontier into a fresh one.

        Example:
            >>> DEFAULT_CONTINUATION.due(Journal(), history_bytes=2_000_000)
            True
        """
        if journal.steps >= self.max_steps:
            return True
        return self.max_history_bytes > 0 and history_bytes >= self.max_history_bytes


DEFAULT_CONTINUATION = ContinuationPolicy()
"""What a deployment gets without choosing. Well inside a Temporal history's limits."""


class Continuation(AdkModel):
    """What crosses from one execution to the next, and nothing else.

    The journal deliberately does not cross: its steps are what the fresh execution is
    entitled to forget, and their results are already folded into the state.

    Args:
        state: Where the run is, as the next execution should start it.
        checkpoint: The same frontier, in the shape the resumer and the CLI read.
    """

    state: WorkflowState
    checkpoint: Checkpoint

    @model_validator(mode="after")
    def _the_approval_binding_survives(self) -> Continuation:
        """Refuse a continuation that dropped what the run is waiting on.

        A run that continued as new without its approval token is a run that will either
        wait forever or act on an approval nobody gave.
        """
        if self.state.pending_approval != self.checkpoint.pending_approval:
            raise ConfigurationError(
                f"{self.state.run_id} is waiting on approval"
                f" {self.state.pending_approval!r} and its checkpoint carries"
                f" {self.checkpoint.pending_approval!r}"
            )
        if self.state.grant != self.checkpoint.grant_id:
            raise ConfigurationError(
                f"{self.state.run_id} is acting under grant {self.state.grant!r} and its"
                f" checkpoint carries {self.checkpoint.grant_id!r}"
            )
        return self


def continued(
    state: WorkflowState,
    *,
    tenant: str,
    agent_name: str,
    model: str = "",
    user: str = "",
    scopes: tuple[str, ...] = (),
) -> Continuation:
    """Build the frontier a fresh execution starts from.

    The transcript does not travel: it is already behind `state.history`, and a handle is
    what a payload limit leaves room for. Whoever resumes resolves it, and fails closed if
    it has been evicted.

    Args:
        state: Where the run got to in the execution that is ending.
        tenant: The isolation boundary, which a checkpoint never infers.
        agent_name: Which agent the run is, for whoever reads the frontier later.
        model: Which model it was calling, recorded so a resume does not silently change it.
        user: Who the run is for, carried so attribution survives the continuation.
        scopes: What the run was authorised to do.

    Returns:
        The state to start the next execution with, and the checkpoint that describes it.

    Raises:
        ConfigurationError: If the frontier would lose the approval or grant binding.

    Example:
        >>> from tesserix_adk.workflows import WorkflowState
        >>> state = WorkflowState(run_id="r1", history="h-9", iteration=4)
        >>> carried = continued(state, tenant="acme", agent_name="booking")
        >>> carried.checkpoint.history_handle, carried.state.iteration
        ('h-9', 4)
    """
    frontier = Checkpoint(
        run_id=state.run_id,
        tenant=tenant,
        agent_name=agent_name,
        model=model,
        boundary=(
            CheckpointBoundary.BEFORE_APPROVAL
            if state.pending_approval
            else CheckpointBoundary.AFTER_MODEL_CALL
        ),
        history_handle=state.history,
        usage=state.usage,
        iterations=state.iteration,
        pending_approval=state.pending_approval,
        grant_id=state.grant,
        scopes=scopes,
        user=user,
    )
    return Continuation(state=state, checkpoint=frontier)
