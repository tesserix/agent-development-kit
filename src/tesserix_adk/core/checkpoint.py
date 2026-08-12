"""What a run has to have written down for another process to carry it on.

A run that dies at iteration nine restarts at zero, repeating every model call and every
tool call it had already made. The cost is one problem; the second booking is the other,
because replaying an effectful call books the same thing twice.

So the loop writes a checkpoint at boundaries where the frontier is unambiguous — a model
answer received, a tool result recorded, an approval about to be waited on — and a resume
reads it and asks, per outstanding call, whether that call ran. The answer comes from the
idempotency store rather than from a guess: a call whose outcome was recorded is finished,
a call nothing holds never started, and a call held in flight by a process that is gone is
`indeterminate`. The kit refuses to decide that last one. Deciding it wrong is exactly the
duplicate booking checkpointing exists to prevent.

The decisions behind these types are in `docs/checkpointing.md`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import Message, ToolCall, Usage

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "CHECKPOINT_FORMAT",
    "Checkpoint",
    "CheckpointBoundary",
    "CheckpointPolicy",
    "CheckpointStore",
    "PendingCall",
    "ResumePlan",
    "Resumption",
    "ToolDisposition",
]

CHECKPOINT_FORMAT = 1
"""The payload version a checkpoint is written at. A reader refuses anything it postdates."""


class CheckpointBoundary(StrEnum):
    """Where in a turn a checkpoint may be taken.

    Each is a point where what has happened and what has not is unambiguous. Anywhere
    else — mid dispatch, mid stream — a checkpoint would record a frontier that never
    existed, and a resume from it would repeat or skip work.
    """

    AFTER_MODEL_CALL = "after_model_call"
    AFTER_TOOL_RESULT = "after_tool_result"
    BEFORE_APPROVAL = "before_approval"


class ToolDisposition(StrEnum):
    """What a resume concluded about one outstanding call.

    `COMPLETED` and `NEVER_RAN` are both safe: replay what was recorded, or dispatch. Only
    `INDETERMINATE` needs a person or the tool's own status endpoint, and it is never
    resolved by retrying — a retry is the second booking.
    """

    COMPLETED = "completed"
    NEVER_RAN = "never_ran"
    INDETERMINATE = "indeterminate"


class PendingCall(AdkModel):
    """A call the model asked for that the checkpoint could not say had finished.

    Args:
        call: What was asked for.
        idempotency_key: The key identifying the side effect, where the tool declared one.
            `None` where the tool is read-only, or where a key argument was absent — and
            the absence of a key is the absence of a guarantee, not permission to retry.
        dispatched: Whether the loop had handed it to the tool before it died.
    """

    call: ToolCall
    idempotency_key: str | None = None
    dispatched: bool = False


class Checkpoint(AdkModel):
    """The frontier of a run, far enough along that another process can carry it on.

    Args:
        run_id: Which run.
        tenant: The isolation boundary.
        agent_name: Which agent was running.
        model: The model the turn was answered by. A resume records it again, so a run
            that finished on a different model says so rather than reading as one run.
        boundary: Where in the turn this was taken.
        format_version: The payload version. A reader refuses one it postdates rather
            than reading fields it has to guess at.
        messages: The conversation as the loop held it, including the assembled prompt.
        pending: Calls outstanding at the boundary. Empty at every boundary but a dispatch
            that did not finish.
        usage: What the run had spent.
        cost_micros: What that cost, in millionths.
        iterations: How many model calls it had made.
        guardrails_passed: Guards that had already returned a verdict on this turn, so a
            resume does not pay for them twice.
        agent_revision: The pinned definition revision, where the run had one.
        prompt_version: The prompt the messages were assembled from.
        created_at: Unix seconds it was written.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    model: str = ""
    boundary: CheckpointBoundary = CheckpointBoundary.AFTER_MODEL_CALL
    format_version: int = CHECKPOINT_FORMAT
    messages: tuple[Message, ...] = ()
    pending: tuple[PendingCall, ...] = ()
    usage: Usage = Field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    cost_micros: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    guardrails_passed: tuple[str, ...] = ()
    agent_revision: str | None = None
    prompt_version: str = ""
    created_at: float = 0.0

    @property
    def size_bytes(self) -> int:
        """How much a store would have to hold, measured the way it will be written."""
        return len(self.model_dump_json().encode())

    def resumable_by(self, format_version: int = CHECKPOINT_FORMAT) -> bool:
        """Whether a reader at `format_version` can read this without guessing."""
        return self.format_version <= format_version


class CheckpointPolicy(AdkModel):
    """When a run is written down, and how much of it may be.

    Args:
        boundaries: Which boundaries are written. Fewer means less write amplification and
            more repeated work on resume; the default writes all three.
        max_bytes: The ceiling on one payload. A run whose conversation outgrows it stops
            being checkpointed rather than silently truncated — half a frontier resumes
            into a conversation that never happened.
        ttl_seconds: How long a checkpoint stays useful. A resume from a very old
            checkpoint is a resume into a world that has moved.

    Example:
        >>> CheckpointPolicy().writes_at(CheckpointBoundary.AFTER_TOOL_RESULT)
        True
    """

    boundaries: frozenset[CheckpointBoundary] = frozenset(CheckpointBoundary)
    max_bytes: int = Field(default=1_048_576, ge=1)
    ttl_seconds: float = Field(default=86_400.0, gt=0)

    def writes_at(self, boundary: CheckpointBoundary) -> bool:
        """Whether a checkpoint is taken at `boundary`."""
        return boundary in self.boundaries


class Resumption(AdkModel):
    """What a resume concluded about one outstanding call.

    Args:
        call: The call in question.
        disposition: What the idempotency record said about it.
        outcome: What the completed call recorded, where it completed. Replayed into the
            conversation rather than re-executed.
    """

    call: ToolCall
    disposition: ToolDisposition
    outcome: str | None = None

    @model_validator(mode="after")
    def _refuse_a_completion_with_nothing_to_replay(self) -> Self:
        """A completed call whose outcome was lost is indeterminate, not completed."""
        if self.disposition is ToolDisposition.COMPLETED and self.outcome is None:
            raise ValueError("a completed call must carry what it recorded")
        return self


class ResumePlan(AdkModel):
    """What carrying a run on would involve, decided before anything is carried on.

    Args:
        checkpoint: What it was planned from.
        resumptions: One per outstanding call, in the order the model asked for them.
    """

    checkpoint: Checkpoint
    resumptions: tuple[Resumption, ...] = ()

    @property
    def indeterminate(self) -> tuple[Resumption, ...]:
        """Calls nobody can say ran or not. A plan with any of these is not safe."""
        return self._with(ToolDisposition.INDETERMINATE)

    @property
    def completed(self) -> tuple[Resumption, ...]:
        """Calls that ran, whose recorded outcome is replayed rather than re-executed."""
        return self._with(ToolDisposition.COMPLETED)

    @property
    def to_dispatch(self) -> tuple[Resumption, ...]:
        """Calls that never ran, which the resumed run makes for the first time."""
        return self._with(ToolDisposition.NEVER_RAN)

    @property
    def safe(self) -> bool:
        """Whether the run can be carried on without anyone deciding anything."""
        return not self.indeterminate

    def _with(self, disposition: ToolDisposition) -> tuple[Resumption, ...]:
        return tuple(one for one in self.resumptions if one.disposition is disposition)


@runtime_checkable
class CheckpointStore(Protocol):
    """Where the frontier of a run is kept between processes.

    One checkpoint per run: a store keeps the latest and nothing else, because an older
    frontier is work already done and resuming from it repeats it. Writes are last-write
    wins by design — unlike run state, a checkpoint has no contended field, and two
    workers writing one are two views of the same monotonic progress.
    """

    async def put(self, checkpoint: Checkpoint) -> None:
        """Store the frontier, replacing any earlier one for the run.

        Raises:
            CheckpointTooLargeError: If the payload exceeds what the store will hold.
            StatePersistenceError: If the store could not take the write.
        """
        ...

    async def latest(self, run_id: str, *, tenant: str) -> Checkpoint | None:
        """Return the frontier of `run_id`, or `None` where nothing was written."""
        ...

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop the frontier. A completed run has nothing left to resume."""
        ...
