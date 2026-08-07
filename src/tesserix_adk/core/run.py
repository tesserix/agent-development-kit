"""A run: what an agent did, what it cost, and how it ended.

A run is data. It holds no provider client, no socket and no callable, because it is
checkpointed mid-flight and rehydrated by a different process. State changes return a new
run rather than mutating this one, so a checkpoint taken before a step still describes
that step.

The state machine is explicit. Every terminal state says *why* the run ended — a budget
ceiling and a provider outage are different failures and belong in different buckets —
and a transition nobody declared legal is refused at the call, not logged and continued.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/primitives.md`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, Self

from pydantic import Field

from tesserix_adk.core.models import AdkModel, OutputT
from tesserix_adk.core.primitives import Message, ToolCall, Usage

__all__ = [
    "Run",
    "RunContext",
    "RunEvent",
    "RunEventKind",
    "RunState",
    "TenantContext",
    "legal_transitions",
]


class RunState(StrEnum):
    """Where a run is, and if it is over, why.

    A string enum so a checkpoint, a log line and a span attribute all carry the same
    readable value.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    LOOP_LIMIT_EXCEEDED = "loop_limit_exceeded"

    @property
    def is_terminal(self) -> bool:
        """Whether the run is over. A terminal run is never reopened."""
        return self in _TERMINAL


_TERMINAL = frozenset(
    {
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.CANCELLED,
        RunState.BUDGET_EXHAUSTED,
        RunState.MAX_ITERATIONS_EXCEEDED,
        RunState.LOOP_LIMIT_EXCEEDED,
    }
)

# A run that never started cannot have exhausted a budget or run out of iterations.
_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.RUNNING: _TERMINAL,
}


def legal_transitions(state: RunState) -> frozenset[RunState]:
    """The states a run in `state` may move to. Empty for every terminal state.

    Example:
        >>> legal_transitions(RunState.PENDING) == frozenset(
        ...     {RunState.RUNNING, RunState.CANCELLED}
        ... )
        True
    """
    return _TRANSITIONS.get(state, frozenset())


class RunEventKind(StrEnum):
    """What happened. One kind per thing the loop does, so a filter is exact.

    A single `step` kind with the detail in a string reads fine in a log and is useless
    to anything that has to count tool failures or total the cost of model calls.
    """

    MODEL_ROUTED = "model_routed"
    PROMPT_ASSEMBLED = "prompt_assembled"
    MODEL_CALL = "model_call"
    MODEL_RESPONSE = "model_response"
    ATTEMPT_FAILED = "attempt_failed"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_RESULT_TRUNCATED = "tool_result_truncated"
    TOOL_ERROR = "tool_error"
    TOOL_REFUSED = "tool_refused"
    TOOL_INDETERMINATE = "tool_indeterminate"
    FAN_OUT_REFUSED = "fan_out_refused"
    REPEAT_DETECTED = "repeat_detected"
    DEPTH_EXCEEDED = "depth_exceeded"
    HOOK_REWRITE = "hook_rewrite"
    HOOK_REFUSAL = "hook_refusal"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    GUARDRAIL_REFUSAL = "guardrail_refusal"
    OUTPUT_UNWRAPPED = "output_unwrapped"
    OUTPUT_VALIDATED = "output_validated"
    SCHEMA_VIOLATION = "schema_violation"
    REPAIR_REQUESTED = "repair_requested"
    REPAIR_ABANDONED = "repair_abandoned"
    CANCELLATION_REQUESTED = "cancellation_requested"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    WORK_ORPHANED = "work_orphaned"
    TERMINATED = "terminated"


class RunEvent(AdkModel):
    """One thing that happened during a run, in the order it happened.

    Args:
        kind: What happened. See `RunEventKind`.
        name: What it happened to — the model, the tool, the guardrail.
        detail: A short human-readable note. Never message content, never credentials.
        at: Unix seconds, where the caller has a clock.
        usage: What this step consumed, for the steps that consume anything. Cost
            attribution totals these rather than re-deriving spend per product.
    """

    kind: RunEventKind
    name: str | None = None
    detail: str | None = None
    at: float | None = None
    usage: Usage | None = None


class TenantContext(AdkModel):
    """Who a run belongs to.

    Args:
        tenant: The isolation boundary. Cost attribution and access both key off it.
        user: The acting principal within the tenant, where there is one.
    """

    tenant: str = Field(min_length=1)
    user: str | None = None


class RunContext(AdkModel):
    """What the runtime threads through every layer of a run.

    A tool never receives the tenant as an argument a caller might forget: the runtime
    carries this, so identity cannot be dropped between the agent and the tool.
    """

    run_id: str = Field(min_length=1)
    tenant: TenantContext
    depth: int = Field(default=0, ge=0)


class Run(AdkModel, Generic[OutputT]):  # noqa: UP046 — PEP 695 syntax cannot carry the parameter's default before 3.13
    """One execution of an agent, from prompt assembly to a terminal state.

    Args:
        id: Identity, carried by every event, span and audit record for this run.
        tenant: The isolation boundary. There is no default; a run with no tenant cannot
            be attributed or isolated.
        user: The acting principal, where there is one.
        agent_name: Which agent ran.
        agent_version: Which version of it, so a behaviour change is attributable.
        model: The model actually used, not the one requested.
        prompt_version: Which prompt produced this run, where prompts are versioned.
        depth: How far down a chain of agents calling agents this run sits. Zero is a run
            nobody called.
        state: Where the run is. See `RunState`.
        messages: The conversation as it stands.
        tool_calls: Calls the model requested, deduplicated by id.
        events: What happened, in order. The record the run loop writes and cost
            attribution, tracing and audit all read.
        output: The validated answer, as an instance of the type the agent declared. A
            checkpoint rehydrates through `Run[TripPlan].model_validate_json`, which is
            where the parameter earns its keep: the wrong payload does not fit.
        usage: What the run has consumed so far.
        started_at: Unix seconds at the transition into `RUNNING`.
        ended_at: Unix seconds at the transition into a terminal state.
    """

    id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    user: str | None = None
    agent_name: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str | None = None
    depth: int = Field(default=0, ge=0)
    state: RunState = RunState.PENDING
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    output: OutputT | None = None
    usage: Usage = Field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def context(self) -> RunContext:
        """The identity this run threads through every layer."""
        return RunContext(
            run_id=self.id,
            tenant=TenantContext(tenant=self.tenant, user=self.user),
            depth=self.depth,
        )

    @property
    def input_tokens_spent(self) -> int:
        """Input tokens consumed so far. A convenience over `usage`, for budget checks."""
        return self.usage.input_tokens

    def transition_to(self, state: RunState, *, at: float | None = None) -> Self:
        """Return this run in `state`.

        Args:
            state: The state to move to.
            at: Unix seconds for the transition, where the caller has a clock.

        Raises:
            ValueError: If the run is already terminal, is already in `state`, or the
                move is not one the transition table declares legal.

        Example:
            >>> run = Run(
            ...     id="run_1",
            ...     tenant="acme",
            ...     agent_name="planner",
            ...     agent_version="1.0.0",
            ...     model="claude-sonnet-5",
            ... )
            >>> run.transition_to(RunState.RUNNING).state is RunState.RUNNING
            True
            >>> run.state is RunState.PENDING
            True
        """
        if self.state.is_terminal:
            raise ValueError(
                f"run {self.id} is terminal ({self.state}); reopening it would make its "
                f"own audit trail a lie"
            )
        if state is self.state:
            raise ValueError(f"run {self.id} is already {self.state}")
        allowed = legal_transitions(self.state)
        if state not in allowed:
            legal = ", ".join(sorted(allowed))
            raise ValueError(f"run {self.id} cannot go {self.state} -> {state}; legal: {legal}")

        # Every legal target is either RUNNING or terminal, so there is no third case.
        timings: dict[str, float | None] = (
            {"started_at": at} if state is RunState.RUNNING else {"ended_at": at}
        )
        return self.model_copy(update={"state": state, **timings})

    def record(self, usage: Usage) -> Self:
        """Return this run with `usage` added to its total."""
        return self.model_copy(update={"usage": self.usage + usage})

    def record_event(self, event: RunEvent) -> Self:
        """Return this run with `event` appended. Order is the record; nothing reorders."""
        return self.model_copy(update={"events": [*self.events, event]})

    def with_output(self, output: OutputT) -> Self:
        """Return this run carrying `output` as its validated answer, typed as declared."""
        return self.model_copy(update={"output": output})
