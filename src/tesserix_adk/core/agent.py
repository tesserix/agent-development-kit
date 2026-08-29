"""What an agent is, as a declaration rather than a running thing.

An `Agent` describes the job: which model or task class does it, what it is told, which
tools it may call, what shape its answer takes, what it may spend and what it is checked
against. It performs no I/O and holds no provider client, so the same declaration can be
built in a test, written into a config file and read in review.

Running one is the runtime's job. Keeping the two apart is what makes an agent
substitutable and a run reproducible.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/primitives.md`.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Generic, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tesserix_adk.core.budget import BudgetLimits  # noqa: TC001 — pydantic needs it at runtime
from tesserix_adk.core.capabilities import CapabilitySet  # noqa: TC001
from tesserix_adk.core.config import (  # noqa: TC001
    ConcurrencyConfig,
    DeadlineConfig,
    LoopConfig,
    RepairConfig,
    RetryConfig,
)

# Runtime import, not type-checking only: pydantic resolves the annotation at class creation.
from tesserix_adk.core.errors import SchemaViolationError
from tesserix_adk.core.models import AdkModel, InputT, OutputT
from tesserix_adk.core.prompts import PromptRef  # noqa: TC001

__all__ = ["Agent", "ToolFailurePolicy", "TypedAgent"]


class ToolFailurePolicy(StrEnum):
    """What a run does when a tool it called fails.

    Neither answer is right everywhere: a model told its search failed can try another
    route, but where the tool was the source of truth, continuing produces a confident
    wrong answer. The agent declares which case it is.
    """

    SURFACE_TO_MODEL = "surface_to_model"
    FAIL_RUN = "fail_run"


class Agent(AdkModel, Generic[OutputT]):  # noqa: UP046 — the parameter carries a default
    """A declarative description of an agent.

    Args:
        name: How the agent is identified in runs, traces and cost attribution.
        version: Which revision of the declaration this is, so a behaviour change is
            attributable to it.
        instructions: The system instruction.
        prompt: Which registered prompt the instructions came from, where they came from
            one. Carried onto the run and every model-call span, so a behaviour change is
            attributable to a prompt version rather than to a deployment date.
        model: The model to use. Mutually exclusive with `task_class`.
        task_class: The kind of work, for a router to resolve to a model. Naming the job
            rather than the model is what lets the routing decision live in one place.
        requires: What this agent needs of whatever model answers, for a router to filter
            candidates by. A run is refused rather than routed to a model that cannot do
            the work, because discovering that from the answer is discovering it late.
        tools: Tool names this agent may call. Empty by default — a tool a consumer did
            not name is a tool the agent may not call.
        idempotent_tools: Which of those tools are safe to call again. A tool cancelled
            after dispatch is reported indeterminate unless it appears here, because the
            kit cannot know whether its side effect landed. Must be a subset of `tools`.
        approval_required_tools: Which of those tools may not be dispatched without a
            human decision. Must be a subset of `tools`.
        scopes: The authority this agent needs, as scope names. What it actually holds is
            this intersected with what the caller holds, resolved once at run start. An
            agent that declares none holds none and is enforced exactly as before.
        tool_scopes: Which of `scopes` each tool needs. A tool whose scopes the caller
            does not hold is not declared to the model and is refused at dispatch, so the
            escalation cannot be attempted rather than being caught downstream.
        output_type: The type the answer must validate against. A Python type, so it is
            excluded from serialisation rather than rendered as a string that cannot be
            read back. Exactly one of this and `free_text` is declared.
        free_text: That this agent answers in prose on purpose. Declared, never reached by
            omission: an agent that forgot to say what shape its answer takes would
            otherwise hand the caller a string to guess at.
        budget: The ceiling for one run.
        deadlines: Wall-clock ceilings for the run and its steps. Unbounded by default.
        loop: Caps on the shape of the run — depth, fan-out, repetition. Narrows what the
            runner allows and never widens it, so an agent cannot vote itself more rope.
        concurrency: How wide this agent's tool batches may run, and what any one tool may
            take. Narrows the runner's lanes and never widens them. Per-tenant and
            per-tool lanes bound a downstream shared with every other run, so those are
            the runner's to declare; what an agent narrows, it narrows for its own turn.
        retry: Which failures are worth a second attempt, and how long to wait first.
            Nothing is retried by default.
        repair: Whether an answer that failed validation is sent back with the reason, and
            how many times. Nothing is repaired by default: a further model call is a
            further charge, so it is asked for rather than assumed.
        on_tool_error: What happens when a tool fails. Surfaced to the model by default,
            so a run does not die on the first recoverable failure.
        guardrails: Guardrail names, applied in order.
        metadata: Consumer-owned annotations. The kit reads nothing from it.

    Example:
        >>> Agent(
        ...     name="planner",
        ...     instructions="Plan trips.",
        ...     model="claude-sonnet-5",
        ...     free_text=True,
        ... ).tools
        ()
    """

    # An output type is a class, which pydantic will not accept as a value without this.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    version: str = "1.0.0"
    instructions: str = Field(min_length=1)
    prompt: PromptRef | None = None
    model: str | None = None
    task_class: str | None = None
    requires: CapabilitySet = frozenset()
    tools: tuple[str, ...] = ()
    idempotent_tools: tuple[str, ...] = ()
    approval_required_tools: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    tool_scopes: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    output_type: type[OutputT] | None = Field(default=None, exclude=True)
    free_text: bool = False
    budget: BudgetLimits | None = None
    deadlines: DeadlineConfig | None = None
    loop: LoopConfig | None = None
    concurrency: ConcurrencyConfig | None = None
    retry: RetryConfig | None = None
    repair: RepairConfig | None = None
    on_tool_error: ToolFailurePolicy = ToolFailurePolicy.SURFACE_TO_MODEL
    guardrails: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_way_to_choose_a_model(self) -> Self:
        if bool(self.model) == bool(self.task_class):
            raise ValueError(
                "declare exactly one of model or task_class: two answers to which model "
                "runs is a routing decision made twice, and none is a run that cannot start"
            )
        return self

    @model_validator(mode="after")
    def _the_shape_of_the_answer_is_declared(self) -> Self:
        if (self.output_type is None) == (not self.free_text):
            raise ValueError(
                "declare exactly one of output_type or free_text: an answer whose shape "
                "nobody declared is prose the caller has to parse by guessing"
            )
        return self

    @model_validator(mode="after")
    def _approval_is_required_of_a_declared_tool(self) -> Self:
        stray = sorted(set(self.approval_required_tools) - set(self.tools))
        if stray:
            raise ValueError(
                f"approval required but not on the allowlist: {', '.join(stray)}. A gate "
                f"in front of a tool the agent cannot call gates nothing"
            )
        return self

    @model_validator(mode="after")
    def _scopes_are_required_of_declared_tools(self) -> Self:
        stray = sorted(set(self.tool_scopes) - set(self.tools))
        if stray:
            raise ValueError(
                f"scopes required of tools not on the allowlist: {', '.join(stray)}. "
                f"A requirement on a tool the agent cannot call requires nothing"
            )
        undeclared = sorted(
            {scope for needed in self.tool_scopes.values() for scope in needed} - set(self.scopes)
        )
        if undeclared:
            raise ValueError(
                f"tool needs a scope this agent does not declare: {', '.join(undeclared)}. "
                f"An undeclared scope is never granted, so the tool could never be called"
            )
        return self

    @model_validator(mode="after")
    def _tools_are_named_once(self) -> Self:
        seen = {name for name in self.tools if self.tools.count(name) > 1}
        if seen:
            raise ValueError(
                f"tool named more than once: {', '.join(sorted(seen))}. One of the "
                f"entries was meant to be a different tool"
            )
        return self

    @model_validator(mode="after")
    def _idempotency_is_declared_of_a_declared_tool(self) -> Self:
        stray = sorted(set(self.idempotent_tools) - set(self.tools))
        if stray:
            raise ValueError(
                f"declared idempotent but not on the allowlist: {', '.join(stray)}. "
                f"A retry policy for a tool the agent cannot call protects nothing"
            )
        return self


class TypedAgent(Agent[OutputT], Generic[InputT, OutputT]):  # noqa: UP046
    """An agent with an application input type in addition to its output type.

    `Agent[OutputT]` remains the stable string-input contract. Use this additive type
    with `AgentRunner.run_typed` or `AgentRunner.stream_typed` when an application hands
    the runtime a Pydantic model instead of text.
    """

    input_type: type[InputT] = Field(default=cast("type[InputT]", str), exclude=True)

    @model_validator(mode="after")
    def _input_is_a_supported_boundary(self) -> Self:
        declared = self.input_type
        if declared is not str and not issubclass(declared, BaseModel):
            raise ValueError("input_type must be str or a Pydantic BaseModel subclass")
        return self

    def render_input(self, value: InputT) -> str:
        """Validate and deterministically render one application input."""
        if not isinstance(value, self.input_type):
            expected = self.input_type.__name__
            actual = type(value).__name__
            problem = f"expected {expected}, got {actual}"
            raise SchemaViolationError(
                f"{expected} rejected the agent input — {problem}",
                model=expected,
                paths=("",),
                problems={"": problem},
                payload=value,
            )
        if isinstance(value, str):
            return value
        return json.dumps(
            value.model_dump(mode="json", by_alias=True, exclude_none=False, round_trip=True),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
