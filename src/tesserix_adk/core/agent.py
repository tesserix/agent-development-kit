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

from enum import StrEnum
from typing import Generic, Self

from pydantic import ConfigDict, Field, model_validator

from tesserix_adk.core.config import (  # noqa: TC001
    BudgetConfig,
    DeadlineConfig,
    LoopConfig,
    RepairConfig,
    RetryConfig,
)

# Runtime import, not type-checking only: pydantic resolves the annotation at class creation.
from tesserix_adk.core.models import AdkModel, OutputT

__all__ = ["Agent", "ToolFailurePolicy"]


class ToolFailurePolicy(StrEnum):
    """What a run does when a tool it called fails.

    Neither answer is right everywhere: a model told its search failed can try another
    route, but where the tool was the source of truth, continuing produces a confident
    wrong answer. The agent declares which case it is.
    """

    SURFACE_TO_MODEL = "surface_to_model"
    FAIL_RUN = "fail_run"


class Agent(AdkModel, Generic[OutputT]):  # noqa: UP046 — PEP 695 syntax cannot carry the parameter's default before 3.13
    """A declarative description of an agent.

    Args:
        name: How the agent is identified in runs, traces and cost attribution.
        version: Which revision of the declaration this is, so a behaviour change is
            attributable to it.
        instructions: The system instruction.
        model: The model to use. Mutually exclusive with `task_class`.
        task_class: The kind of work, for a router to resolve to a model. Naming the job
            rather than the model is what lets the routing decision live in one place.
        tools: Tool names this agent may call. Empty by default — a tool a consumer did
            not name is a tool the agent may not call.
        idempotent_tools: Which of those tools are safe to call again. A tool cancelled
            after dispatch is reported indeterminate unless it appears here, because the
            kit cannot know whether its side effect landed. Must be a subset of `tools`.
        approval_required_tools: Which of those tools may not be dispatched without a
            human decision. Must be a subset of `tools`.
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
    model: str | None = None
    task_class: str | None = None
    tools: tuple[str, ...] = ()
    idempotent_tools: tuple[str, ...] = ()
    approval_required_tools: tuple[str, ...] = ()
    output_type: type[OutputT] | None = Field(default=None, exclude=True)
    free_text: bool = False
    budget: BudgetConfig | None = None
    deadlines: DeadlineConfig | None = None
    loop: LoopConfig | None = None
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
