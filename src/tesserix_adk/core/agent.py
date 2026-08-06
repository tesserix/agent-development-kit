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

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Runtime import, not type-checking only: pydantic resolves the annotation at class creation.
from tesserix_adk.core.config import BudgetConfig  # noqa: TC001

__all__ = ["Agent", "ToolFailurePolicy"]


class ToolFailurePolicy(StrEnum):
    """What a run does when a tool it called fails.

    Neither answer is right everywhere: a model told its search failed can try another
    route, but where the tool was the source of truth, continuing produces a confident
    wrong answer. The agent declares which case it is.
    """

    SURFACE_TO_MODEL = "surface_to_model"
    FAIL_RUN = "fail_run"


class Agent(BaseModel):
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
        output_type: The type the answer must validate against. A Python type, so it is
            excluded from serialisation rather than rendered as a string that cannot be
            read back.
        budget: The ceiling for one run.
        on_tool_error: What happens when a tool fails. Surfaced to the model by default,
            so a run does not die on the first recoverable failure.
        guardrails: Guardrail names, applied in order.
        metadata: Consumer-owned annotations. The kit reads nothing from it.

    Example:
        >>> Agent(name="planner", instructions="Plan trips.", model="claude-sonnet-5").tools
        ()
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    version: str = "1.0.0"
    instructions: str = Field(min_length=1)
    model: str | None = None
    task_class: str | None = None
    tools: tuple[str, ...] = ()
    output_type: type[BaseModel] | None = Field(default=None, exclude=True)
    budget: BudgetConfig | None = None
    on_tool_error: ToolFailurePolicy = ToolFailurePolicy.SURFACE_TO_MODEL
    guardrails: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_way_to_choose_a_model(self) -> Agent:
        if bool(self.model) == bool(self.task_class):
            raise ValueError(
                "declare exactly one of model or task_class: two answers to which model "
                "runs is a routing decision made twice, and none is a run that cannot start"
            )
        return self

    @model_validator(mode="after")
    def _tools_are_named_once(self) -> Agent:
        seen = {name for name in self.tools if self.tools.count(name) > 1}
        if seen:
            raise ValueError(
                f"tool named more than once: {', '.join(sorted(seen))}. One of the "
                f"entries was meant to be a different tool"
            )
        return self
