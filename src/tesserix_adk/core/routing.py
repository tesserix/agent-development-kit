"""Naming the job instead of the model.

A model id at a call site is a cost decision compiled into a product: retuning it means a
code change in every consumer that wrote it down, and there is no way to say that a
classification step wants the cheap model while a planning step wants the reasoning one.
A task class names the job; where it resolves to is configuration.

The vocabulary is open. `CHEAP`, `SMART` and `REASONING` are the three the kit ships
because they are the three every consumer reinvents, but a class is a string and a
consumer that meters `transcription` separately says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import Field
from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

from tesserix_adk.core.capabilities import CapabilitySet, ModelRef  # noqa: TC001
from tesserix_adk.core.models import AdkModel

__all__ = [
    "CHEAP",
    "REASONING",
    "SMART",
    "ModelRequirements",
    "ModelRouter",
    "RejectedCandidate",
    "RoutingDecision",
    "TaskClass",
]


class TaskClass(str):
    """The kind of work, named rather than the model that does it.

    A `str` subclass rather than an enum because the set is open: the kit's three classes
    are a starting vocabulary, and a consumer that routes `transcription` or `extraction`
    separately should not have to subclass anything to say so.

    Example:
        >>> TaskClass("extraction") == "extraction"
        True
    """

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Validate as a non-empty string: an unnamed class routes to nothing."""
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema(min_length=1)
        )


CHEAP = TaskClass("cheap")
SMART = TaskClass("smart")
REASONING = TaskClass("reasoning")


class ModelRequirements(AdkModel):
    """What the caller needs of whatever model answers.

    Stated as a requirement rather than checked afterwards: a router that picks first and
    fails on the capability check has already recorded the wrong model against the run.

    Args:
        capabilities: Every capability the work needs. A candidate that has not declared
            one of them is not eligible, because silence is not a claim.
        min_context_window_tokens: The smallest window that could hold the prompt. A
            candidate declaring no window at all cannot satisfy this — an unknown window
            is not evidence of a large one.
        min_output_tokens: The smallest output ceiling the answer could fit in.
    """

    capabilities: CapabilitySet = frozenset()
    min_context_window_tokens: int | None = Field(default=None, gt=0)
    min_output_tokens: int | None = Field(default=None, gt=0)

    def unsatisfied_by(self, declared: object) -> tuple[str, ...]:
        """Name every requirement `declared` does not meet, in a stable order.

        Args:
            declared: A `ModelCapabilities` record.

        Returns:
            The requirement names, empty where the candidate is eligible.
        """
        missing = [
            capability.value
            for capability in sorted(self.capabilities)
            if not getattr(declared, capability.value, False)
        ]
        missing += [
            f"{name} >= {needed}"
            for name, needed in (
                ("context_window_tokens", self.min_context_window_tokens),
                ("max_output_tokens", self.min_output_tokens),
            )
            if needed is not None and (getattr(declared, name, None) or 0) < needed
        ]
        return tuple(missing)


class RejectedCandidate(AdkModel):
    """A model the router passed over, and what it could not do.

    Args:
        ref: The candidate, as `provider:model`.
        reason: Which requirements it failed, so the answer to "why not that one" is in
            the record rather than in somebody's head.
    """

    ref: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RoutingDecision(AdkModel):
    """Which model answered a task class, and why that one.

    Recorded on the run: a model choice nobody can explain after the fact is a bill nobody
    can explain after the fact.

    Args:
        task_class: What was asked for.
        chosen: What answered.
        considered: Every candidate the matching rule offered, in order.
        rejected: The ones ruled out, with the requirement each failed.
        rule: Which rule answered, as `class@tenant/agent`.
        pinned: Whether the caller named the model directly, bypassing the order.
    """

    task_class: TaskClass
    chosen: ModelRef
    considered: tuple[str, ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    rule: str = ""
    pinned: bool = False

    def explain(self) -> str:
        """One line for a run event: what was asked, what answered, on whose authority."""
        how = "pinned" if self.pinned else f"rule {self.rule}"
        return f"{self.task_class} -> {self.chosen} ({how}, {len(self.considered)} considered)"


@runtime_checkable
class ModelRouter(Protocol):
    """Resolves a task class and a requirement set to one concrete model."""

    def resolve(
        self,
        task_class: TaskClass,
        *,
        requirements: ModelRequirements | None = None,
        tenant: str | None = None,
        agent: str | None = None,
        pinned: ModelRef | None = None,
    ) -> RoutingDecision:
        """Choose the model for this piece of work.

        Args:
            task_class: The kind of work.
            requirements: What the work needs of the model. Nothing by default.
            tenant: Who it is for, where a rule is scoped to one.
            agent: Which agent is asking, where a rule is scoped to one.
            pinned: A model named directly, for reproducing a run. Still checked against
                the requirements — a pin is a choice among configured models, not a way
                past the checks.

        Raises:
            NoEligibleModelError: If nothing configured can do the work. Never a
                downgrade: a model that cannot do the job is not a cheaper way to do it.
        """
        ...
