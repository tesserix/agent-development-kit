"""Who spent what, read off the run rather than tagged by the caller.

A vendor invoice is one number per API key. Everything that would make it answerable —
tenant, agent, agent version, model, prompt version, task class — is known while the run is
happening and gone by the time the bill arrives. This reads it back out of the run, so
chargeback is a query rather than a reconstruction from application logs that were never
designed for it.

Attribution is derived, never supplied. A consumer that has to remember to tag its own
spend is a consumer that will forget on one code path and mis-tag it on another.
"""

from __future__ import annotations

from enum import StrEnum
from functools import reduce
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tesserix_adk.core import AdkModel, Cost, CostConfidence, Run, RunEventKind, Usage

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "ATTRIBUTE_PREFIX",
    "UNKNOWN",
    "Attribution",
    "Outcome",
    "SpendRecord",
    "Step",
    "Totals",
    "attributes_of",
    "spend_of",
    "totals_by",
]

# What a run could not say. An explicit bucket, because spend attributed to a blank is
# spend nobody chases and a hole nobody notices.
UNKNOWN = "unknown"

ATTRIBUTE_PREFIX = "adk."


class Step(StrEnum):
    """Which kind of step spent it."""

    MODEL = "model_call"
    TOOL = "tool_call"


class Outcome(StrEnum):
    """Whether the step that spent it produced anything."""

    ANSWERED = "answered"
    FAILED = "failed"


class Attribution(AdkModel):
    """The dimensions a bill is broken down by, all of them stated.

    Args:
        tenant: The isolation boundary the spend belongs to. A run acting on behalf of
            another tenant's request still bills the tenant it ran as; attributing it
            elsewhere would move money across a boundary that exists to stop exactly that.
        user: The acting principal, where there was one.
        agent: Which agent ran.
        agent_version: Which version of it, so a cost change is attributable to a change.
        definition: The `AgentDefinition` revision the run declared, where it ran from one.
            A version can be edited in place; a revision names the exact reviewed artifact.
        model: The model that actually answered, not the one first asked for.
        prompt_version: Which prompt produced the call.
        task_class: The kind of work routing was asked for.
        run_id: The run, for joining spend back to what happened.
    """

    tenant: str
    user: str
    agent: str
    agent_version: str
    definition: str
    model: str
    prompt_version: str
    task_class: str
    run_id: str

    @property
    def unknowns(self) -> tuple[str, ...]:
        """Which dimensions fell back to the bucket, sorted. Empty is a complete row."""
        return tuple(sorted(f for f in type(self).model_fields if getattr(self, f) == UNKNOWN))


class SpendRecord(AdkModel):
    """One metered step, and who it belongs to.

    Args:
        attribution: The dimensions this step is billed under.
        step: Whether a model or a tool spent it.
        outcome: Whether the step answered. A failed attempt still burned tokens.
        usage: What it consumed.
    """

    attribution: Attribution
    step: Step
    outcome: Outcome
    usage: Usage

    @property
    def cost(self) -> Cost:
        """What the step came to.

        A model step nobody priced is `Cost.unknown()`, not zero: a model call that
        reported no usage is a call at a price nobody knows, and recording it as free is a
        false statement that totals up into a bill. A tool step genuinely has no price.
        """
        if self.usage.cost is not None:
            return self.usage.cost
        return Cost.nothing() if self.step is Step.TOOL else Cost.unknown()

    @property
    def estimated(self) -> bool:
        """Whether any part of this row was counted rather than metered.

        An estimated row will never appear on a vendor invoice, so reconciliation has to
        be able to set it aside rather than chase a difference that is not a difference.
        A tool step has no price rather than an unknown one, which is why it is not
        counted as a guess here.
        """
        unpriced = self.step is Step.MODEL and self.usage.cost is None
        return (
            unpriced or self.usage.estimated or self.cost.confidence is not CostConfidence.COUNTED
        )


class Totals(AdkModel):
    """What one group of records came to.

    Args:
        cost: The money, kept in one currency.
        input_tokens: Tokens sent, cache reads included.
        output_tokens: Tokens generated.
        cached_tokens: Of the input, how much the providers served from their caches.
        cache_write_tokens: Tokens charged to write into a provider cache, totalled apart
            because they are priced apart and often at a premium.
        calls: How many metered steps are behind the number.
        estimated: Whether any part of it was counted rather than metered.
    """

    cost: Cost
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int
    estimated: bool

    @property
    def measured(self) -> bool:
        """Whether anything was sent, so the hit ratio over this group means something."""
        return self.input_tokens > 0

    @property
    def hit_ratio(self) -> float:
        """How much of this group's input came from a provider cache, from 0 to 1.

        The one number that says whether prefix stability is paying off. Zero rather than
        a division error where nothing was sent — check `measured` to tell "no hits" from
        "no data", which a dashboard otherwise draws identically.

        Example:
            >>> Totals(cost=Cost.nothing(), input_tokens=1000, output_tokens=0,
            ...        cached_tokens=450, calls=1, estimated=False).hit_ratio
            0.45
        """
        if not self.measured:
            return 0.0
        return min(self.cached_tokens / self.input_tokens, 1.0)


def _named(value: str | None) -> str:
    return value or UNKNOWN


def _attribution(run: Run[Any], model: str) -> Attribution:
    return Attribution(
        tenant=_named(run.tenant),
        user=_named(run.user),
        agent=_named(run.agent_name),
        agent_version=_named(run.agent_version),
        definition=_named(run.definition_revision),
        model=_named(model),
        prompt_version=_named(run.prompt_version),
        task_class=_named(run.task_class),
        run_id=_named(run.id),
    )


def spend_of[OutputT: BaseModel](run: Run[OutputT]) -> tuple[SpendRecord, ...]:
    """Every metered step of `run`, in order, attributed.

    Failed attempts are included: tokens the vendor read before it failed are on the
    invoice whether or not anything came back, and dropping them is how a run that never
    answered reads as a run that cost nothing. A step is attributed to the model that
    actually burned it, so a run that fell back to a second vendor does not report both
    against the first.

    Args:
        run: The run to read. Nothing is called and nothing is mutated.

    Returns:
        One record per metered step, in the order the run recorded them.

    Example:
        >>> spend_of(Run(id="r1", tenant="acme", agent_name="a", agent_version="1", model="m"))
        ()
    """
    records: list[SpendRecord] = []
    model = run.model
    for event in run.events:
        if event.kind is RunEventKind.MODEL_CALL and event.name:
            model = event.name
        elif event.kind is RunEventKind.MODEL_RESPONSE and event.usage is not None:
            records.append(
                SpendRecord(
                    attribution=_attribution(run, model),
                    step=Step.MODEL,
                    outcome=Outcome.ANSWERED,
                    usage=event.usage,
                )
            )
        elif event.kind is RunEventKind.ATTEMPT_FAILED and event.usage is not None:
            records.append(
                SpendRecord(
                    attribution=_attribution(run, model),
                    step=Step.MODEL,
                    outcome=Outcome.FAILED,
                    usage=event.usage,
                )
            )
        elif event.kind is RunEventKind.TOOL_RESULT:
            records.append(
                SpendRecord(
                    attribution=_attribution(run, model),
                    step=Step.TOOL,
                    outcome=Outcome.ANSWERED,
                    usage=Usage(input_tokens=0, output_tokens=0),
                )
            )
    return tuple(records)


def attributes_of(record: SpendRecord) -> dict[str, str]:
    """The standard attribute set for one record, ready to hang on a span.

    One set of names rather than one per product: a cost question that has to know which
    team exported the span is a cost question nobody answers twice the same way.

    Args:
        record: The step to describe.

    Returns:
        Attribute names under the `adk.` prefix, every value a string so no exporter has
        to guess a type.
    """
    attribution = record.attribution
    return {
        f"{ATTRIBUTE_PREFIX}{name}": str(getattr(attribution, name))
        for name in type(attribution).model_fields
    } | {
        f"{ATTRIBUTE_PREFIX}step": str(record.step),
        f"{ATTRIBUTE_PREFIX}outcome": str(record.outcome),
        f"{ATTRIBUTE_PREFIX}input_tokens": str(record.usage.input_tokens),
        f"{ATTRIBUTE_PREFIX}output_tokens": str(record.usage.output_tokens),
        f"{ATTRIBUTE_PREFIX}cost": str(record.cost.total),
        f"{ATTRIBUTE_PREFIX}currency": record.cost.currency,
        f"{ATTRIBUTE_PREFIX}estimated": "true" if record.estimated else "false",
    }


def totals_by(records: Iterable[SpendRecord], *by: str) -> dict[tuple[str, ...], Totals]:
    """Group `records` by attribution dimensions and total each group.

    Args:
        records: The records to total.
        by: Dimension names, as declared on `Attribution`. The key of each group is a
            tuple in the order given, whether one dimension or four, so a caller never
            has to branch on how many it asked for.

    Returns:
        One `Totals` per distinct combination of the requested dimensions.

    Raises:
        ValueError: If a name is not a dimension, or if a group spans two currencies —
            adding those would produce a number that is true in neither.

    Example:
        >>> totals_by((), "tenant")
        {}
    """
    unknown = [name for name in by if name not in Attribution.model_fields]
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)} is not an attribution dimension; "
            f"available: {', '.join(sorted(Attribution.model_fields))}"
        )
    grouped: dict[tuple[str, ...], list[SpendRecord]] = {}
    for record in records:
        key = tuple(str(getattr(record.attribution, name)) for name in by)
        grouped.setdefault(key, []).append(record)
    return {key: _totalled(group) for key, group in grouped.items()}


def _totalled(group: Sequence[SpendRecord]) -> Totals:
    return Totals(
        cost=reduce(lambda one, two: one + two, (record.cost for record in group)),
        input_tokens=sum(record.usage.input_tokens for record in group),
        output_tokens=sum(record.usage.output_tokens for record in group),
        cached_tokens=sum(record.usage.cached_tokens for record in group),
        cache_write_tokens=sum(record.usage.cache_write_tokens for record in group),
        calls=len(group),
        estimated=any(record.estimated for record in group),
    )
