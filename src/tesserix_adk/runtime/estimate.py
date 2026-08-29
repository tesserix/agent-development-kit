"""What a run will cost, answered before the first provider call.

A caller deciding whether to start a deep-research run, or whether to put it to a human
first, needs a number in advance. The number is a guess, so the shape of it matters more
than the value: a point estimate alone reads as a promise, and a promise nobody made is
what turns an estimate into a complaint. Every estimate here carries a range, the
assumptions it rests on and how much is actually known — and where nothing is known, it
refuses rather than inventing a figure that looks confident.

Estimating costs nothing beyond a local token count. Where the estimate turns out wrong,
`calibrate` says by how much, so the estimator's accuracy is measured rather than assumed.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from tesserix_adk.core import (
    AdkModel,
    ApprovalRecord,
    BudgetDecision,
    BudgetLimits,
    Cost,
    CostConfidence,
    EstimateUnavailableError,
    Run,
    RunEventKind,
    RunState,
    Usage,
)
from tesserix_adk.runtime.prompt import assemble_prompt

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tesserix_adk.core import Agent, TypedAgent
    from tesserix_adk.core.protocols import ModelProvider
    from tesserix_adk.runtime.prompt import ToolDeclaration

__all__ = [
    "Assumptions",
    "Calibration",
    "Confidence",
    "CostEstimate",
    "InMemoryHistory",
    "Observed",
    "Pricer",
    "RunHistory",
    "Scope",
    "Spread",
    "affordable",
    "approval_for",
    "calibrate",
    "estimate_run",
    "estimate_run_typed",
    "refuse_unaffordable",
]

TOOL_RESULT_TOKENS = 800


class Confidence(StrEnum):
    """How much of the estimate is measurement and how much is assumption."""

    MEASURED = "measured"
    """Built from this agent version's own finished runs."""

    INFERRED = "inferred"
    """Built from another version's runs, or from the kit's defaults."""

    UNKNOWN = "unknown"
    """Nothing prices this model. Returned only where the caller asked for it by name."""


_WEAKEST = {Confidence.MEASURED: 0, Confidence.INFERRED: 1, Confidence.UNKNOWN: 2}


class Scope(StrEnum):
    """Whose spend the estimate covers."""

    PARENT_ONLY = "parent_only"
    """This run alone. A run that delegates will cost more than this says."""

    WITH_CHILDREN = "with_children"
    """This run and the child estimates folded into it."""


class Spread(AdkModel):
    """A distribution reduced to the three numbers an estimate needs.

    Args:
        low: The tenth percentile of what was observed.
        typical: The median.
        high: The ninetieth percentile.
    """

    low: Decimal
    typical: Decimal
    high: Decimal

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.low <= self.typical <= self.high:
            raise ValueError(
                f"a spread runs low ({self.low}) through typical ({self.typical}) to high "
                f"({self.high}); out of order, the range means the opposite of what it says"
            )
        return self

    @classmethod
    def of(cls, samples: Sequence[Decimal]) -> Spread:
        """The spread of `samples`, by nearest-rank percentile."""
        ordered = sorted(samples)
        return cls(
            low=_at(ordered, Decimal("0.1")),
            typical=_at(ordered, Decimal("0.5")),
            high=_at(ordered, Decimal("0.9")),
        )

    @classmethod
    def around(cls, typical: int, *, spread: str) -> Spread:
        """A spread of `spread` either side of `typical`, for what nobody has measured."""
        width = Decimal(typical) * Decimal(spread)
        return cls(
            low=max(Decimal(typical) - width, Decimal(1)),
            typical=Decimal(typical),
            high=Decimal(typical) + width,
        )


class Observed(AdkModel):
    """What an agent's finished runs actually did.

    Args:
        runs: How many runs this rests on. A distribution over three runs is a rumour, and
            the number is here so a caller can see which it has.
        iterations: Turns of the loop.
        output_tokens: Tokens generated per iteration.
        tool_calls: Tools invoked over the whole run.
        cached_fraction: How much of the prompt the provider served from its cache.
        exact: Whether these are this agent version's own runs. A prompt change moves the
            numbers, so last version's runs are a hint rather than a count.
    """

    runs: int = Field(ge=0)
    iterations: Spread
    output_tokens: Spread
    tool_calls: Spread
    cached_fraction: Decimal = Decimal(0)
    exact: bool = True


@runtime_checkable
class RunHistory(Protocol):
    """Where finished runs are kept, in whatever store a deployment already has."""

    def observed(self, agent_name: str, version: str) -> Observed | None:
        """What runs of `agent_name` at `version` did, or `None` where there are none."""
        ...


@runtime_checkable
class Pricer(Protocol):
    """What a usage comes to on a model.

    Stated as a callable rather than a price list so this layer holds no opinion about
    where prices live. `tesserix_adk.models.pricing.pricing_at` is the shipped one.
    """

    def __call__(self, usage: Usage, model: str) -> Cost:
        """The cost of `usage` on `model`, `Cost.unknown()` where nothing prices it."""
        ...


class Assumptions(AdkModel):
    """Everything the estimate rests on, in a form somebody can argue with.

    Args:
        model: Which model was priced.
        prompt_tokens: The first turn's prompt, by the provider's own count.
        iterations: Turns of the loop assumed.
        output_tokens: Tokens generated per iteration.
        tool_calls: Tools assumed to be invoked over the run.
        iterations_high: Turns in the ninetieth-percentile case, which is what a ceiling
            built from this estimate is set to.
        tool_calls_high: Tools invoked in that same case.
        tool_result_tokens: What one tool result is assumed to add to the next prompt.
        cached_fraction: How much of a repeated prompt is assumed to be served from cache.
        runs_observed: How many finished runs the distribution came from. Zero means the
            numbers above are the kit's defaults, not this agent's behaviour.
    """

    model: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    iterations: int = Field(ge=1)
    output_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    iterations_high: int = Field(ge=1)
    tool_calls_high: int = Field(ge=0)
    tool_result_tokens: int = Field(ge=0)
    cached_fraction: Decimal = Decimal(0)
    runs_observed: int = Field(default=0, ge=0)


class CostEstimate(AdkModel):
    """What a run is expected to cost, and how much of that is known.

    Args:
        point: The typical case.
        low: The tenth-percentile case.
        high: The ninetieth-percentile case. This is what a ceiling is checked against: a
            ceiling the typical run fits and a bad one does not is not a ceiling.
        input_tokens: Prompt tokens expected across the whole run, typical case.
        output_tokens: Generated tokens expected across the whole run, typical case.
        input_tokens_high: Prompt tokens in the high case.
        output_tokens_high: Generated tokens in the high case.
        assumptions: What the numbers rest on.
        confidence: How much of it is measurement.
        scope: Whether child runs are included.
    """

    point: Cost
    low: Cost
    high: Cost
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    input_tokens_high: int = Field(ge=0)
    output_tokens_high: int = Field(ge=0)
    assumptions: Assumptions
    confidence: Confidence
    scope: Scope = Scope.PARENT_ONLY

    def as_limits(self, *, headroom: Decimal = Decimal(1)) -> BudgetLimits:
        """This estimate as a ceiling: the high case, with `headroom` on the money.

        An estimate is not a limit until somebody says it is. Turning one into the other
        implicitly is how a run gets killed for being averagely expensive. Headroom applies
        to the price rather than to the shape, because what varies between an estimate and
        an invoice is what the tokens cost, not how many turns the agent was allowed.
        """
        return BudgetLimits(
            max_cost=(self.high.total * headroom),
            currency=self.point.currency,
            max_input_tokens=self.input_tokens_high,
            max_output_tokens=self.output_tokens_high,
            max_iterations=self.assumptions.iterations_high,
            max_tool_calls=self.assumptions.tool_calls_high,
        )

    def with_children(self, *children: CostEstimate) -> CostEstimate:
        """This estimate plus the runs it will start, totalled and marked as such."""
        if not children:
            return self
        return self.model_copy(
            update={
                "point": _sum(self.point, [child.point for child in children]),
                "low": _sum(self.low, [child.low for child in children]),
                "high": _sum(self.high, [child.high for child in children]),
                "input_tokens": self.input_tokens + sum(c.input_tokens for c in children),
                "output_tokens": self.output_tokens + sum(c.output_tokens for c in children),
                "input_tokens_high": self.input_tokens_high
                + sum(c.input_tokens_high for c in children),
                "output_tokens_high": self.output_tokens_high
                + sum(c.output_tokens_high for c in children),
                "confidence": max(
                    (self.confidence, *(c.confidence for c in children)), key=_WEAKEST.__getitem__
                ),
                "scope": Scope.WITH_CHILDREN,
            }
        )


class Calibration(AdkModel):
    """An estimate held against what the run actually cost.

    Args:
        estimated: The point estimate that was given.
        actual: What the finished run consumed.
        ratio: Actual over estimated, or `None` where nothing priced the run. A run that
            cost fifty times its estimate reads as fifty here; nothing clamps it, because
            the outliers are the ones the estimator has something to learn from.
        within_range: Whether the actual landed between the low and high cases.
    """

    estimated: Cost
    actual: Cost
    ratio: Decimal | None
    within_range: bool


def estimate_run[OutputT: BaseModel](
    agent: Agent[OutputT],
    user_input: str,
    *,
    provider: ModelProvider,
    pricing: Pricer,
    history: RunHistory | None = None,
    tools: Iterable[ToolDeclaration] = (),
    allow_unknown: bool = False,
) -> CostEstimate:
    """What `agent` is expected to spend answering `user_input`, before it starts.

    Args:
        agent: The declaration about to run.
        user_input: What is being asked.
        provider: Who would answer it. Asked only to count tokens — an estimate that needs
            a paid round trip is a bill for asking about a bill.
        pricing: What that usage comes to on this model.
        history: Where this agent's finished runs are kept. Without it the kit's own
            defaults are used and the estimate says so.
        tools: The tool declarations that would be in the prompt.
        allow_unknown: Whether to return an unpriced estimate rather than refusing.

    Returns:
        A `CostEstimate` carrying a point, a range, its assumptions and its confidence.

    Raises:
        EstimateUnavailableError: If nothing prices this model and `allow_unknown` is
            false. The kit does not return a confident-looking invented figure.
    """
    model = agent.model or agent.task_class or ""
    prompt = assemble_prompt(agent, user_input, tools=tools)
    prompt_tokens = provider.count_tokens(prompt.messages)

    seen = history.observed(agent.name, agent.version) if history is not None else None
    shape = seen if seen is not None else _defaults()
    assumptions = Assumptions(
        model=model,
        prompt_tokens=prompt_tokens,
        iterations=int(shape.iterations.typical),
        output_tokens=int(shape.output_tokens.typical),
        tool_calls=int(shape.tool_calls.typical),
        iterations_high=max(int(shape.iterations.high), 1),
        tool_calls_high=int(shape.tool_calls.high),
        tool_result_tokens=TOOL_RESULT_TOKENS,
        cached_fraction=shape.cached_fraction,
        runs_observed=shape.runs,
    )

    typical = _usage(prompt_tokens, shape, "typical", assumptions.cached_fraction)
    point = pricing(typical, model)
    if point.confidence is CostConfidence.UNKNOWN and not allow_unknown:
        raise EstimateUnavailableError(
            f"nothing prices {model}, so this run cannot be estimated. Pass "
            f"allow_unknown=True to get the token counts with the cost marked unknown",
            model=model,
            reason="no price for the model",
        )

    low = _usage(prompt_tokens, shape, "low", assumptions.cached_fraction)
    high = _usage(prompt_tokens, shape, "high", assumptions.cached_fraction)
    return CostEstimate(
        point=point,
        low=pricing(low, model),
        high=pricing(high, model),
        input_tokens=typical.input_tokens,
        output_tokens=typical.output_tokens,
        input_tokens_high=high.input_tokens,
        output_tokens_high=high.output_tokens,
        assumptions=assumptions,
        confidence=_confidence(seen, point),
    )


def estimate_run_typed[InputT: str | BaseModel, OutputT: BaseModel](
    agent: TypedAgent[InputT, OutputT],
    user_input: InputT,
    *,
    provider: ModelProvider,
    pricing: Pricer,
    history: RunHistory | None = None,
    tools: Iterable[ToolDeclaration] = (),
    allow_unknown: bool = False,
) -> CostEstimate:
    """Estimate a structured-input agent through the stable text estimator."""
    return estimate_run(
        agent,
        agent.render_input(user_input),
        provider=provider,
        pricing=pricing,
        history=history,
        tools=tools,
        allow_unknown=allow_unknown,
    )


def affordable(estimate: CostEstimate, limits: BudgetLimits) -> BudgetDecision:
    """Whether `limits` has room for `estimate`, checked against its high case.

    Returns:
        A permitted decision, or the first dimension that does not fit.
    """
    against: dict[str, tuple[Decimal | None, Decimal]] = {
        "max_cost": (limits.max_cost, estimate.high.total),
        "max_input_tokens": (
            _decimal(limits.max_input_tokens),
            Decimal(estimate.input_tokens_high),
        ),
        "max_output_tokens": (
            _decimal(limits.max_output_tokens),
            Decimal(estimate.output_tokens_high),
        ),
        "max_iterations": (
            _decimal(limits.max_iterations),
            Decimal(estimate.assumptions.iterations_high),
        ),
        "max_tool_calls": (
            _decimal(limits.max_tool_calls),
            Decimal(estimate.assumptions.tool_calls_high),
        ),
    }
    for name, (ceiling, wanted) in against.items():
        if ceiling is not None and wanted > ceiling:
            return BudgetDecision(
                permitted=False,
                breached=name,
                limit=ceiling,
                consumed=wanted,
                remaining=Decimal(0),
                priced=estimate.confidence is not Confidence.UNKNOWN,
            )
    return BudgetDecision(permitted=True, priced=estimate.confidence is not Confidence.UNKNOWN)


def refuse_unaffordable(estimate: CostEstimate, limits: BudgetLimits) -> None:
    """Raise if `estimate` does not fit `limits`.

    Raises:
        BudgetExceededError: Before the run starts, which is the whole point: a ceiling
            reached mid-run has already spent whatever it took to get there.
    """
    decision = affordable(estimate, limits)
    if not decision.permitted:
        raise decision.as_error()


def calibrate[OutputT: BaseModel](estimate: CostEstimate, run: Run[OutputT]) -> Calibration:
    """Hold `estimate` against what `run` actually cost."""
    actual = run.usage.cost if run.usage.cost is not None else Cost.unknown(estimate.point.currency)
    known = run.usage.cost is not None and estimate.point.total > 0
    return Calibration(
        estimated=estimate.point,
        actual=actual,
        ratio=(actual.total / estimate.point.total) if known else None,
        within_range=known and estimate.low.total <= actual.total <= estimate.high.total,
    )


def approval_for[OutputT: BaseModel](
    estimate: CostEstimate,
    agent: Agent[OutputT],
    *,
    run_id: str,
    tenant: str,
    requested_at: float = 0.0,
) -> ApprovalRecord:
    """The estimate as something a human can be asked to approve.

    The reason carries the range and the confidence rather than a single number, because a
    single number shown to a person reads as a promise nobody made.
    """
    currency = estimate.point.currency
    return ApprovalRecord.for_call(
        run_id=run_id,
        tenant=tenant,
        agent_name=agent.name,
        tool_name="start_run",
        arguments=estimate.assumptions.model_dump(mode="json"),
        reason=(
            f"starting {agent.name} is estimated at {estimate.point.total} {currency}, "
            f"{estimate.low.total} to {estimate.high.total} {currency} "
            f"({estimate.confidence} over {estimate.assumptions.runs_observed} runs)"
        ),
        requested_at=requested_at,
    )


class InMemoryHistory:
    """Finished runs kept in the process, for a deployment that has nowhere else yet.

    Enough to make the estimator honest in tests and small deployments. A real one reads
    the runs a deployment already stores; the protocol is `RunHistory`.
    """

    def __init__(self) -> None:
        self._samples: dict[tuple[str, str], list[Run[Any]]] = {}

    def record[OutputT: BaseModel](self, run: Run[OutputT]) -> None:
        """Keep `run` as a sample, if it is one.

        A run that was refused or died partway says nothing about what a run of that agent
        costs, so only completed runs count.
        """
        if run.state is not RunState.COMPLETED:
            return
        self._samples.setdefault((run.agent_name, run.agent_version), []).append(run)

    def observed(self, agent_name: str, version: str) -> Observed | None:
        """The distribution over recorded runs, this version's own where there are any."""
        exact = self._samples.get((agent_name, version), [])
        runs = exact or [
            run for (name, _), kept in self._samples.items() if name == agent_name for run in kept
        ]
        if not runs:
            return None
        iterations = [Decimal(_iterations(run)) for run in runs]
        return Observed(
            runs=len(runs),
            iterations=Spread.of(iterations),
            output_tokens=Spread.of(
                [Decimal(run.usage.output_tokens) / Decimal(_iterations(run)) for run in runs]
            ),
            tool_calls=Spread.of([Decimal(len(run.tool_calls)) for run in runs]),
            cached_fraction=_cached_fraction(runs),
            exact=bool(exact),
        )


def _defaults() -> Observed:
    """What a run of an agent nobody has run is assumed to do.

    Deliberately wide rather than deliberately low: an estimate that guesses low is one a
    caller finds out about from the invoice.
    """
    return Observed(
        runs=0,
        iterations=Spread.around(4, spread="1.0"),
        output_tokens=Spread.around(400, spread="1.0"),
        tool_calls=Spread.around(3, spread="1.0"),
        exact=False,
    )


def _usage(prompt_tokens: int, shape: Observed, case: str, cached: Decimal) -> Usage:
    """What the whole run reads and writes, in the `case` corner of the distribution.

    Each turn re-sends what came before it, so the prompt grows by the last answer and any
    tool result: a run's input is the triangle under that growth, not the first prompt
    times the turn count.
    """
    iterations = max(int(getattr(shape.iterations, case)), 1)
    per_turn_output = int(getattr(shape.output_tokens, case))
    tool_calls = int(getattr(shape.tool_calls, case))
    growth = per_turn_output + (tool_calls * TOOL_RESULT_TOKENS // max(iterations, 1))
    accumulated = growth * iterations * (iterations - 1) // 2
    return Usage(
        input_tokens=prompt_tokens * iterations + accumulated,
        output_tokens=per_turn_output * iterations,
        cached_tokens=int(Decimal(prompt_tokens * (iterations - 1)) * cached),
    )


def _confidence(seen: Observed | None, point: Cost) -> Confidence:
    if point.confidence is CostConfidence.UNKNOWN:
        return Confidence.UNKNOWN
    if seen is not None and seen.exact:
        return Confidence.MEASURED
    return Confidence.INFERRED


def _sum(first: Cost, rest: Iterable[Cost]) -> Cost:
    total = first
    for other in rest:
        total = total + other
    return total


def _at(ordered: Sequence[Decimal], quantile: Decimal) -> Decimal:
    index = int(quantile * (len(ordered) - 1))
    return ordered[index]


def _iterations(run: Run[Any]) -> int:
    return max(sum(1 for event in run.events if event.kind is RunEventKind.MODEL_CALL), 1)


def _cached_fraction(runs: Sequence[Run[Any]]) -> Decimal:
    read = sum(run.usage.cached_tokens for run in runs)
    total = sum(run.usage.input_tokens for run in runs)
    return Decimal(read) / Decimal(total) if total else Decimal(0)


def _decimal(value: int | None) -> Decimal | None:
    return None if value is None else Decimal(value)
