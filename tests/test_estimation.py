"""What a run will cost, said before the money is spent.

A caller deciding whether to start a deep-research run needs a number it can defend, and
the way that goes wrong is a confident-looking figure with nothing behind it. These are the
tests that the estimate says how much it knows, and that being wrong is measurable.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    BudgetLimits,
    Cost,
    CostConfidence,
    EstimateUnavailableError,
    Run,
    RunEvent,
    RunEventKind,
    RunState,
    Usage,
)
from tesserix_adk.models.pricing import pricing_at
from tesserix_adk.runtime import (
    Assumptions,
    Confidence,
    CostEstimate,
    InMemoryHistory,
    Observed,
    Scope,
    Spread,
    affordable,
    approval_for,
    calibrate,
    estimate_run,
    refuse_unaffordable,
)
from tesserix_adk.testing import CAPABLE, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core import Message

PER_MTOK = Decimal("3")


class FlatRate:
    """Three dollars per million tokens, whichever way they went, cached at a tenth."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, usage: Usage, model: str) -> Cost:
        self.asked.append(model)
        tokens = Decimal(usage.input_tokens - usage.cached_tokens + usage.output_tokens)
        cached = Decimal(usage.cached_tokens) / 10
        return Cost(
            input=(tokens + cached) * PER_MTOK / Decimal(1_000_000),
            currency="USD",
            confidence=CostConfidence.ESTIMATED,
        )


class NoPrice:
    """A model nobody has priced."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, usage: Usage, model: str) -> Cost:
        self.asked.append(f"{model}:{usage.input_tokens}")
        return Cost.unknown()


def agent(name: str = "researcher", version: str = "1.0.0") -> Agent[Any]:
    return Agent(
        name=name,
        version=version,
        instructions="Research the question and answer it.",
        model="scripted-1",
        free_text=True,
    )


class CountingProvider(ScriptedProvider):
    """A provider that will count tokens and fail loudly if asked to complete anything."""

    def __init__(self) -> None:
        super().__init__(name="scripted", capabilities=CAPABLE)
        self.counted: list[int] = []

    def count_tokens(self, messages: Sequence[Message]) -> int:
        self.counted.append(len(messages))
        return 1_000


def observed(
    runs: int = 20,
    iterations: tuple[str, str, str] = ("2", "6", "12"),
    output_tokens: tuple[str, str, str] = ("100", "200", "400"),
    tool_calls: tuple[str, str, str] = ("1", "4", "9"),
    cached_fraction: str = "0",
    exact: bool = True,
) -> Observed:
    def spread(values: tuple[str, str, str]) -> Spread:
        return Spread(low=Decimal(values[0]), typical=Decimal(values[1]), high=Decimal(values[2]))

    return Observed(
        runs=runs,
        iterations=spread(iterations),
        output_tokens=spread(output_tokens),
        tool_calls=spread(tool_calls),
        cached_fraction=Decimal(cached_fraction),
        exact=exact,
    )


class Recalled:
    """A history that answers with whatever a test handed it."""

    def __init__(self, seen: Observed | None) -> None:
        self.seen = seen
        self.asked: list[tuple[str, str]] = []

    def observed(self, agent_name: str, version: str) -> Observed | None:
        self.asked.append((agent_name, version))
        return self.seen


def an_estimate(**overrides: Any) -> CostEstimate:
    return estimate_run(
        agent(),
        "what happened to the Antikythera mechanism?",
        provider=overrides.pop("provider", CountingProvider()),
        pricing=overrides.pop("pricing", FlatRate()),
        history=overrides.pop("history", Recalled(observed())),
        **overrides,
    )


class TestAnEstimateSaysWhatItAssumed:
    def test_the_point_estimate_comes_with_a_range_and_the_assumptions_behind_it(self) -> None:
        estimate = an_estimate()
        assert estimate.low.total < estimate.point.total < estimate.high.total
        assert estimate.assumptions.iterations == 6
        assert estimate.assumptions.tool_calls == 4
        assert estimate.assumptions.prompt_tokens == 1_000

    def test_history_for_this_agent_version_is_a_measured_confidence(self) -> None:
        assert an_estimate().confidence is Confidence.MEASURED

    def test_history_for_another_version_of_the_agent_is_inferred_rather_than_measured(
        self,
    ) -> None:
        """A prompt change moves the numbers; last version's runs are a hint, not a count."""
        estimate = an_estimate(history=Recalled(observed(exact=False)))
        assert estimate.confidence is Confidence.INFERRED

    def test_an_agent_nobody_has_run_falls_back_to_stated_defaults(self) -> None:
        estimate = an_estimate(history=Recalled(None))
        assert estimate.confidence is Confidence.INFERRED
        assert estimate.assumptions.runs_observed == 0

    def test_the_range_widens_when_there_is_less_to_go_on(self) -> None:
        """Certainty nobody has is the thing an estimate must not manufacture."""
        measured = an_estimate()
        guessed = an_estimate(history=Recalled(None))
        assert _width(guessed) > _width(measured)

    def test_the_estimate_names_the_model_it_priced(self) -> None:
        assert an_estimate().assumptions.model == "scripted-1"


class TestItCostsNothingToAsk:
    def test_estimating_counts_tokens_locally_and_calls_no_provider(self) -> None:
        """An estimate that needs a paid round trip is a bill for asking about a bill."""
        provider = CountingProvider()
        an_estimate(provider=provider)
        assert len(provider.counted) == 1
        assert provider.requests == []

    def test_prompt_caching_is_priced_rather_than_ignored(self) -> None:
        """A naive estimate over a cached prompt is high enough to refuse a cheap run."""
        cold = an_estimate(history=Recalled(observed(cached_fraction="0")))
        warm = an_estimate(history=Recalled(observed(cached_fraction="0.9")))
        assert warm.point.total < cold.point.total
        assert warm.assumptions.cached_fraction == Decimal("0.9")


class TestTheKitNeverInventsAFigure:
    def test_a_model_with_no_price_refuses_rather_than_guessing(self) -> None:
        with pytest.raises(EstimateUnavailableError) as raised:
            an_estimate(pricing=NoPrice(), history=Recalled(None))
        assert "scripted-1" in str(raised.value)
        assert raised.value.model == "scripted-1"

    def test_an_unpriced_estimate_is_returned_only_when_asked_for_by_name(self) -> None:
        estimate = an_estimate(pricing=NoPrice(), history=Recalled(None), allow_unknown=True)
        assert estimate.confidence is Confidence.UNKNOWN
        assert estimate.point.confidence is CostConfidence.UNKNOWN

    def test_the_token_count_survives_an_unknown_price(self) -> None:
        """What it will read is known even when what it will cost is not."""
        estimate = an_estimate(pricing=NoPrice(), history=Recalled(None), allow_unknown=True)
        assert estimate.input_tokens > 0
        assert estimate.output_tokens > 0


class TestRefusingBeforeTheFirstCall:
    def test_an_estimate_over_the_remaining_budget_is_refused_pre_flight(self) -> None:
        with pytest.raises(Exception, match="max_cost") as raised:
            refuse_unaffordable(an_estimate(), _limits("0.000001"))
        assert type(raised.value).__name__ == "BudgetExceededError"

    def test_a_fitting_estimate_is_permitted(self) -> None:
        assert affordable(an_estimate(), _limits("100")).permitted is True

    def test_a_fitting_estimate_passes_the_refusal_without_a_word(self) -> None:
        refuse_unaffordable(an_estimate(), _limits("100"))

    def test_the_refusal_names_the_dimension_that_did_not_fit(self) -> None:
        decision = affordable(an_estimate(), BudgetLimits(max_iterations=2))
        assert decision.permitted is False
        assert decision.breached == "max_iterations"

    def test_the_high_end_is_what_a_ceiling_is_checked_against(self) -> None:
        """A ceiling that the typical run fits and a bad one does not is not a ceiling."""
        estimate = an_estimate()
        between = (estimate.point.total + estimate.high.total) / 2
        assert affordable(estimate, _limits(str(between))).permitted is False

    def test_turning_an_estimate_into_a_ceiling_is_the_caller_saying_so(self) -> None:
        limits = an_estimate().as_limits(headroom=Decimal("1.5"))
        assert limits.max_cost is not None
        assert limits.max_cost > an_estimate().high.total
        assert limits.max_iterations == 12

    def test_a_budget_with_room_for_everything_refuses_nothing(self) -> None:
        assert affordable(an_estimate(), BudgetLimits()).permitted is True


class TestBeingWrongIsMeasured:
    def test_estimate_versus_actual_is_recorded_per_run(self) -> None:
        estimate = an_estimate()
        actual = _finished(cost="1.00")
        calibration = calibrate(estimate, actual)
        assert calibration.actual.total == Decimal("1.00")
        assert calibration.estimated == estimate.point
        assert calibration.ratio is not None

    def test_a_run_far_over_its_estimate_reads_as_far_over_rather_than_being_smoothed(
        self,
    ) -> None:
        """A tool returning a very large document is the estimator's problem to know about."""
        calibration = calibrate(an_estimate(), _finished(cost="500.00"))
        assert calibration.within_range is False
        assert calibration.ratio is not None
        assert calibration.ratio > 100

    def test_a_run_inside_the_range_says_so(self) -> None:
        estimate = an_estimate()
        assert calibrate(estimate, _finished(cost=str(estimate.point.total))).within_range is True

    def test_an_unpriced_run_has_no_ratio_rather_than_a_ratio_of_zero(self) -> None:
        calibration = calibrate(an_estimate(), _finished(cost=None))
        assert calibration.ratio is None
        assert calibration.within_range is False


class TestHistoryIsBuiltFromRunsThatHappened:
    def test_recorded_runs_become_the_distribution_the_next_estimate_uses(self) -> None:
        history = InMemoryHistory()
        for iterations in (2, 4, 6, 8, 10):
            history.record(_finished(cost="1.00", iterations=iterations))
        seen = history.observed("researcher", "1.0.0")
        assert seen is not None
        assert seen.runs == 5
        assert seen.iterations.low < seen.iterations.typical < seen.iterations.high

    def test_a_version_with_no_runs_of_its_own_borrows_the_agent_s_and_says_so(self) -> None:
        history = InMemoryHistory()
        history.record(_finished(cost="1.00", iterations=4))
        seen = history.observed("researcher", "2.0.0")
        assert seen is not None
        assert seen.exact is False

    def test_an_agent_nobody_has_run_is_absent_rather_than_empty(self) -> None:
        assert InMemoryHistory().observed("researcher", "1.0.0") is None

    def test_a_run_that_never_started_is_not_a_sample(self) -> None:
        """A run refused pre-flight tells you nothing about what a run of it costs."""
        history = InMemoryHistory()
        history.record(_finished(cost="1.00", state=RunState.BUDGET_EXHAUSTED))
        assert history.observed("researcher", "1.0.0") is None


class TestMultiAgentRunsAreScopedOutLoud:
    def test_an_estimate_is_parent_only_unless_children_are_folded_in(self) -> None:
        assert an_estimate().scope is Scope.PARENT_ONLY

    def test_folding_a_child_in_totals_both_and_says_the_scope_widened(self) -> None:
        parent, child = an_estimate(), an_estimate()
        together = parent.with_children(child)
        assert together.scope is Scope.WITH_CHILDREN
        assert together.point.total == parent.point.total + child.point.total
        assert together.high.total == parent.high.total + child.high.total

    def test_the_weakest_confidence_of_the_tree_is_the_confidence_of_the_total(self) -> None:
        guessed = an_estimate(history=Recalled(None))
        assert an_estimate().with_children(guessed).confidence is Confidence.INFERRED

    def test_folding_nothing_in_leaves_the_estimate_alone(self) -> None:
        parent = an_estimate()
        assert parent.with_children() == parent


class TestShowingItToAHuman:
    def test_an_estimate_can_be_put_to_an_approval_gate(self) -> None:
        record = approval_for(
            an_estimate(), agent(), run_id="run_1", tenant="acme", requested_at=1.0
        )
        assert record.tool_name == "start_run"
        assert "USD" in record.reason
        assert len(record.arguments_digest) == 64

    def test_the_reason_carries_the_range_and_the_confidence_not_just_a_number(self) -> None:
        """A single number shown to a human reads as a promise nobody made."""
        reason = approval_for(an_estimate(), agent(), run_id="run_1", tenant="acme").reason
        assert "measured" in reason
        assert " to " in reason


class TestTheShapeOfAnAssumption:
    def test_a_spread_that_is_not_ordered_is_a_mistake_rather_than_a_wide_range(self) -> None:
        with pytest.raises(ValueError, match="low"):
            Spread(low=Decimal(9), typical=Decimal(2), high=Decimal(4))

    def test_assumptions_are_readable_by_whoever_has_to_defend_the_number(self) -> None:
        assumptions = an_estimate().assumptions
        assert set(Assumptions.model_fields) >= {
            "model",
            "prompt_tokens",
            "iterations",
            "output_tokens",
            "tool_calls",
            "tool_result_tokens",
            "cached_fraction",
            "runs_observed",
        }
        assert assumptions.tool_result_tokens > 0


def _width(estimate: CostEstimate) -> Decimal:
    return (estimate.high.total - estimate.low.total) / estimate.point.total


def _limits(cost: str) -> BudgetLimits:
    return BudgetLimits(max_cost=Decimal(cost), currency="USD")


def _finished(
    *,
    cost: str | None,
    iterations: int = 6,
    state: RunState = RunState.COMPLETED,
) -> Run[Any]:
    spent = Usage(
        input_tokens=8_000,
        output_tokens=1_200,
        cost=Cost(input=Decimal(cost), currency="USD") if cost is not None else None,
    )
    return Run(
        id="run_1",
        tenant="acme",
        agent_name="researcher",
        agent_version="1.0.0",
        model="scripted-1",
        state=state,
        usage=spent,
        events=[RunEvent(kind=RunEventKind.MODEL_CALL, name="scripted-1")] * iterations,
    )


class TestTheShippedPricesAsAnEstimator:
    def test_an_estimate_can_be_built_against_the_kit_s_own_price_list(self) -> None:
        estimate = estimate_run(
            agent().model_copy(update={"model": "anthropic:claude-opus-4-1"}),
            "what happened to the Antikythera mechanism?",
            provider=CountingProvider(),
            pricing=pricing_at(date(2026, 8, 7)),
            history=Recalled(observed()),
        )
        assert estimate.point.total > 0
        assert estimate.point.currency == "USD"

    def test_a_model_the_shipped_list_does_not_cover_reads_as_unknown_not_free(self) -> None:
        """A silent zero here is a run that looks free right up until the invoice."""
        assert (
            pricing_at(date(2026, 8, 7))(
                Usage(input_tokens=100, output_tokens=10), "nobody:priced-this"
            ).confidence
            is CostConfidence.UNKNOWN
        )
