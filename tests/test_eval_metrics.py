"""Quality, spend and speed measured together, and an unknown reported as unknown."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import Message, NoOutput, Run, RunState, TextPart, ToolCall, Usage
from tesserix_adk.core.cost import Cost, CostConfidence
from tesserix_adk.evals import (
    Aggregate,
    CacheHitRate,
    CaseResult,
    CaseStatus,
    CostPerCase,
    EvalCase,
    EvalSuite,
    ExactMatch,
    Groundedness,
    LatencyMs,
    MetricValue,
    RefusalRate,
    SuiteResult,
    Threshold,
    TokensIn,
    TokensOut,
    ToolSequenceMatch,
    measure,
)
from tesserix_adk.evals.metrics import SchemaValidity

if TYPE_CHECKING:
    from tesserix_adk.evals import Metric

ANSWER = "delayed by nine minutes"


def _case(case_id: str = "c1", **overrides: Any) -> EvalCase:
    fields: dict[str, Any] = {"id": case_id, "input": "how late", "tenant": "acme"}
    return EvalCase(**(fields | overrides))


def _run(
    *,
    answer: str = ANSWER,
    state: RunState = RunState.COMPLETED,
    usage: Usage | None = None,
    tool_calls: list[ToolCall] | None = None,
    started_at: float | None = 100.0,
    ended_at: float | None = 100.4,
) -> Run[NoOutput]:
    return Run[NoOutput](
        id="run-1",
        tenant="acme",
        agent_name="timetable",
        agent_version="1.0.0",
        model="fake-1",
        state=state,
        messages=[Message(role="assistant", content=[TextPart(text=answer)])],
        tool_calls=tool_calls or [],
        usage=usage or Usage(input_tokens=100, output_tokens=20),
        started_at=started_at,
        ended_at=ended_at,
    )


def _measured(
    metric: Metric, case: EvalCase | None = None, run: Run[NoOutput] | None = None
) -> MetricValue:
    return metric.compute(case or _case(), run or _run())


def _report(
    *metrics: Metric,
    cases: tuple[EvalCase, ...] = (),
    runs: tuple[Run[NoOutput], ...] = (),
    thresholds: tuple[Threshold, ...] = (),
) -> Any:
    cases = cases or (_case(expected=ANSWER),)
    runs = runs or tuple(_run() for _ in cases)
    suite = EvalSuite(name="timetable", version="1", cases=cases)
    results = tuple(
        CaseResult(case_id=case.id, run_id=f"r-{case.id}", status=CaseStatus.COMPLETED, run=run)
        for case, run in zip(cases, runs, strict=True)
    )
    outcome = SuiteResult(suite_name="timetable", suite_version="1", results=results)
    return measure(suite, outcome, metrics, thresholds=thresholds)


def _priced(amount: str, currency: str = "USD") -> Usage:
    return Usage(
        input_tokens=100,
        output_tokens=20,
        cost=Cost(output=Decimal(amount), currency=currency),
    )


class TestAValueThatIsNotKnown:
    def test_a_value_nobody_could_compute_is_not_zero(self) -> None:
        """Zero is a measurement; unknown is the absence of one, and they gate differently."""
        unpriced = _measured(CostPerCase(), run=_run(usage=Usage(input_tokens=1, output_tokens=1)))
        assert unpriced.value is None
        assert not unpriced.known
        assert "pric" in unpriced.reason

    def test_a_self_hosted_model_reports_cost_unknown_rather_than_free(self) -> None:
        usage = Usage(
            input_tokens=100,
            output_tokens=20,
            cost=Cost(currency="USD", confidence=CostConfidence.UNKNOWN),
        )
        assert _measured(CostPerCase(), run=_run(usage=usage)).value is None

    def test_a_response_without_usage_makes_the_token_metrics_unavailable(self) -> None:
        blank = _run(usage=Usage(input_tokens=0, output_tokens=0))
        assert _measured(TokensIn(), run=blank).value is None
        assert _measured(TokensOut(), run=blank).value is None

    def test_a_cancelled_run_is_not_scored_as_a_wrong_answer(self) -> None:
        cancelled = _run(answer="", state=RunState.CANCELLED)
        scored = _measured(ExactMatch(), _case(expected=ANSWER), cancelled)
        assert scored.value is None
        assert "cancelled" in scored.reason


class TestCorrectness:
    def test_every_correctness_metric_declines_to_judge_an_unfinished_run(self) -> None:
        stopped = _run(state=RunState.MAX_ITERATIONS_EXCEEDED)
        case = _case(expected=ANSWER, expected_tools=("search",), expected_sources=("a",))
        for metric in (SchemaValidity(), ToolSequenceMatch(), Groundedness(), RefusalRate()):
            assert metric.compute(case, stopped).value is None

    def test_the_answer_is_the_assistant_turn_not_whatever_came_last(self) -> None:
        run = _run()
        run.messages.append(Message(role="user", content=[TextPart(text="thanks")]))
        assert _measured(ExactMatch(), _case(expected=ANSWER), run).value == 1.0

    def test_an_exact_answer_scores_one_and_a_different_one_scores_zero(self) -> None:
        assert _measured(ExactMatch(), _case(expected=ANSWER)).value == 1.0
        assert _measured(ExactMatch(), _case(expected="on time")).value == 0.0

    def test_a_case_with_no_expected_answer_is_unknown_not_a_failure(self) -> None:
        assert _measured(ExactMatch(), _case()).value is None

    def test_exact_match_ignores_surrounding_whitespace_and_case(self) -> None:
        assert _measured(ExactMatch(), _case(expected="  Delayed By Nine Minutes ")).value == 1.0

    def test_the_declared_tool_sequence_has_to_match_in_order(self) -> None:
        case = _case(expected_tools=("search", "book"))
        called = _run(
            tool_calls=[
                ToolCall(id="1", name="search", arguments={}),
                ToolCall(id="2", name="book", arguments={}),
            ]
        )
        assert ToolSequenceMatch().compute(case, called).value == 1.0
        backwards = _run(
            tool_calls=[
                ToolCall(id="1", name="book", arguments={}),
                ToolCall(id="2", name="search", arguments={}),
            ]
        )
        assert ToolSequenceMatch().compute(case, backwards).value == 0.0

    def test_a_case_declaring_no_tool_sequence_is_unknown_not_a_match(self) -> None:
        assert _measured(ToolSequenceMatch()).value is None

    def test_a_run_with_nothing_from_the_assistant_answers_nothing(self) -> None:
        silent = _run()
        silent.messages.clear()
        assert _measured(ExactMatch(), _case(expected=ANSWER), silent).value == 0.0

    def test_structured_output_that_never_arrived_scores_zero(self) -> None:
        """A run that completed without the declared shape is a schema failure, not unknown."""
        assert _measured(SchemaValidity()).value == 0.0

    def test_a_citation_naming_a_source_the_case_declared_is_grounded(self) -> None:
        case = _case(expected_sources=("timetable-7", "notices-2"))
        assert Groundedness().compute(case, _run(answer="late [timetable-7]")).value == 1.0
        assert Groundedness().compute(case, _run(answer="late [invented-9]")).value == 0.0

    def test_an_answer_citing_nothing_is_unknown_rather_than_ungrounded(self) -> None:
        assert Groundedness().compute(_case(expected_sources=("a",)), _run()).value is None

    def test_a_refusal_is_counted_and_is_not_an_improvement(self) -> None:
        assert RefusalRate().higher_is_better is False
        assert _measured(RefusalRate(), run=_run(answer="I cannot help with that")).value == 1.0
        assert _measured(RefusalRate()).value == 0.0


class TestOperationalMetricsAreNotExtras:
    def test_tokens_come_from_the_run_the_provider_reported(self) -> None:
        assert _measured(TokensIn()).value == 100.0
        assert _measured(TokensOut()).value == 20.0

    def test_cost_carries_its_currency_so_two_providers_are_never_added_up(self) -> None:
        measured = _measured(CostPerCase(), run=_run(usage=_priced("0.02")))
        assert measured.value == pytest.approx(0.02)
        assert measured.currency == "USD"

    def test_latency_is_the_wall_clock_the_caller_waited(self) -> None:
        assert _measured(LatencyMs()).value == pytest.approx(400.0)
        assert LatencyMs().higher_is_better is False

    def test_a_run_that_never_started_has_no_latency_to_report(self) -> None:
        assert _measured(LatencyMs(), run=_run(started_at=None)).value is None

    def test_a_run_that_sent_no_prompt_has_no_cache_rate_to_report(self) -> None:
        empty = _run(usage=Usage(input_tokens=0, output_tokens=3))
        assert _measured(CacheHitRate(), run=empty).value is None

    def test_the_cache_hit_rate_is_the_share_of_input_served_from_cache(self) -> None:
        cached = _run(usage=Usage(input_tokens=100, cached_tokens=25, output_tokens=5))
        assert _measured(CacheHitRate(), run=cached).value == pytest.approx(0.25)


class TestAggregation:
    def test_the_mean_and_the_tail_are_both_reported(self) -> None:
        cases = tuple(_case(f"c{index}") for index in range(4))
        runs = tuple(_run(started_at=0.0, ended_at=each) for each in (0.1, 0.2, 0.3, 1.0))
        report = _report(LatencyMs(), cases=cases, runs=runs)
        latency = report.aggregate("latency_ms")
        assert latency.mean == pytest.approx(400.0)
        assert latency.p95 == pytest.approx(1000.0)

    def test_a_tiny_sample_says_so_rather_than_offering_an_interval(self) -> None:
        """One case cannot have a confidence interval, and printing one would invite trust."""
        single = _report(TokensIn()).aggregate("tokens_in")
        assert single.n == 1
        assert single.ci_low is None
        assert not single.reliable
        assert "sample" in single.note

    def test_a_larger_sample_carries_an_interval_around_its_mean(self) -> None:
        cases = tuple(_case(f"c{index}") for index in range(10))
        runs = tuple(
            _run(usage=Usage(input_tokens=100 + index, output_tokens=5)) for index in range(10)
        )
        tokens = _report(TokensIn(), cases=cases, runs=runs).aggregate("tokens_in")
        assert tokens.reliable
        assert tokens.ci_low is not None
        assert tokens.ci_low < tokens.mean < (tokens.ci_high or 0.0)

    def test_unknown_values_are_counted_apart_and_never_averaged_in(self) -> None:
        cases = (_case("c1", expected=ANSWER), _case("c2"))
        report = _report(ExactMatch(), cases=cases, runs=(_run(), _run()))
        scored = report.aggregate("exact_match")
        assert scored.n == 1
        assert scored.unknown == 1
        assert scored.mean == 1.0

    def test_a_metric_no_case_could_answer_has_no_mean_at_all(self) -> None:
        empty = _report(ExactMatch(), cases=(_case("c1"),)).aggregate("exact_match")
        assert empty.n == 0
        assert empty.mean is None

    def test_two_currencies_in_one_suite_refuse_to_be_summed(self) -> None:
        cases = (_case("c1"), _case("c2"))
        runs = (_run(usage=_priced("0.02")), _run(usage=_priced("0.03", "EUR")))
        cost = _report(CostPerCase(), cases=cases, runs=runs).aggregate("cost_per_case")
        assert cost.mean is None
        assert "EUR" in cost.note
        assert "USD" in cost.note

    def test_results_break_down_by_tag_so_one_slice_cannot_hide(self) -> None:
        cases = (
            _case("c1", expected=ANSWER, tags=("refunds",)),
            _case("c2", expected="something else", tags=("search",)),
        )
        report = _report(ExactMatch(), cases=cases, runs=(_run(), _run()))
        assert report.aggregate("exact_match", tag="refunds").mean == 1.0
        assert report.aggregate("exact_match", tag="search").mean == 0.0


class TestThresholds:
    def test_a_breached_cost_threshold_fails_the_report_a_quality_gain_would_have_passed(
        self,
    ) -> None:
        """The primary scenario: better answers, worse unit economics, and the report says so."""
        report = _report(
            ExactMatch(),
            CostPerCase(),
            cases=(_case(expected=ANSWER),),
            runs=(_run(usage=_priced("0.05")),),
            thresholds=(Threshold(metric="cost_per_case", maximum=0.02),),
        )
        assert report.aggregate("exact_match").mean == 1.0
        assert not report.ok
        assert report.exit_code == 1
        breached = report.verdict("cost_per_case")
        assert breached.verdict == "fail"
        assert breached.value == pytest.approx(0.05)
        assert breached.limit == pytest.approx(0.02)
        assert "cost_per_case" in report.summary()

    def test_a_metric_inside_its_threshold_passes(self) -> None:
        report = _report(
            CostPerCase(),
            runs=(_run(usage=_priced("0.001")),),
            thresholds=(Threshold(metric="cost_per_case", maximum=0.02),),
        )
        assert report.ok
        assert report.verdict("cost_per_case").verdict == "pass"

    def test_a_value_close_to_the_limit_warns_without_failing(self) -> None:
        report = _report(
            CostPerCase(),
            runs=(_run(usage=_priced("0.019")),),
            thresholds=(Threshold(metric="cost_per_case", maximum=0.02, warn_within=0.005),),
        )
        assert report.verdict("cost_per_case").verdict == "warn"
        assert report.ok

    def test_a_minimum_catches_correctness_falling_through_the_floor(self) -> None:
        report = _report(
            ExactMatch(),
            cases=(_case(expected="on time"),),
            thresholds=(Threshold(metric="exact_match", minimum=0.9),),
        )
        assert report.verdict("exact_match").verdict == "fail"

    def test_a_threshold_nobody_could_evaluate_fails_rather_than_passing_quietly(self) -> None:
        """A gate that clears itself when the number is missing is not a gate."""
        report = _report(
            CostPerCase(), thresholds=(Threshold(metric="cost_per_case", maximum=0.02),)
        )
        failed = report.verdict("cost_per_case")
        assert failed.verdict == "fail"
        assert "nothing" in failed.reason

    def test_a_threshold_with_no_bound_at_all_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tokens_in"):
            Threshold(metric="tokens_in")

    def test_a_value_just_above_a_floor_warns_before_it_breaches(self) -> None:
        report = _report(
            ExactMatch(),
            cases=(_case(expected=ANSWER),),
            thresholds=(Threshold(metric="exact_match", minimum=0.95, warn_within=0.1),),
        )
        assert report.verdict("exact_match").verdict == "warn"

    def test_asking_for_a_verdict_nobody_declared_is_a_lookup_failure(self) -> None:
        with pytest.raises(KeyError, match="tokens_in"):
            _report(TokensIn()).verdict("tokens_in")

    def test_a_verdict_lookup_does_not_answer_with_another_metrics_verdict(self) -> None:
        report = _report(
            TokensIn(), ExactMatch(), thresholds=(Threshold(metric="tokens_in", maximum=1000.0),)
        )
        with pytest.raises(KeyError, match="exact_match"):
            report.verdict("exact_match")

    def test_a_threshold_on_a_metric_nobody_registered_is_refused(self) -> None:
        with pytest.raises(KeyError, match="latency_ms"):
            _report(ExactMatch(), thresholds=(Threshold(metric="latency_ms", maximum=1.0),))


class TestWhenAMetricItselfBreaks:
    def test_a_custom_metric_that_raises_errors_that_case_and_fails_the_report(self) -> None:
        class Exploding:
            name = "exploding"
            higher_is_better = True

            def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
                raise ZeroDivisionError("division by zero")

        report = _report(Exploding())
        assert not report.ok
        assert report.exit_code == 1
        broke = report.failures[0]
        assert broke.metric == "exploding"
        assert broke.case_id == "c1"
        assert "ZeroDivisionError" in broke.traceback

    def test_a_broken_metric_is_never_coerced_to_zero(self) -> None:
        class Exploding:
            name = "exploding"
            higher_is_better = True

            def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
                raise ZeroDivisionError("division by zero")

        aggregated = _report(Exploding()).aggregate("exploding")
        assert aggregated.n == 0
        assert aggregated.mean is None

    def test_one_broken_metric_does_not_stop_the_others_being_reported(self) -> None:
        class Exploding:
            name = "exploding"
            higher_is_better = True

            def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
                raise ZeroDivisionError("division by zero")

        report = _report(Exploding(), TokensIn())
        assert report.aggregate("tokens_in").mean == 100.0

    def test_two_metrics_sharing_a_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="exact_match"):
            _report(ExactMatch(), ExactMatch())


class TestTheReport:
    def test_a_case_that_never_ran_is_reported_apart_from_the_scores(self) -> None:
        suite = EvalSuite(name="timetable", version="1", cases=(_case("c1"), _case("c2")))
        outcome = SuiteResult(
            suite_name="timetable",
            suite_version="1",
            results=(
                CaseResult(case_id="c1", run_id="r1", status=CaseStatus.COMPLETED, run=_run()),
                CaseResult(
                    case_id="c2", run_id="r2", status=CaseStatus.ERRORED, reason="stale cassette"
                ),
            ),
        )
        report = measure(suite, outcome, (TokensIn(),))
        assert report.unscored == ("c2",)
        assert report.aggregate("tokens_in").n == 1

    def test_a_report_within_its_limits_says_so_in_one_line(self) -> None:
        report = _report(TokensIn(), thresholds=(Threshold(metric="tokens_in", maximum=1000.0),))
        assert "all within limits" in report.summary()

    def test_a_metric_that_raised_is_named_in_the_summary(self) -> None:
        class Exploding:
            name = "exploding"
            higher_is_better = True

            def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the protocol's shape
                raise ZeroDivisionError("division by zero")

        assert "raised exploding" in _report(Exploding()).summary()

    def test_the_table_names_every_metric_and_its_unit(self) -> None:
        table = _report(CostPerCase(), LatencyMs(), runs=(_run(usage=_priced("0.02")),)).table()
        assert "cost_per_case" in table
        assert "USD" in table
        assert "ms" in table

    def test_the_json_report_is_what_ci_reads(self) -> None:
        report = _report(
            TokensIn(), thresholds=(Threshold(metric="tokens_in", maximum=1000.0),)
        ).as_dict()
        assert report["suite"] == "timetable"
        assert report["metrics"]["tokens_in"]["mean"] == 100.0
        assert report["verdicts"]["tokens_in"]["verdict"] == "pass"
        assert report["ok"] is True

    def test_an_aggregate_nobody_asked_for_is_a_lookup_failure_not_an_empty_row(self) -> None:
        with pytest.raises(KeyError, match="cost_per_case"):
            _report(TokensIn()).aggregate("cost_per_case")


class TestTheProtocol:
    def test_a_consumer_metric_needs_only_the_three_names(self) -> None:
        class WordCount:
            name = "word_count"
            higher_is_better = False

            def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
                return MetricValue(value=float(len(str(run.messages[-1].content[0]))), unit="words")

        assert _report(WordCount()).aggregate("word_count").mean > 0

    def test_a_built_in_metric_declares_which_direction_is_better(self) -> None:
        assert ExactMatch().higher_is_better is True
        assert CostPerCase().higher_is_better is False

    def test_an_aggregate_is_readable_on_its_own(self) -> None:
        aggregate = Aggregate(metric="tokens_in", n=2, unknown=0, mean=10.0, p50=10.0, p95=10.0)
        assert "tokens_in" in str(aggregate.line())
