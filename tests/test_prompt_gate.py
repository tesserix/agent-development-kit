"""Whether a prompt change is allowed to reach the alias production reads."""

from __future__ import annotations

import pytest

from tesserix_adk.core import ConfigurationError, EvalIncompleteError, IncomparableEvalError
from tesserix_adk.evals import (
    DEFAULT_POLICY,
    Bypass,
    GatePolicy,
    Measured,
    Tolerance,
    gate,
)

BASELINE = Measured(
    prompt="itinerary_system",
    version="4",
    digest="aaaa",
    examples=200,
    scored=200,
    metrics={
        "task_success": 0.91,
        "schema_validity": 1.0,
        "judge_score": 4.2,
        "p95_latency_ms": 1800.0,
        "cost_per_run": 0.012,
    },
    variables=("budget", "city"),
    judge="judge-v3",
)


def candidate(**metrics: float) -> Measured:
    """The baseline's twin at version 5, with the named metrics moved."""
    return BASELINE.model_copy(
        update={
            "version": "5",
            "digest": "bbbb",
            "metrics": {**BASELINE.metrics, **metrics},
        }
    )


class TestJudgingTheChange:
    """The comparison a pull request wants: did anything get worse, and by how much."""

    def test_a_change_that_moves_nothing_passes(self) -> None:
        report = gate(BASELINE, candidate())

        assert report.verdict == "pass"

    def test_a_quality_drop_beyond_the_tolerance_fails(self) -> None:
        report = gate(BASELINE, candidate(task_success=0.84))

        assert report.verdict == "fail"

    def test_a_drop_within_the_tolerance_passes(self) -> None:
        report = gate(BASELINE, candidate(task_success=0.905))

        assert report.verdict == "pass"

    def test_an_improvement_is_never_a_regression(self) -> None:
        report = gate(BASELINE, candidate(task_success=0.97, cost_per_run=0.004))

        assert report.verdict == "pass"

    def test_a_metric_where_lower_is_better_is_read_the_right_way_round(self) -> None:
        slower = gate(BASELINE, candidate(p95_latency_ms=4000.0))
        faster = gate(BASELINE, candidate(p95_latency_ms=900.0))

        assert slower.verdict == "fail"
        assert faster.verdict == "pass"

    def test_schema_validity_admits_no_regression_at_all(self) -> None:
        report = gate(BASELINE, candidate(schema_validity=0.999))

        assert report.verdict == "fail"

    def test_the_report_names_the_metric_that_moved_and_by_how_much(self) -> None:
        report = gate(BASELINE, candidate(task_success=0.80))

        moved = next(move for move in report.moves if move.verdict == "fail")
        assert moved.metric == "task_success"
        assert moved.before == pytest.approx(0.91)
        assert moved.after == pytest.approx(0.80)
        assert moved.regression == pytest.approx(0.11)

    def test_only_declared_metrics_decide_anything(self) -> None:
        policy = GatePolicy(tolerances=(Tolerance(metric="task_success", tolerance=0.01),))

        report = gate(BASELINE, candidate(cost_per_run=9.99), policy=policy)

        assert report.verdict == "pass"
        assert [move.metric for move in report.moves] == ["task_success"]

    def test_a_policy_declaring_a_metric_twice_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            GatePolicy(
                tolerances=(Tolerance(metric="task_success"), Tolerance(metric="task_success"))
            )


class TestCostIsAGate:
    """Spend is not a footnote under the quality numbers."""

    def test_a_cost_regression_fails_on_its_own(self) -> None:
        report = gate(BASELINE, candidate(cost_per_run=0.02))

        assert report.verdict == "fail"
        assert "cost_per_run" in report.summary()

    def test_a_cost_regression_still_fails_when_quality_improved(self) -> None:
        report = gate(BASELINE, candidate(task_success=0.98, cost_per_run=0.05))

        assert report.verdict == "fail"

    def test_taking_the_cost_deliberately_needs_a_recorded_exception(self) -> None:
        report = gate(
            BASELINE,
            candidate(task_success=0.98, cost_per_run=0.05),
            bypass=Bypass(
                metrics=("cost_per_run",),
                by="ada",
                reason="PLAT-102, worth the spend until the shorter rewrite lands",
            ),
        )

        assert report.verdict == "pass"
        assert report.bypassed == ("cost_per_run",)
        assert "bypassed" in report.summary()

    def test_a_bypass_only_excuses_what_it_names(self) -> None:
        report = gate(
            BASELINE,
            candidate(task_success=0.4, cost_per_run=0.05),
            bypass=Bypass(metrics=("cost_per_run",), by="ada", reason="PLAT-102"),
        )

        assert report.verdict == "fail"

    def test_a_bypass_nobody_signed_or_justified_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            Bypass(metrics=("cost_per_run",), by="ada")

        with pytest.raises(ConfigurationError):
            Bypass(metrics=("cost_per_run",), by="", reason="because")


class TestFailingClosed:
    """Every way the comparison can be meaningless, and the refusal for each."""

    def test_a_partly_scored_candidate_refuses_rather_than_passes(self) -> None:
        with pytest.raises(EvalIncompleteError) as raised:
            gate(BASELINE, candidate().model_copy(update={"scored": 150}))

        assert raised.value.coverage == pytest.approx(0.75)
        assert raised.value.version == "5"

    def test_a_partly_scored_baseline_refuses_too(self) -> None:
        with pytest.raises(EvalIncompleteError):
            gate(BASELINE.model_copy(update={"scored": 1}), candidate())

    def test_a_dataset_nobody_ran_is_not_full_coverage(self) -> None:
        empty = BASELINE.model_copy(update={"examples": 0, "scored": 0})

        with pytest.raises(EvalIncompleteError):
            gate(empty, candidate())

    def test_a_declared_metric_nobody_computed_fails(self) -> None:
        thin = candidate().model_copy(update={"metrics": {"task_success": 0.99}})

        report = gate(BASELINE, thin)

        assert report.verdict == "fail"
        assert [move.metric for move in report.moves if move.verdict == "fail"] == [
            "schema_validity",
            "judge_score",
            "p95_latency_ms",
            "cost_per_run",
        ]

    def test_a_judge_that_moved_since_the_baseline_is_incomparable(self) -> None:
        with pytest.raises(IncomparableEvalError) as raised:
            gate(BASELINE, candidate().model_copy(update={"judge": "judge-v4"}))

        assert raised.value.reason == "judge"
        assert raised.value.retryable is False

    def test_a_prompt_that_changed_its_variables_invalidates_the_dataset(self) -> None:
        with pytest.raises(IncomparableEvalError) as raised:
            gate(BASELINE, candidate().model_copy(update={"variables": ("city",)}))

        assert raised.value.reason == "variables"
        assert "budget" in str(raised.value)

    def test_the_incomplete_refusal_is_retryable_once_the_rest_is_scored(self) -> None:
        with pytest.raises(EvalIncompleteError) as raised:
            gate(BASELINE, candidate().model_copy(update={"scored": 199}))

        assert raised.value.retryable is True


class TestVerdictsOnTheBoundary:
    """A metric that flakes across the line costs a rerun, not a coin flip."""

    def test_a_regression_inside_the_noise_band_asks_for_a_repeat(self) -> None:
        report = gate(BASELINE, candidate(judge_score=4.14))

        assert report.verdict == "repeat"

    def test_a_regression_past_the_noise_band_fails(self) -> None:
        report = gate(BASELINE, candidate(judge_score=4.0))

        assert report.verdict == "fail"

    def test_a_failure_anywhere_outweighs_a_repeat(self) -> None:
        report = gate(BASELINE, candidate(judge_score=4.14, task_success=0.5))

        assert report.verdict == "fail"

    def test_the_policy_says_how_many_runs_to_average(self) -> None:
        assert DEFAULT_POLICY.repeats == 3


class TestWhatTheResultIsGoodFor:
    """The record, so promotion and rollback read the same verdict CI did."""

    def test_a_pass_permits_promotion_of_the_digest_it_measured(self) -> None:
        report = gate(BASELINE, candidate())

        assert report.permits("bbbb") is True

    def test_it_does_not_permit_a_digest_it_never_saw(self) -> None:
        report = gate(BASELINE, candidate())

        assert report.permits("cccc") is False
        assert report.permits("") is False

    def test_a_failing_result_permits_nothing(self) -> None:
        report = gate(BASELINE, candidate(task_success=0.1))

        assert report.permits("bbbb") is False

    def test_the_record_carries_the_version_digest_and_verdict(self) -> None:
        report = gate(BASELINE, candidate())

        assert report.attributes() == {
            "adk.prompt": "itinerary_system@5",
            "adk.prompt_digest": "bbbb",
            "adk.eval_verdict": "pass",
            "adk.eval_baseline": "4",
        }

    def test_the_summary_reads_as_a_ci_comment(self) -> None:
        summary = gate(BASELINE, candidate(cost_per_run=0.02)).summary()

        assert summary.splitlines()[0] == "itinerary_system@5 vs 4: FAIL"
        assert "cost_per_run: 0.012 -> 0.02 (fail, tolerance 0.0005)" in summary

    def test_coverage_is_reported_as_a_share(self) -> None:
        assert BASELINE.coverage == pytest.approx(1.0)
        assert BASELINE.model_copy(update={"scored": 50}).coverage == pytest.approx(0.25)
