"""Savings reported as measured or estimated, with a holdout that validates the estimate."""

from __future__ import annotations

import pytest

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.observability import (
    Arm,
    Basis,
    HoldoutPolicy,
    ShapedRun,
    account,
    by_tenant,
)


def _run(
    run_id: str,
    *,
    arm: Arm = Arm.SHAPED,
    before: int = 1000,
    after: int = 400,
    output: int = 100,
    tenant: str = "acme",
) -> ShapedRun:
    """One accounted run, in whichever arm it was assigned."""
    return ShapedRun(
        run_id=run_id,
        tenant=tenant,
        arm=arm,
        input_tokens_before=before,
        input_tokens_after=after,
        output_tokens=output,
    )


def _both_arms(shaped: int = 60, holdout: int = 60) -> tuple[ShapedRun, ...]:
    """Traffic through both arms, shaping producing shorter answers."""
    return (
        *(_run(f"s-{index}", output=100) for index in range(shaped)),
        *(_run(f"h-{index}", arm=Arm.HOLDOUT, after=1000, output=150) for index in range(holdout)),
    )


class TestInputSavingsAreMeasured:
    """Both counts exist at the same moment, so nothing here is a counterfactual."""

    def test_it_is_the_difference_between_the_paired_counts(self) -> None:
        report = account((_run("a"), _run("b")), policy=HoldoutPolicy(fraction=0.1))
        assert report.input.tokens == 1200
        assert report.input.basis is Basis.MEASURED

    def test_a_holdout_run_saved_no_input(self) -> None:
        held = _run("h", arm=Arm.HOLDOUT, after=1000)
        assert held.input_saved == 0

    def test_the_label_says_measured(self) -> None:
        report = account((_run("a"),), policy=HoldoutPolicy(fraction=0.1))
        assert "measured" in report.input.label()


class TestOutputSavingsAreEstimated:
    """The response that was never received cannot be measured, only compared against."""

    def test_it_comes_from_the_holdout_comparison_with_an_interval(self) -> None:
        report = account(_both_arms(), policy=HoldoutPolicy(fraction=0.5))
        assert report.output.basis is Basis.ESTIMATED
        assert report.output.low is not None
        assert report.output.high is not None
        assert report.output.low < report.output.tokens < report.output.high

    def test_it_is_never_labelled_measured(self) -> None:
        report = account(_both_arms(), policy=HoldoutPolicy(fraction=0.5))
        assert "estimated" in report.output.label()
        assert "measured" not in report.output.label()

    def test_shorter_answers_under_shaping_read_as_a_saving(self) -> None:
        report = account(_both_arms(), policy=HoldoutPolicy(fraction=0.5))
        assert report.output.tokens > 0


class TestWithoutAControl:
    """A holdout of zero is a number nobody can defend, and it has to say so."""

    def test_it_is_estimated_and_says_no_control_exists(self) -> None:
        report = account((_run("a"), _run("b")), policy=HoldoutPolicy(fraction=0.0))
        assert report.output.basis is Basis.ESTIMATED
        assert "no holdout" in report.output.reason

    def test_the_absence_is_surfaced_rather_than_omitted(self) -> None:
        report = account((_run("a"),), policy=HoldoutPolicy(fraction=0.0))
        assert "no holdout" in report.summary()

    def test_input_savings_are_still_measured_without_a_control(self) -> None:
        report = account((_run("a"),), policy=HoldoutPolicy(fraction=0.0))
        assert report.input.basis is Basis.MEASURED


class TestTooSmallASample:
    """A wide interval reads as a result; insufficient data reads as what it is."""

    def test_it_reports_insufficient_rather_than_a_wide_interval(self) -> None:
        report = account(_both_arms(shaped=3, holdout=2), policy=HoldoutPolicy(fraction=0.4))
        assert report.output.basis is Basis.INSUFFICIENT
        assert report.output.low is None
        assert report.output.high is None

    def test_it_says_how_many_runs_each_arm_needs(self) -> None:
        report = account(_both_arms(shaped=3, holdout=2), policy=HoldoutPolicy(fraction=0.4))
        assert "30" in report.output.reason

    def test_an_empty_arm_is_insufficient_not_a_hundred_percent_saving(self) -> None:
        report = account(
            tuple(_run(f"s-{index}") for index in range(60)),
            policy=HoldoutPolicy(fraction=0.5),
        )
        assert report.output.basis is Basis.INSUFFICIENT


class TestLoweringTheMinimum:
    """A caller may compare smaller arms, and gets a band rather than false certainty."""

    def test_a_single_run_per_arm_still_carries_an_interval(self) -> None:
        runs = (_run("s"), _run("h", arm=Arm.HOLDOUT, after=1000, output=150))
        report = account(runs, policy=HoldoutPolicy(fraction=0.5), minimum=1)
        assert report.output.basis is Basis.ESTIMATED
        assert report.output.low is not None
        assert report.output.high is not None
        assert report.output.low < report.output.tokens < report.output.high


class TestAssigningTheArm:
    """Which arm a run is in is decided once and recorded, shaping enabled or not."""

    def test_the_same_run_always_lands_in_the_same_arm(self) -> None:
        policy = HoldoutPolicy(fraction=0.5)
        assert policy.arm("run-7") is policy.arm("run-7")

    def test_a_retry_does_not_switch_arms_mid_flight(self) -> None:
        policy = HoldoutPolicy(fraction=0.5)
        arms = {policy.arm("run-7") for _ in range(20)}
        assert len(arms) == 1

    def test_roughly_the_configured_share_is_held_out(self) -> None:
        policy = HoldoutPolicy(fraction=0.2)
        held = sum(policy.arm(f"run-{index}") is Arm.HOLDOUT for index in range(2000))
        assert 300 < held < 500

    def test_a_zero_fraction_holds_nothing_out(self) -> None:
        policy = HoldoutPolicy(fraction=0.0)
        assert all(policy.arm(f"run-{index}") is Arm.SHAPED for index in range(200))

    def test_an_arm_is_recorded_even_where_shaping_is_off(self) -> None:
        run = _run("s", after=1000)
        assert run.arm is Arm.SHAPED
        assert run.input_saved == 0

    def test_a_fraction_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 1"):
            HoldoutPolicy(fraction=1.4)


class TestTenantIsolation:
    """One tenant's traffic never contributes to another's figure."""

    def test_by_tenant_never_mixes_two_tenants(self) -> None:
        runs = (_run("a", tenant="acme"), _run("b", tenant="beta", before=2000, after=500))
        reports = by_tenant(runs, policy=HoldoutPolicy(fraction=0.1))
        assert reports["acme"].input.tokens == 600
        assert reports["beta"].input.tokens == 1500

    def test_each_tenant_keeps_its_own_run_count(self) -> None:
        runs = (_run("a", tenant="acme"), _run("b", tenant="acme"), _run("c", tenant="beta"))
        reports = by_tenant(runs, policy=HoldoutPolicy(fraction=0.1))
        assert reports["acme"].runs == 2
        assert reports["beta"].runs == 1


class TestItCountsTokensNeverContent:
    """An accounting record that carries text is a copy of the conversation."""

    def test_no_field_holds_content(self) -> None:
        assert set(ShapedRun.model_fields) == {
            "run_id",
            "tenant",
            "arm",
            "input_tokens_before",
            "input_tokens_after",
            "output_tokens",
        }

    def test_the_report_carries_counts_and_figures_only(self) -> None:
        document = account((_run("a"),), policy=HoldoutPolicy(fraction=0.0)).as_dict()
        assert set(document) == {"runs", "shaped", "holdout", "input", "output"}


class TestWhatARunReports:
    """The report is what a dashboard draws, so it has to be honest at a glance."""

    def test_it_counts_both_arms(self) -> None:
        report = account(_both_arms(shaped=40, holdout=10), policy=HoldoutPolicy(fraction=0.2))
        assert report.runs == 50
        assert report.shaped == 40
        assert report.holdout == 10

    def test_the_summary_labels_both_figures(self) -> None:
        summary = account(_both_arms(), policy=HoldoutPolicy(fraction=0.5)).summary()
        assert "measured" in summary
        assert "estimated" in summary

    def test_the_machine_readable_form_carries_the_basis(self) -> None:
        document = account(_both_arms(), policy=HoldoutPolicy(fraction=0.5)).as_dict()
        measured, estimated = document["input"], document["output"]
        assert isinstance(measured, dict)
        assert isinstance(estimated, dict)
        assert measured["basis"] == "measured"
        assert estimated["basis"] == "estimated"

    def test_a_run_counted_twice_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="one record per run"):
            account((_run("a"), _run("a")), policy=HoldoutPolicy(fraction=0.1))

    def test_after_may_not_exceed_before(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot grow"):
            ShapedRun(
                run_id="a",
                tenant="acme",
                arm=Arm.SHAPED,
                input_tokens_before=100,
                input_tokens_after=200,
                output_tokens=10,
            )
