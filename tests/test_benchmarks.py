"""What the benchmark harness measures, and when it refuses to draw a conclusion.

A performance gate that fails on a noisy runner gets disabled within a fortnight, and one
that passes anything a noisy runner produces defends nothing. These pin both halves: a
regression larger than the measured noise is named, and a delta the noise could have
produced is reported as inconclusive with what a conclusive run would need.
"""

from __future__ import annotations

import json
import tracemalloc
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.testing.benchmarks import (
    Baseline,
    BenchmarkReport,
    Comparison,
    Measurement,
    Metric,
    Scenario,
    Thresholds,
    Verdict,
    compare,
    load_baseline,
    measure,
    run_suite,
    write_baseline,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path


# The harness collects garbage while it measures, which finalises whatever else the test
# session left unclosed. Those finalisers are not this module's to answer for.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


class Ticks:
    """A clock handing out the durations the test wrote, one per measured iteration."""

    def __init__(self, *durations: float, otherwise: float = 0.001) -> None:
        self.queued = list(durations)
        self.otherwise = otherwise
        self.now = 0.0
        self.reads = 0
        self.tracing: list[bool] = []

    def __call__(self) -> float:
        self.tracing.append(tracemalloc.is_tracing())
        self.reads += 1
        if self.reads % 2 == 0:
            self.now += self.queued.pop(0) if self.queued else self.otherwise
        return self.now


class Work:
    """A scenario body that counts how many times it ran and allocates a little."""

    def __init__(self, *, allocate: int = 0) -> None:
        self.runs = 0
        self.allocate = allocate
        self.held: list[object] = []

    async def __call__(self) -> None:
        self.runs += 1
        if self.allocate:
            self.held = [object() for _ in range(self.allocate)]


class Spiky:
    """A body that allocates and holds on the passes the test names, counting from one."""

    def __init__(self, *heavy: int) -> None:
        self.heavy = set(heavy)
        self.passes = 0
        self.held: list[object] = []

    async def __call__(self) -> None:
        self.passes += 1
        self.held = [object() for _ in range(50_000)] if self.passes in self.heavy else []


class Garbage:
    """A body that drops a reference cycle every pass, which only a collection reclaims."""

    def __init__(self, *, cyclic: bool) -> None:
        self.cyclic = cyclic

    async def __call__(self) -> None:
        dropped: list[object] = [object() for _ in range(5_000)]
        if self.cyclic:
            dropped.append(dropped)


def scenario(body: Callable[[], Awaitable[None]], **overrides: object) -> Scenario:
    """One scenario, small enough that a test can spell out every duration it takes."""
    fields: dict[str, object] = {
        "name": "single-turn",
        "run": body,
        "iterations": 2,
        "warmup": 1,
        "rounds": 2,
    }
    return Scenario(**{**fields, **overrides})  # type: ignore[arg-type]


def measured(**values: float) -> Measurement:
    """A measurement of the values a comparison test cares about."""
    metrics: dict[Metric, float] = {Metric(name): value for name, value in values.items()}
    return Measurement(
        scenario="single-turn",
        python="3.13",
        values=metrics,
        spread=0.01,
        rounds=5,
        iterations=50,
    )


def baseline(**values: float) -> Baseline:
    """A committed baseline for the same scenario on the same interpreter."""
    return Baseline({("single-turn", "3.13"): {Metric(n): v for n, v in values.items()}})


def verdicts(report: BenchmarkReport) -> Mapping[Metric, Verdict]:
    """The verdict per metric, which is what most of these assert on."""
    return {one.metric: one.verdict for one in report.comparisons}


class TestMeasuring:
    """What one scenario run produces, and what it deliberately does not time."""

    async def test_every_measured_iteration_of_every_round_is_timed(self) -> None:
        body = Work()
        clock = Ticks()

        await measure(scenario(body), timer=clock)

        assert clock.reads == 2 * 2 * 2

    async def test_warm_up_iterations_run_but_are_not_measured(self) -> None:
        body = Work()
        clock = Ticks()

        result = await measure(scenario(body, warmup=3), timer=clock)

        # Two timed rounds of warm-up plus measured iterations, then the memory rounds.
        assert body.runs == (3 + 2) * 2 * 2
        assert clock.reads == 2 * 2 * 2
        assert result.iterations == 2

    async def test_the_percentiles_come_from_the_iterations_that_were_kept(self) -> None:
        clock = Ticks(1.0, 2.0, 3.0, 4.0)

        result = await measure(scenario(Work(), rounds=2, drop_slowest=False), timer=clock)

        assert result.values[Metric.LATENCY_P50] == pytest.approx(2.0)
        assert result.values[Metric.LATENCY_P99] == pytest.approx(4.0)

    async def test_throughput_is_the_iterations_over_the_time_they_took(self) -> None:
        clock = Ticks(0.5, 0.5, 0.5, 0.5)

        result = await measure(scenario(Work(), rounds=2, drop_slowest=False), timer=clock)

        assert result.values[Metric.THROUGHPUT] == pytest.approx(2.0)

    async def test_the_spread_is_how_far_the_rounds_disagreed(self) -> None:
        clock = Ticks(1.0, 1.0, 3.0, 3.0)

        result = await measure(scenario(Work(), rounds=2, drop_slowest=False), timer=clock)

        assert result.spread == pytest.approx(0.5)

    async def test_a_steady_run_has_no_spread(self) -> None:
        clock = Ticks(1.0, 1.0, 1.0, 1.0)

        result = await measure(scenario(Work(), rounds=2, drop_slowest=False), timer=clock)

        assert result.spread == pytest.approx(0.0)

    async def test_the_slowest_round_is_dropped_where_there_are_rounds_to_spare(self) -> None:
        clock = Ticks(1.0, 1.0, 1.0, 1.0, 9.0, 9.0)

        result = await measure(scenario(Work(), rounds=3), timer=clock)

        assert result.values[Metric.LATENCY_P99] == pytest.approx(1.0)

    async def test_a_single_round_keeps_what_it_measured(self) -> None:
        clock = Ticks(9.0, 9.0)

        result = await measure(scenario(Work(), rounds=1), timer=clock)

        assert result.values[Metric.LATENCY_P50] == pytest.approx(9.0)

    async def test_memory_is_recorded(self) -> None:
        result = await measure(scenario(Work(allocate=2_000)), timer=Ticks())

        assert result.values[Metric.PEAK_BYTES] > 0
        assert result.values[Metric.ALLOCATIONS] > 0

    async def test_one_greedy_memory_round_does_not_move_the_figure(self) -> None:
        # Nine timed passes, then memory rounds of a warm-up and two counted passes: the
        # first of those counts passes 11 and 12.
        steady = await measure(scenario(Spiky(), rounds=3), timer=Ticks())
        spiky = await measure(scenario(Spiky(11, 12), rounds=3), timer=Ticks())

        assert spiky.values[Metric.ALLOCATIONS] == pytest.approx(
            steady.values[Metric.ALLOCATIONS], rel=0.5
        )

    async def test_the_memory_warm_up_is_not_billed_to_the_round_it_warmed(self) -> None:
        # Three timed passes, then the memory round's own warm-up: pass 4.
        steady = await measure(scenario(Spiky(), rounds=1), timer=Ticks())
        warming = await measure(scenario(Spiky(4), rounds=1), timer=Ticks())

        assert warming.values[Metric.PEAK_BYTES] == pytest.approx(
            steady.values[Metric.PEAK_BYTES], rel=0.5
        )

    async def test_garbage_nobody_has_swept_yet_is_not_billed_as_live(self) -> None:
        plain = await measure(scenario(Garbage(cyclic=False)), timer=Ticks())
        cyclic = await measure(scenario(Garbage(cyclic=True)), timer=Ticks())

        # Both counts sit near zero, where a relative tolerance is ambient noise; the
        # claim is that 5_000 uncollected objects do not appear, so give it a floor.
        assert cyclic.values[Metric.ALLOCATIONS] == pytest.approx(
            plain.values[Metric.ALLOCATIONS], rel=0.5, abs=50
        )

    async def test_allocation_tracing_is_off_while_the_timings_are_taken(self) -> None:
        clock = Ticks()

        await measure(scenario(Work(allocate=100)), timer=clock)

        assert not any(clock.tracing)

    async def test_token_overhead_is_recorded_where_a_scenario_reports_it(self) -> None:
        body = Work()

        result = await measure(scenario(body, tokens=lambda: body.runs * 420), timer=Ticks())

        assert result.values[Metric.TOKENS] == pytest.approx(420.0)

    async def test_a_scenario_reporting_no_tokens_has_no_token_metric(self) -> None:
        result = await measure(scenario(Work()), timer=Ticks())

        assert Metric.TOKENS not in result.values

    async def test_the_interpreter_is_recorded_because_numbers_are_not_comparable(
        self,
    ) -> None:
        result = await measure(scenario(Work()), timer=Ticks(), python="3.12")

        assert result.python == "3.12"

    async def test_a_suite_measures_every_scenario_it_was_given(self) -> None:
        first, second = Work(), Work()

        results = await run_suite(
            (scenario(first), scenario(second, name="streaming")), timer=Ticks()
        )

        assert [one.scenario for one in results] == ["single-turn", "streaming"]


class TestComparing:
    """A measurement against a committed baseline, per metric and per interpreter."""

    def test_a_metric_inside_its_threshold_is_within(self) -> None:
        report = compare((measured(latency_p95=1.05),), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.WITHIN

    def test_a_metric_past_its_threshold_is_a_regression(self) -> None:
        report = compare((measured(latency_p95=1.4),), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.REGRESSED

    def test_the_regression_carries_the_delta_that_caused_it(self) -> None:
        report = compare((measured(latency_p95=1.4),), baseline(latency_p95=1.0))

        assert report.comparisons[0].delta == pytest.approx(0.4)

    def test_a_metric_comfortably_better_is_an_improvement(self) -> None:
        report = compare((measured(latency_p95=0.5),), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.IMPROVED

    def test_falling_throughput_is_a_regression_though_the_number_went_down(self) -> None:
        report = compare((measured(throughput=800.0),), baseline(throughput=1_000.0))

        assert verdicts(report)[Metric.THROUGHPUT] is Verdict.REGRESSED

    def test_rising_throughput_is_an_improvement(self) -> None:
        report = compare((measured(throughput=1_400.0),), baseline(throughput=1_000.0))

        assert verdicts(report)[Metric.THROUGHPUT] is Verdict.IMPROVED

    def test_any_extra_token_overhead_is_a_regression(self) -> None:
        report = compare((measured(tokens=421.0),), baseline(tokens=420.0))

        assert verdicts(report)[Metric.TOKENS] is Verdict.REGRESSED

    def test_a_metric_with_no_baseline_is_unrecorded_rather_than_passed(self) -> None:
        report = compare((measured(peak_bytes=1_000.0),), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.PEAK_BYTES] is Verdict.UNRECORDED

    def test_a_baseline_for_another_interpreter_is_not_borrowed(self) -> None:
        recorded = Baseline({("single-turn", "3.12"): {Metric.LATENCY_P95: 1.0}})

        report = compare((measured(latency_p95=4.0),), recorded)

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.UNRECORDED

    def test_a_threshold_may_be_widened_deliberately_per_metric(self) -> None:
        thresholds = Thresholds(limits={Metric.LATENCY_P95: 0.5})

        report = compare((measured(latency_p95=1.4),), baseline(latency_p95=1.0), thresholds)

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.WITHIN

    def test_a_baseline_of_zero_does_not_divide_by_it(self) -> None:
        report = compare((measured(allocations=5.0),), baseline(allocations=0.0))

        assert verdicts(report)[Metric.ALLOCATIONS] is Verdict.REGRESSED

    def test_a_zero_measurement_against_a_zero_baseline_is_within(self) -> None:
        report = compare((measured(allocations=0.0),), baseline(allocations=0.0))

        assert verdicts(report)[Metric.ALLOCATIONS] is Verdict.WITHIN


class TestVariance:
    """A noisy runner must produce neither a spurious failure nor a silent pass."""

    def test_peak_memory_is_not_judged_against_a_baseline_of_another_size(self) -> None:
        recorded = Baseline(
            {("single-turn", "3.13"): {Metric.PEAK_BYTES: 1e6}}, {("single-turn", "3.13"): 5}
        )

        report = compare((measured(peak_bytes=4e6),), recorded)

        assert verdicts(report)[Metric.PEAK_BYTES] is Verdict.INCONCLUSIVE

    def test_a_per_iteration_metric_is_judged_whatever_size_it_was_recorded_at(self) -> None:
        recorded = Baseline(
            {("single-turn", "3.13"): {Metric.TOKENS: 420.0}}, {("single-turn", "3.13"): 5}
        )

        report = compare((measured(tokens=420.0),), recorded)

        assert verdicts(report)[Metric.TOKENS] is Verdict.WITHIN

    def test_a_change_too_small_to_be_worth_a_percentage_is_within(self) -> None:
        report = compare((measured(allocations=2.8),), baseline(allocations=2.0))

        assert verdicts(report)[Metric.ALLOCATIONS] is Verdict.WITHIN

    def test_the_floor_does_not_hide_a_change_that_is_large_in_absolute_terms(self) -> None:
        report = compare((measured(allocations=40.0),), baseline(allocations=2.0))

        assert verdicts(report)[Metric.ALLOCATIONS] is Verdict.REGRESSED

    def test_a_floor_may_be_set_per_metric(self) -> None:
        thresholds = Thresholds(floors={Metric.LATENCY_P95: 0.5})

        report = compare((measured(latency_p95=1.4),), baseline(latency_p95=1.0), thresholds)

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.WITHIN

    def test_a_negative_floor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            Thresholds(floors={Metric.ALLOCATIONS: -1.0})

    def test_a_delta_the_noise_could_have_produced_is_inconclusive(self) -> None:
        noisy = Measurement("single-turn", "3.13", {Metric.LATENCY_P95: 1.2}, 0.3, 5, 50)

        report = compare((noisy,), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.INCONCLUSIVE

    def test_an_inconclusive_result_says_what_a_conclusive_one_would_need(self) -> None:
        noisy = Measurement("single-turn", "3.13", {Metric.LATENCY_P95: 1.2}, 0.3, 5, 50)

        report = compare((noisy,), baseline(latency_p95=1.0))

        assert "30.0%" in report.comparisons[0].reason
        assert "more rounds" in report.comparisons[0].reason

    def test_a_regression_larger_than_the_noise_is_still_named(self) -> None:
        noisy = Measurement("single-turn", "3.13", {Metric.LATENCY_P95: 3.0}, 0.3, 5, 50)

        report = compare((noisy,), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.REGRESSED

    def test_a_quiet_run_inside_the_ceiling_is_judged_normally(self) -> None:
        quiet = Measurement("single-turn", "3.13", {Metric.LATENCY_P95: 1.2}, 0.02, 5, 50)

        report = compare((quiet,), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.REGRESSED

    def test_noise_does_not_make_an_unchanged_metric_inconclusive(self) -> None:
        noisy = Measurement("single-turn", "3.13", {Metric.LATENCY_P95: 1.0}, 0.3, 5, 50)

        report = compare((noisy,), baseline(latency_p95=1.0))

        assert verdicts(report)[Metric.LATENCY_P95] is Verdict.WITHIN


class TestTheReport:
    """What a maintainer reads, and what the process exits with."""

    def test_a_clean_report_exits_zero(self) -> None:
        report = compare((measured(latency_p95=1.0),), baseline(latency_p95=1.0))

        assert report.exit_code == 0

    def test_a_regression_exits_one_so_ci_fails(self) -> None:
        report = compare((measured(latency_p95=2.0),), baseline(latency_p95=1.0))

        assert report.exit_code == 1

    def test_an_inconclusive_run_exits_two_rather_than_failing_the_build(self) -> None:
        noisy = Measurement("single-turn", "3.13", {Metric.LATENCY_P95: 1.2}, 0.3, 5, 50)

        report = compare((noisy,), baseline(latency_p95=1.0))

        assert report.exit_code == 2

    def test_a_regression_outranks_an_inconclusive_metric(self) -> None:
        noisy = Measurement(
            "single-turn", "3.13", {Metric.LATENCY_P95: 1.2, Metric.TOKENS: 900.0}, 0.3, 5, 50
        )

        report = compare((noisy,), baseline(latency_p95=1.0, tokens=420.0))

        assert report.exit_code == 1

    def test_the_rendered_report_names_the_scenario_metric_and_delta(self) -> None:
        report = compare((measured(latency_p95=1.4),), baseline(latency_p95=1.0))

        rendered = report.render()
        assert "single-turn" in rendered
        assert "latency_p95" in rendered
        assert "+40.0%" in rendered

    def test_the_rendered_report_says_when_nothing_moved(self) -> None:
        report = compare((measured(latency_p95=1.0),), baseline(latency_p95=1.0))

        assert "within threshold" in report.render()

    def test_an_unrecorded_metric_is_listed_rather_than_hidden(self) -> None:
        report = compare((measured(peak_bytes=10.0),), Baseline({}))

        assert "no baseline" in report.render()

    def test_the_regressions_are_available_without_reading_the_text(self) -> None:
        report = compare((measured(latency_p95=2.0, tokens=1.0),), baseline(latency_p95=1.0))

        assert [one.metric for one in report.regressions] == [Metric.LATENCY_P95]


class TestTheBaselineFile:
    """Baselines are a reviewed commit, so what is written has to be readable by a human."""

    def test_a_baseline_round_trips(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"
        write_baseline(where, (measured(latency_p95=1.5),))

        assert load_baseline(where).value("single-turn", "3.13", Metric.LATENCY_P95) == 1.5

    def test_a_missing_file_is_an_empty_baseline_not_a_crash(self, tmp_path: Path) -> None:
        assert load_baseline(tmp_path / "absent.json").value("a", "3.13", Metric.TOKENS) is None

    def test_a_malformed_baseline_is_refused_rather_than_read_as_empty(
        self, tmp_path: Path
    ) -> None:
        where = tmp_path / "baseline.json"
        where.write_text("[]")

        with pytest.raises(ValueError, match="baseline"):
            load_baseline(where)

    def test_writing_one_interpreter_leaves_another_alone(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"
        write_baseline(where, (measured(latency_p95=1.0),))

        other = Measurement("single-turn", "3.12", {Metric.LATENCY_P95: 9.0}, 0.01, 5, 50)
        write_baseline(where, (other,))

        recorded = load_baseline(where)
        assert recorded.value("single-turn", "3.13", Metric.LATENCY_P95) == 1.0
        assert recorded.value("single-turn", "3.12", Metric.LATENCY_P95) == 9.0

    def test_what_is_written_is_sorted_so_a_diff_is_reviewable(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"
        second = Measurement("a-streaming", "3.13", {Metric.THROUGHPUT: 2.0}, 0.01, 5, 50)

        write_baseline(where, (measured(latency_p95=1.0), second))

        assert list(json.loads(where.read_text())["scenarios"]) == ["a-streaming", "single-turn"]

    def test_a_written_baseline_records_how_it_was_measured(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"
        write_baseline(where, (measured(latency_p95=1.0),))

        held = json.loads(where.read_text())["scenarios"]["single-turn"]["3.13"]
        assert held["rounds"] == 5
        assert held["iterations"] == 50


class TestRefusedConfigurations:
    """A harness that quietly accepts a nonsense setting reports nonsense numbers."""

    async def test_a_scenario_with_no_iterations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            Scenario(name="empty", run=Work(), iterations=0)

    async def test_a_scenario_with_no_rounds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="rounds"):
            Scenario(name="empty", run=Work(), rounds=0)

    async def test_a_negative_warm_up_is_refused(self) -> None:
        with pytest.raises(ValueError, match="warmup"):
            Scenario(name="empty", run=Work(), warmup=-1)

    async def test_two_scenarios_sharing_a_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="single-turn"):
            await run_suite((scenario(Work()), scenario(Work())), timer=Ticks())

    def test_a_negative_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            Thresholds(limits={Metric.LATENCY_P95: -0.1})

    def test_a_noise_ceiling_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="noise_ceiling"):
            Thresholds(noise_ceiling=1.5)


class TestComparisonIsReadable:
    """The dataclass a report is made of is part of the surface consumers read."""

    def test_a_comparison_states_the_delta_as_a_percentage(self) -> None:
        one = Comparison("single-turn", Metric.LATENCY_P95, 1.0, 1.4, 0.4, Verdict.REGRESSED)

        assert one.percentage == "+40.0%"

    def test_growth_from_a_baseline_of_zero_is_reported_as_unbounded(self) -> None:
        report = compare((measured(allocations=5.0),), baseline(allocations=0.0))

        assert report.comparisons[0].percentage == "+inf%"

    def test_the_unrecorded_metrics_are_available_without_reading_the_text(self) -> None:
        report = compare((measured(peak_bytes=10.0),), Baseline({}))

        assert [one.metric for one in report.unrecorded] == [Metric.PEAK_BYTES]

    def test_a_comparison_with_no_baseline_has_no_percentage(self) -> None:
        one = Comparison("single-turn", Metric.PEAK_BYTES, None, 10.0, 0.0, Verdict.UNRECORDED)

        assert one.percentage == "n/a"
