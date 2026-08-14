"""The numbers that decide whether CPU inference feels usable, and the gate that holds them."""

from __future__ import annotations

import pytest

from tesserix_adk.observability import (
    CACHE_HIT_RATIO,
    LATENCY_SECONDS,
    TIME_TO_FIRST_TOKEN,
    TOKENS_PER_SECOND,
    CacheHits,
    LatencyReport,
    RunTimer,
)
from tesserix_adk.testing import FakeClock, FakeMeter
from tesserix_adk.testing.benchmarks import (
    DEFAULT_FLOORS,
    DEFAULT_LIMITS,
    Baseline,
    Measurement,
    Metric,
    Thresholds,
    Verdict,
    compare,
)

pytestmark = pytest.mark.anyio


def timed(*, streaming: bool = True, cold: bool = False) -> tuple[RunTimer, FakeClock]:
    """A timer over a clock a test can move."""
    clock = FakeClock()
    return RunTimer(clock=clock, streaming=streaming, cold=cold), clock


class TestWhatOneRunReports:
    """Time to first token, total latency, sustained rate — per run, not per sample."""

    async def test_time_to_first_token_is_measured_from_the_start_of_the_run(self) -> None:
        timer, clock = timed()
        clock.advance(2.5)
        timer.first_token()
        clock.advance(4.0)

        report = timer.finished(output_tokens=130)

        assert report.time_to_first_token == pytest.approx(2.5)
        assert report.seconds == pytest.approx(6.5)

    async def test_the_sustained_rate_excludes_the_wait_for_the_first_token(self) -> None:
        timer, clock = timed()
        clock.advance(2.0)
        timer.first_token()
        clock.advance(4.0)

        report = timer.finished(output_tokens=40)

        assert report.tokens_per_second == pytest.approx(10.0)

    async def test_a_non_streaming_run_has_no_time_to_first_token(self) -> None:
        timer, clock = timed(streaming=False)
        clock.advance(9.0)

        report = timer.finished(output_tokens=90)

        assert report.time_to_first_token is None
        assert report.tokens_per_second == pytest.approx(10.0)

    async def test_a_stream_that_never_produced_a_token_says_so(self) -> None:
        timer, clock = timed()
        clock.advance(3.0)

        report = timer.finished(output_tokens=0)

        assert report.time_to_first_token is None
        assert report.tokens_per_second is None

    async def test_only_the_first_token_of_a_stream_counts(self) -> None:
        timer, clock = timed()
        clock.advance(2.0)
        for _ in range(5):
            timer.first_token()
            clock.advance(1.0)

        report = timer.finished(output_tokens=5)

        assert report.time_to_first_token == pytest.approx(2.0)

    async def test_a_clock_that_did_not_start_at_zero_reports_the_same_numbers(self) -> None:
        clock = FakeClock(start=1_786_712_730.0)
        timer = RunTimer(clock=clock)
        clock.advance(2.0)
        timer.first_token()
        clock.advance(4.0)

        report = timer.finished(output_tokens=40)

        assert report.time_to_first_token == pytest.approx(2.0)
        assert report.tokens_per_second == pytest.approx(10.0)

    async def test_a_cold_run_is_labelled_a_cold_run(self) -> None:
        cold, _ = timed(cold=True)
        warm, _ = timed(cold=False)

        assert cold.finished(output_tokens=1).cold is True
        assert warm.finished(output_tokens=1).cold is False


class TestTheCacheHitRatio:
    """Prefill dominates CPU latency, so the ratio is tracked beside the latency."""

    async def test_the_ratio_is_cached_over_what_was_sent(self) -> None:
        hits = CacheHits(input_tokens=1000, cached_tokens=800)

        assert hits.ratio == pytest.approx(0.8)
        assert hits.known is True

    async def test_a_provider_that_reports_nothing_is_unknown_not_zero(self) -> None:
        hits = CacheHits(input_tokens=1000, cached_tokens=None)

        assert hits.ratio is None
        assert hits.known is False

    async def test_a_run_with_no_input_at_all_is_unknown_too(self) -> None:
        hits = CacheHits(input_tokens=0, cached_tokens=0)

        assert hits.ratio is None

    async def test_more_cached_than_sent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cached_tokens"):
            CacheHits(input_tokens=10, cached_tokens=11)


class TestWhatIsEmitted:
    """Metrics, not spans: a sampled-away trace still has to leave the numbers behind."""

    async def test_every_number_is_counted(self) -> None:
        meter = FakeMeter()
        timer, clock = timed()
        clock.advance(1.0)
        timer.first_token()
        clock.advance(2.0)

        timer.finished(output_tokens=20, hits=CacheHits(input_tokens=100, cached_tokens=50)).emit(
            meter, model="local-cpu", streaming="true"
        )

        counted = {point.name for point in meter.points}
        assert {TIME_TO_FIRST_TOKEN, LATENCY_SECONDS, TOKENS_PER_SECOND, CACHE_HIT_RATIO} <= counted

    async def test_an_unknown_hit_ratio_is_not_counted_as_zero(self) -> None:
        meter = FakeMeter()
        timer, clock = timed()
        clock.advance(1.0)
        timer.first_token()
        clock.advance(1.0)

        timer.finished(output_tokens=10).emit(meter, model="local-cpu")

        assert CACHE_HIT_RATIO not in {point.name for point in meter.points}

    async def test_the_dimensions_separate_cold_from_warm_and_stream_from_block(self) -> None:
        meter = FakeMeter()
        timer, clock = timed(cold=True)
        clock.advance(1.0)
        timer.first_token()
        clock.advance(1.0)

        timer.finished(output_tokens=10).emit(meter, model="local-cpu")

        point = next(one for one in meter.points if one.name == LATENCY_SECONDS)
        assert point.dimensions["cold"] == "true"
        assert point.dimensions["streaming"] == "true"

    async def test_the_span_attributes_carry_counts_and_never_content(self) -> None:
        timer, clock = timed()
        clock.advance(1.0)
        timer.first_token()
        clock.advance(1.0)

        attributes = timer.finished(
            output_tokens=10, hits=CacheHits(input_tokens=80, cached_tokens=40)
        ).attributes()

        assert attributes[TIME_TO_FIRST_TOKEN] == "1.000"
        assert attributes[CACHE_HIT_RATIO] == "0.500"
        assert all(isinstance(value, str) for value in attributes.values())

    async def test_a_run_with_nothing_to_report_carries_no_empty_attributes(self) -> None:
        attributes = LatencyReport(seconds=3.0, streaming=False).attributes()

        assert TIME_TO_FIRST_TOKEN not in attributes
        assert TOKENS_PER_SECOND not in attributes
        assert CACHE_HIT_RATIO not in attributes

    async def test_a_meter_that_falls_over_does_not_stop_the_run(self) -> None:
        class Broken:
            def count(self, name: str, value: float, **dimensions: str) -> None:
                del name, value, dimensions
                raise RuntimeError("collector is down")

        timer, clock = timed()
        clock.advance(1.0)
        timer.first_token()
        clock.advance(1.0)

        timer.finished(output_tokens=10).emit(Broken(), model="local-cpu")


class TestTheGate:
    """A regression is caught by the benchmark, naming the metric that moved."""

    def measured(self, **values: float) -> Measurement:
        """One measurement, at a spread low enough for a verdict to be drawn."""
        return Measurement(
            scenario="cpu-warm",
            python="3.13",
            values={Metric(name): value for name, value in values.items()},
            spread=0.01,
            rounds=4,
            iterations=20,
        )

    def recorded(self, **values: float) -> Baseline:
        """The committed numbers for that scenario."""
        return Baseline(
            entries={("cpu-warm", "3.13"): {Metric(name): value for name, value in values.items()}},
            sizes={("cpu-warm", "3.13"): 20},
        )

    async def test_the_new_metrics_are_judged_in_the_right_direction(self) -> None:
        assert Metric.TOKENS_PER_SECOND.higher_is_better is True
        assert Metric.CACHE_HIT_RATIO.higher_is_better is True
        assert Metric.TIME_TO_FIRST_TOKEN.higher_is_better is False

    async def test_every_new_metric_has_a_threshold_and_a_floor(self) -> None:
        for metric in (
            Metric.TIME_TO_FIRST_TOKEN,
            Metric.TOKENS_PER_SECOND,
            Metric.CACHE_HIT_RATIO,
        ):
            assert metric in DEFAULT_LIMITS
            assert metric in DEFAULT_FLOORS

    async def test_a_broken_prefix_fails_naming_the_two_numbers_that_moved(self) -> None:
        report = compare(
            [self.measured(time_to_first_token=4.0, cache_hit_ratio=0.05)],
            self.recorded(time_to_first_token=2.0, cache_hit_ratio=0.80),
        )

        named = {one.metric for one in report.regressions}
        assert named == {Metric.TIME_TO_FIRST_TOKEN, Metric.CACHE_HIT_RATIO}
        assert report.exit_code == 1

    async def test_a_faster_first_token_is_an_improvement_not_a_regression(self) -> None:
        report = compare(
            [self.measured(time_to_first_token=1.0)], self.recorded(time_to_first_token=2.0)
        )

        assert report.comparisons[0].verdict is Verdict.IMPROVED

    async def test_a_hit_ratio_wobble_is_below_the_floor(self) -> None:
        report = compare(
            [self.measured(cache_hit_ratio=0.79)],
            self.recorded(cache_hit_ratio=0.80),
            Thresholds(),
        )

        assert report.comparisons[0].verdict is Verdict.WITHIN


class TestTheSuiteMeasuresThemSeparately:
    """Cold from warm, streaming from blocking — averaged together they say nothing."""

    async def test_the_shipped_suite_names_a_scenario_for_each(self) -> None:
        from benchmarks.suite import scenarios

        named = {one.name for one in scenarios()}

        assert {"first-token-cold", "first-token-warm", "sustained-stream"} <= named

    async def test_a_scenario_can_report_the_latency_metrics_it_observed(self) -> None:
        from tesserix_adk.testing.benchmarks import Scenario, measure

        async def nothing() -> None:
            return None

        scenario = Scenario(
            name="observed",
            run=nothing,
            iterations=1,
            warmup=0,
            rounds=1,
            observed=lambda: {Metric.TIME_TO_FIRST_TOKEN: 2.0, Metric.CACHE_HIT_RATIO: 0.9},
        )

        report = await measure(scenario)

        assert report.values[Metric.TIME_TO_FIRST_TOKEN] == pytest.approx(2.0)
        assert report.values[Metric.CACHE_HIT_RATIO] == pytest.approx(0.9)


class TestALatencyReportIsReadable:
    """The report is what an operator reads before deciding the machine is too small."""

    async def test_it_renders_every_number_it_has(self) -> None:
        report = LatencyReport(
            time_to_first_token=2.0,
            seconds=6.0,
            output_tokens=40,
            tokens_per_second=10.0,
            hits=CacheHits(input_tokens=100, cached_tokens=90),
            streaming=True,
            cold=False,
        )

        rendered = report.render()

        assert "2.000" in rendered
        assert "10.0" in rendered
        assert "90%" in rendered

    async def test_an_unknown_ratio_reads_as_unknown(self) -> None:
        report = LatencyReport(seconds=1.0, output_tokens=1, streaming=False, cold=True)

        assert "unknown" in report.render()
