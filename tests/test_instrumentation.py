"""Spans a run emits without anybody wiring them, and what happens when export cannot."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    SPAN_NAMES,
    BudgetExceededError,
    Instrumentation,
    RunSpan,
    Sampling,
    SpanKind,
    SpanLimits,
    SpanStatus,
    Trace,
)
from tesserix_adk.testing import FakeClock, FakeTracer

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

pytestmark = pytest.mark.anyio


class ExplodingTracer:
    """A collector that is down, in the way a collector is actually down."""

    def __init__(self) -> None:
        self.attempts = 0

    def span(self, name: str, **attributes: object) -> AbstractContextManager[None]:
        """Fail the moment the exporter is asked for anything."""
        self.attempts += 1
        message = f"collector queue is full, dropping {name} {attributes}"
        raise RuntimeError(message)

    def event(self, name: str, **attributes: object) -> None:
        """Fail here too, because a failing exporter fails everywhere."""
        message = f"collector queue is full, dropping {name} {attributes}"
        raise RuntimeError(message)


def _instrument(tracer: object = None, **kwargs: object) -> Instrumentation:
    return Instrumentation(tracer, clock=FakeClock(), **kwargs)  # type: ignore[arg-type]


class TestTheShapeOfATrace:
    def test_a_run_emits_one_root_span(self) -> None:
        tracer = FakeTracer()
        with _instrument(tracer).run("run-1", tenant="acme"):
            pass
        assert tracer.names() == [SPAN_NAMES[SpanKind.RUN]]

    def test_every_step_is_parented_to_the_run_without_being_passed_anything(self) -> None:
        instrument = _instrument(FakeTracer())
        with instrument.run("run-1") as run:
            with instrument.step(SpanKind.MODEL, "gpt"):
                pass
            with instrument.step(SpanKind.TOOL, "refund"):
                pass
        children = [record for record in run.trace.recordings if record.kind is not SpanKind.RUN]
        assert {record.parent_id for record in children} == {run.span_id}

    def test_a_step_inside_a_step_is_parented_to_that_step(self) -> None:
        instrument = _instrument(FakeTracer())
        with (
            instrument.run("run-1"),
            instrument.step(SpanKind.TOOL, "search") as tool,
            instrument.step(SpanKind.RETRIEVAL, "chunks") as retrieval,
        ):
            pass
        assert retrieval.parent_id == tool.span_id

    def test_a_retry_is_a_sibling_rather_than_a_reopened_span(self) -> None:
        instrument = _instrument(FakeTracer())
        with instrument.run("run-1") as run:
            for attempt in (1, 2):
                with instrument.step(SpanKind.TOOL, "refund", attempt=attempt):
                    pass
        tools = run.trace.of_kind(SpanKind.TOOL)
        assert [record.attempt for record in tools] == [1, 2]
        assert tools[0].span_id != tools[1].span_id

    def test_the_primary_scenario_emits_one_run_rooted_trace(self) -> None:
        """Two model calls, a retrieval and three tool calls of which one retries."""
        instrument = _instrument(FakeTracer())
        with instrument.run("run-1") as run:
            for _ in range(2):
                with instrument.step(SpanKind.MODEL, "gpt"):
                    pass
            with instrument.step(SpanKind.RETRIEVAL, "chunks"):
                pass
            for name in ("refund", "lookup", "notify"):
                with instrument.step(SpanKind.TOOL, name):
                    pass
            with instrument.step(SpanKind.TOOL, "notify", attempt=2):
                pass
        assert len(run.trace.recordings) == 8
        assert len(run.trace.roots) == 1
        assert all(record.duration_seconds >= 0.0 for record in run.trace.recordings)


class TestTiming:
    def test_the_run_span_carries_time_to_first_token(self) -> None:
        clock = FakeClock()
        instrument = Instrumentation(FakeTracer(), clock=clock)
        with instrument.run("run-1") as run:
            clock.set(0.25)
            run.first_token()
            clock.set(1.25)
        assert run.trace.roots[0].attributes["adk.time_to_first_token_seconds"] == "0.25"  # noqa: S105 — a duration, not a secret

    def test_only_the_first_token_counts_as_the_first_token(self) -> None:
        clock = FakeClock()
        instrument = Instrumentation(FakeTracer(), clock=clock)
        with instrument.run("run-1") as run:
            clock.set(0.25)
            run.first_token()
            clock.set(5.0)
            run.first_token()
        assert run.trace.roots[0].attributes["adk.time_to_first_token_seconds"] == "0.25"  # noqa: S105 — a duration, not a secret

    def test_the_run_span_counts_its_iterations(self) -> None:
        instrument = _instrument(FakeTracer())
        with instrument.run("run-1") as run:
            for _ in range(3):
                run.iterated()
        assert run.trace.roots[0].attributes["adk.iterations"] == "3"

    def test_attributes_set_during_a_span_reach_the_export(self) -> None:
        instrument = _instrument(FakeTracer())
        with instrument.run("run-1") as run, instrument.step(SpanKind.MODEL, "gpt") as span:
            span.set(model="gpt-5", tokens="1200")
        assert run.trace.of_kind(SpanKind.MODEL)[0].attributes["model"] == "gpt-5"

    def test_a_span_records_the_duration_the_clock_measured(self) -> None:
        clock = FakeClock()
        instrument = Instrumentation(FakeTracer(), clock=clock)
        with instrument.run("run-1") as run, instrument.step(SpanKind.TOOL, "slow"):
            clock.set(2.0)
        assert run.trace.of_kind(SpanKind.TOOL)[0].duration_seconds == 2.0


def _trace_of_failure(
    instrument: Instrumentation, expected: type[Exception], body: Callable[[], None]
) -> Trace:
    """Run `body` inside a run span that is expected to fail, and return what it recorded."""
    spans: list[RunSpan] = []
    with pytest.raises(expected), instrument.run("run-1") as span:  # noqa: PT012 — the span is captured before it fails
        spans.append(span)
        body()
    return spans[0].trace


class TestFailures:
    def test_a_failing_step_records_the_error_class_and_leaves_the_status_wrong(self) -> None:
        instrument = _instrument(FakeTracer())

        def body() -> None:
            with instrument.step(SpanKind.MODEL, "gpt"):
                raise BudgetExceededError("out of money")

        failed = _trace_of_failure(instrument, BudgetExceededError, body).of_kind(SpanKind.MODEL)[0]
        assert failed.status is SpanStatus.ERROR
        assert failed.attributes["adk.error.type"] == "BudgetExceededError"

    def test_the_failure_propagates_rather_than_being_swallowed_by_the_span(self) -> None:
        instrument = _instrument(FakeTracer())
        with pytest.raises(BudgetExceededError, match="out of money"), instrument.run("run-1"):
            raise BudgetExceededError("out of money")

    def test_a_failure_says_on_the_span_whether_asking_again_could_help(self) -> None:
        instrument = _instrument(FakeTracer())

        def body() -> None:
            raise BudgetExceededError("out of money")

        trace = _trace_of_failure(instrument, BudgetExceededError, body)
        assert trace.roots[0].attributes["adk.error.retryable"] == "false"

    def test_a_failure_that_is_not_a_kit_error_is_recorded_the_same_way(self) -> None:
        instrument = _instrument(FakeTracer())

        def body() -> None:
            raise ZeroDivisionError("nope")

        trace = _trace_of_failure(instrument, ZeroDivisionError, body)
        assert trace.roots[0].attributes["adk.error.type"] == "ZeroDivisionError"
        assert "adk.error.retryable" not in trace.roots[0].attributes

    async def test_cancellation_closes_the_open_spans_with_the_right_status(self) -> None:
        instrument = _instrument(FakeTracer())
        traces: list[Trace] = []

        spans: list[RunSpan] = []

        async def work() -> None:
            try:
                with instrument.run("run-1") as run:
                    spans.append(run)
                    with instrument.step(SpanKind.TOOL, "slow"):
                        await asyncio.sleep(60)
            finally:
                traces.append(spans[0].trace)

        task = asyncio.ensure_future(work())
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert {record.status for record in traces[0].recordings} == {SpanStatus.CANCELLED}


class TestDegradedExport:
    def test_an_unavailable_collector_does_not_fail_the_run(self) -> None:
        tracer = ExplodingTracer()
        instrument = _instrument(tracer)
        with instrument.run("run-1"), instrument.step(SpanKind.TOOL, "refund"):
            pass
        assert tracer.attempts > 0
        assert instrument.loss.export_failures > 0

    def test_the_loss_is_counted_locally_rather_than_retried_forever(self) -> None:
        instrument = _instrument(ExplodingTracer())
        for index in range(3):
            with instrument.run(f"run-{index}"):
                pass
        assert instrument.loss.export_failures == 3

    def test_no_tracer_at_all_is_a_supported_configuration(self) -> None:
        instrument = _instrument()
        with instrument.run("run-1") as run, instrument.step(SpanKind.MODEL, "gpt") as span:
            span.set(model="gpt-5")
        assert len(run.trace.recordings) == 2

    def test_a_step_outside_any_run_records_nowhere_rather_than_raising(self) -> None:
        tracer = FakeTracer()
        instrument = _instrument(tracer)
        with instrument.step(SpanKind.TOOL, "orphan") as span:
            span.set(anything="goes")
        assert span.parent_id is None
        assert tracer.recorded == []


class TestSampling:
    def test_a_sampled_out_run_emits_nothing_at_all_rather_than_orphan_children(self) -> None:
        tracer = FakeTracer()
        instrument = _instrument(tracer, sampling=Sampling(ratio=0.0))
        with instrument.run("run-1"), instrument.step(SpanKind.TOOL, "refund"):
            pass
        assert tracer.recorded == []
        assert instrument.loss.sampled_out == 1

    def test_a_failing_run_is_kept_however_the_sampler_decided(self) -> None:
        tracer = FakeTracer()
        instrument = _instrument(tracer, sampling=Sampling(ratio=0.0))
        with (
            pytest.raises(BudgetExceededError),
            instrument.run("run-1"),
            instrument.step(SpanKind.MODEL, "gpt"),
        ):
            raise BudgetExceededError("out of money")
        assert len(tracer.recorded) == 2

    def test_a_deployment_that_wants_nothing_kept_can_say_so(self) -> None:
        tracer = FakeTracer()
        instrument = _instrument(tracer, sampling=Sampling(ratio=0.0, always_on_error=False))
        with pytest.raises(BudgetExceededError), instrument.run("run-1"):
            raise BudgetExceededError("out of money")
        assert tracer.recorded == []

    def test_the_decision_is_per_run_and_stable_so_a_replay_decides_the_same_way(self) -> None:
        sampling = Sampling(ratio=0.5)
        assert sampling.keeps("run-1") == sampling.keeps("run-1")

    def test_a_full_ratio_keeps_every_run(self) -> None:
        sampling = Sampling(ratio=1.0)
        assert all(sampling.keeps(f"run-{index}") for index in range(50))

    def test_a_half_ratio_keeps_roughly_half(self) -> None:
        sampling = Sampling(ratio=0.5)
        kept = sum(sampling.keeps(f"run-{index}") for index in range(500))
        assert 200 < kept < 300

    def test_a_ratio_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ratio"):
            Sampling(ratio=1.5)


class TestBounds:
    def test_a_wide_fan_out_is_bounded_rather_than_unbounded(self) -> None:
        instrument = _instrument(FakeTracer(), limits=SpanLimits(max_spans=5))
        with instrument.run("run-1") as run:
            for index in range(20):
                with instrument.step(SpanKind.TOOL, f"tool-{index}"):
                    pass
        assert len(run.trace.recordings) == 5

    def test_truncation_is_visible_on_the_root_rather_than_silent(self) -> None:
        instrument = _instrument(FakeTracer(), limits=SpanLimits(max_spans=5))
        with instrument.run("run-1") as run:
            for index in range(20):
                with instrument.step(SpanKind.TOOL, f"tool-{index}"):
                    pass
        assert run.trace.dropped == 16
        assert run.trace.roots[0].attributes["adk.spans.dropped"] == "16"

    def test_the_root_survives_truncation_so_the_trace_is_never_headless(self) -> None:
        instrument = _instrument(FakeTracer(), limits=SpanLimits(max_spans=1))
        with instrument.run("run-1") as run, instrument.step(SpanKind.TOOL, "refund"):
            pass
        assert len(run.trace.roots) == 1

    def test_a_limit_of_no_spans_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_spans"):
            SpanLimits(max_spans=0)


class TestReplay:
    def test_replaying_a_run_does_not_export_its_spans_twice(self) -> None:
        tracer = FakeTracer()
        instrument = _instrument(tracer)
        for _ in range(2):
            with instrument.run("run-1"), instrument.step(SpanKind.TOOL, "refund"):
                pass
        assert len(tracer.recorded) == 2
        assert instrument.loss.replayed == 1

    def test_the_replay_memory_is_bounded_so_a_long_lived_process_does_not_grow(self) -> None:
        tracer = FakeTracer()
        limits = SpanLimits(remembered_runs=2)
        instrument = _instrument(tracer, limits=limits)
        for run_id in ("run-0", "run-1", "run-2", "run-0"):
            with instrument.run(run_id):
                pass
        assert instrument.loss.replayed == 0
        assert len(tracer.recorded) == 4

    def test_a_different_run_is_not_mistaken_for_a_replay(self) -> None:
        tracer = FakeTracer()
        instrument = _instrument(tracer)
        for index in range(2):
            with instrument.run(f"run-{index}"):
                pass
        assert len(tracer.recorded) == 2


class TestTheTrace:
    def test_the_trace_is_only_offered_once_the_run_has_finished(self) -> None:
        """A trace read mid-run is missing the root, which is the one span nobody may lose."""
        instrument = _instrument(FakeTracer())
        with instrument.run("run-1") as run, pytest.raises(RuntimeError, match="finished"):
            _ = run.trace


class TestConcurrency:
    async def test_two_runs_at_once_do_not_take_each_others_spans(self) -> None:
        instrument = _instrument(FakeTracer())
        traces: dict[str, Trace] = {}

        async def one(run_id: str, name: str) -> None:
            with instrument.run(run_id) as run:
                await asyncio.sleep(0)
                with instrument.step(SpanKind.TOOL, name):
                    await asyncio.sleep(0)
            traces[run_id] = run.trace

        await asyncio.gather(one("run-a", "refund"), one("run-b", "lookup"))
        tools = {
            record.name for record in traces["run-a"].recordings if record.kind is SpanKind.TOOL
        }
        assert tools == {"refund"}
