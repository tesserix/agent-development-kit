"""A run that traces itself, a collector that is down, and what one span costs.

Run it with `uv run python examples/auto_instrumentation.py`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tesserix_adk.core import (
    BudgetExceededError,
    Instrumentation,
    Sampling,
    SpanKind,
    SpanLimits,
)
from tesserix_adk.testing import FakeClock, FakeTracer

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from tesserix_adk.core import RunSpan


class DownCollector:
    """An exporter whose queue is full, which is how an exporter usually fails."""

    def span(self, name: str, **attributes: object) -> AbstractContextManager[None]:
        """Refuse everything, loudly."""
        message = f"queue full, dropping {name} {len(attributes)} attributes"
        raise RuntimeError(message)

    def event(self, name: str, **attributes: object) -> None:
        """Refuse this too."""
        message = f"queue full, dropping {name} {len(attributes)} attributes"
        raise RuntimeError(message)


def work(instrument: Instrumentation, run: RunSpan) -> None:
    """Two model calls, a retrieval, and a tool that needs a second try."""
    for _ in range(2):
        with instrument.step(SpanKind.MODEL, "gpt-5") as span:
            span.set(tokens="1200")
            run.first_token()
        run.iterated()
    with instrument.step(SpanKind.RETRIEVAL, "policies"):
        pass
    for attempt in (1, 2):
        with instrument.step(SpanKind.TOOL, "refund", attempt=attempt):
            pass


def main() -> None:
    """The same run traced, dropped, degraded and truncated."""
    tracer = FakeTracer()
    instrument = Instrumentation(tracer, clock=FakeClock())
    with instrument.run("run-1", tenant="acme") as run:
        work(instrument, run)
    kinds = [record.kind for record in run.trace.recordings]
    print(f"one run, no wiring: {len(tracer.recorded)} spans {[str(k) for k in kinds]}")  # noqa: T201
    print(f"iterations={run.trace.roots[0].attributes['adk.iterations']}")  # noqa: T201

    sampled = Instrumentation(FakeTracer(), clock=FakeClock(), sampling=Sampling(ratio=0.0))
    with sampled.run("run-2") as run:
        work(sampled, run)
    print(f"\nsampled out: {sampled.loss.sampled_out} run, no orphan children left behind")  # noqa: T201

    kept = Instrumentation(FakeTracer(), clock=FakeClock(), sampling=Sampling(ratio=0.0))
    try:
        with kept.run("run-3") as run, kept.step(SpanKind.TOOL, "refund"):
            raise BudgetExceededError("out of money")
    except BudgetExceededError:
        print(f"the failure was kept anyway: sampled_out={kept.loss.sampled_out}")  # noqa: T201

    down = Instrumentation(DownCollector(), clock=FakeClock())
    with down.run("run-4") as run:
        work(down, run)
    print(f"\ncollector down, run still finished: losses={down.loss.export_failures}")  # noqa: T201

    tight = Instrumentation(FakeTracer(), clock=FakeClock(), limits=SpanLimits(max_spans=4))
    with tight.run("run-5") as tight_run:
        work(tight, tight_run)
    dropped = tight_run.trace.roots[0].attributes["adk.spans.dropped"]
    print(f"truncated visibly: {len(tight_run.trace.recordings)} kept, {dropped} dropped")  # noqa: T201

    quiet = Instrumentation(clock=FakeClock(), limits=SpanLimits(max_spans=100_001))
    started = time.perf_counter()
    with quiet.run("run-6") as quiet_run:
        for index in range(100_000):
            with quiet.step(SpanKind.TOOL, f"tool-{index}"):
                pass
    each = (time.perf_counter() - started) / len(quiet_run.trace.recordings)
    print(f"\noverhead: {each * 1_000_000:.2f}µs per span, budget 10µs")  # noqa: T201


if __name__ == "__main__":
    main()
