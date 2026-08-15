"""Spans a run emits by itself, so no product has to wire its own instrumentation.

Recording is in-memory and export happens once the run has finished. That ordering is the
whole point: a full exporter queue, a slow collector or a misconfigured endpoint cannot
reach into the run that produced the numbers. It also means a trace is kept or dropped
whole, so a sampled-away run leaves no orphan children behind, and a run that failed can
be kept after the fact however the sampler decided at the start.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from tesserix_adk.core.errors import AdkError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from tesserix_adk.core.protocols import Clock, Tracer

__all__ = [
    "SPAN_NAMES",
    "Instrumentation",
    "Recording",
    "RunSpan",
    "Sampling",
    "Span",
    "SpanKind",
    "SpanLimits",
    "SpanStatus",
    "TelemetryLoss",
    "Trace",
]


class SpanKind(StrEnum):
    """What a span covers. One kind per stage a run can spend time in."""

    RUN = "run"
    MODEL = "model"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    GUARDRAIL = "guardrail"


SPAN_NAMES: Mapping[SpanKind, str] = {kind: f"adk.{kind.value}" for kind in SpanKind}
"""The exported name per kind. The operation goes in an attribute, not the name, because
a span name carrying a tool name is a cardinality problem in every backend."""


class SpanStatus(StrEnum):
    """How a span ended."""

    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class SpanLimits(AdkModel):
    """Bounds on what one run may record.

    Args:
        max_spans: The most spans one trace may hold, the root included. A recursive tool
            fan-out is otherwise unbounded, and an unbounded trace is a memory leak.
        remembered_runs: How many finished run ids are remembered for replay suppression.
    """

    max_spans: int = Field(default=1000, ge=1)
    remembered_runs: int = Field(default=1024, ge=1)


class Sampling(AdkModel):
    """Which runs are exported.

    Args:
        ratio: The share of runs kept, decided from the run id so the same run decides the
            same way on a workflow replay.
        always_on_error: Whether a run that failed is kept whatever the ratio said. The
            traces worth having are the ones nobody would have sampled.
    """

    ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    always_on_error: bool = True

    def keeps(self, run_id: str) -> bool:
        """Whether `run_id` is sampled in. Deterministic, and stable across processes."""
        digest = hashlib.sha256(run_id.encode()).hexdigest()[:8]
        return int(digest, 16) / 0x1_0000_0000 < self.ratio


@dataclass(slots=True)
class TelemetryLoss:
    """What was not exported, counted locally.

    Observability degrading is not a run failing, so the losses are counted here rather
    than raised. A consumer that cares alerts on these.
    """

    sampled_out: int = 0
    replayed: int = 0
    export_failures: int = 0


@dataclass(frozen=True, slots=True)
class Recording:
    """One finished span.

    Args:
        span_id: Unique within the process.
        sequence: When it opened relative to its siblings. Ordering by the clock is wrong
            for spans that opened inside the same tick.
        parent_id: The span this one happened inside, or None for the root.
        kind: What stage it covers.
        name: The operation — the tool, the model, the index queried.
        attempt: Which try this was. A retry is a sibling span, never a reopened one.
        started: Clock reading when it opened.
        ended: Clock reading when it closed.
        status: How it ended.
        attributes: Everything the caller set on it, plus what the span recorded itself.
    """

    span_id: str
    sequence: int
    parent_id: str | None
    kind: SpanKind
    name: str
    attempt: int
    started: float
    ended: float
    status: SpanStatus
    attributes: Mapping[str, str]

    @property
    def duration_seconds(self) -> float:
        """How long it took, by the injected clock."""
        return self.ended - self.started

    def exported(self, run_id: str) -> dict[str, str]:
        """The attributes as the tracer receives them."""
        return {
            **self.attributes,
            "adk.run_id": run_id,
            "adk.span_id": self.span_id,
            "adk.parent_id": self.parent_id or "",
            "adk.operation": self.name,
            "adk.attempt": str(self.attempt),
            "adk.status": self.status.value,
            "adk.duration_seconds": f"{self.duration_seconds:g}",
        }


@dataclass(frozen=True, slots=True)
class Trace:
    """Everything one run recorded.

    Args:
        run_id: The run it covers.
        recordings: Its spans, in the order they opened.
        dropped: Spans the limit refused. Non-zero means the trace is incomplete, which
            the root span says out loud rather than leaving to be inferred.
    """

    run_id: str
    recordings: tuple[Recording, ...]
    dropped: int = 0

    @property
    def roots(self) -> tuple[Recording, ...]:
        """The spans with no parent. One, for a trace of a finished run."""
        return tuple(record for record in self.recordings if record.parent_id is None)

    @property
    def failed(self) -> bool:
        """Whether anything in the run ended badly."""
        return any(record.status is not SpanStatus.OK for record in self.recordings)

    def of_kind(self, kind: SpanKind) -> tuple[Recording, ...]:
        """The spans of one kind, in the order they opened."""
        return tuple(record for record in self.recordings if record.kind is kind)


@dataclass(slots=True)
class _Collector:
    """The in-flight recordings for one run."""

    run_id: str
    limits: SpanLimits
    recordings: list[Recording] = field(default_factory=list)
    dropped: int = 0

    def record(self, recording: Recording) -> None:
        """Keep a finished span, unless the limit is reached and it is not the root."""
        room = self.limits.max_spans - (0 if recording.parent_id is None else 1)
        if len(self.recordings) >= room and recording.parent_id is not None:
            self.dropped += 1
            return
        self.recordings.append(recording)

    def trace(self) -> Trace:
        """The finished trace, spans in the order they opened."""
        ordered = tuple(sorted(self.recordings, key=lambda record: record.sequence))
        return Trace(run_id=self.run_id, recordings=ordered, dropped=self.dropped)


_COLLECTOR: ContextVar[_Collector | None] = ContextVar("adk_collector", default=None)
_PARENT: ContextVar[str | None] = ContextVar("adk_parent_span", default=None)
_IDS = itertools.count(1)


class Span:
    """A span while it is open. Setting anything on it is a dictionary write, never I/O."""

    __slots__ = (
        "_attributes",
        "_clock",
        "_sequence",
        "_started",
        "attempt",
        "kind",
        "name",
        "parent_id",
        "span_id",
    )

    def __init__(
        self, *, kind: SpanKind, name: str, attempt: int, parent_id: str | None, clock: Clock
    ) -> None:
        self._sequence = next(_IDS)
        self.span_id = f"s{self._sequence}"
        self.parent_id = parent_id
        self.kind = kind
        self.name = name
        self.attempt = attempt
        self._clock = clock
        self._started = clock.now()
        self._attributes: dict[str, str] = {}

    def set(self, **attributes: str) -> None:
        """Put `attributes` on the span. Later writes to the same key win."""
        self._attributes.update(attributes)

    def first_token(self) -> None:
        """Mark the first streamed token. Only the first call counts; the rest are latency."""
        if "adk.time_to_first_token_seconds" not in self._attributes:
            elapsed = self._clock.now() - self._started
            self._attributes["adk.time_to_first_token_seconds"] = f"{elapsed:g}"

    def iterated(self, count: int = 1) -> None:
        """Advance the iteration counter by `count`."""
        current = int(self._attributes.get("adk.iterations", "0"))
        self._attributes["adk.iterations"] = str(current + count)

    def _close(self, status: SpanStatus, extra: Mapping[str, str]) -> Recording:
        return Recording(
            span_id=self.span_id,
            sequence=self._sequence,
            parent_id=self.parent_id,
            kind=self.kind,
            name=self.name,
            attempt=self.attempt,
            started=self._started,
            ended=self._clock.now(),
            status=status,
            attributes={**self._attributes, **extra},
        )


class RunSpan(Span):
    """The root span, plus the trace it collected once the run is over."""

    __slots__ = ("_trace",)

    def __init__(
        self, *, kind: SpanKind, name: str, attempt: int, parent_id: str | None, clock: Clock
    ) -> None:
        super().__init__(kind=kind, name=name, attempt=attempt, parent_id=parent_id, clock=clock)
        self._trace: Trace | None = None

    @property
    def trace(self) -> Trace:
        """What the run recorded.

        Raises:
            RuntimeError: Before the run has finished, when the trace is still missing the
                spans that have not closed — including the root.
        """
        if self._trace is None:
            message = "the trace is only complete once the run has finished"
            raise RuntimeError(message)
        return self._trace


def _error_attributes(error: BaseException) -> dict[str, str]:
    """What a failure puts on its span. The class, and whether asking again could help."""
    attributes = {"adk.error.type": type(error).__name__}
    if isinstance(error, AdkError):
        attributes["adk.error.retryable"] = "true" if error.retryable else "false"
    return attributes


class Instrumentation:
    """Emits a run-rooted trace with no per-agent wiring.

    Args:
        tracer: Where finished traces go. None records without exporting, which is the
            supported way to run with tracing off.
        clock: What times the spans.
        sampling: Which runs are exported.
        limits: What one trace may hold.
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        *,
        clock: Clock,
        sampling: Sampling | None = None,
        limits: SpanLimits | None = None,
    ) -> None:
        self._tracer = tracer
        self._clock = clock
        self._sampling = sampling or Sampling()
        self._limits = limits or SpanLimits()
        self._recent: deque[str] = deque(maxlen=self._limits.remembered_runs)
        self.loss = TelemetryLoss()

    @contextlib.contextmanager
    def run(self, run_id: str, **attributes: str) -> Iterator[RunSpan]:
        """Open the root span for `run_id` and export the trace when it closes."""
        collector = _Collector(run_id=run_id, limits=self._limits)
        span = RunSpan(kind=SpanKind.RUN, name=run_id, attempt=1, parent_id=None, clock=self._clock)
        span.set(**attributes)
        collector_token = _COLLECTOR.set(collector)
        parent_token = _PARENT.set(span.span_id)
        try:
            yield span
        except BaseException as error:
            self._finish(collector, span, _status_of(error), _error_attributes(error))
            raise
        else:
            self._finish(collector, span, SpanStatus.OK, {})
        finally:
            _PARENT.reset(parent_token)
            _COLLECTOR.reset(collector_token)

    @contextlib.contextmanager
    def step(
        self, kind: SpanKind, name: str, *, attempt: int = 1, **attributes: str
    ) -> Iterator[Span]:
        """Open a child span for one stage, parented to whatever span is already open.

        Outside any run the span records nowhere. Instrumentation that refused to run
        without a run would be instrumentation every caller had to guard.
        """
        collector = _COLLECTOR.get()
        parent = _PARENT.get() if collector is not None else None
        span = Span(kind=kind, name=name, attempt=attempt, parent_id=parent, clock=self._clock)
        span.set(**attributes)
        token = _PARENT.set(span.span_id)
        try:
            yield span
        except BaseException as error:
            self._record(collector, span, _status_of(error), _error_attributes(error))
            raise
        else:
            self._record(collector, span, SpanStatus.OK, {})
        finally:
            _PARENT.reset(token)

    def _record(
        self, collector: _Collector | None, span: Span, status: SpanStatus, extra: Mapping[str, str]
    ) -> None:
        if collector is not None:
            collector.record(span._close(status, extra))

    def _finish(
        self, collector: _Collector, span: RunSpan, status: SpanStatus, extra: Mapping[str, str]
    ) -> None:
        dropped = {"adk.spans.dropped": str(collector.dropped)} if collector.dropped else {}
        collector.record(span._close(status, {**extra, **dropped}))
        trace = collector.trace()
        span._trace = trace
        self._export(trace)

    def _export(self, trace: Trace) -> None:
        """Hand the trace to the tracer, or record why it did not go."""
        if trace.run_id in self._recent:
            self.loss.replayed += 1
            return
        self._recent.append(trace.run_id)
        if not self._keeps(trace):
            self.loss.sampled_out += 1
            return
        if self._tracer is None:
            return
        try:
            for record in trace.recordings:
                with self._tracer.span(SPAN_NAMES[record.kind], **record.exported(trace.run_id)):
                    pass
        except Exception:
            self.loss.export_failures += 1

    def _keeps(self, trace: Trace) -> bool:
        if trace.failed and self._sampling.always_on_error:
            return True
        return self._sampling.keeps(trace.run_id)


def _status_of(error: BaseException) -> SpanStatus:
    """Cancellation is not a fault, and a span that says it was is a false alarm."""
    return SpanStatus.CANCELLED if isinstance(error, asyncio.CancelledError) else SpanStatus.ERROR
