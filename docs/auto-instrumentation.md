# Spans without wiring

A run instruments itself. `Instrumentation` opens a root span for the run and a child span
for every stage inside it, so a product gets a run-rooted trace without writing a line of
per-agent instrumentation.

```python
from tesserix_adk.core import Instrumentation, SpanKind

instrument = Instrumentation(tracer, clock=clock)

with instrument.run(run_id, tenant=tenant) as run:
    with instrument.step(SpanKind.MODEL, "gpt-5") as span:
        span.set(tokens="1200")
    with instrument.step(SpanKind.TOOL, "refund"):
        ...
```

Nothing is passed down. `step` finds whatever span is already open through a context
variable, so a retrieval opened inside a tool is a child of that tool, and a step opened
in a task spawned inside a run belongs to that run.

## Recording now, exporting later

Spans are recorded in memory while the run happens and handed to the tracer once it
finishes. That ordering is the point:

- A full exporter queue, a slow collector or a wrong endpoint cannot reach into the run.
  Export failures are counted on `Instrumentation.loss` and never raised.
- A trace is kept or dropped whole, so a sampled-away run leaves no orphan children.
- A run that failed can be kept after the fact, however the sampler decided at the start.

`Instrumentation(tracer=None)` records and exports nothing, which is the supported way to
run with tracing off. A `step` opened outside any run records nowhere rather than raising:
instrumentation that refused to run without a run is instrumentation every caller has to
guard.

## Names and attributes

Span names are one per kind — `adk.run`, `adk.model`, `adk.tool`, `adk.retrieval`,
`adk.memory`, `adk.guardrail`. The operation goes in `adk.operation`, not in the name,
because a span name carrying a tool name is a cardinality problem in every backend.

Every exported span carries `adk.run_id`, `adk.span_id`, `adk.parent_id`, `adk.attempt`,
`adk.status` and `adk.duration_seconds`, plus whatever the caller set on it.

A retry is a sibling span with a higher `attempt`, never a reopened one. Reopening loses
the timing of the try that failed, which is the one worth looking at.

## Streaming and failure

`span.first_token()` records `adk.time_to_first_token_seconds` — only the first call
counts, the rest are latency. `span.iterated()` advances `adk.iterations` on the run span.

A failing span records `adk.error.type`, and for a kit error `adk.error.retryable`, then
re-raises. Cancellation closes the open spans with status `cancelled` rather than leaving
them unended or reporting a fault that did not happen.

## Sampling

```python
from tesserix_adk.core import Sampling

Sampling(ratio=0.05)                          # keep a twentieth, plus every failure
Sampling(ratio=0.05, always_on_error=False)   # keep a twentieth, full stop
```

The decision comes from the run id, so a workflow replay decides the same way in a
different process. Failures are kept whatever the ratio said, because the traces worth
having are the ones nobody would have sampled.

## Bounds

`SpanLimits(max_spans=...)` caps one trace. A recursive tool fan-out is otherwise
unbounded, and an unbounded trace is a memory leak. The root always survives truncation,
and the dropped count goes on it as `adk.spans.dropped` — a truncated trace says so rather
than looking merely quiet.

`SpanLimits(remembered_runs=...)` bounds the replay memory. Exporting a run id that was
exported recently is suppressed and counted on `loss.replayed`, so a durable workflow
replaying its history does not double the spans.

## Overhead

Recording one span is a dictionary write, a clock read and a list append — no I/O, no
serialisation and no lock. The budget is under 10µs per span on a modern laptop core;
`examples/auto_instrumentation.py` measures it and prints what it got. The exporter runs
once per run, outside the work being measured.

## Related

- [Cost attribution](cost-attribution.md) — the counters that accompany these spans.
- [Multi-agent traces](multi-agent-trace.md) — carrying the trace across a hop.
