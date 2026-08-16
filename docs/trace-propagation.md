# Trace context across hops and workflows

A durable multi-agent flow produces one disconnected trace per participant unless the
context travels with the work. `tesserix_adk.observability.propagation` carries it in the
W3C format every backend already speaks, so a peer that knows nothing about this kit still
lands in the same trace.

```python
from tesserix_adk.observability import W3CContext, joined

context = W3CContext.rooted("run-1")
headers = {**request_headers, **context.carried()}   # outbound

downstream, broken = joined(inbound_headers, run_id="run-2")   # inbound
```

## The interoperability contract

`carried()` emits `traceparent` and `tracestate` exactly as the W3C recommendation defines
them, and nothing else is required of a peer:

- `traceparent` is `00-<32 hex>-<16 hex>-<flags>`. A peer running OpenTelemetry, Datadog or
  a hand-rolled propagator reads it without configuration.
- `extracted()` reads headers case-insensitively, because a broker that normalises header
  case has not broken the trace, and accepts a **future version prefix** — a version this
  code does not recognise still has a readable trace id in the first three fields, and
  discarding it would cost the trace for no gain.
- An all-zero trace or span id is refused, as the spec requires.
- `tracestate` round-trips other vendors' entries. This vendor's entry is written first, as
  the spec's ordering requires, and the rest follow in arrival order.

`MAX_STATE_ENTRIES` (32) and `MAX_STATE_LENGTH` (512) cap what is rendered. The cap matters
because a transport that silently truncates an oversized header produces a header that is
no longer parseable — a lost trace rather than a shortened one. Entries are dropped from
the far end, keeping the nearest participants, which are the ones an investigation needs.

`sampled` is **carried, not re-decided**. A peer that samples for itself is how a trace
arrives truncated at the service boundary: half the spans recorded, the other half absent
with no signal that they ever existed.

## Replay safety

Nothing here reads a clock or a random source. `trace_id_of(run_id)` and
`span_id_of(run_id, name, attempt)` are sha256 digests of what the run already knows:

```python
W3CContext.rooted("run-1").trace_id == W3CContext.rooted("run-1").trace_id   # True
```

A Temporal workflow replayed after a worker restart re-executes its code from history. If
identifiers came from a clock or `random`, the replay would mint a second trace alongside
the first and export it — one flow, two traces, neither complete. Deriving them means the
replay rebuilds exactly what the first attempt built, so a duplicate export is
indistinguishable from the original and collapses into it.

An activity retry is a **different** span: `attempt` is part of the digest, so try 2 is a
sibling of try 1 rather than a reopening of it. Both attempts stay visible, which is the
point of instrumenting a retry at all.

## When parent–child would lie

A parent–child edge claims the parent was running while the child ran. Three common cases
break that claim, and each gets a `Link` instead:

| Case | Kind |
|---|---|
| A queue message delivered hours after the producer's span closed | `FOLLOWS_FROM` |
| One branch of a parallel fan-out | `FAN_OUT` |
| Work resumed after a suspension, such as a human-gated approval | `CONTINUATION` |
| A hop that arrived with no readable context | `BROKEN` |

```python
link = context.link(LinkKind.FAN_OUT, branch="north")
attributes = link.rendered()   # adk.link.trace_id, adk.link.span_id, adk.link.kind, ...
```

`rendered()` exists for backends with no first-class link type: the relationship becomes
attributes that can still be queried, rather than being dropped on the floor.

### A long suspension

A workflow that waits a day for an approval cannot hold a span open across it — a span open
for 24 hours distorts every latency percentile it appears in, and most backends drop it.
Close the span at the suspension, and open a `CONTINUATION`-linked span on resume. The
trace id is unchanged, so both halves are one trace.

## A hop that broke

`joined()` never raises. A message that arrives without a readable context is still work
that has to run; refusing it costs the work as well as the trace.

```python
downstream, link = joined(headers, run_id="run-2")
```

Where nothing readable arrived but the caller set the `adk-correlation` header,
`joined()` starts a new root **and** returns a `BROKEN` link back to the correlated
context. A visible gap someone can follow is worth more than two traces that look
unrelated — the second one otherwise reads as a flow that started from nowhere.

## Related

- [`docs/telemetry-convention.md`](telemetry-convention.md) — the attribute names spans carry.
- [`docs/export-redaction.md`](export-redaction.md) — what may leave the process on them.
