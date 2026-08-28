# Reading a run without a collector

Iterating locally, the choice is usually between standing up a trace backend and debugging
by print. The telemetry already exists; what is missing is a way to read it.
`tesserix_adk.observability.local_view` draws the spans a run emits as a tree, and
`python -m tesserix_adk.cli trace` reads one back from a file.

```python
from tesserix_adk.observability import assembled, machine_readable, rendered

print(rendered(assembled(spans)))
document = machine_readable(assembled(spans))   # for a test assertion
```

```console
$ uv run python -m tesserix_adk.cli trace trace.json --depth 3
```

## What a step shows

Each line carries the span name, how long it took, and what the step reported — attempt
number, outcome, typed error class, guard verdict, budget state, tokens and cost:

```
adk.run 9.000s outcome=failed
  adk.tool 3.000s attempt=1
  adk.tool 3.000s attempt=2
! adk.tool 3.000s attempt=3 outcome=failed error.type=ToolTimeout
```

A step that reported no cost shows no cost. It is not drawn as `0.00`, because a zero is
summed by whoever reads it and an unmeasured step then reads as a free one.

Every retry is drawn, not only the last: a run that succeeded on attempt three took three
attempts, and collapsing them hides the reason it was slow.

## A failed run is never drawn as a tidy tree

`!` marks the step a run stopped at. The mark comes from the exported attributes —
`adk.outcome` in `cancelled`, `failed` or `refused`, or an `adk.error.type` being present —
so a failure the pipeline recorded is a failure the local view shows.

Filters do not hide it. `only=` narrows a wide trace to the span names asked for, but a
step that failed, or that has a failure underneath it, is kept regardless: a filtered view
that looks clean is how a failure gets missed.

`depth=` caps how deep the tree is drawn and states what it hid (`... 4 hidden`) rather
than trimming silently.

## A span whose parent never arrived

A span whose parent is missing is drawn as a marked (`?`) root rather than dropped — the
missing step is often exactly the one being looked for. A cycle in the parent pointers is
drawn once and does not hang the renderer.

## Sharing a trace file

`TraceFile.of(spans)` puts every span through the same `RedactingSpanProcessor` the export
path uses, so a file cannot carry what an export would not. The file states the format
version that produced it and which attributes were dropped, so a reader knows a gap is a
decision.

```python
Path("trace.json").write_text(TraceFile.of(spans).model_dump_json())
```

A file written by a newer format version is refused rather than partially read, and
`python -m tesserix_adk.cli trace` reports that with exit code `3` — a file this build cannot read is a
different problem from a path somebody typed wrong (`1`).

## One source of truth

The renderer reads the exported attribute set. It defines no attribute names of its own,
because a local view with its own vocabulary drifts from the production one and then
disagrees with it at the worst moment. What is drawn locally is what the pipeline receives.

## Known limitations

- Live streaming during a run is not here; this renders spans a run has already recorded.
- Hosted trace visualisation is out of scope for the kit.

## Related

- [`docs/telemetry-convention.md`](telemetry-convention.md) — the attribute names read here.
- [`docs/export-redaction.md`](export-redaction.md) — the redaction a saved file goes through.
- [`docs/trace-propagation.md`](trace-propagation.md) — how the spans came to share a trace.
