# Gating a prompt change on the eval suite

A prompt edit ships because it reads well. The quality regression is found by users, and the
cost regression is found a fortnight later on a spend graph nobody connects to a wording
change. Both are measurable before the change lands, against the version already in
production, on the same dataset.

```python
from tesserix_adk.evals import Measured, gate

report = gate(baseline, candidate)
print(report.summary())
if report.verdict != "pass":
    raise SystemExit(1)
```

This module is the comparison and the verdict. Running the suite, scoring the examples and
defining the metrics belong to the eval framework and to CI; what belongs here is the part
that has to be uniform, because it decides what ships.

## What is compared

`Measured` is one version's numbers on one dataset: the concrete `version` and its `digest`,
how many `examples` the dataset holds and how many were `scored`, the `metrics`, the
`variables` the version declares, and the `judge` that scored it. Two of those exist to make
a bad comparison impossible rather than to be reported.

The candidate must be measured by **deterministic replay** — same inputs, same fixtures, same
judge — so that the prompt is the only difference between the two sets of numbers.

## The metrics, cost among them

`DEFAULT_POLICY` declares five:

| Metric | Direction | Default tolerance |
|---|---|---|
| `task_success` | higher is better | 0.01 |
| `schema_validity` | higher is better | 0.0 — no regression at all |
| `judge_score` | higher is better | 0.05, with a 0.02 noise band |
| `p95_latency_ms` | lower is better | 250 |
| `cost_per_run` | lower is better | 0.0005 |

Cost is a gate, not a footnote under the quality numbers. A candidate that improves task
success and raises cost per run fails: spending more is a decision somebody makes on purpose,
with a record, and a `Bypass` naming `cost_per_run`, who took it and why is that record.

A project declares its own `GatePolicy` where these are wrong for it. A metric nobody declares
is not judged; a metric declared twice is refused, because ordering would decide it.

## What it refuses

**A partly scored dataset.** `EvalIncompleteError`, carrying coverage. The gate never infers
a score for an unscored example, and never passes on the half that ran — the examples that
were skipped are exactly where a new prompt breaks.

**A judge that moved.** `IncomparableEvalError(reason="judge")`. Calibration drift moves the
numbers on its own, so re-measure the baseline with the current judge before comparing.

**A prompt that changed its variables.** `IncomparableEvalError(reason="variables")`. The
golden dataset supplies the old inputs; a schema change invalidates it, and the fix is to
update the dataset with the change rather than to compare two different prompts.

**A declared metric nobody computed.** Reported as a failing move for that metric. Missing is
not passing.

## Boundary verdicts

A verdict is `pass`, `fail`, or `repeat`. `repeat` means a metric landed in the noise band
just past its tolerance — an example that flakes across the line. CI should rerun and average
`policy.repeats` runs rather than accept a coin flip in either direction. A `fail` anywhere
outweighs a `repeat`.

## What the result is for

`GateReport.attributes()` is the record, keyed to the exact `digest` measured, so the prompt
registry and [`docs/prompt-rollback.md`](prompt-rollback.md) read the verdict CI produced
rather than a second opinion. `report.permits(digest)` is the promotion check: an alias may
move only to text with a passing result for **that digest**. A version label is not enough,
because the same label over edited text is a different prompt.

## Known limitations

* The gate compares numbers; it does not produce them. A suite that scores inconsistently
  produces a consistent-looking verdict from inconsistent inputs.
* Tolerances are absolute, in each metric's own units. A relative tolerance on a metric whose
  scale changes between datasets is not expressible.
* Flipped individual examples are visible in the suite's own output, not here — this module
  sees per-metric aggregates.
