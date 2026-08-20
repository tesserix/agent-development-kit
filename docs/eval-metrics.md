# Metrics: quality, spend and speed in one report

A prompt change that answers better while tripling cost per case and doubling p95 latency
currently reads as an unqualified win, because only one of those three is measured at review
time and the other two arrive on the bill. Here they are peers: same protocol, same
aggregation, same thresholds, same table.

```python
report = measure(
    suite,
    result,
    (ExactMatch(), CostPerCase(), LatencyMs()),
    thresholds=(
        Threshold(metric="exact_match", minimum=0.9),
        Threshold(metric="cost_per_case", maximum=0.02, warn_within=0.005),
        Threshold(metric="latency_ms", maximum=2500.0),
    ),
)
print(report.table())
sys.exit(report.exit_code)
```

## The protocol

Three names, and custom metrics are ordinary consumer-owned objects:

```python
class AnswerLength:
    name = "answer_length"
    higher_is_better = False

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:
        return MetricValue(value=float(len(answer_of(run))), unit="chars")
```

`Metric` is a stability commitment: it is consumer code that implements it, so it moves only
under [`docs/versioning.md`](versioning.md).

## Built in

| Correctness | Operational |
|---|---|
| `ExactMatch` | `TokensIn`, `TokensOut` |
| `SchemaValidity` | `CostPerCase` (with its currency) |
| `ToolSequenceMatch` (against `EvalCase.expected_tools`) | `LatencyMs` |
| `Groundedness` (against `EvalCase.expected_sources`) | `CacheHitRate` |
| `RefusalRate` | |

`Groundedness` measures citation discipline: what share of the answer's `[source-id]`
markers name a source the case declared. Whether the cited source actually supports the
claim is the judge's question, not this one.

## Unknown is not zero

The rule that makes the rest trustworthy. `MetricValue` carries either a number or the
reason there isn't one:

- A self-hosted vLLM or Ollama deployment has no price list, so `cost_per_case` is unknown.
  Reporting it as zero would let a migration read as free rather than as unmeasured.
- A provider response with no usage block makes the token metrics unavailable for that case.
- A cancelled or iteration-capped run has no answer, so the correctness metrics decline to
  judge it instead of scoring it wrong.

Unknowns are counted in the aggregate's `unknown` column and kept out of the mean. Mixed
currencies within one suite are the same idea one level up: dollars and euros are not
averaged, and the aggregate says which two it saw.

## Small samples say so

Every aggregate carries `n`, `p50`, `p95` and — with at least five scored cases — a 95%
interval. Below that, `ci_low` and `ci_high` are `None` and `reliable` is false, with a note
saying why. Printing an interval over two cases invites a confidence the data cannot support.

Per-tag breakdowns come free: `report.aggregate("exact_match", tag="refunds")` is the same
aggregate over one slice, which is where a regression concentrated in one category shows up
before the overall mean moves.

## Thresholds fail closed

`pass`, `warn` or `fail` per declared threshold, and `warn_within` gives a metric drifting
towards its ceiling somewhere to appear before it breaches.

Two deliberate refusals:

- A threshold naming a metric nobody computed raises rather than being ignored.
- A threshold whose metric produced no value at all **fails**, with the reason "nothing
  measured it". A gate that clears itself when the number is missing is not a gate.

## A metric that breaks is a broken case, not a zero

Custom metrics are consumer code and consumer code raises. That case is recorded as a
`MetricFailure` with the metric name and the formatted traceback, `report.ok` is false and
`exit_code` is `1`. The value is never coerced to zero, and the other metrics still report.

## Two reports, one measurement

`report.table()` is the fixed-width table for a human, `report.summary()` the single line for
a PR comment, and `report.as_dict()` the JSON the CI gate reads. All three come from the same
aggregates, so the number in the comment and the number in the gate cannot disagree.

## Known limitations

- `ExactMatch` is string equality after casefolding and stripping; semantic equivalence is
  the judge's job.
- `RefusalRate` reads the answer for refusal phrasing in English.
- Confidence intervals use a normal approximation, which is what a suite of tens of cases
  supports; it is not a significance test between two runs — that is the regression gate.

A runnable version of all of the above is `examples/eval_metrics.py`.
