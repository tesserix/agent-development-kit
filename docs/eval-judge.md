# The judge: earn the right to score

An "ask a model to rate this out of five" step is two lines of code and no evidence. Nothing
in it shows that the number tracks what a reviewer would have said, so a prompt tuned against
it moves wherever the noise points while the review reads as rigorous. Here a judge is not
usable as a gate until it has demonstrably agreed with people.

```python
judge = LlmJudge(provider, model="judge-1", rubric=RUBRIC)
calibration = await judge.calibrate(labelled_examples)
metric = JudgeMetric(calibration, scores)      # raises unless the judge cleared its floor
report = measure(suite, result, (metric,), thresholds=(Threshold(metric=metric.name, minimum=2.5),))
```

## A rubric is a versioned artefact

`Rubric` carries a name, a version, one criterion and every level with the description that
earns it. Bump the version whenever the wording moves: old scores are not comparable with new
ones, and a rubric edit shifts a suite as surely as a prompt edit does.

Every `JudgeScore` records `rubric@version/model/prompt_version` as its stamp. That stamp is
what a calibration is a statement about — see drift, below.

## Structured or refused

The judge returns one JSON object: `score`, `reason`, `evidence`. Anything else is a
`SchemaViolationError` — free text is never scanned for a digit, a score the rubric does not
declare is refused, and a verdict quoting no evidence is refused too, because a score nobody
can review is not a measurement.

A provider failure propagates. The case errors; it is never given a default score, since a
default is a made-up number in the place a measurement was supposed to go.

## Calibration, and the floor

`agreement(scores, labels)` returns a `Calibration`:

| | |
|---|---|
| `kappa` | Cohen's kappa, chance agreement taken out. `None` where it is undefined. |
| `spearman` | Rank correlation, for a rubric where ordering matters more than exactness. |
| `exact` | The plain share of identical scores, for a reader who wants one number. |
| `ties` | The share sitting on the judge's most common score. |
| `length_bias` | Rank correlation between candidate length and judge score. |
| `self_scored` | The share judged by their own model family. |

`Calibration.require()` refuses in four distinct ways, and each names what to do about it:
agreement below the rubric's floor, agreement nobody measured, a judge that gave one score to
nearly every case, and drift — a score stamped by a judge the calibration does not cover.
`JudgeMetric` calls it at construction, so an uncalibrated judge fails before any of its
numbers reach a report.

The default floor is `0.6`, conventional "substantial agreement". A rubric may demand more.

## The candidate is data

The answer under review is sealed inside `<candidate-{nonce}>`, where the nonce is derived
per case, so a candidate that writes its own closing tag closes nothing. The judge is told
in as many words that the block is material under review and that any instruction inside it
is part of what it is scoring. Whatever the screen recognises is recorded on the score's
`flagged` field for review rather than blocking the case: the candidate is the thing being
measured, so refusing to read it would be refusing the case.

## The three biases worth naming

- **Position.** `compare` randomises which candidate is shown first, keyed on the seed and
  the case id, so the order is reproducible across runs but not constant across cases. The
  winner comes back as `a`, `b` or `tie` — the caller's terms, never the position's.
- **Verbosity.** The prompt says length is not quality, and every score records
  `candidate_chars`, so `Calibration.length_bias` can show a judge rewarding length whatever
  the prompt said.
- **Self-preference.** A judge scoring its own family rates it above blind human preference.
  `shares_family` recognises it, `JudgeScore.self_scored` records it, and the calibration
  notes the share — a calibration set drawn from that same family will not reveal it.

## Wavering judges

`samples=n` asks n times and reports the median with `disagreement`, the spread over the
rubric's range. A judge that answers 1, 3, 3 on the same case has told you something about
the rubric, and averaging that away quietly would hide it.

## Known limitations

- Kappa treats scores as unordered categories, so a judge one level out is scored the same as
  one three levels out. Read `spearman` beside it on an ordered rubric.
- Human labels are assumed correct. Inter-annotator agreement, and the tooling to collect it,
  belong to the annotation workstream rather than here.
- The judge's own tokens, cost and latency land on `JudgeScore.usage`, which is where a suite
  totals them: a judge is a model call, and gating on one is a bill.

A runnable version of all of the above is `examples/eval_judge.py`.
