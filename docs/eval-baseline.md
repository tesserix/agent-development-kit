# The baseline a change is measured against

A prompt edit merges because it reads well. The regression it caused is found by users a
fortnight later, in a week of commits nobody wants to bisect. The gate here compares the
change against the numbers already in production, on the same dataset, and blocks the merge
on a real regression.

```python
report = compare(Baseline.read(stored), Baseline.of(measured, provenance=now), policy=policy)
print(report.comment(artefacts=run_url))
sys.exit(report.exit_code)
```

## What a baseline holds

Per-metric aggregates, every per-case value, and the provenance that makes the numbers mean
something: suite, dataset version, agent version, prompt version, model and a digest of the
cassettes replayed. The per-case values are the part that matters at review time — they are
how the report names the six cases that got worse rather than reporting that a mean moved.

`Baseline.of(report, provenance=...)` freezes what `measure` produced. Nothing else in the
kit knows the agent's version or which cassettes were replayed, so the consumer supplies it,
and a provenance naming a different dataset version than the report measured is refused.

## The gate fails closed

Four refusals, each with the command that fixes it:

- **No baseline.** The first run on a new suite is bootstrapped deliberately —
  `python -m tesserix_adk.cli.evals bootstrap` — and never inferred from the run under review.
- **Not a baseline.** Another tool's JSON, or one from a newer format, refuses rather than
  being read for whichever keys happen to match.
- **A different dataset version.** A dataset edit and a prompt edit in one pull request make
  every number a mixture of two causes, so the comparison is refused rather than reported.
- **A declared metric nobody measured.** The delta is a `fail`, not an absence.

Each raises `BaselineUnusableError` with a `reason` of `missing`, `format`, `dataset` or
`suite`, and the CLI exits `3` — distinct from the `1` that means "measured, and worse".

## The noise band

A metric fails when it moves further than `tolerance + band`, where the band is the wider of
the noise the policy declares and half the baseline's own 95% interval. Inside the tolerance
is a clean pass; between the tolerance and the band is a `warn` that reports without
blocking. A gate with no band reports every rerun as a regression and gets switched off,
which is the failure mode this exists to avoid.

## Flaky cases

`quarantined` case ids — on the policy or on the baseline itself — are kept out of the
metric comparison and still listed in the comment, marked as quarantined. They can never
block, and they stay visible, so quarantine is a debt that gets paid rather than a hiding
place. A real regression next to a quarantined case still fails.

## Overrides argue in the pull request

`Bypass(metrics=..., by=..., reason=...)` turns a fail into a warn for the metrics it names
and only those. It refuses to exist without a name and a reason, and it is rendered in the
comment, so a fourth override in a month is visible to the reviewer rather than buried in a
config file.

## What CI runs

`.github/workflows/eval-gate.yml` is callable from a consumer's workflow. It compares,
writes the comment, edits the existing comment rather than adding a fifth, and exits with
the gate's code. Cassette-backed replay is the intended mode, so a pull request costs no
provider spend.

On merge to the default branch, `promote` makes the merged run the baseline and keeps the
one it replaced at `<name>.previous.json` for a rollback.

## Known limitations

- The band is a heuristic, not a significance test; with a small suite the interval is
  withheld by `measure` and the declared noise is all there is.
- A price change with no agent change is detected as "nothing about the run changed", which
  points at the provider — it does not fetch a price list to prove it.
- Comparison is per metric and per case. A regression that only shows up in a combination
  of metrics is not modelled.

A runnable version of all of the above is `examples/eval_baseline.py`.
