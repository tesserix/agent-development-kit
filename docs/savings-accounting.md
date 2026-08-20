# Reporting what shaping saved, without inventing the part nobody measured

Input savings are a fact: the unshaped and the shaped token counts exist at the same moment,
so their difference can be printed as a measurement. Output savings are not. The system never
sees the response it would have received without shaping, so every output figure is a
counterfactual — and a dashboard that prints a counterfactual in the same typeface as a
measurement is how cost reporting stops being believed.

```python
report = account(runs, policy=HoldoutPolicy(fraction=0.05))
print(report.summary())
```

## Two figures, two bases

`Figure.basis` is `measured`, `estimated` or `insufficient`, and `label()` prints it beside
the number, always. Input savings are `measured`. Output savings are `estimated` from the
holdout comparison and carry an interval, never a bare point figure.

## The holdout

`HoldoutPolicy(fraction=...)` holds a random slice of traffic out of shaping entirely.
Assignment is a hash of the run id, so it is stable: a retried run stays in the arm it began
in rather than being shaped on the second attempt and compared against itself. The `salt`
namespaces the assignment so two experiments do not hold out the same runs and confound each
other. The arm is recorded on every run, including where shaping was globally off, because
an unlabelled run cannot be put on either side of the comparison later.

The slice costs a few percent of the available savings. It buys the only output figure that
survives a review.

## What it refuses to claim

- **No holdout.** With `fraction=0.0` the output figure is `estimated` with `0` tokens and
  says so: no control exists, so nothing about the output can be attributed to shaping. It is
  never presented as measured, and the absence is in `summary()` rather than omitted.
- **Too small a sample.** Below `MINIMUM_ARM` runs in either arm the basis is `insufficient`
  and there is no interval. A very wide interval reads as a result; "not enough data" does
  not.
- **A run counted twice.** `account` raises `ConfigurationError` rather than double-counting.
- **A prompt that grew.** `ShapedRun` refuses counts where the shaped prompt is larger than
  the original, which is a counting bug rather than a negative saving.

## Tenancy and content

`by_tenant` groups before it totals, so no tenant's figure ever draws on another's traffic.
`ShapedRun` holds a run id, a tenant, an arm and four integers — no content, no prompt, no
response — because an accounting record that carries text is a second copy of the
conversation living in the metrics pipeline.

## Known limitations

- The interval is a normal approximation over the two arms' means, not a model of the
  workload. It says how noisy the comparison is, not that shaping caused the difference.
- The comparison assumes the arms see similar traffic. A holdout that happens to catch the
  long-running tenants will read as a saving that is not one; slice `by_tenant` where that
  is plausible.
- Bypassing shaping for the holdout arm is the caller's to wire up — the policy says which
  runs, the shaping path decides what to do about it.

A runnable version of all of the above is `examples/savings_accounting.py`.
