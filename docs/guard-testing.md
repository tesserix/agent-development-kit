# Testing a guard

"Guards enabled" is not a safety property. Without a corpus nobody can say what fraction of
real payloads a guard catches, and a weakened detector regex is invisible until an incident.
The harness turns that into three numbers, produced offline with no model and no network.

```python
metrics = await measure(PIIGuard(tenant="acme"), families={GuardFamily.PII})
assert not metrics.failures(GuardThresholds(recall=1.0, false_positives=0.0))
```

## Recall and false positives are reported together

A guard that blocks everything has perfect recall and is useless. Only the benign control
set shows it, so every measurement runs the control set whatever families the guard claims,
and `GuardMetrics` carries both numbers plus a p95 evaluation time.

A guard is judged only on the families it declares. A detector of identifiers is not a worse
detector for letting an injection payload past — that is a different guard's job.

## Thresholds are declared per guard

```python
INJECTION_BAR = GuardThresholds(recall=0.71, false_positives=0.3, p95_seconds=0.05)
```

An internal agent's guard is deliberately permissive, and holding it to a customer-facing
guard's bar means either the bar or the guard is wrong. Each guard states its own, and
`failures()` names the specific cases that regressed rather than reporting a number:

```
recall 0.57 below 0.71
missed: a forged chat turn borrowed from a model's template
```

The bars for the kit's own guards live in `tests/test_guard_harness.py` and are a ratchet.
Raising one when a guard improves is the point. Lowering one is a change a reviewer sees,
which is the whole reason it is a number in a file rather than a claim in a README.

A run that measured nothing never passes: an empty corpus produces a recall of 1.0, which is
exactly the shape of a passing gate, so `failures()` reports it as a failure instead.

## The corpus

`GUARD_CORPUS` ships with the kit, versioned by `CORPUS_VERSION`. Comparing recall across
versions without saying so lets an easier corpus masquerade as an improved guard, so every
report carries the version it came from.

Cases cover injection, identifiers, content policy, tool escalation, and a benign control
set that deliberately includes near-misses — a policy document quoting the words an
injection uses, a number that is not a card number.

Everything in it is synthetic. `assert_synthetic` refuses a case carrying a live-looking
credential, an email outside the reserved example domains, or a card-shaped number that
passes Luhn, and it runs over the shipped corpus in the kit's own tests.

**Adding a case.** When a real bypass is found, add the payload to `GUARD_CORPUS` with the
family it belongs to and the least restrictive verdict that would have prevented it,
neutralised into a synthetic form, and bump `CORPUS_VERSION`. Expect the guard's recall to
drop and its bar to fail — that is the corpus doing its job, and the fix is the guard.

## Sampling stays deterministic

```python
subset = sampled(GUARD_CORPUS, 200, seed=commit_sha)
```

A growing corpus that makes CI unaffordable gets sampled, but a sample that varies per run
turns a real regression into a flake somebody reruns. Seeded by the commit, a given commit
always measures the same subset.

## A remote classifier stays offline

A guard calling a hosted classifier cannot be measured in CI without a network and cannot be
measured reproducibly with one. `RecordedGuard` replays recorded verdicts; content nobody
recorded blocks rather than being guessed at, because a missing recording is a check that
did not happen.

## The contract suite

```python
class TestOurGuard(GuardrailConformance):
    def make_guard(self): return OurGuard()
    def families(self): return {GuardFamily.INJECTION}
    def thresholds(self): return GuardThresholds(recall=0.8, false_positives=0.1)
```

A third-party guard inherits the protocol cases, the corpus run against its own bar, and the
obligation that a refusal carries a matchable code and never quotes the payload it refused.

## Known limitations

- Fail-closed behaviour cannot be induced from outside a guard, so the suite cannot assert
  it. `assert_fails_closed` is provided for a guard author who can inject a fault into
  their own implementation.
- No shipped guard claims the escalation family: those payloads are refused structurally by
  the resolved allowlist rather than by reading content. See
  [`docs/tool-allowlists.md`](tool-allowlists.md).
- The shipped heuristic content classifier matches a short term list, so its recall against
  a paraphrase is measured and low. Replacing it with a real classifier is the intended
  path, and the corpus is how that replacement is compared.

## Related

- [`docs/guardrails.md`](guardrails.md) — the pipeline these guards run in
- [`docs/prompt-injection.md`](prompt-injection.md) — what the injection cases are testing
- [`examples/guard_testing.py`](../examples/guard_testing.py)
