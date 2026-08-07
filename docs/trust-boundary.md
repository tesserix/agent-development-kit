# Trust boundaries and fail-closed fallback

Resilience and confidentiality pull in opposite directions the moment a model is
unavailable. A chain that promotes a hosted vendor because the self-hosted endpoint is
down, or the standard tier because the sealed one is, has traded a data-handling guarantee
for an availability one — silently, and in the direction nobody would have approved.

```python
ModelSpec(
    provider="vllm",
    model="qwen",
    capabilities=BIG,
    trust=TrustBoundary(tier="sealed", hosting="self-hosted", residency="in-central"),
)
```

Worked through with no network: `examples/trust_boundary.py`.

## What a boundary is

Three axes: `tier`, `hosting`, `residency`. All three must match for two models to be
interchangeable, and `differs_from` names the ones that do not — "not equivalent" on its
own leaves an operator to work out which of three to change.

They are free strings rather than enums. The tiers, hosting arrangements and residency
classes that matter are a deployment's own vocabulary; a kit-level enum would either be
wrong or force a fork to add a value to it.

| Source states | Target states | Fallback |
|---|---|---|
| nothing | anything | allowed — the kit cannot enforce a boundary nobody declared |
| something | the same thing | allowed |
| something | something different | refused, naming the axes |
| something | nothing | refused — an undeclared boundary is an unknown one, and unknown is not equal |

The first row is why adding this breaks no existing deployment, and the last is why adding
a boundary to one model is enough to protect it.

## What routing does with it

The chosen model's boundary is the run's. Candidates that could do the work but sit outside
it are dropped from the chain and recorded in two places: `excluded_by_boundary`, and
`rejected` with a reason naming the axes. They are kept apart from ordinary rejections
because the distinction matters — these models *could* have answered, and were not allowed
to.

Trust is not the only floor. Every link had already passed the capability floor, so a
fallback still cannot quietly lose structured output and leave the caller parsing prose.

## Failing closed

When the chain is spent and the failure was one another vendor might have answered,
`FallbackChain.refuse_the_excluded()` raises `TrustBoundaryError` naming what it would not
send to. The run ends `FAILED` with that on the record, and no request is made to the
out-of-boundary provider — the refusal happens before a link is chosen, not after a call.

A chain with nothing excluded refuses nothing, so a run that simply ran out of models still
fails with `FallbackExhaustedError` as it did before.

## The recorded rationale

`RoutingDecision` records what chose the model, not just what was chosen:

| Field | What it says |
|---|---|
| `required` | the capability names the work asked for |
| `min_context_window_tokens` | the context floor, zero where none |
| `boundary` | the trust boundary the run is inside |
| `considered` / `chain` / `rejected` | what was offered, what is legal, what was ruled out |
| `excluded_by_boundary` | what the boundary refused |

`explain()` reads as one line in a run record. Everything in it is drawn from a closed
vocabulary — model references, capability names, boundary axes, the rule scope. No prompt
content reaches the rationale, which is what lets a sealed matter keep the trace at all.

## Known limitations

- The boundary is declared on the model, not derived from the provider. A deployment that
  points one provider name at two different hosting arrangements has to say so itself.
- There is no boundary *hierarchy*: `sealed` does not imply `standard`. Equality is the
  whole rule, deliberately — a partial order here is a place for a mistake to hide.
