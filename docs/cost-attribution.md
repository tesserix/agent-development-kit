# Cost attribution

A vendor invoice is one number per API key. Which tenant, which product, which agent and
which version of it are all known while the run is happening and gone by the time the bill
arrives, which is why cost questions end in a spreadsheet built from application logs that
were never designed for one. This surface reads the answer back off the run.

Two rules make it usable. **Attribution is derived, never supplied** — a consumer that has
to remember to tag its own spend will forget on one path and mis-tag it on another. And
**metrics are not traces** — a cost total taken from sampled spans is a number that looks
precise and is wrong.

## Reading a run

```python
from tesserix_adk.observability import spend_of, totals_by

records = spend_of(run)
records[0].attribution.tenant   # "acme"
records[0].cost.total           # Decimal("0.20")
```

`spend_of` returns one `SpendRecord` per metered step, in the order the run recorded them.
Each carries an `Attribution`: tenant, user, agent, agent version, model, prompt version,
task class and run id. Three of those are worth stating explicitly:

- **The model is the one that actually burned it.** A run that fell back to a second vendor
  mid-way reports each step against the model that answered it, not both against the first.
- **A failed attempt is a record, not a gap.** Tokens a vendor read before it failed are on
  the invoice whether or not anything came back; dropping them makes a run that never
  answered read as a run that cost nothing.
- **A tenant is never widened.** A run acting on behalf of another tenant's request bills
  the tenant it ran as. Attributing it to the requester would move money across the
  boundary that exists to stop exactly that.

Anything the run could not say resolves to the explicit `unknown` bucket and is listed by
`attribution.unknowns`, because spend attributed to a blank is spend nobody chases.

## Chargeback

```python
totals = totals_by(records, "tenant", "agent")
totals[("acme", "planner")].cost.total    # Decimal("0.70")
totals[("acme", "planner")].calls         # 3
totals[("acme", "planner")].estimated     # False
```

The key is a tuple in the order asked for, whether one dimension or four, so a caller never
branches on how many it requested. Any field of `Attribution` is a valid dimension and
anything else is refused by name rather than silently grouping everything into one bucket.
A group spanning two currencies raises: the sum would be a number that is true in neither.

Common questions, as one call each:

| Question | Call |
|---|---|
| What did each tenant spend? | `totals_by(records, "tenant")` |
| Which agent is the expensive one? | `totals_by(records, "agent")` |
| What did the model change cost us? | `totals_by(records, "model", "agent_version")` |
| Where is the reasoning spend going? | `totals_by(records, "task_class", "tenant")` |
| Did the new prompt get cheaper? | `totals_by(records, "prompt_version")` |
| Which reviewed artifact spent it? | `totals_by(records, "definition")` |

## The prompt-cache hit ratio

```python
totals = totals_by(records, "agent")
totals[("clerk",)].cached_tokens    # 900
totals[("clerk",)].hit_ratio        # 0.45
totals[("clerk",)].measured         # True
```

Cache hit ratio is the single number that says whether the context-engineering work is
paying off. Without it, prefix stability is unfalsifiable — every change to prompt
assembly reads as an improvement if nothing counts what the server re-evaluated. On CPU
that is not a cost-optimisation nicety: prefill dominates, so a prefix that stopped being
stable is a deployment that stopped being usable.

Cached input is tracked apart from fresh input at every level:

| Level | Where |
|---|---|
| One call | `Usage.cached_tokens`, `Usage.fresh_input_tokens`, `Usage.cache_hit_ratio` |
| One group | `Totals.cached_tokens`, `Totals.cache_write_tokens`, `Totals.hit_ratio` |
| A metric store | `adk.input_tokens` and `adk.cached_tokens`, divided in the query |

The counters are emitted separately rather than as one ratio because a ratio cannot be
re-aggregated: averaging the hit ratios of two series of different sizes gives a number
that is true of neither. Two counters divide correctly at any grouping.

`fresh_input_tokens` never goes negative. Vendors disagree about whether a cache read sits
inside the reported input count, and a negative token count is believed by whatever divides
it next.

**Nothing read is a ratio of zero, not a division error** — and `measured` says which of
the two you are looking at. `hit_ratio == 0.0` with `measured is False` means nobody sent
anything; with `measured is True` it means the cache missed every time. A dashboard that
cannot tell those apart reports an outage as perfect behaviour, or the reverse.

Cache *writes* total apart from reads because they are priced apart, often at a premium: a
total that folds them together makes caching look free on the turn that is paying for it.

## Reconciling with an invoice

`Totals.estimated` is true when any row behind the number was counted rather than metered —
a self-hosted model, a call with no price card, a token count the kit worked out itself.
Those rows will never appear on a vendor invoice, so reconciliation sets them aside rather
than chasing a difference that is not one. The practical procedure:

1. Total the billing period by model with `totals_by(records, "model")`.
2. Drop groups where `estimated` is true and account for them separately; self-hosted spend
   is infrastructure cost, not vendor cost.
3. Compare what is left against the invoice per model. The kit counts a call when the
   response arrives, so calls in flight across the period boundary explain a small
   difference in either direction.
4. A difference larger than that is a price card that is out of date — see
   [`docs/cost.md`](cost.md), where a card is added rather than edited.

## Exporting

```python
from tesserix_adk.observability import Dimensions, record_spend

record_spend(run, tracer=tracer, meter=meter, dimensions=Dimensions(tenants=known))
```

Nothing is wired into the run loop. `record_spend` reads a finished run, so a collector
outage or a slow exporter cannot reach into the run that produced the numbers.

**Spans** get the full attribute set under the `adk.` prefix — every dimension above plus
step, outcome, input and output tokens, cost, currency and whether it was estimated. One
set of names rather than one per product: a cost question that has to know which team
exported the span is one nobody answers twice the same way.

**Counters** are emitted whatever the trace did. `adk.cost`, `adk.tokens`,
`adk.input_tokens`, `adk.cached_tokens` and `adk.calls` carry a deliberately smaller
dimension set — tenant, agent, model, task class, outcome,
estimated, currency — because a tenant id per series is a metric store that falls over at
the worst possible moment. `Dimensions` states which tenants, agents and models a
deployment wants kept as their own series; everything else lands in `other` and is still
counted. The money is never dropped, only the ability to break it out from a metric, and
the span still carries the full identity for an investigation that needs it.

Passing `sampled=False` exports no spans and still counts every increment.

## Redaction

Cost data is queried by people who were never cleared to read prompts, so attributes a
consumer attached are pattern-scrubbed on the way out: addresses, vendor keys, bearer
tokens, JWTs and long opaque hex strings, plus whatever `Redactor(extra_patterns=...)` a
deployment adds for its own references. A masked attribute is recorded on an `adk.redacted`
event naming the keys, so a missing value reads as a decision rather than as a bug in the
exporter.

The kit's own `adk.` attributes are passed through unscrubbed. They are structural identity
— a tenant id, an agent name, a token count — and redacting them would leave spend
attributed to nobody, which is the failure this whole surface exists to prevent. Case or
prompt content never reaches them in the first place.

## Known limitations

- Metric dimension bucketing is an allow-list stated in configuration. There is no
  automatic top-N: bucketing whatever arrived after the store filled up gives a different
  answer every week.
- Spend is read from a finished run. A long run reports nothing until it ends; streaming
  partial attribution is a separate story.
- Latency is not attributed here. It is a span duration, and the exporter already has it.
- The hit ratio is only as good as what the vendor reports. A provider that sends no cache
  fields reads as a total miss, which is why `estimated` travels with every group.

Exercised by [`examples/cost_attribution.py`](../examples/cost_attribution.py) and
`tests/test_cost_attribution.py`.

## The definition dimension

`agent_version` is declared and can be edited in place, so two runs claiming one version
may not be the same agent. `definition` is the `AgentDefinition` revision the run
declared — a digest of everything that definition says — so a cost change is attributable
to an exact reviewed artifact. A run started from a bare agent reports `unknown` and says
so through `unknowns`. See [`agent-definition.md`](agent-definition.md).
