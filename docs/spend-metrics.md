# Spend and performance as metrics

Spend discovered from the provider's monthly invoice is discovered a month late, and a
runaway retry loop is expensive by then. `tesserix_adk.observability.spend_metrics` emits
the series to alert on, off the same `SpendRecord`s the spans are built from, so metrics
and traces cannot disagree about a run.

```python
from tesserix_adk.observability import ModelRate, PricingTable, SpendMeter

table = PricingTable(
    version="2026-08-01",
    rates={"gpt-5": ModelRate(input_per_million=Decimal("1.25"), output_per_million=Decimal("10"))},
)
meter = SpendMeter(collector, pricing=table)
meter.record(record, seconds=elapsed, tool_calls=4, key=f"{run_id}/model/{attempt}")
```

## The series

| Series | What it counts |
|---|---|
| `adk.tokens` | Input plus output tokens the provider reported |
| `adk.cost` | Money, in the currency it was priced in |
| `adk.latency_seconds` | Time per metered step |
| `adk.iterations` | Loop iterations per run |
| `adk.tool_calls` | Tool fan-out |
| `adk.input_tokens`, `adk.cached_tokens` | Counted apart, so a hit ratio is a division in the store |
| `adk.budget_breaches` | Ceilings crossed. Enforcement is elsewhere; this reports it |
| `adk.unknown_usage` | Calls the provider reported no usage for |
| `adk.unpriced` | Calls nobody could price |
| `adk.metrics_dropped` | Counted locally — a metric lost on the way out cannot report its own loss |

## Nothing is estimated and presented as measured

A call whose usage never arrived is counted in `adk.unknown_usage`, and contributes
**nothing** to `adk.cost`. A zero would be summed by whoever reads it and would understate
the bill without saying so.

A call whose tokens are known but whose model the table does not price still has its
tokens counted — they were spent whether or not anyone can price them — and lands in
`adk.unpriced`.

Where a cost is computed here rather than billed by anyone, the cost series carries
`cost_confidence=estimated` — kept apart from `estimated`, which says whether the *token
counts* were measured. That includes self-hosted models: a self-hosted model priced at nothing
reads as free capacity, so give it a rate and mark its source.

## Cardinality policy

Dimensions are `tenant`, `agent`, `agent_version`, `model`, `task_class`, `outcome`,
`currency`, `estimated`, `cost_confidence` and `pricing_version`. That is the whole list.

- **Never a dimension:** user id, run id, prompt text, tool arguments, error message. One
  series per user is the blow-up that takes the metric store down at the worst moment.
- `Dimensions` is an allowlist, not a cap. A deployment names the tenants, agents and
  models it wants separated; everything else lands in `other` and is **still counted**.
  What is lost is the ability to break it out, never the money — and the span still
  carries the full identity for an investigation that needs it.

## Pricing versions and currency

`pricing_version` is a dimension on the cost series, so a mid-window price change starts a
new series rather than retroactively rewriting what was already recorded.

Currency is carried, never converted. A provider billing in EUR is reported in EUR; the
kit will not convert on its own authority, because a converted figure with no recorded rate
is true in neither currency. Conversion belongs where somebody records the rate they used.

## Not counting a run twice

Pass `key=` — typically `f"{run_id}/{step}/{attempt}"`. A worker restart replays a run; the
invoice does not. A repeat under a key already counted is ignored and shows up in
`stats.duplicates`. A retry is a *different* attempt and is counted, because it really did
spend tokens.

Without a key every call counts: a caller that cannot identify a step is better served by a
visible double count than by a silent drop.

## When the collector is down

`record()` never raises. A store that refuses a counter costs that counter, not the run,
and each refusal increments `stats.dropped` — a partial failure still records everything
the store did accept.

## Recommended alerts

Definitions to deploy where dashboards live, not in the kit:

| Alert | Condition |
|---|---|
| Tenant spend rate | `rate(adk.cost[15m])` by tenant above the tenant's agreed ceiling |
| Runaway loop | `rate(adk.iterations[5m])` by agent above its normal band |
| Budget breaches | `increase(adk.budget_breaches[15m]) > 0` |
| Error rate | `rate(adk.tokens{outcome="failed"}[15m]) / rate(adk.tokens[15m])` |
| p95 latency | `adk.latency_seconds` p95 by agent above its objective |
| Cache-hit collapse | `adk.cached_tokens / adk.input_tokens` falling sharply — the first sign of a prompt change that broke prefix caching |
| Unmeasured spend | `increase(adk.unknown_usage[1h]) > 0` or `adk.unpriced` — money is being spent that nobody is reporting |
| Telemetry loss | `increase(adk.metrics_dropped[15m]) > 0` — every other alert here is unreliable while this is firing |

## Known limitations

- Budget enforcement is not here; this observes and reports what enforcement decided.
- Dashboard and alert deployment belongs in the infrastructure repository.

## Related

- [`docs/cost-attribution.md`](cost-attribution.md) — the records these series come from.
- [`docs/telemetry-convention.md`](telemetry-convention.md) — the attribute contract.
