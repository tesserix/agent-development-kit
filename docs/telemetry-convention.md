# One set of attribute names

Two products instrument the same agent and name the same field `tenant`, `tenant_id` and
`org`. The cost question then gets answered twice, differently, and neither number can be
compared with the other. `AttributeSet` is the contract that stops that: a versioned set of
names under the `adk.` prefix, populated from what the run already knows.

```python
from tesserix_adk.observability import AttributeSet, CacheStatus, Measured, Outcome

attributes = AttributeSet.here(
    agent="refunds",
    agent_version="3.1.0",
    model="gpt-5",
    provider="openai",
    run_id=run_id,
    outcome=Outcome.ANSWERED,
    cache=CacheStatus.MISS,
    input_tokens=Measured.of(1200),
    output_tokens=Measured.of(340),
    latency_seconds=Measured.of(2.5),
    cost=cost,
    pricing_version="2026-08-01",
)
span.set(**attributes.rendered())
```

`here` takes the tenant and user from the bound tenant scope, so neither is a field a
caller can forget or get wrong. Outside a scope it raises: a span attributed to no tenant
is spend nobody chases.

## Nothing is estimated

A provider that reported no usage produces `adk.input_tokens.unavailable = "not reported"`,
not a number. A model with no price list produces `adk.cost.unavailable = "not priced"`,
not zero. Both would otherwise be totalled by a dashboard as if they had been measured,
and zero in particular reads as free.

```python
Measured.of(1200)                            # measured
Measured.missing(Unavailability.NOT_REPORTED)  # not measured, and why
```

A cost that was derived rather than billed carries `adk.cost.confidence = "estimated"`, and
the price list that produced it travels as `adk.pricing_version` so a historical figure
stays interpretable after a price change.

## Cardinality

`CARDINALITY` declares each name low or high. `metric_dimensions()` returns only the low
ones, because a counter split by user id or run id is one time series per user, which is
how a metrics backend falls over. The high-cardinality names stay on spans, where they cost
one write.

## Versioning

`adk.convention` travels on every span. Within a major version the set only gains names — a
rename breaks a dashboard, an alert and a saved query on the same afternoon. A name that
has to go is deprecated and kept alongside its replacement for a full major version.

`conforms(attributes)` refuses two things: a missing mandatory name, and a name in the
`adk.` namespace the convention has not defined. Squatting is how the next version of the
convention breaks somebody's dashboard, so a product's own attributes go in `extra`, under
its own prefix.

The names deliberately avoid `gen_ai.`, so kit attributes and OpenTelemetry's GenAI
conventions can sit on the same span without either overwriting the other.

## Related

- [Cost attribution](cost-attribution.md) — where the cost on a span comes from.
- [Spans without wiring](auto-instrumentation.md) — what emits the spans these describe.
