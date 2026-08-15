# Proving no cross-tenant leakage

A single-tenant test suite cannot fail the way a multi-tenant system does. The code
reads correctly, every assertion passes, and the defect only appears once two tenants
hold data similar enough to be confused for each other. `tesserix_adk.testing.isolation`
supplies the second tenant, the confusable data, and the markers that turn "looks fine"
into an assertion.

## The scenario

```python
from tesserix_adk.testing import IsolationScenario

scenario = IsolationScenario.confusable("acme", "rival")
```

Both tenants are seeded with documents that share titles, profile keys that collide,
and an identical cache input. Only the sentinel embedded in each body differs — so a
similarity search that forgot its filter ranks the neighbour's copy just as highly, and
a cache keyed without the tenant produces a hit across them.

`confusable()` refuses fewer than two tenants. One tenant proves nothing, which is
exactly the suite that misses this class of defect. Passing no tenants uses
`DEFAULT_TENANTS`.

Every payload is invented. The fixtures ship with the kit and get copied into consumer
repositories, so nothing in them is an extract of anything real.

## Sentinels

`sentinel_for(tenant, kind)` is the marker embedded in one tenant's data of one kind
(`"document"`, `"profile"`, `"cache"`). It is lowercase and alphanumeric so it survives
the mangling a summariser or tokeniser does to it — a marker split on a hyphen is a
marker the leak check no longer finds. Finding one where it does not belong is a leak,
not a heuristic.

## Checking a run

```python
from tesserix_adk.testing import Observed, Surface, assert_no_leak

assert_no_leak(
    scenario,
    tenant="acme",
    observed=[
        Observed(Surface.OUTPUT, answer),
        Observed(Surface.MEMORY, str(memory.dump())),
        Observed(Surface.SEARCH, str(hits)),
        Observed(Surface.CACHE, str(cache.keys())),
        Observed(Surface.SPANS, str(exporter.attributes())),
        Observed(Surface.EVENTS, str(bus.drained())),
    ],
)
```

Content is stringified by the caller, because a leak in a key is as much a leak as one
in a value. The failure names every leaking surface, the marker found, and its owner —
not just the first one.

## Surfaces nobody read

A report over surfaces that were never inspected is not clean. `assert_no_leak` fails
naming them, because a suite reporting green because it never looked is worse than no
suite. A deployment that genuinely has no cache declares it:

```python
assert_no_leak(scenario, tenant="acme", observed=[...], absent=(Surface.CACHE,))
```

The report then states which guarantee it actually verified, via
`LeakReport.declared_absent`.

## Concurrency

Context bleed shows up under interleaving, and an interleaving left to the event loop
is a test that fails one run in twenty. `interleaved` hands control over deterministically
— each tenant's run proceeds exactly one step, then the next tenant does — so the defect
fails every time:

```python
from tesserix_adk.testing import Step, interleaved

async def run(tenant: str, step: Step) -> list[LeakReport]:
    seeded = scenario.fixture(tenant)
    await step()               # the other tenant runs here
    answer = await agent.ask(seeded.cache_input)
    return [LeakReport.over(scenario, tenant=tenant, observed=[...])]

reports = await interleaved(scenario, run)
```

The body is called inside `tenant_scope`, so `current_tenant()` inside it is the tenant
being driven. A task spawned inside that scope inherits it; one spawned outside is
refused with `MissingTenantContextError` rather than silently picking up a neighbour's.

## Related

- [Tenancy](tenancy.md) — the context itself and how it propagates.
- [Store isolation](store-isolation.md) — the adapter-side partitioning this suite tests.
