# Per-tenant configuration

What a tenant is permitted is data, resolved once, not a branch in agent code.

A plan tier written as an `if` inside an agent is in one code path and not the next. The
one it misses is a background task or a second entry point, and the symptom is a run on
the cheap plan reaching the reasoning model with nobody able to say afterwards what that
tenant was actually allowed. This page is about removing the conditional.

## The shape

```python
from tesserix_adk.core import (
    FileTenantConfig, CachingTenantConfig, resolve_tenant_policy, tenant_policy, tenant_scope,
)

source = CachingTenantConfig(FileTenantConfig("/etc/adk/tenants"))

with tenant_scope("acme"):
    policy = await resolve_tenant_policy(source)   # one read, at the boundary
    with tenant_policy(policy):
        run = await runner.run(agent, "plan a trip", tenant="acme")
```

`resolve_tenant_policy` reads the tenant from the bound context — resolving limits for
nobody in particular is resolving nobody's limits, so outside a `tenant_scope` it raises
`MissingTenantContextError`. `tenant_policy` binds the answer for the block;
`current_policy()` is what code below reads and `policy_here()` returns `None` for code
that legitimately runs with no entitlement in force.

## What a tenant states

```toml
# /etc/adk/tenants/acme.toml
[limits]
models = ["gpt-4o-mini"]
tools = ["search"]
region = "eu-west-1"
memory_retention = "P30D"

[limits.task_class_models]
cheap = "gpt-4o-mini"

[limits.budget]
max_cost = 1.00
currency = "USD"
max_model_calls = 5

[secrets]
provider_key = { name = "ACME_PROVIDER_KEY" }
```

An absent field states nothing rather than permitting everything: an absent allowlist
leaves the decision to whatever else is in force, an empty one is a stated refusal of the
lot. `task_class_models` is checked against `models` where both are given, so two settings
that contradict each other fail where they are written rather than on the run that hits
them.

Secrets are named, never carried. `SecretRef` holds the name and `resolve(secrets)` asks
the `SecretProvider` in force; a literal where a reference belongs does not validate,
because a tenant file is a file somebody commits and a row somebody dumps.

## Enforcement

| Call | Refuses with |
|---|---|
| `policy.check_model(model)` | `TenantLimitError(limit="models")` |
| `policy.check_tool(name)` | `ToolNotPermittedError` |
| `policy.check_region(region)` | `TenantLimitError(limit="region")` |
| `policy.model_for(task_class)` | nothing — returns `None` where the tenant maps no model |

A ceiling catches spend after the call; the allowlist is what stops the call.

## Budget is the exception to "the tenant layer is highest"

Everything else the tenant layer states outright. Money does not work that way: a tenant
ceiling arrives as one more `ScopedLimits` under `BudgetScope.TENANT` and
`most_restrictive` decides, so a tenant configuration can narrow a run's ceiling and never
widen it. A runner given no explicit `BudgetPolicy` folds in the bound policy's ceiling
itself, so a plan tier holds without the agent knowing which tenant it is running for:

```python
with tenant_scope("acme"), tenant_policy(policy):
    run = await runner.run(agent, "plan a trip", tenant="acme")
    run.state    # RunState.BUDGET_EXHAUSTED where the tenant's ceiling was reached
```

Where a `TenantLedger` is wired the tenant's dimensions are checked against the shared
total across runs — see [`ledger.md`](ledger.md). Where none is, they hold each run
individually: a deployment that has not wired a ledger gets a real ceiling rather than
none.

## Failing closed

Every path that could end in a permissive default ends in a refusal instead.

- An unknown tenant is `UnknownTenantError`, never the global defaults: a tenant nobody
  has entitled is a tenant nobody has priced.
- An unreadable store is `TenantUnconfiguredError` — a distinct type, because an unknown
  tenant is a request to reject and an unreachable store is an outage to page on.
- A cache entry past its window is not served even when the store behind it is down. A
  ceiling nobody can confirm may have been lowered since.
- A missing bound policy raises rather than reading as absence.

## The cache

`CachingTenantConfig` sits in front of any provider so a limit lookup is not a round trip
per run. It is bounded by `keep` (least recently used first) rather than by the customer
list, its entries expire after `ttl`, and `invalidate(tenant)` drops exactly one tenant —
flushing the cache to apply one tenant's change is an outage for everybody else's.
Refusals are not cached: caching "unknown" makes a newly onboarded tenant unknown for the
rest of the window.

## Known limitations

- **A limit lowered mid-run applies to the next run.** The policy is resolved once and
  bound, so completed work is not retroactively over a ceiling that arrived after it. Code
  that wants the new ceiling sooner resolves again at the next iteration boundary.

- **An explicit `BudgetPolicy` on the runner owns the ceiling.** A runner constructed with
  `budget=` uses it as given and the tenant's ceiling is not folded in — the deployment
  said what the ceiling is. Compose it yourself with `most_restrictive` where both should
  apply.

- **`FileTenantConfig` is the simplest store there is.** One TOML file per tenant, read on
  resolution, no watch. A deployment with a real entitlement database implements
  `TenantConfigProvider` — one tenant per call, because a hundred thousand tenants do not
  fit in memory and should not have to.

- **Strictness gives way at the file edge.** TOML has no frozenset and no decimal, so
  `FileTenantConfig` validates leniently. A value that is not the declared type is still
  refused, with the file named.
