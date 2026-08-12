# Delegation — how far a run may go, and what a child may hold

Multi-agent runs fail in two loud ways and one quiet one. They recurse until the budget is
gone; two agents hand the same task back and forth forever; or — the quiet one — a
sub-agent three levels down ends up with the allowlist from its own configuration rather
than the narrowed scope of the caller it acts for. The third is a privilege escalation
dressed as a default.

So the shape of a run is bounded and its scope only ever narrows:

```python
scope = DelegationScope(tools=frozenset({"search", "summarise", "file_bug"}))
root = Delegation.root(run_id="run_1", tenant="acme", agent="supervisor", scope=scope)

researcher = root.to("researcher", tools={"search"})
researcher.path          # ('supervisor', 'researcher')
researcher.scope.tools   # frozenset({'search'})
```

A delegation comes from `Delegation.root` or from `parent.to(...)` and from nowhere else.
The constructor raises `ConfigurationError`, because one built by hand would carry a scope
nobody narrowed, a depth nobody counted and a ledger nobody shares.

## The ceilings

`DelegationLimits` bounds three different things, and a run needs all three:

| | Default | Bounds |
|---|---|---|
| `max_depth` | 3 | How many agents may sit between the root and the deepest worker. |
| `max_fan_out` | 8 | How many children one agent may hand work to. |
| `max_delegations` | 24 | How many delegations the whole run may make. |

Depth and fan-out each bound one lineage. `max_delegations` bounds their product, which is
where a shallow, very wide tree would otherwise escape both.

Limits narrow downward only. A child may declare tighter ceilings for itself and they
apply; a child that declares roomier ones keeps its parent's, since a limit that can be
raised from inside is not a limit.

An agent already on the current path may not be delegated to again — `reason="cycle"` —
even when the depth ceiling would still permit it. Two agents alternating below the
ceiling is a loop that terminates only by exhausting the budget. The same agent on two
separate branches is not a cycle and is allowed.

## What a refusal is

Every refusal happens before the child is created, so it costs nothing and spends nothing
from the run's allowance.

```python
try:
    parent.to("researcher", tools={"wire_transfer"})
except ScopeEscalationError as refused:
    refused.requested   # ('wire_transfer',)
    refused.path        # ('supervisor', 'researcher')
```

`DelegationLimitError` carries a `reason` — `depth`, `fan_out`, `run`, `cycle` or
`expired` — and the `path` it happened on. `ScopeEscalationError` carries what was asked
for that the parent did not hold.

Neither is retryable: the same call refused for the same reason refuses again, so a parent
that retries it is looping rather than recovering. Both reach the parent as something it
can reason about, because an agent that cannot delegate can often still answer.

## What a child holds

`to(agent)` with no `tools` passes the parent's allowlist through unchanged. `to(agent,
tools={...})` intersects. Asking for anything the parent does not hold is
`ScopeEscalationError` rather than a grant — refused whether or not the child's own
configuration would permit it. `mutations` works the same way, for deployments that
separate reading from writing.

A child asking for no tool at all is refused too: a delegation that could call nothing was
a mistake at the call site, not a maximally safe setting.

The tenant is not a parameter of `to()` at all. A child inherits its parent's
`TenantContext`, so crossing a tenant boundary by delegation is unrepresentable rather
than merely checked.

## Expiry

A scope may declare `expires_at`, read against the `Clock` passed to `root`. Delegating on
an expired scope raises `DelegationLimitError(reason="expired")`. Time that cannot be read
fails closed, and `root` refuses a scope that declares an expiry with no clock to read it
against — an expiry nothing evaluates is a comment.

## Known limitation

Budgets are not narrowed here. A child inherits the run budget its parent resolved, so
depth and delegation count are what bound spend on this path today. Per-delegation budget
narrowing belongs with the spend ledger (`docs/ledger.md`) and is not implemented.
