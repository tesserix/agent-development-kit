# Tool allowlists

Three layers have an opinion about which tools a run may call, and enforcing them
separately means the effective permission at the moment of a call is written down nowhere:

| Layer | States | Where it comes from |
|-------|--------|---------------------|
| Agent | What it was built to call | `Agent.tools` |
| Tenant | What the plan entitles them to | `TenantLimits.tools` |
| Caller | What their scopes cover | the request that started the run |

`ToolAllowlist.resolve` intersects them once, at the boundary:

```python
allowlist = ToolAllowlist.resolve(agent.tools, tenant=policy.limits.tools, caller=scopes)
allowlist.check(call.name)   # raises ToolNotPermittedError, before anything is dispatched
```

The narrowest source wins and nothing widens it at runtime. A source that states nothing
passes `None` and narrows nothing; a source that states an empty set refuses everything.
Those are different, which is why they are not both the empty set.

## Where it is enforced

In the dispatch path, after argument validation and before execution. A refusal that lands
after the call is not a refusal — the side effect has already happened by the time the
decision is recorded. `AgentRunner` resolves the allowlist per dispatch from the agent's
declaration and the tenant policy in force, and a call outside it fails the run.

It applies identically to native tools, MCP tools and peer agents. A tool is a name and a
schema; where it executes does not change who may ask for it.

## The refusal is not negotiable

`ToolNotPermittedError` names the tool as it was asked for and which layer refused:

```python
refused.tool          # 'refund'
refused.details       # {'tool': 'refund', 'agent': 'concierge', 'reason': 'tenant'}
```

The model is told the call was refused and is not offered a way to appeal it. A refused
call still spent a turn, so it counts against the run's iteration and budget caps —
`ToolAllowlistGuard` keeps the tally on `.attempts` and `.refusals`. A refusal that is free
is one a looping model will attempt indefinitely.

## Names are compared in one normal form

`canonical` applies NFKC, strips and case-folds, so `Search`, a full-width spelling and
`search` are one permission rather than three. Two declared names that collapse to the same
form are a `ConfigurationError` at resolution: which of them a call reached would otherwise
depend on iteration order.

A namespaced MCP tool is a different name from the native tool with the same stem —
`mcp:search` and `search` do not collide, and neither grants the other.

## Delegation only ever narrows

```python
peer = allowlist.narrowed(peer_agent.tools, agent=peer_agent.name)
```

A peer agent whose own declaration is broader is held to the intersection, so handing work
to it is not a way around the layer that refused. Chains compose the same way: each hop can
only shrink.

## Known limitations

- Caller scopes are tool names here. A deployment that models scopes as capabilities maps
  them to names at the boundary.
- The allowlist is resolved per dispatch, so a tenant setting changed mid-run applies to the
  next tool call rather than being pinned for the whole run.

## Related

- [`docs/tools.md`](tools.md) — declaring and registering tools
- [`docs/tenancy.md`](tenancy.md) — where tenant limits come from
- [`examples/tool_allowlist.py`](../examples/tool_allowlist.py)
