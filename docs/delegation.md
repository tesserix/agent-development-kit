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

## What a delegated run inherits

`Delegation` is the model of a call graph. What follows is what the run loop enforces when
one run actually calls another, with `runner.run(child, parent=parent_run.context)`.

A kit with two dispatch paths grows a control that covers one of them. Guardrails covering
tool calls but not delegation leave the cheapest bypass in the system open: hand the work
to a sub-agent that declared no guard. So a run states what it was allowed to do, and a run
below it inherits that rather than its own configuration.

```python
parent = await runner.run(supervisor, "plan the work", tenant="acme")
parent.grant.tools                     # ('search',)
parent.grant.guardrails                # ('no_pii', 'no_prompt_leak')

child = await runner.run(researcher, "sub-task", tenant="acme", parent=parent.context)
child.grant.guardrails                 # ('no_pii', 'no_prompt_leak') — it declared none
```

- **Guards are inherited, in the caller's order,** followed by any of the child's own that
  were not already there. A child cannot drop one, and a guard named at both levels is
  asked once. A guard the child's runner was never given is a `ConfigurationError` at the
  boundary rather than a skipped check.
- **Reach only narrows.** A tool the caller did not hold is `ScopeEscalationError`,
  recorded as `SCOPE_REFUSED` and terminal, before a model is called. It is refused rather
  than intersected away, because the difference is a wiring mistake and a silent
  intersection is how nobody finds out about it. This holds at every depth: a grandchild
  cannot recover what the level above it gave up.
- **Approval is inherited.** A tool a human had to clear at the top is not cleared by being
  called one level down.
- **Budget is shared, not reset.** A parent passes `bounds.budget.child()`, so a delegation
  spends the caller's remaining allowance.

A `RunContext` built by hand outside the loop carries no grant and narrows nothing: the
absence of a record is not a claim that the caller held nothing. Every context the loop
produces carries one.

## What comes back

```python
messages.append(Message(role="user", content=[TextPart(text=handed_back(child))]))
```

A sub-agent's answer is model output that read whatever the sub-agent read. Pasted into the
caller's conversation bare, it is an instruction channel for whatever wrote it, so it
crosses in the same `<untrusted-data>` envelope a tool result crosses in.

A child a guard stopped hands back the guard and its code rather than an empty string, so a
refusal inside a delegation reaches the caller as a refusal it can reason about rather than
as an unexplained silence.

## Handing work to a roster

`Delegation` says what a child may hold; `Supervisor` is the thing that actually hands the
work over, so the narrowing above is not something each product rebuilds around its own
`runner.run` call.

```python
roster = Roster((
    Specialist(agent=researcher, capabilities=frozenset({"flights", "research"})),
    Specialist(agent=accountant, capabilities=frozenset({"refund"}),
               budget=BudgetLimits(max_input_tokens=2_000)),
))

supervisor = Supervisor(
    runner, roster,
    agent=planner,
    delegation=Delegation.root(run_id="run_1", tenant="acme", agent="planner", scope=scope),
    budget=run_budget,
    guardrails=guardrails,
)

result = await supervisor.delegate("find two refundable flights", needs={"flights"})
result.data                 # the answer, inside <untrusted-data>
result.answered             # False if the worker was stopped
supervisor.spent["researcher"]   # what that worker cost, under its own name
```

**Routing is by declared capability.** A `Specialist` declares what it can do, and
`delegate(needs=...)` picks the narrowest worker that covers all of it — narrowest, so a
generalist does not absorb work a specialist declared. A roster with nobody in it, or with
nobody matching, is a `DelegationError(reason="no_worker")` rather than the supervisor
quietly doing the work itself with its own wider access.

**A worker holds the intersection** of its own tools and what the supervisor holds under
its scope. A worker sharing no tool with its caller never starts —
`DelegationError(reason="no_tools")` — because a run that could call nothing would burn
tokens to say so.

**The allowance is a slice of the caller's ledger.** `budget.sliced(limits)` is a tighter
ceiling that is still deducted from the parent, so a worker cannot spend what the run does
not have, and the run cannot spend more because it delegated. A worker that hits its slice
ends in `BUDGET_EXHAUSTED` and comes back as a refusal the supervisor can read; it does not
end the supervisor's run unless the call declared `fatal=True`. Spend is attributed by
worker name whether the work finished or not, so a cancelled worker's partial cost is still
on the ledger.

**Cancellation flows down.** `supervisor.cancel(reason)` cancels every worker in flight,
including the provider call one is waiting on.

**Two workers, one key.** `delegate(..., writes="itinerary")` claims a key for the run.
A second worker claiming the same key is refused with `reason="conflict"` rather than
overwriting the first, since concurrent workers writing one key silently is the failure
nobody sees until the answer is wrong.

Every hand-over lands on `supervisor.events` as `DELEGATED` or `DELEGATION_REFUSED`, with
the worker's name, usage and reason — the record of what was handed to whom, for a run
whose events span more than one agent.

## Known limitations

`Supervisor` slices budgets for the workers it dispatches; `Delegation` itself still does
not. A run that calls `runner.run(child, parent=...)` directly inherits the parent's
resolved budget, so depth and delegation count are what bound spend on that path.

A write claim is held in the supervisor for the life of the supervisor, not in a store, so
it coordinates workers under one supervisor and not two supervisors on one memory.

`DelegationScope.mutations` is not part of `RunGrant`: an agent declares tools, not
mutation classes, so there is nothing at the run boundary to narrow. Deployments that
separate reading from writing express it through the tool allowlist.
