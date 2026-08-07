# Budgets

Spend control invented per product is spend control that disagrees with itself: one team
counts iterations, one counts nothing, and the agent that looped overnight is discovered on
the invoice. The kit states a ceiling once, in a vocabulary every scope can use, and
enforces it before the spend rather than reporting it afterwards.

Two rules make it a ceiling rather than a suggestion. **A limit nobody set is not no
limit** — an unstated dimension resolves to a conservative default, because forgetting is
not a way to opt out. And **the tightest applicable scope wins, with its source recorded** —
a ceiling nobody can attribute is a ceiling nobody can raise.

## `BudgetLimits` — the vocabulary

```python
BudgetLimits(
    max_cost=Decimal("1.00"),   # money, never a float
    currency="USD",
    max_input_tokens=100_000,   # prompt tokens, cache reads included
    max_output_tokens=20_000,
    max_model_calls=20,         # failed and retried attempts included
    max_tool_calls=40,
    max_iterations=10,
    max_seconds=300.0,
)
```

Every field is optional to write and none is optional in effect: `filled()` replaces what
was left unsaid with `BudgetLimits.conservative()`, shown above. A ceiling of zero is
refused — a run that may do nothing is a run nobody meant to configure, and it is almost
always a field somebody meant to disable rather than a limit somebody meant to set.

Saying "no ceiling" takes `BudgetLimits.unbounded()`, which is a sentence a reviewer can
see in configuration rather than a field somebody forgot. Marking limits unlimited *and*
stating a ceiling is refused: one of the two is a mistake and the kit will not guess which.

## Scopes

Limits attach to a run, an agent, a tenant, or a tenant within a rolling window.
`most_restrictive` resolves them:

```python
resolved = most_restrictive(
    ScopedLimits(scope=BudgetScope.TENANT, limits=BudgetLimits(max_cost=Decimal("1.00"))),
    ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_cost=Decimal("5.00"))),
)
resolved.limits.max_cost      # Decimal("1.00")
resolved.sources["max_cost"]  # BudgetScope.TENANT
```

Nearness does not decide this. A run asking for 5.00 under a tenant capped at 1.00 gets
1.00, or the boundary that matters could be widened from inside it. Resolution is per
dimension, so a tenant may cap money while a run caps calls, and each dimension records the
scope that won it.

Two mistakes are found here rather than mid-run:

- **Two scopes of the same kind.** One boundary cannot have two ceilings, and picking one
  would be a guess.
- **Two currencies capping money.** The kit will not convert; a rate it invented is a rate
  nobody agreed to.

## `RunBudget` — the default policy

`RunBudget` holds one run to its resolved ceiling. `reserve` holds an estimate *before* the
call, so a call cannot start that the ceiling could not cover; `record` replaces the hold
with what was actually consumed. `check()` answers whether there is room without raising.

```python
budget = RunBudget(resolved=resolved, clock=clock)
await budget.reserve(1_200)                 # BudgetExceededError if it would not fit
await budget.record(usage, model_calls=1)   # releases the hold, keeps the real number
```

`BudgetExceededError` names the limit breached, the scope it came from, the ceiling, what
was consumed and what is left — enough to act on without reading the code that raised it.

`BudgetDecision.priced` reports whether every call so far had a price. A money ceiling
checked against calls nobody could price is not enforcement, and this is how a caller finds
out rather than being reassured by a total of zero.

### A child spends what the parent has left

```python
child = await runner.run(sub_agent, "…", tenant="acme", budget=parent_budget.child())
```

A sub-agent handed a fresh allowance is a way to spend one ceiling twice. `child()` shares
the parent's ledger: what the child spends the parent has spent, and the child cannot widen
what the parent was given.

## Tenant ceilings and the ledger

A ceiling wider than one run has to live outside the process, or two concurrent runs each
see the whole allowance as free. `TenantLedger` is that seam:

```python
budget = RunBudget(
    resolved=resolved,          # a dimension whose source is TENANT is checked against the ledger
    clock=clock,
    ledger=ledger,
    tenant="acme",
    window_seconds=3600.0,
)
```

Which dimensions are shared is not configured twice: a dimension the resolution attributed
to `TENANT` or `TENANT_WINDOW` is checked against the ledger, and the rest stay local.
`consume` adds and returns the new total in one call, because a ledger that is read, then
decided against, then written has already handed the same allowance to somebody else.

The window is pinned when the run starts and does not move under it. A run beginning at
10:59 must not become a way to spend two hours of one hour's allowance.

**A ledger that cannot be reached fails closed.** `BudgetUnavailableError` is distinct from
exceeding a budget: nobody knows whether this run would exceed one, and carrying on is how
one outage becomes an unbounded bill. Proceeding anyway is possible and is a choice somebody
makes in configuration — `on_ledger_failure=LedgerFailure.PROCEED` — recorded on the run, so
the runs that took it can be found afterwards.

## There is no runtime without a ceiling

A runner constructed without a policy does not run unbounded. Each run gets a `RunBudget`
resolved from the agent's own limits and the conservative defaults, and the resolution is
recorded on the run:

```python
run.budget.limits.max_model_calls   # 20
run.budget.sources                  # which scope stated each dimension
```

Removing the ceiling takes `UnlimitedBudget(reason=...)`, which refuses to be built without
a stated reason and puts that reason on `run.budget.unlimited_reason`. A ceiling removed for
a reason nobody wrote down is one nobody can review.

Breaching a ceiling ends the run in `BUDGET_EXHAUSTED`; losing the ledger ends it in
`FAILED`. Neither is a warning.

## Where the loop enforces it

A ceiling checked at the boundaries does not stop the run that discovered its spend on the
fortieth iteration, so the loop checks it where the money goes:

| Point | What happens |
|---|---|
| Top of each iteration | Every dimension re-checked, and the iteration charged before it runs |
| Before a model call | Input tokens estimated and reserved; a call that would not fit is never dispatched |
| After a model call | The reservation settled against what actually came back |
| After a failed attempt | The kit's own estimate of what the vendor read is charged |
| Before a tool call | The tool charged, so a ceiling refuses the dispatch rather than discovering it |

Money is the one dimension that can only stop the run *after* the call that broke it: the
price is not known until the response carries it. Pre-flight cost estimation and caller
refusal are a separate story. Token, iteration, tool and wall-clock ceilings all refuse
before dispatch.

Nothing is squeezed under a ceiling. The prompt is not truncated, tools are not dropped and
the model is not downgraded to make a call fit — a degraded answer presented as a real one is
worse than no answer and cheaper only in money.

### Retries, cancellation and side effects

A failed attempt is spend. Whatever the vendor read before it failed is charged against the
ceiling, so retries and fallback cannot be used to spend past a limit.

Cancellation wins over a breach reached in the same breath: the aborted call settles what it
had already sent, and the run still ends `CANCELLED`. One deterministic terminal state, and a
ledger that still reconciles.

A tool that already ran is never re-dispatched while the run unwinds — that is how one side
effect becomes two. Instead each non-idempotent tool that ran on a run that did not complete
gets a `COMPENSATION_REQUIRED` event naming it, for whoever has to undo it. Tools listed in
`Agent.idempotent_tools` are left alone.

When a call costs more than the estimate that reserved for it, the run lands marginally over
and `BudgetDecision.overshoot` says by how much. The terminating event carries it, because
rounding it away is how a ceiling comes to mean something other than what it says.

### Streams

`budgeted_stream` holds a stream to the same ceiling. Each running total the vendor reports is
charged as an increment, so a stream repeating a total is billed once, and passing the ceiling
mid-stream raises `BudgetExceededError` rather than letting the stream end quietly:

```python
async for event in budgeted_stream(provider.stream(request), run_budget):
    ...
```

A consumer that sees a stream simply stop reads it as a finished answer. The source stream is
closed on the way out, so an aborted stream does not leave the vendor sending tokens charged
to nobody.

## Testing

`FakeBudgetPolicy` is a counting budget with a hard token limit and no ledger, and
`FakeTenantLedger` is a tenant ledger in one process, with a `reachable=False` switch for
the fail-closed path. Both are network-free. `BudgetPolicyConformance` is the suite any
other implementation of the protocol has to pass.

## What this does not do

Pricing is not here — see [`docs/cost.md`](cost.md) for how a `Usage` becomes a `Cost`. A
cross-process ledger implementation is not here either; the protocol is the contract and a
deployment supplies the store.

Exercised by [`examples/budget.py`](../examples/budget.py) and `tests/test_budget.py`;
enforcement by [`examples/budget_enforcement.py`](../examples/budget_enforcement.py) and
`tests/test_budget_enforcement.py`.
