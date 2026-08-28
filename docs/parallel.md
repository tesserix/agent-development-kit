# Parallel fan-out

`fan_out` runs several branches at once through a `Supervisor` and adds them up under a
rule that was declared rather than implied. It exists because the obvious version —
`asyncio.gather` over a list of sub-agents — loses three properties at once.

- **Nothing caps concurrency.** The fan-out's real width becomes whatever the provider's
  rate limiter allows, which is a number nobody in the codebase chose and which changes
  when the account tier does.
- **Nothing orders the spend.** Four branches that each respect the run ceiling breach it
  collectively, because each checks before the others have recorded.
- **The result is a list.** An aggregate built from three branches out of five is
  indistinguishable from one built from all five. That is the failure that matters: a
  partial answer presented as a whole one.

Fan-out adds no authority. Every branch goes through `Supervisor.delegate`, so scope
intersection, the shared ledger, the memory-key claim and the guardrail chain on the way
back are all the supervisor's rules, unchanged. See [`delegation.md`](delegation.md).

## The shape

```python
from tesserix_adk.runtime import Branch, Quorum, fan_out

done = await fan_out(
    supervisor,
    (
        Branch(name="fares", task="What does the LHR–JFK leg cost?", needs={"research"}),
        Branch(name="hotels", task="What is near the venue?", needs={"research"}),
        Branch(name="visas", task="What does a UK passport need?", needs={"research"}),
    ),
    into=Quorum(2),
    max_concurrency=2,
)

done.value          # what the strategy formed
done.contributed    # ("fares", "hotels") — in declared order
done.excluded       # {"visas": "failed: 'researcher' did not answer: the run ended failed"}
done.spent          # per-branch Usage
done.peak_in_flight # what the cap actually achieved, not what it permitted
done.usage          # the whole fan-out, excluded branches included
```

## Aggregation strategies

An `Aggregation` answers two questions separately: which branches are in, and what to
build from them. Keeping them apart is what lets a refusal carry the same provenance a
success does.

| Strategy | In the aggregate | Refuses when |
|---|---|---|
| `All()` (default) | Every branch | Any branch did not answer (`failed`) |
| `FirstSuccess()` | The first that answered, **in declared order** | Nothing answered (`none`) |
| `Quorum(n)` | Every branch that answered | Fewer than `n` did (`quorum`) |
| `Reduce(fn)` | Every branch that answered, passed to `fn` | Nothing answered (`none`) |

`All` is the default because a missing branch is usually a bug, and a default that hides
it is a default that ships it.

`FirstSuccess` takes the first in *declared* order, not the first to finish. Finishing
order is the obvious rule and the wrong one: it makes the answer depend on scheduling, so
the same fan-out gives different answers on different days for reasons nobody can
reproduce.

`Reduce` is handed the contributing `BranchResult`s rather than their strings, so a
reducer can attribute what it used to the branch it came from.

## Refusals

An aggregate that cannot be formed is an `AggregationError`, never a smaller aggregate.
It carries `strategy`, `reason` (`failed` / `quorum` / `none` / `cancelled`),
`contributed` and `excluded` — the same provenance a successful `Aggregate` carries, so
the two are comparable in a log.

## Determinism

`results`, `contributed` and everything derived from them are in declared order regardless
of completion order. Two runs of the same fan-out over the same answers aggregate
identically.

## Branch outcomes

| Outcome | Means |
|---|---|
| `ok` | Answered, and the answer passed the supervisor's guardrails |
| `failed` | The run failed, nobody on the roster could do it, the specialist held none of its caller's tools, another branch already claimed `writes`, or the delegation ceiling was reached |
| `budget_exhausted` | The branch ran out of its slice, or the shared ledger ran out |
| `cancelled` | The fan-out was stopped |

Every branch appears in `results` whichever it is, and `spent` attributes what each
consumed — including the ones excluded from the answer. Work that was paid for and then
left out is exactly the spend that goes missing from a cost report.

## The shared ledger

Branches spend against one ledger, not one each. A branch that exhausts a **slice of its
own** (`Branch(budget=...)`) is that branch's problem and the rest carry on. A branch with
no slice of its own that exhausts is a signal the *shared* ledger has gone, so the
branches that had not started are refused as `budget_exhausted` without running — there is
nothing left for them to spend, and starting them only produces more refusals to read.

Branches already in flight when that happens are not killed. Cancelling paid-for work
halfway is not obviously better than letting it finish, and the run's own budget policy
already refuses the next call.

## Cancellation

A cancelled fan-out refuses. It does not aggregate whatever happened to have arrived —
that is precisely the "three branches out of five, reading like five" failure, arrived at
by a different route. Branches that had answered before the stop landed are still listed
in `excluded`, with their answers unused, so the provenance does not read as though the
refusal was about the other branches.

Branches still queued behind the concurrency cap when the stop lands never start.

## Known limitations

- The concurrency cap is per fan-out, not per provider or per tenant. Two fan-outs running
  at once are two caps. Runner-wide lane limits live in
  [`tesserix_adk.runtime.fanout`](https://github.com/tesserix/agent-development-kit/blob/main/src/tesserix_adk/runtime/fanout.py) and bound tool
  calls, not branches.
- Nested fan-out is bounded only by `DelegationLimits` (`max_depth`, `max_fan_out`,
  `max_delegations`), which bound breadth × depth for the whole run. There is no separate
  ceiling on how many fan-outs a run may perform.
- `spent` is attributed per branch, but a branch that was refused before it ran is
  attributed zero rather than the cost of the refusal itself, which is not zero in
  wall-clock terms.
