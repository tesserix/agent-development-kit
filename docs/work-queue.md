# Work queues — work whose owner can die

A run dispatched to a background worker has no owner once the pod is rolled. Nothing times
it out, nothing retries it, and nothing can say afterwards whether it finished — it simply
sits in whatever state it was in when the process went away, until somebody notices.

So work here is claimed under a lease rather than taken. A living worker renews the lease;
a dead one's lapses, and the reaper puts the item back with its attempt counted.

```python
queue = MemoryWorkQueue(QueuePolicy(lease_seconds=30.0, max_attempts=5), clock)

await queue.enqueue(WorkItem(id="run_1", tenant="acme", payload={"agent": "planner"}))

item = await queue.claim(worker="worker-7")
if item is not None:
    await queue.complete(item.id, tenant=item.tenant, worker="worker-7")
```

## Delivery is at-least-once

A worker that is merely slow — a long garbage-collection pause, a stalled socket — has its
lease lapse while it is still working. The item is redelivered, and for a moment two workers
hold the same work. This is not a bug to be designed out; it is what a queue that survives
a dead worker costs. Handlers are idempotent or they are wrong, and
[`docs/tool-idempotency.md`](tool-idempotency.md) is where that machinery lives.

Lease expiry is evaluated by the store, never by a worker. Two workers' clocks disagree, and
a queue that trusted theirs would free work that is being done and keep work that is not.

## Attempts, backoff and the dead letter

| What happened | What the queue does |
|---|---|
| `complete` | Done. The item is not claimable again. |
| `fail(retryable=True)` | Attempt counted, requeued after a capped exponential backoff. |
| `fail(retryable=False)` | Straight to the dead letter. Waiting will not make it succeed. |
| Lease lapsed | Attempt counted, requeued — the worker is presumed dead. |
| Attempts exhausted | Dead letter, with the failure from every attempt attached. |

The dead letter is the point. A handler that crashes on the same item forever is a loop that
spends model budget on failing, and an item that is quietly dropped instead is work somebody
is still waiting for. Neither happens: the item stops, and it is somewhere an operator can
look at it, carrying the failures that explain it.

Backoff is deterministic rather than jittered. A queue redelivers one item to one worker, so
there is no thundering herd to spread, and a test that has to allow for jitter stops
asserting the interval at all.

## One worker at a time, for as long as is reasonable

`heartbeat` says the worker is alive and extends the claim, but only up to
`max_lease_seconds` in total. A worker that renews forever holds work nothing else can pick
up, and a stuck run is indistinguishable from a busy one to everything except that bound.
Past it, the renewal is refused with `LeaseLostError(reason="capped")` and the item comes
back for somebody else.

A worker acting on a claim it no longer holds — lapsed, taken, or capped — is refused rather
than allowed to write its result over the one that counts. `LeaseLostError` is deliberately
**not retryable**: the item belongs to another worker now.

## Boot reconciliation

A worker that restarts under its own name cannot know what it was doing. `adopt` gives back
everything it held, immediately rather than after waiting out a lease nobody is renewing, so
a rolled deployment orphans nothing. The attempt still counts — the work was tried, and the
cap exists to bound tries.

```python
for item in await queue.adopt(worker="worker-7"):
    log.info("gave back %s after restart", item.id)
```

## Fairness before priority

Tenants are served in rotation, and priority orders a tenant's own work and nothing else. A
priority that crossed tenants would be a priority every tenant sets to `URGENT`, and they
would be right to: one tenant's backlog must not be able to starve everybody else's queue.

## The same job twice

`dedupe_key` collapses a second enqueue of the same logical job into the first while that
job is still live. Once it completes or is dead-lettered the key is free again — nightly
means nightly, not once. Re-enqueueing an id that is still in the queue returns what is
stored rather than replacing it: an item whose attempts were reset by a well-meaning retry
is an item the cap can never catch.

## What a deployment watches

`stats()` carries depth, the age of the oldest waiting item, how many items are claimed, the
dead-letter count, and cumulative reaper and duplicate-suppression counters. Depth alone says
nothing — ten items ten seconds old is healthy, one item an hour old is an incident — which
is why age is beside it.

An enqueue that could not reach the store raises `QueueUnavailableError` rather than
returning. Work that was silently dropped is work nobody is waiting for and nothing will
reap, because nothing ever recorded that it existed.

## Implementing a queue

`WorkQueueConformance` carries the guarantees: exclusive claims, lapsed claims that come back
with their attempt counted, a dead letter nothing falls out of, and no tenant able to starve
or read another. `MemoryWorkQueue` exists so those can be exercised without Redis — nothing
in it outlives the process, which is what surviving a rolled pod is not.

## See also

- [`docs/state-adapters.md`](state-adapters.md) — the Redis queue that does outlive the process, and what its server has to be configured as.
- [`docs/checkpointing.md`](checkpointing.md) — how a redelivered run carries on from its frontier rather than from zero.
- [`docs/tool-idempotency.md`](tool-idempotency.md) — what makes a handler safe to deliver twice.
