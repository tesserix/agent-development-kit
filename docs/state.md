# State — what survives a process, and what keeps two workers honest

A run that exists only in one process ends when that process does. So state goes to a
store — and then the interesting failure appears: two workers holding the same run both
write it back, the second wins, and the first worker's iteration, spend and cursor are
gone with nothing recording that they ever happened. Nobody notices, because a lost update
looks exactly like work that was never done.

`StateStore` is the one shape that state has here. The tenant is part of the key rather
than an argument beside it, every write states the version it read, and amounts that
accumulate go through a patch that adds rather than sets.

```python
store = MemoryStateStore()

run = await store.put_run(RunRecord(run_id="run_1", tenant="acme", agent_name="planner"))
run.version                       # 1

await store.patch_run(run.key, StateDelta(iterations=1, cost_micros=1_200))
```

## Versions

A record read at version *n* writes at *n*, and the store commits it at *n + 1* or raises
`StateConflictError` carrying both numbers. The loser re-reads and decides again rather
than retrying against the same stale copy.

Version zero is a create. Two workers racing to start the same run therefore resolve to
one winner and one conflict, instead of one of them quietly overwriting a run the other had
already begun.

```python
try:
    await store.put_run(stale)
except StateConflictError as refused:
    refused.expected_version   # 1
    refused.actual_version     # 2
```

## Patches add

`patch_run` takes no version, because it does not need one: `StateDelta` carries amounts to
add, and additions commute. Ten workers each adding what they spent produce the sum;
ten workers each writing a total produce whichever one arrived last.

| | |
|---|---|
| `usage` | Tokens to add. |
| `cost_micros` | Cost to add, in millionths — an integer, because money accumulating in a float stops adding up. |
| `iterations` | Loop passes to add. |
| `messages_read` | How far to advance the message cursor. |

None of them may be negative. A negative amount is how one worker unspends what another
worker spent.

## What a run record holds

Enough to resume, and no more. The message cursor, so a resumed run neither repeats nor
skips a turn. The tool calls that were asked for and never came back — a run abandoned mid
tool call is a state, not a corruption, and `record.mid_tool_call` says so. What it has
spent, how many times round the loop it has been, and the approval it is held on.

Tool-call arguments are scrubbed on the way in. Persisted state is a queryable store, so a
token that reached an argument is a token an operator can later grep for; every
implementation writes `record.scrubbed()` rather than what it was handed, and the
conformance suite fails one that does not.

## Listing

```python
page = await store.list_runs(StateQuery(tenant="acme", state=RunState.RUNNING, limit=100))
while page.cursor is not None:
    page = await store.list_runs(StateQuery(tenant="acme", cursor=page.cursor))
```

Listings are ordered by the store's own insertion counter, never by a timestamp: two
workers' clocks disagree, and a listing that pages by clock skips records written during
the disagreement. `updated_before` still exists, because finding abandoned work means
asking for everything last touched before some moment — but it filters, it does not order.

A page that ends the walk carries no cursor. An empty page carrying one is rejected at
construction, since a reaper handed that asks for the next page forever.

## Sessions

A session is the conversation across runs. Deleting one that still has unfinished runs
raises `StateInUseError` naming them: a live run whose session has gone cannot be resumed
and cannot be reached from any listing that starts at the session, so it becomes work
nothing will ever reap. Callers that mean it pass `cascade=True`.

## Failures

| | |
|---|---|
| `StateConflictError` | The version moved. Carries both numbers. Not retryable as-is — re-read first. |
| `StateNotFoundError` | A patch named a run that is not there. Creating one would invent a run that never started. |
| `StatePersistenceError` | The store refused or could not be reached. `reason="unavailable"` is retryable; `too_large` is not, because the record stays too large. |
| `StateInUseError` | Deleting a session would orphan live runs. |

Nothing is partially applied. A store that could not be reached is not a store that
accepted the write, and a run that carries on regardless is a run whose recorded spend is
fiction.

## Implementing one

Subclass `StateStoreConformance`, return the store under test from `make_store`, and
inherit the suite. It covers the version rules, patch commutativity under concurrency,
scrubbing, cursor paging that walks every record exactly once, and tenant isolation on both
reads and listings.

```python
class TestRedisStateStore(StateStoreConformance):
    def make_store(self) -> StateStore:
        return RedisStateStore(url="redis://localhost")
```

## Known limitations

- `MemoryStateStore` is one process. It exists so the rules can be exercised without a
  database, not so a deployment can skip having one.
- Checkpoint payloads and resume semantics are not here — a record says where a run got
  to, not how to rebuild its messages.
- The run loop does not write to a store itself yet; a consumer that wants durable runs
  reads and writes around it.
