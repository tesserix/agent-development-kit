# Checkpointing — carrying a run on after the process holding it died

A run that dies at iteration nine restarts at zero. The wasted model calls are the cheap
half of the problem; the expensive half is that every tool call it had already made gets
made again. For a search that is noise. For a booking it is a second seat, charged to
somebody who asked for one.

Checkpointing writes the run's frontier as it goes, and `AgentRunner.resume` continues from
there. The hard part is not storing the conversation — it is deciding, for each call that
was outstanding when the process died, whether that call happened.

```python
runner = AgentRunner(
    provider=provider,
    idempotency=MemoryIdempotencyStore(),
    checkpoints=Checkpointer(MemoryCheckpointStore()),
)

run = await runner.run(agent, "book the flight", tenant="acme", run_id="run_1")
# ... the process dies mid-run ...
run = await runner.resume(agent, "run_1", tenant="acme")
```

## Only at unambiguous boundaries

| Boundary | What is settled |
|---|---|
| `after_model_call` | The model answered. What it asked for is known; none of it has run. |
| `after_tool_result` | Every call from the last turn came back and is in the conversation. |
| `before_approval` | The run is about to wait on a person. Nothing is in flight. |

Anywhere else — mid-dispatch, mid-stream — the frontier is a moving thing, and a checkpoint
of it describes a state the run was never in. A resume from that either repeats work or
skips it, and nothing downstream can tell which happened.

`CheckpointPolicy` picks among the three. Writing at fewer boundaries costs less and repeats
more work on resume; the default writes at all three.

## The cap refuses rather than truncates

`max_bytes` (1 MiB by default) is a ceiling on one payload. A run whose conversation grows
past it stops being checkpointed. It is not trimmed to fit: a resume from a truncated
frontier continues a conversation that never happened, and neither the model nor the audit
record can see that anything is missing.

A checkpoint that could not be written — over the cap, or a store that is unreachable —
never fails the live run. Losing the ability to resume is worth knowing about; it is not
worth killing a run that is otherwise fine. The runner carries on and `Checkpointer.last_error`
holds what went wrong.

## Deciding what already ran

`plan_resume` asks the idempotency store about each outstanding call and returns a
`ResumePlan`. It decides; it does not act.

| `ToolDisposition` | What the record said | What the resume does |
|---|---|---|
| `completed` | An outcome was recorded under the call's key. | Replays that outcome into the conversation. |
| `never_ran` | Nothing holds the key. | Dispatches the call for the first time. |
| `indeterminate` | The key is held in flight by a process that is gone. | Nothing. |

`indeterminate` is the case the design exists for, and the kit refuses to guess at it.
Retrying might book a second seat; skipping might strand the run having promised something
it never did. `refuse_if_undecidable` raises `IndeterminateToolCallError`, naming every
call, and that error is deliberately **not retryable** — the answer comes from the tool's
own status endpoint or from a person, never from asking again.

A dispatched call with no idempotency key is indeterminate too. The absence of a key is the
absence of a guarantee, not permission to run it again.

```python
plan = await plan_resume(checkpoint, idempotency)
plan.completed        # replayed
plan.to_dispatch      # called for the first time
plan.indeterminate    # nobody can say
plan.safe             # False if there is anything in indeterminate
```

## One worker at a time

Two workers resuming one run is two runs: one budget spent twice, every outstanding call
dispatched twice. A resume takes an at-most-once claim on the run through the same
idempotency machinery a tool call uses, and the second worker gets `ResumeConflictError`.
The claim expires after five minutes, so one crashed worker does not strand the run forever.

`Resumer` puts a `RunLease` in front of that, because a claim that only covers the moment of
resuming does not cover the hour of work after it. The lease is taken **before the frontier is
read**, so the worker that loses never sees the run at all:

```python
resumer = Resumer(checkpoints=checkpointer, leases=lease_store, history=transcripts)
carried = await resumer.resume("run_1", tenant="acme", worker="worker-7")
carried.lease.fence   # rises on every acquisition
carried.plan.safe     # the same decision plan_resume makes
```

Three rules make it safe under clock skew:

- **The store owns the clock.** A caller's idea of `now` is never used to decide expiry —
  the Redis scripts call `TIME`, the SQL uses `extract(epoch FROM now())`. A worker whose
  clock is an hour fast cannot take a lease that has not lapsed.
- **Every acquisition raises the fence.** A superseded holder is refused on the number rather
  than on the time, so a write from a worker that was paused past its TTL is rejected by
  whatever it writes to. `RunLease.superseded_by` is the comparison.
- **Renewal keeps the fence.** `Leaseholder` is an async context manager that renews inside
  `LeasePolicy.renew_within` and releases on the way out, so a run that finishes hands the
  lease back rather than leaving the next worker to wait out the TTL.

`LeaseStoreConformance` carries all of that. `MemoryLeaseStore` is the in-process
implementation the suite is first run against.

## What a checkpoint must not contain

A frontier is durable, and a run's conversation is the most likely place in the kit for an
API key a user pasted or a bearer token a tool was handed. `Checkpointer` masks before it
sizes, so the redaction happens once and every store inherits it — including one a consumer
writes:

```python
Checkpointer(store, extra_patterns=(r"EMP-\d+",))   # on top of the built-in patterns
Checkpointer(store, redact=False)                   # only where the store is already trusted
```

Binary parts are left alone: masking bytes produces a payload that is neither the original
nor honestly empty.

## Storing the transcript out of line

`history_handle` is a pointer to the conversation rather than the conversation. A run that
has been going for a day carries a transcript far past `max_bytes`, and a frontier that
refuses to be written is a run that cannot be resumed. `HistoryStore.fetch` resolves the
handle at resume time, and a handle that no longer resolves raises `HistoryUnavailableError`
— the run stops rather than continuing against a conversation with a hole in it.

## Durable stores

`tesserix_adk.adapters` ships the two most deployments already run:

| Store | Backed by | Notes |
|---|---|---|
| `RedisCheckpointStore` | one key per run | `max_value_bytes` refuses an oversized payload before the call |
| `RedisLeaseStore` | Lua, server-clocked | `ACQUIRE` returns the holder to refuse with |
| `PostgresCheckpointStore` | `adk_checkpoints` | `verify()` reads the schema version before use |
| `PostgresLeaseStore` | `adk_run_leases` | one upsert decides acquisition; nothing races |

The DDL is published as `EXPECTED_CHECKPOINT_SCHEMA` and applied by nobody here. Schema
lifecycle belongs to whatever owns migrations; a kit that creates its own tables at import
time is a kit that cannot be reviewed before it runs.

## Starting the execution again without losing the run

A workflow engine's history is finite. `ContinuationPolicy.due` says when an execution has
run long enough to be worth starting again — `max_steps` on the journal, `max_history_bytes`
on whatever the engine reports — and `continued` builds the `Continuation` that crosses over:

```python
if DEFAULT_CONTINUATION.due(state.journal, history_bytes=info.history_size):
    carry = continued(state, tenant="acme", agent_name="booking")
```

The transcript crosses as a handle, the ledger and the iteration count cross as numbers, and
the journal deliberately does not — its results are already folded into the state, and
replaying them is the cost being cut. A `Continuation` that dropped the approval or grant the
run was acting under cannot be constructed: that is a `ConfigurationError`, not a warning,
because the new execution would otherwise carry on unapproved.

## From a terminal

The run waiting on an approval since Friday is usually resumed by an on-call engineer:

```console
$ adk inspect run_1 --tenant acme          # takes no lease
run run_1  tenant acme  agent booking
iteration 3  1200 in, 300 out  4100 micros
waiting on approval req-9
```

`adk run --resume run_1 --tenant acme` prints the same summary — the same `describe`, so the
two cannot drift — then takes the lease and reports the fence it holds. Its exit codes are
what a script reads: `0` carried on, `1` nothing checkpointed under that id, `2` a command
line it could not read, `3` another worker holds it, `4` it must not be carried on — an
undecidable call, an evicted transcript, or a frontier this kit cannot read.

## What is not carried

Spend. The ledger already survives the process, and a second copy of a number that only ever
goes up is a second copy to disagree with. The checkpoint carries usage for reporting, not
as the authority on what a run may still spend.

A definition whose revision differs from the one the checkpoint pinned is refused.
Resuming into a changed agent is a different run wearing the first one's identity, and a
past run that names a revision it did not actually use is an audit record that lies.

## Formats

`CHECKPOINT_FORMAT` is the payload version. A reader refuses a checkpoint written by a newer
kit with `CheckpointFormatError` rather than reading fields it would have to guess at — the
run is left where it is, for a worker on the newer version to pick up.

## Implementing a store

`CheckpointStore` is three methods: `put`, `latest`, `forget`. One frontier per run —
keeping an older one alongside the newest means resuming into work already done. Writes are
last-write-wins by design: unlike run state, a checkpoint has no contended field, and two
workers writing one are two views of the same monotonic progress.

`CheckpointStoreConformance` carries the guarantees. `MemoryCheckpointStore` exists so they
can be exercised without a database — nothing in it outlives the process, which is precisely
what surviving a restart is not.

## See also

- [`docs/state.md`](state.md) — run and session state, and the versioning that keeps two writers honest.
- [`docs/tool-idempotency.md`](tool-idempotency.md) — where the record of an executed side effect lives.
