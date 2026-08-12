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
