# Running an agent as a durable workflow

A run that takes minutes to hours lives in one process today. A pod roll, an OOM kill or a
scale-down loses it, and the choices are to start from zero — paying for every model call a
second time, and re-applying every tool call — or to hand-roll durable wiring, which is how
two teams end up with two different answers to the same problem.

`AgentWorkflow` drives the same loop as the in-process runtime, with one difference: it
performs no I/O. Model calls and tool calls are the only execution points, and each one is
recorded in a `Journal` under a step id derived from the run's position. A worker that dies
and restarts replays the loop, finds the completed steps, and pays for none of them again.

```python
from tesserix_adk.workflows import ActivityContext, AgentWorkflow, WorkflowState

workflow = AgentWorkflow(activities=activities, model="claude-opus-5", journal=replayed)
final = await workflow.run(
    WorkflowState(run_id=run_id, history=handle),
    context=ActivityContext(run_id=run_id, tenant=tenant, user=user, trace_id=trace),
)
```

Nothing here imports a workflow engine. The kit supplies the deterministic driver and the
activity payloads; binding them to Temporal activities is the deployment's job, through the
optional `tesserix-adk[temporal]` extra. The package imports cleanly with nothing installed,
so a consumer that does not want durability does not pay for it.

## The two activities

| Activity | Input | Result |
|---|---|---|
| `model_call` | `ModelCallInput` — context, step, attempt, model, history handle, tools | `ModelCallResult` — the validated `ModelResponse` and the new history handle |
| `tool_call` | `ToolCallInput` — context, step, attempt, tool, call id, arguments | `ToolCallResult` — content or a handle, whether it failed, the new history handle |

Everything else on the workflow path is arithmetic: the iteration counter, the usage ledger,
the pending approval, the autonomy grant id and the terminal decision. That is what makes a
replay produce the same decisions as the original run.

## What travels, and what does not

**The transcript never travels.** It outgrows the payload limit long before a long run ends,
so a model activity receives a *handle* to the history in the run's store, resolves it, calls
the provider, appends the turn and returns the new handle. An input with an empty handle is
refused at construction rather than being sent with no history.

**Nothing is truncated to fit.** An input or result over `PAYLOAD_LIMIT_BYTES` raises
`PayloadTooLargeError` naming the step and the size. A retrieval result cut in half is still a
plausible-looking answer, and the run would continue on evidence nobody chose.

**Tenant is never inferred.** A worker serves every tenant, so an `ActivityContext` without one
raises `MissingTenantContextError` — the worker's own identity is not a default. User, scopes
and trace id ride along on every input, so an activity cannot widen its own authority and its
span attaches to the run's trace rather than the worker's.

## Failure, retry and cancellation

An unavailable provider is retried up to `attempts` times; when they are exhausted the run
fails with `ProviderUnavailableError` carrying the attempt count and the step. The state and
the journal stay inspectable, and no invented completion is ever returned.

Cancellation is both checked before each activity and *raced against* the one in flight: a
streaming completion nobody is listening to still consumes tokens and still bills. Pass the
runtime's `CancellationToken` as `token`.

## Known limitations

* **Token streaming is not available on the workflow path.** An activity returns once. A run
  that needs tokens as they arrive stays in process; `STREAMING_UNSUPPORTED` says so rather
  than the kit degrading quietly.
* Replay-safety enforcement and the CI guard that catches non-deterministic workflow code are
  a separate concern, as is compensation of partially applied work.
* Worker charts and Temporal namespace provisioning live in `tesserix-k8s`, not here.
* The journal is held by the workflow object. Persisting it is the engine's job — under
  Temporal that is the event history, and under a test it is the object itself.
