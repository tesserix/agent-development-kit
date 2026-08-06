# The run loop

`AgentRunner` drives one agent from prompt assembly to exactly one terminal state, and
returns the `Run` that records how it got there. Every product that hand-rolled this loop
disagreed about what "finished" meant; here it is one thing.

```python
runner = AgentRunner(provider=provider, tools=registry)
run = await runner.run(agent, "Four nights near Kyoto.", tenant="acme", user="ada")
assert run.state.is_terminal
```

A worked end-to-end run, tool call included and no network: `examples/run_loop.py`.

## Prompt assembly is fixed and documented

`assemble_prompt` composes one turn in this order, always:

| # | Part | Notes |
|---|---|---|
| 1 | `agent.instructions` | The system message. |
| 2 | Memory | Wrapped as untrusted data. |
| 3 | History | In the order given. |
| 4 | The new input | Always last. |

Tool declarations keep the registry's order and are filtered to `agent.tools` — the model
is never told about a tool it may not call.

`Prompt.version` is a short digest of the *cacheable prefix* (instructions plus tool
declarations) and lands on `Run.prompt_version`. Two runs sharing a version were shaped by
the same prompt whatever was asked of them, which is what makes a regression attributable.

### Untrusted content is handed over as data

Memory, tool results and anything else the agent did not author go through
`wrap_untrusted`, which fences them and names their origin:

```
<untrusted-data source="tool_result">
{"trains": 4}
</untrusted-data>
```

A forged marker inside the content is escaped, so content cannot close its own fence and
continue as instruction. This is a boundary, not a defence: real injection classification
is #127.

## Terminal states

Every path ends in exactly one of these, and the run always comes back — a failure is a
state, not an escaped exception.

| State | Cause |
|---|---|
| `completed` | The model stopped without tool calls, and any declared `output_type` validated. |
| `failed` | Provider error, guardrail refusal, schema violation, an allowlist breach, zero content and zero tool calls, or a tool failure under `FAIL_RUN`. |
| `budget_exhausted` | The budget policy refused a reservation. |
| `max_iterations_exceeded` | The cap was reached without settling. |
| `cancelled` | A `CancelledError` from the kit's hierarchy, a cancelled token, or an elapsed deadline. |

`asyncio.CancelledError` is deliberately *not* converted: a cancelled task that returns
normally leaves its canceller waiting forever, so it propagates.

## Cancellation and deadlines

Two ways a run stops early, both ending in `cancelled` with the record intact:

```python
token = CancellationToken()
run = await runner.run(agent, "…", tenant="acme", cancellation=token)  # token.cancel() stops it

runner = AgentRunner(provider=provider, deadlines=DeadlineConfig(model_call_seconds=30))
run = await runner.run(agent, "…", tenant="acme", deadline=Deadline.in_seconds(60, now=time()))
```

Worked end to end, no network: `examples/cancellation.py`.

**A deadline is an instant, not a duration.** `Deadline` carries the wall-clock moment the
run must be over by, so it survives being passed down: a duration restarts at every hop,
and five agents each given "30 seconds" take two and a half minutes. `narrowed_to` takes
the earlier of two, so an inherited ceiling can be tightened and never extended.

**Nothing is bounded by default.** `DeadlineConfig` leaves `run_seconds`,
`model_call_seconds` and `tool_call_seconds` as `None`. A model call on CPU inference
legitimately takes minutes where the same call on a GPU takes a second, so a ceiling the
kit invented would kill good runs on exactly the hardware this kit is aimed at. A ceiling
of zero is refused at construction: it reads as "no time at all" and cancels every run
before it starts, which is never what was meant. `grace_seconds` (5s) is the only one with
a default, because it bounds the kit's own waiting rather than the deployment's work. An
agent that declares its own `DeadlineConfig` replaces the runner's: the agent's author
knows what that agent does, where the runner only knows what it drives.

**Cancellation is checked between steps and raced against them.** Every iteration checks
the token and the deadline before the next model call, and each model call, guardrail
check and tool call is raced against both. The race uses the injected `Clock`, so a test
with `FakeClock(auto_advance=False)` drives a timeout deterministically and never sleeps.

**Uncooperative work is dropped, not waited for.** Aborted work is cancelled, given the
grace window to unwind, and then abandoned — the run resolves and records `work_orphaned`
rather than blocking on a provider that keeps streaming into a socket nobody reads. The
abandoned task's reference is retained so it cannot be destroyed mid-flight unobserved.

**A tool cut off after dispatch is indeterminate, not failed.** `tool_indeterminate`
records that the call was stopped *after* it went out, so whether its effect landed cannot
be known. Naming it that is the point: the kit never claims a payment did not go through
when it has no way to tell. A tool listed in `Agent.idempotent_tools` records an ordinary
`tool_error` marked safe to retry instead — the declaration is what makes retry safe, and
it is checked against the allowlist so a policy cannot name a tool the agent cannot call.

## Events

Every step is appended to `Run.events` in the order it happened: `prompt_assembled`,
`model_call`, `model_response` (carrying its `Usage`), `tool_call`, `tool_result`,
`tool_result_truncated`, `tool_error`, `tool_refused`, `tool_indeterminate`,
`guardrail_refusal`, `output_validated`, `schema_violation`, `cancellation_requested`,
`deadline_exceeded`, `work_orphaned`, `terminated`. Cost attribution totals the usage on
those events rather than being wired per project.

## Decisions

**A failure returns the run, a misconfiguration does not.** Anything that happens *during*
a run is recorded and returned, because a failure that discards the record leaves nobody
able to say what happened. But an agent declaring a guardrail, a budget or a tool registry
the runner was never given is refused *before* the run starts — starting anyway would run
it without a check it declared, and the first tool call is a worse time to find out.

**A tool failure is shown to the model by default.** `ToolFailurePolicy.SURFACE_TO_MODEL`
feeds the error back as a tool result so the model can choose another route; a run that
dies on the first recoverable failure cannot. Where a tool was the source of truth,
`FAIL_RUN` says so. Either way the exception is wrapped in `ToolExecutionError` and no
result is invented.

**A tool outside the allowlist ends the run.** The model was only told about allowed
tools, so a call outside them means something upstream is wrong. Nothing is dispatched.

**Truncation is an event.** An oversized tool result is cut at `max_tool_result_chars` and
records `tool_result_truncated`. Silently dropping half a result is a wrong answer nobody
can account for.

**Guardrails fail closed.** A check that raised is a denial, not a pass. They are keyed by
name and must answer to the name they are filed under, or construction fails.

**Zero content and zero tool calls is terminal.** Asking again for the same nothing is how
a loop wedges.

## Known limitations

- **No streaming.** `run` returns a finished `Run`; incremental events are #38–#42.
- **No durable resumption.** A `Run` serialises, but nothing resumes one mid-flight yet —
  that is the durable orchestration epic.
- **Tool arguments are not validated against the tool's schema before dispatch.** The
  registry validates them; the loop does not pre-check. Hard allowlist enforcement and
  schema validation at dispatch are #130 and #129.
- **Tools run one at a time.** Bounded parallelism is #44.
- **No retry or repair on a schema violation.** The run fails; bounded validation repair
  with the error fed back is #35.
- **`task_class` is refused.** Routing by task class needs a router (#53); guessing a
  model would attribute the run to one that never ran it.
- **`ModelRequest`, `ModelResponse` and `ToolDeclaration` are provisional**, owned by the
  runtime until the provider protocol (#49) and the tool registry (#130) land theirs.
- **`Run.output` is a dict, not a typed instance.** `Agent[TripPlan]` returning `TripPlan`
  is #36; a checkpoint is JSON, so the payload is stored and re-parsed.
- **The budget estimate is characters over four**, not a tokeniser. Normalised accounting
  is #55.
- **The token is not threaded into tool or provider signatures.** `ToolRegistry.invoke`
  and `ModelProvider.complete` take no cancellation argument, so their work is raced and
  cancelled from outside rather than asked to stop. A tool that wants to unwind its own
  work cooperatively needs the token on `RunContext` — that arrives with the context
  object (#33).
- **Indeterminacy is a recorded event, not a raised type.** `tool_indeterminate` is on the
  run for a caller to branch on; there is no `ToolIndeterminateError` to catch, because
  the run does not raise.
- **A per-run `deadline` only narrows.** Passing one later than the runner's
  `run_seconds` ceiling changes nothing, by design — a caller cannot buy more time than
  the deployment allows.
