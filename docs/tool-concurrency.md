# Tool concurrency

A model that asks for four independent lookups in one turn should cost about one lookup.
Running them one after another pays four round trips for work that had no order to it.
Running them all at once is how a single agent turn becomes a rate-limit breach at a
partner, an exhausted connection pool, or a bill nobody predicted.

So the batch runs concurrently, inside lanes somebody declared.

## The three lanes

`ConcurrencyConfig` names them. Each call stands in all of the ones that apply, and the
tightest decides.

| Lane | Field | Bounds | Declared by |
|---|---|---|---|
| The turn | `max_concurrent_tools` | One model response's batch | Runner, narrowable by the agent |
| The tool | `per_tool` | One tool across every run on the runner | Runner |
| The tenant | `per_tenant` | One tenant across all of its runs | Runner |

```python
runner = AgentRunner(
    provider=provider,
    tools=registry,
    concurrency=ConcurrencyConfig(
        max_concurrent_tools=8,
        per_tool={"search": 4, "booking": 1},
        per_tenant=16,
    ),
)
```

An agent may narrow what the runner declared and may never widen it:

```python
Agent(name="planner", instructions="...", model="claude-sonnet-5", free_text=True,
      concurrency=ConcurrencyConfig(max_concurrent_tools=2))
```

`ConcurrencyConfig.narrowed_to` is the composition rule, and it takes the smaller of each
field. Per-tool and per-tenant lanes bound a downstream shared with every other run on the
process — an agent cannot know about the others, so those stay the runner's to declare.
What an agent narrows, it narrows for its own turn.

Every call takes the lanes in one fixed order — tenant, turn, tool — so two calls cannot
each hold what the other is waiting for. A tool body that starts a sub-agent is a special
case of the same thing: the nested run spends the lane its caller is already standing in
rather than queueing behind itself, so nested concurrency counts against the parent's
limits exactly once.

## Order out is the order asked for

Results go back to the model in call order regardless of finish order. Each call is
dispatched against the same run and records only what it changed; the deltas are merged in
call order once the batch resolves. A batch that finished in whatever sequence the network
allowed therefore reads — and replays — as one deterministic transcript, and a cassette
recorded from it does not depend on which lookup happened to be quickest that day.

Approvals, guardrails and hooks stay serial and stay in call order. Only the invocations
fan out, because a human deciding about four tool calls should see them one at a time in
the order they were asked for.

## Failure is per call

A tool that fails loses its own call and nothing else. There is no placeholder result: the
model sees exactly which calls failed and why.

- A raising tool is reported against its own call id as a tool error.
- A tool that outran `per_tool_seconds[name]` fails with `ToolTimedOutError`, naming the
  tool and its ceiling. That ceiling is the tool's own, so one slow lookup does not spend
  the batch's time. The run-wide `DeadlineConfig.tool_call_seconds` is unchanged and still
  ends the run — it is a ceiling on the run, not on a call.
- Whether the batch survives at all is the agent's existing `on_tool_error` declaration.
  `SURFACE_TO_MODEL` keeps the siblings and sends the failures back for the model to work
  around. `FAIL_RUN` ends the run, and the siblings are stopped.

When a batch stops — cancelled, or ended by a sibling — what it did is recorded
distinctly from what it never started:

| What happened | Event | Reported as |
|---|---|---|
| Still queued behind a lane | `TOOL_ERROR` | `never dispatched: <why>` |
| In flight, tool declared idempotent | `TOOL_ERROR` | Safe to call again |
| In flight, not declared idempotent | `TOOL_INDETERMINATE` | The side effect may have landed |

The distinction is the point. A call that never went out is undone; a call that went out
and was cancelled is unknown, and claiming it was undone is how a booking gets made twice.

## Tools that cannot be parallelised

The runtime cannot infer order-dependence from a signature, so a tool declares it:

```python
ToolDeclaration(name="booking", parallel_safe=False)
```

A call to such a tool is given a phase of its own: everything the model asked for before
it has resolved, nothing else is in flight while it runs, and everything after it starts
once it is done. Consecutive parallel-safe calls share a phase. Declaring a tool
`parallel_safe=False` also serialises two concurrent calls to it with identical arguments,
which is the answer for a non-idempotent tool the model asked for twice.

`parallel_safe` is not sent to any vendor and is not part of a determinism fingerprint, so
declaring it changes scheduling and nothing else.

## Blocking tool bodies

A synchronous body that blocks blocks the event loop, and its siblings with it. Offloading
is the registry's job, not the batch's: run it on a `WorkerPool`
(`docs/async-and-sync.md`), which bounds the threads and refuses rather than growing
without limit. The loop monitor catches an undeclared blocking body and fails the call
with `EventLoopStalledError` naming the tool, rather than letting the whole batch stall
anonymously.

## Worked example

`examples/tool_concurrency.py` runs the whole of the above against scripted providers —
four lookups at once, the same four two at a time, one failure among three successes, a
hanging tool cut off by its own ceiling, and an order-dependent tool run alone.
