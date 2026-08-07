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

`assemble_prompt` composes one turn in this order, always — `PROMPT_LAYERS` is the same
list in code, and `Prompt.layers` labels each assembled message with the one it belongs to:

| # | Layer | Notes | In the prefix |
|---|---|---|---|
| 1 | `PromptLayer.SYSTEM` | `agent.instructions`, then the output contract where the provider cannot enforce it. | Yes |
| 2 | `PromptLayer.TOOLS` | Declarations, **sorted by name**. They travel in their own field, not as a message. | Yes |
| 3 | `PromptLayer.PINNED` | Context that holds for the conversation — a case file, a style guide. Wrapped as untrusted data. | Yes |
| 4 | `PromptLayer.RETRIEVED` | Context fetched for this turn — recalled memory, retrieved documents. Wrapped as untrusted data. | No |
| 5 | `PromptLayer.CONVERSATION` | History in the order given, then the new input, always last. | No |

Declarations are filtered to `agent.tools` — the model is never told about a tool it may
not call — and sorted by name, so a registry that iterates a dict in a different order
cannot cost a prefix refill. Two tools sharing a name are refused: sorting would hide the
duplicate and the model could not tell which it was calling.

### The prefix, and why the order is an invariant

`Prompt.fingerprint` is a short digest of the prefix **as bytes** — the layers marked above,
pinned context included. Equal fingerprints across two turns mean the inference server can
reuse the prefix it already evaluated. On CPU that is the difference between usable and
unusable: prefill dominates, and a 40k-token prompt that costs about a second on an H100
costs tens of seconds on a CPU. A prefix that shifts by one byte pays that again, every
turn.

So the layer order is guarded by a test rather than left to convention. Anything that
reorders the prefix fails `tests/test_prompt.py::TestTheLayerOrderIsFixed` by name, instead
of doubling a bill quietly. Retrieved context deliberately sits *outside* the prefix:
documents fetched per turn would invalidate the cache per turn. What is worth retrieving at
all, and what leaves when it no longer fits, is [`context.md`](context.md).

`Prompt.prefix` is the messages that digest covers, and `Prompt.prefix_tokens` is how large
they are. The count comes from `approximate_tokens` — four characters to a token, which is
fine for a log line and wrong for a context-window check. Pass the server's own tokenizer
as `tokenizer=` where the number has to be right; anything matching the `Tokenizer`
protocol will do.

`Prompt.version` is the other digest, and answers a different question: not "will the cache
hit" but "which prompt design produced this run". It covers instructions, tool declarations
and the output contract, and lands on `Run.prompt_version`, which is what makes a regression
attributable.

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
| `loop_limit_exceeded` | A cap on the run's shape bound: depth, fan-out width, per-run total, or repetition. |
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

## Retries

```python
runner = AgentRunner(provider=provider, retry=RetryConfig(max_attempts=3))
agent = Agent(name="planner", …, retry=RetryConfig(max_attempts=5), idempotent_tools=("lookup_fare",))
```

Worked end to end, no network and no sleeping: `examples/retry.py`.

**Nothing is retried by default.** `RetryConfig().max_attempts` is 1. A retry is a second
charge on someone's account and a second write to someone's database; the kit does not
assume either on a caller's behalf. As with deadlines, an agent's own `RetryConfig`
replaces the runner's.

**Retryability is a property of the error, not of the call site.** `AdkError.retryable` is
`False` and is overridden only where a failure is a *fault* rather than an *answer*: a
timeout and a transient status (408, 409, 425, 429, 500, 502, 503, 504) are faults; a 400,
a guardrail refusal, a budget ceiling and a schema violation are answers, and asking again
spends more to be told the same thing. A `ProviderError` with no status at all is retried,
because a request the provider rejected always comes back with one — no status means the
call never got there. Anything outside the kit's hierarchy is never retried: the kit did
not raise it, so it cannot know what repeating it repeats.

**The backoff is full jitter.** The delay is drawn uniformly from `[0, min(base ×
multiplier^(n-1), cap))`. A fixed schedule makes every process in a fleet retry the same
blip at the same instant and knock the recovering provider over again. The source is an
injected `Random`, so a test seeds it and asserts the exact schedule instead of waiting it
out, and each `RetryPlan` seeds its own rather than sharing one per process.

**A provider that names a time is believed, up to a ceiling.** A `Retry-After` is used in
preference to the computed window — the provider knows its own recovery better than a
multiplier does — but one beyond `max_retry_after_seconds` (60s) ends the run rather than
being clamped. A provider asking for an hour is reporting a quota, not a blip; waiting
stalls the run and retrying sooner ignores what it said.

**A retry never outlives the run.** A backoff that would land past the deadline is not
taken; the run fails with the attempt's own error instead of sleeping through its ceiling.
Every attempt also reserves against the budget, so a budget bounds retries without knowing
what a retry is.

**A tool is retried on its declaration, never on the shape of its exception.** Only tools
in `Agent.idempotent_tools` are tried again, because an exception says nothing about
whether the side effect landed — a gateway timeout on `charge_card` is exactly the case
where the charge went through. Every failed attempt records `attempt_failed` with what
failed and either the delay before the next attempt or why there is not one.

## Caps on the shape of a run

```python
runner = AgentRunner(provider=provider, budget=RunBudget(resolved=resolved, clock=clock))
agent = Agent(name="planner", …, budget=BudgetLimits(max_delegation_depth=2))
agent = Agent(name="planner", …, loop=LoopConfig(max_repeated_calls=2))
```

Worked end to end, no network: `examples/loops.py`.

**How deep and how wide are stated with the money.** `max_delegation_depth`,
`max_parallel_tool_calls`, `max_peer_invocations` and `max_tool_calls` are dimensions of
`BudgetLimits`, resolved and attributed like any other ceiling — see
[`docs/budget.md`](budget.md). Two policies would be two places to raise a cap, and the
one nobody read is the one that lets a run away. `LoopConfig` keeps only
`max_repeated_calls`, which counts a run going round rather than a run spending.

**They are bounded by default,** unlike deadlines and retries. `BudgetLimits.conservative()`
caps delegation depth at 4, fan-out at 8, peer invocations at 8 and tool calls at 40. A
wall-clock ceiling the kit invented would kill good runs on slow hardware; a cap on shape
only ever stops a run that has stopped making progress, and costs nothing when it does not
bind. A cap of zero is refused at construction — it reads as "never do this at all", which
is not a bound on a run but a run that cannot work.

**A cap narrows and never widens.** Resolution takes the tightest applicable scope, so an
agent declaring its own tightens what the deployment gave it and can never vote itself more
rope; a delegated child spends its parent's remaining allowance through `child()`. This is
the opposite of `DeadlineConfig` and `RetryConfig`, which an agent *replaces*: how long to
wait and what to retry are properties of the work, but how far a chain of agents may
recurse is a property of the deployment paying for it.

**A turn that would break a cap is refused entire, before any dispatch.** A fan-out wider
than `max_parallel_tool_calls`, a fan-out that would only partly fit under
`max_tool_calls`, and repetition are all checked against the whole turn first. Trimming a
fan-out to fit leaves half a plan executed — a set of side effects nobody chose — so the
run terminates `loop_limit_exceeded` with nothing dispatched. A *single* call that will not
fit is spend rather than shape, and ends the run `budget_exhausted` at the ceiling that
charges for it.

**Tool calls go out one at a time.** Dispatching a cleared fan-out concurrently would save
wall-clock and lose the check between calls, and that check is what stops the second call
of a turn the caller cancelled during the first. Latency is worth less than a side effect
nobody is waiting for.

**Depth and the call path are checked before a prompt is assembled.** Pass the caller's
`RunContext` as `parent` and both the depth and the path carry down the chain; a run past
the ceiling fails closed without a model call, naming the path it took — `alpha→beta→alpha`
is the shape of the bug, and printing it is how somebody finds it. Failing closed at the
bottom is the point: a level that invents a substitute result keeps the cycle alive one
layer up, where nothing can see it.

**A delegation is counted against the whole tree.** `max_peer_invocations` is charged on
the shared ledger, so a graph of agents each under the cap cannot break it together — which
is exactly how per-hop counting fails.

**Repeats are counted by request, not by tool.** The signature is the tool name plus its
arguments, order-independent, so paging through results is progress and asking the same
question five times is not. A tool in `Agent.idempotent_tools` is exempt: polling one
status endpoint with the same arguments is the design.

Which cap bound is in the type — `RecursionLimitError`, `FanOutLimitError`,
`RepeatedCallError`, `MaxIterationsError`, all under `LoopLimitError` — and named in the
`terminated` event, because a run that stops without saying which bound it hit is a run
nobody can tune. None of them are retryable: a cap is a decision, not a fault.

## Hooks, approvals and where policy attaches

```python
runner = AgentRunner(provider=provider, hooks=HookChain([Redactor(), ModelAllowList()]))
runner = AgentRunner(provider=provider, tools=tools, approvals=desk, approval_ttl_seconds=900)
agent = Agent(name="clerk", …, tools=("wire_funds",), approval_required_tools=("wire_funds",))
```

Worked end to end, no network: `examples/hooks.py`.

**The seven points are the loop's own.** `before_prompt_assembly`, `before_model_call`,
`after_model_response`, `before_tool_dispatch`, `after_tool_result`,
`before_output_validation`, `on_terminal`. A check declared once is enforced on every path
out of a run, which is what stops an agent being safe in one product and unsafe in the next
because the check lived in application code and the next caller did not write it.

**A hook returns a decision, never a mutation.** Four words and no fifth — `continue`,
`rewrite`, `require_approval`, `refuse` — and a `HookSubject` of facts rather than handles.
There is no run, no config and no chain in what a hook is handed, so widening a tenant
scope, disabling another hook or raising a cap is not a thing it can be talked into.

**The most restrictive answer wins, ties to the first declared.** Two hooks disagreeing is
not a coin to toss: the same chain resolves the same way on every process, and the tighter
answer is the one nobody has to justify afterwards.

**Hooks fail closed.** One that raises or outruns `DeadlineConfig.hook_seconds` stops the
run, because a check that did not run is not a check that passed. The exception is
`on_terminal`, where the run is already over: a failure there is recorded rather than acted
on, since there is nothing left to fail closed to.

**The chain is sealed when a runner takes it.** Sealing is one-way and in place, so a hook
holding the chain it was declared in finds it shut. The chain a run started with is the
chain it is judged by; otherwise a hook could register a permissive one behind itself.

**A rewrite is logged as digests, not content.** `hook_rewrite` records
`before → after` as SHA-256 prefixes. A replay recomputes them and knows it assembled the
same prompt, without the redacted text living on in the log that was supposed to remove it.

**An approval is permission at a moment, not a standing licence.** `ApprovalRecord` carries
a digest of the arguments and never the arguments, because an approval queue outlives the
run and is read by people who are not party to it. A decision is honoured only if it echoes
the record's id and lands inside `approval_ttl_seconds`; a gate that fails or never answers
is not a grant. `require_approval` with no gate wired is a `ConfigurationError`, not a
call that goes out unapproved.

## Events

Every step is appended to `Run.events` in the order it happened: `prompt_assembled`,
`model_call`, `model_response` (carrying its `Usage`), `tool_call`, `tool_result`,
`tool_result_truncated`, `tool_error`, `tool_refused`, `tool_indeterminate`,
`attempt_failed`, `fan_out_refused`, `repeat_detected`, `depth_exceeded`,
`hook_rewrite`, `hook_refusal`, `approval_required`, `approval_granted`, `approval_denied`,
`guardrail_refusal`, `output_validated`, `schema_violation`,
`cancellation_requested`,
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
- **No fallback to a second provider.** A retry re-sends the same request to the same
  provider; routing a fault to another one is the gateway epic.
- **A failed attempt's usage is only counted if the provider attached it.** `ProviderError`
  carries no `Usage`, so tokens spent on an attempt that raised are invisible until the
  provider protocol (#49) defines how a failed call reports what it burned.
- **Depth is only counted where it is passed.** A caller that starts a nested run without
  handing down the parent `RunContext` starts it at depth 0. Agent-to-agent delegation
  inside the kit threads it automatically once the workflows epic owns the call.
- **Repetition is exact-match only.** Two calls that differ by a whitespace-only argument
  read as different requests. Semantic near-duplicate detection is not planned.
- **`task_class` is refused without a router.** A runner given no router cannot resolve a
  class, and guessing a model would attribute the run to one that never ran it. Wire a
  router and a provider per vendor it can choose — see [`docs/routing.md`](routing.md).
- **`ModelRequest`, `ModelResponse` and `ToolDeclaration` are provisional**, owned by the
  runtime until the provider protocol (#49) and the tool registry (#130) land theirs.
- **A rehydrated run needs its type parameter named.** Nothing on the wire says which type
  the answer was, so `Run[TripPlan].model_validate_json` rehydrates it and a bare `Run` is
  refused rather than handed back with the answer dropped. See
  [`docs/typing.md`](typing.md).
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
- **An approval blocks the run in process.** The gate is awaited, so a run waiting on a
  human holds its task and its memory. Suspending a run to durable storage and resuming it
  when the decision arrives is the durable orchestration epic.
- **Hooks see text, not structure.** `HookSubject.content` is the text of the message or
  response; non-text parts are passed through untouched and a rewrite replaces the text
  parts wholesale. Redacting inside an image or a structured part is out of scope here.
- **`on_terminal` cannot rewrite anything.** It is asked, and its decision is recorded, but
  a run that has already ended has nothing left to swap and nothing left to refuse.
- **A per-run `deadline` only narrows.** Passing one later than the runner's
  `run_seconds` ceiling changes nothing, by design — a caller cannot buy more time than
  the deployment allows.
