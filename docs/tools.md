# Tools

A tool is one typed function. The schema the model reads is derived from its signature, so
there is nothing to keep in step with it:

```python
from tesserix_adk.tools import tool

@tool
async def fare(origin: str, nights: int = 1) -> str:
    """Price a stay.

    Args:
        origin: Where the traveller boards.
        nights: How long they stay.
    """
    return f"{origin}: {nights * 40} EUR"

fare.parameters_schema   # {"type": "object", "properties": {...}, "required": ["origin"]}
fare.description         # "Price a stay."
await fare.invoke({"origin": "Osaka", "nights": 2})
```

The alternative — a hand-written dictionary advertising the tool and a Python function
implementing it — is two declarations of one shape, and the second one is the one nobody
updates. The drift is silent: the model keeps sending the argument that was renamed six
months ago, and it surfaces as a call the code refuses in production.

Descriptions come from the docstring, on the same rules as
[`docs/schemas.md`](schemas.md#descriptions-come-from-the-docstring): the summary line
becomes the tool's description, the Google-style `Args:` block describes the parameters, and
a missing docstring costs the descriptions and never the tool. `@tool(name=..., description=...)`
overrides either.

Nested models are inlined by default — `dialect=INLINE_REFS` — because a provider that will
not follow a `$ref` is the common case for tool declarations. Pass `dialect=JSON_SCHEMA` to
keep `$defs`, or any other [dialect](schemas.md#provider-dialects).

## Refusals happen at import time

Every failure below is raised by the decorator, at the line where the tool is declared:

| Situation | Raised |
|---|---|
| An unannotated parameter, `*args`, `**kwargs`, `Any` at any depth | `ToolDefinitionError` naming the parameter |
| A type with no JSON Schema representation | `ToolDefinitionError` naming the parameter and the type |
| An annotation that does not resolve | `ToolDefinitionError` quoting the `NameError` |
| A generator or async generator | `ToolDefinitionError` |
| A self-referencing model under the default dialect | `ToolDefinitionError` naming the type |
| Two `ToolContext` parameters | `ToolDefinitionError` naming the second |
| A name another live tool answers to | `ToolDefinitionError` naming the holder |
| A callable with no `__name__` and no `name=` | `ToolDefinitionError` |

A parameter no schema can describe is a tool the model calls wrongly for as long as the
process lives, and the call that suffers is nowhere near the definition that caused it. The
same argument as the schema layer's, applied one level up: import time is where a
misdescribed tool is cheap.

## One name, one live tool

Two tools answering to one name means the model reaches whichever the registry happened to
keep. The second declaration is refused.

The claim is held for as long as the tool is, not for as long as the process is: a tool that
goes out of scope gives its name back, and re-decorating the same function — a module
reloaded, a fixture rebuilt per test — takes the name it already held. `Tool.release()`
gives a name back explicitly, for a tool being replaced while something still holds a
reference to it.

## Sync and async are one path

Decorating makes every tool awaitable. A synchronous body is run off the event loop, so a
tool that blocks does not stall every other run sharing the loop:

```python
await lookup.invoke({"code": "OSA"})                    # a thread, unbounded
await lookup.invoke({"code": "OSA"}, workers=pool)      # a bounded WorkerPool
```

Without a `workers=`, the body goes to `asyncio.to_thread` and nothing bounds how many run
at once. Passing a [`WorkerPool`](async-and-sync.md#going-out-a-body-that-blocks) bounds it and attributes
the work to the tool's name. A body that hands back an awaitable is awaited, so the result
type is the one the signature promised either way.

`Tool.__call__` is the typed path — `await fare("Osaka", nights=2)` type-checks against the
original signature. `Tool.invoke` is the model-facing path, taking whatever a provider chose
to send, and it is the path that validates it.

## What the model sends is checked before the body runs

Tool arguments are model output, and model output is untrusted input. `invoke` reads the
payload into the tool's own signature first, so the body is entered with the declared types
or not at all:

| The model sent | What happens |
|---|---|
| A field the tool does not declare | `ToolArgumentValidationError` — refused, never dropped |
| A required field left out | `ToolArgumentValidationError` — nothing is filled in |
| `"2"` where an `int` is declared | `ToolArgumentValidationError` under the default policy |
| A nested object where a model is declared | Read into that model, so the body gets the type |
| Arguments as JSON text, or inside an `{"arguments": ...}` envelope | Normalised, then validated |
| Two of the same key | Refused: which one wins is the parser's opinion |
| More bytes than `max_bytes` | Refused before it is parsed |

The failure names every field that failed, not the first — one field per round trip is how a
four-argument call takes four model calls to get right. It never repeats the values: a
rejected argument may be a password or someone's address, and a repair prompt that quotes it
back has copied it into the next request and the provider's logs. `error.payload` keeps what
arrived for a debugger, and the kit never logs it.

```python
try:
    await book_leg.invoke({"lg": {...}, "seats": "two", "class": "first"})
except ToolArgumentValidationError as rejected:
    rejected.paths        # ("class", "leg", "lg", "seats")
    rejected.problems     # {"class": "Extra inputs are not permitted", ...}
    rejected.feedback()   # what can be said back to the model
```

Inside a run, that is what happens: the tool did not run, so the refusal is correctable
rather than fatal. `feedback()` goes back as the tool's result on the agent's
[repair budget](repair.md), the attempt is recorded as `REPAIR_REQUESTED` and counted with
every other repair the run has made, and a model that spends the budget still calling the
tool wrongly fails the run rather than continuing without it. The call is never retried
against the tool: the same payload is the same refusal.

### The coercion policy is declared once

Strict is the default. One provider sends `2` and another `"2"`, and a kit that quietly reads
both makes a tool's contract depend on which vendor answered:

```python
from tesserix_adk.tools import ArgumentPolicy, LENIENT, tool

@tool(arguments=LENIENT)                       # the documented JSON coercions
@tool(arguments=ArgumentPolicy(max_bytes=8192))  # a tighter ceiling
```

`LENIENT` reads `"2"` as an integer and `"yes"` as a boolean — Pydantic's JSON coercions, and
nothing else. It does not relax the unknown-field refusal, the missing-field refusal or the
ceiling. `ToolArgumentValidator` is the same check, usable directly wherever a registry holds
tools it did not build with `@tool`.

## Context is injected, never chosen

A tool that needs the run's tenant asks for it by annotating a parameter `ToolContext`. That
parameter is excluded from `parameters_schema`, so it is not in what the model is told, and
an argument of that name the model sent anyway is a field the tool does not declare —
refused with the rest of the call rather than quietly overwritten:

```python
@tool
async def archive(document: str, ctx: ToolContext) -> str:
    """File a document against the run's tenant."""
    ctx.raise_if_cancelled()
    return f"{document} filed for {ctx.tenant}"
```

`ToolContext` carries `run_id`, `tenant`, `user`, a `trace` mapping and the run's
cancellation token. `ToolContext.current()` reads the ambient the runtime bound, which is
what `invoke` falls back to when no context is passed.

Whether the parameter may be left unfilled is the signature's decision. `ctx: ToolContext`
has no default, so a call outside a run raises `ToolExecutionError` — a guessed tenant is
worse than a refused call. `ctx: ToolContext | None = None` has one, so the tool also works
outside a run and receives `None`.

## What a tool exposes

| Attribute | |
|---|---|
| `name` | What the model calls it |
| `description` | One line, from the docstring summary or the override |
| `parameters_schema` | The model-facing arguments, without the context |
| `returns_schema` | The result, where the function annotates one, awaited |
| `is_async` | Whether the body is a coroutine function |
| `function` | The undecorated callable |
| `context_parameter` / `context_required` | The injected parameter, where there is one |
| `validator` | What `invoke` holds the model's arguments to |
| `timeout` | The ceiling the author declared, where one was declared |
| `parallel_safe` | Whether two calls to it may overlap |

Run [`examples/tools.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/tools.py) for each of these end to end, and
[`examples/tool_arguments.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/tool_arguments.py) for the refusals and the
feedback that goes back to the model.

## The registry is what an agent may call

`ToolRegistry` holds the tools a process has and hands each agent a view of the subset it
may call. A tool registered for one agent is not reachable by every other agent sharing the
process, which is what a filtered dict built at construction cannot promise once a second
agent is added to the same service.

```python
registry = ToolRegistry((fare_for, rooms_in, refund))
planner = registry.view(allow=("fare_for", "rooms_in"), agent="planner")
desk = registry.view(allow=("fare_for", "refund"), agent="desk")

await planner.invoke("refund", {"booking": "AB-1"})  # ToolNotPermittedError, never called
```

An allowlist is resolved once, when the view is made. A name nobody registered fails there,
naming what is registered, rather than at the first call in production; the view is frozen,
so an agent's callable set cannot widen mid-run. A refusal is raised *before* dispatch, not
checked after it — an allowlist enforced after the call has already had its side effect.

The two errors are deliberately different types. `ToolNotFoundError` means nothing is
registered under that name, which is a wiring mistake; `ToolNotPermittedError` means the
tool exists and this agent may not call it, which is a permission decision. Neither is
retried by the run loop even for a tool the agent declared idempotent: asking again gets
the same answer, so a retry only spends the budget.

An empty allowlist raises `ConfigurationError` at construction. An agent with no tools is
either a misconfiguration or an agent that should not have been given a view.

## A tool's ceiling is the author's decision

```python
@tool(timeout=5.0)
async def partner_lookup(reference: str) -> str: ...
```

The ceiling travels with the tool, because the person who wrote the network call knows what
a healthy one costs. A registry may override it per deployment —
`ToolRegistry(tools, timeouts={"partner_lookup": 2.0})` — and the override wins. `timeout=0`
is refused at the line that declared it.

When the ceiling elapses the underlying task is cancelled, not orphaned, and
`ToolTimedOutError` reaches the run loop. A body that swallows cancellation and keeps
running is bounded by a hard abandonment path: the call stops waiting, the span is marked
`abandoned`, and the late result is discarded rather than injected into the run. The run
loop never receives an invented result for a call that timed out.

## How wide the calls may run

`ConcurrencyConfig` bounds the registry as a whole and each tool individually:

```python
ToolRegistry(tools, concurrency=ConcurrencyConfig(max_concurrent_tools=2, per_tool={"refund": 1}))
```

The registry-wide limit is how much the process may do at once; the per-tool limit is how
much one downstream may be asked to take. A tool declared `parallel_safe=False` is held to
one call at a time whatever the config says. Fan-out caps — how many calls one model turn
may make, and how many a run may make in total — are the run loop's, documented in
[`run-loop.md`](run-loop.md), and are refused before any of the turn's calls run.

## What is recorded about a call

Every invocation emits a `ToolCallSpan` to each registered observer:

| Field | |
|---|---|
| `tool` / `agent` | What was called, and by whom |
| `permitted` | The permission decision, recorded for refusals too |
| `outcome` | `ok`, `refused`, `not_found`, `timed_out` or `error` |
| `duration_seconds` | How long the call was waited on |
| `failure` | The class of the failure, never its message |
| `abandoned` | Whether a timed-out body ignored cancellation |

A span carries neither the arguments nor the result. A tool's payload is model output and
its result is business data; both belong in the audit trail the run loop keeps, not in
telemetry that leaves the process. An observer that raises is ignored — a broken exporter
does not fail a tool call.

## Stability

* Additive tool metadata is non-breaking. A new field on `Tool` or `ToolCallSpan` does not
  require a major version.
* Allowlist semantics are versioned: what `allow=` means, and what a view resolves to, may
  only change with a documented version bump.
* Any change to default-deny — a view permitting anything it was not given — is a major
  version. There is no configuration that turns it off.

Run [`examples/tool_registry.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/tool_registry.py) for allowlists, refusals,
ceilings and spans end to end.
