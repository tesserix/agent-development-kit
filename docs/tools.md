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

Run [`examples/tools.py`](../examples/tools.py) for each of these end to end, and
[`examples/tool_arguments.py`](../examples/tool_arguments.py) for the refusals and the
feedback that goes back to the model.
