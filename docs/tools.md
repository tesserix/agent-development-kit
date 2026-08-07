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
original signature. `Tool.invoke` is the model-facing path, taking the mapping a provider
chose. `invoke` does not validate that mapping against `parameters_schema`: validation is a
separate step, and doing it inside `invoke` would put the check where a caller could skip it
by calling the function directly.

## Context is injected, never chosen

A tool that needs the run's tenant asks for it by annotating a parameter `ToolContext`. That
parameter is excluded from `parameters_schema`, so it is not in what the model is told, and
`invoke` overwrites any argument of the same name the model sent anyway:

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

Run [`examples/tools.py`](../examples/tools.py) for each of these end to end.
