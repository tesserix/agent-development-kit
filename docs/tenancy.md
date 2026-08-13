# Tenancy

Tenant identity is a property of the execution context, not of developer discipline.

A tenant threaded through call sites arrives on most of them. The one it misses is a
background task, a thread offload, or a helper somebody wrote in a hurry, and the symptom
is a query with no tenant filter that nothing in the type system notices and nothing at
runtime refuses. This page is about removing the call site that had to remember.

## The shape

```python
from tesserix_adk.core import TenantContext, current_tenant, tenant_scope

with tenant_scope(TenantContext(tenant="acme", user="ada")):
    ...                       # everything below reads the tenant from here
    current_tenant().tenant   # 'acme'
```

`TenantContext` is frozen: `tenant`, `user`, and the optional `locale`, `region`,
`correlation_id` and `crossing`. A context a helper can edit is a context one helper can
rewrite for everything running below it. `acting_as` returns a new one.

`tenant_scope` binds for the duration of the block and restores what was there before,
including when the block leaves by an exception. `current_tenant()` is what egress reads;
`tenant_here()` returns `None` instead of raising, for code that legitimately runs outside
a run — a health check, a corpus job — and has to branch on that.

## Absence is a refusal, never a default

```python
current_tenant(where="memory.recall")   # MissingTenantContextError
```

There is no default tenant and no "all tenants", because a default is one typo away from
being every tenant. `where` names the egress point, so the refusal says what was about to
happen rather than only which accessor raised.

## A run binds it

`AgentRunner.run`, `resume` and `resume_with_decision` bind the tenant they were given for
the whole run. A tool body reads it with no argument:

```python
@tool
async def looking_up(what: str) -> str:
    """Read the tenant from the context rather than from an argument."""
    return f"{what} for {current_tenant(where='looking_up').tenant}"
```

Two runs executing concurrently cannot observe each other's context whatever the scheduler
does — the binding is a `contextvars.ContextVar`, so `await`, `asyncio.gather`,
`TaskGroup`, `asyncio.to_thread` and `create_task` all carry it, and a task started under
one tenant keeps that tenant when it is resumed.

A handler that has bound tenant A and then starts a run for tenant B is refused with
`TenantCrossingError` rather than quietly rebinding.

## Crossing tenants has to be said out loud

An administrative operation across tenants is legitimate. One nobody declared is the
incident, so the declaration is required rather than assumed:

```python
with tenant_scope("acme"):
    with tenant_scope("globex", crossing="registry backfill"):
        current_tenant().crossing   # 'registry backfill'
```

The reason is recorded on the bound context, so anything auditing below the crossing can
see it was deliberate and why.

## Executors that copy no context

`loop.run_in_executor` and a bare `ThreadPoolExecutor.submit` do not copy contextvars, and
a body landing on a pool thread with whatever the previous job left bound is the failure
this module exists to prevent. `bound` takes a snapshot now for a call that runs later:

```python
await loop.run_in_executor(pool, bound(lambda: current_tenant().tenant))
```

`asyncio.to_thread`, `create_task` and `TaskGroup` copy the context themselves and need
nothing.

## Egress

`MemoryScope.here()` builds a scope from the bound context, defaulting `user_id` to the
acting principal:

```python
with tenant_scope(TenantContext(tenant="acme", user="ada")):
    MemoryScope.here(session_id="s-1")   # tenant_id='acme', user_id='ada'
```

Outside a scope it raises rather than widening: an unscoped recall reads every tenant's
memory and looks like an answer.

## Known limitations

- **Store types keep their explicit tenant arguments.** `MemoryScope` still carries its
  tenant as a value somebody can read in a diff, and the store protocols still take a
  scope. Removing the argument would move the tenant out of the record as well as out of
  the call, and a stored memory whose tenant is only implicit cannot be audited after the
  fact. What `here()` removes is the call site that had to decide which tenant to put
  there, which is the call site that forgets.

- **An async generator holds its binding between yields.** A scope entered inside a
  generator is still bound in whoever resumes it, because a generator is resumed on the
  caller's stack. That cannot be prevented by contextvars alone; the crossing rule turns
  it from a silent mis-scope into a loud `TenantCrossingError`. A generator resumed
  outside the scope that created it reads no tenant rather than a stale one.

- **Cross-process propagation is not this.** Carrying identity over MCP, A2A or a workflow
  boundary is a separate concern — see [`multi-agent-trace.md`](multi-agent-trace.md) for
  what crosses a process today.
