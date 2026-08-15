# Acting for a caller

The service account an agent runs on holds the union of everything any user of the product
might need, because that is the account somebody could get provisioned. A low-privilege
conversation then executes against it. Nothing in the code says the agent is acting for
that customer, so nothing refuses the tool call that reads every tenant's bookings.

Here that statement is a value, and the runtime enforces it:

```python
caller = Principal(subject="ada", tenant="acme", scopes=frozenset({"bookings:read"}))
desk = Agent(
    name="desk",
    scopes=("bookings:read", "bookings:write"),
    tool_scopes={"search": ("bookings:read",), "refund": ("bookings:write",)},
    ...,
)

with principal_scope(caller):
    run = await runner.run(desk, "refund my fare", tenant="acme")
```

The run holds `bookings:read` and nothing else. `refund` is never declared to the model,
and a call to it is refused before it is dispatched.

## The effective set is an intersection, computed once

`AgentIdentity.resolve` intersects what the agent declares with what the principal holds,
at run start, and the result is frozen for the run. Nothing in this module widens a scope
set — there is no `union`, and `narrowed_to` is the only combinator — so "what could this
run reach" is one value somebody can read rather than a history of settings.

Scope names are compared in one normal form. `Bookings:Read` from one product and
`bookings:read` from another is otherwise an intersection that silently empties, and an
agent refused everything for a reason nobody can see in a diff.

## Absence fails closed

An agent that declares `scopes` and finds no principal bound raises `AuthorisationError`
at construction. The kit never falls back to the process identity to keep the run going,
because that fallback is the escalation. A principal whose authority has already lapsed is
refused the same way, and one that lapses mid-run is refused at the next dispatch: a run
must not outlive the session that authorised it.

An agent that declares no scopes is unchanged. It holds no authority to spend, and every
run written before this existed behaves exactly as it did.

## Refusals name the scope

```
'desk' acting for 'ada' does not hold 'bookings:write' (refund); it holds bookings:read
```

`AuthorisationError` carries `scope`, `agent`, `subject` and `where`. A generic permission
failure read back from a downstream service sends somebody to the wrong system to fix it —
and by then the request has already been made with somebody's credentials.

## Delegation only narrows

```python
peer = identity.narrowed(agent="billing", declared=("bookings:read", "bookings:write"))
```

A peer's effective set is its own declaration intersected with the caller's effective set,
so an agent holding more than its caller cannot be invoked as a way around the refusal.
The identity carries the `chain` it was narrowed through, and narrowing again from a
narrowed identity cannot recover what was cut.

## A system-initiated run has a principal too

"No user" is not an absence of authority. A scheduled job is a `Principal` with
`AuthMethod.SCHEDULED_JOB` holding exactly the scopes somebody granted that job — which is
a thing a reviewer can read and trim, unlike a run with no caller at all.

A durable workflow resuming hours later runs on a recorded delegation with its own expiry:

```python
later = caller.delegating(until=clock.now() + 3600, scopes=("bookings:read",))
```

`until` is required rather than inherited, because a continuation that silently keeps the
original session's authority is the long-lived credential this exists to avoid.

## Trimming a declaration

`AgentIdentity.unused` names declared scopes that nothing reached for over a set of runs,
so a declaration written defensively can be cut back with evidence rather than by nerve.

## Known limitations

- Expiry is in clock seconds, the same currency the runner's deadlines use, not a
  wall-clock date. A consumer holding a token expiry as a datetime converts it once at the
  boundary.
- Scopes are opaque strings. There is no hierarchy: holding `bookings` does not imply
  `bookings:read`, because an implication rule is a place for an unintended grant to hide.

## Related

- [`docs/tool-allowlists.md`](tool-allowlists.md) — which tools exist for this run at all
- [`docs/tenancy.md`](tenancy.md) — the tenant the principal acts within
- [`examples/agent_identity.py`](../examples/agent_identity.py)
