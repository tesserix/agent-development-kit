# Tool idempotency

A tool call that times out has not necessarily failed. The seat may be booked, the mail may be
sent, the row may be written — the runtime only knows that nothing came back. Retrying is a
second booking; not retrying leaves the run stuck. Neither is a decision the kit can make from
the exception alone, so the tool declares what repeating it would do and the dispatcher holds
a record of what already happened.

## Declaring it

```python
from tesserix_adk.core import Idempotency, IdempotencyPolicy
from tesserix_adk.tools import tool


@tool(idempotency=IdempotencyPolicy(Idempotency.READ_ONLY))
async def lookup_fare(flight: str) -> str:
    """Read the current fare.

    Args:
        flight: Which flight.
    """
    ...


@tool(idempotency=IdempotencyPolicy(Idempotency.EFFECTFUL, key_arguments=("flight",)))
async def book(flight: str, request_id: str) -> str:
    """Take a seat.

    Args:
        flight: Which flight.
        request_id: Fresh on every attempt, which is why it is not a key argument.
    """
    ...
```

| Kind | What it means | Retried on failure | Deduplicated |
|---|---|---|---|
| `READ_ONLY` | Reads, changes nothing | Yes | No |
| `IDEMPOTENT` | Writes, but repeating lands the same state | Yes | Yes |
| `EFFECTFUL` | Repeating it is a second effect | No — see below | Yes |

`key_arguments` names the arguments that identify the effect. Everything else — a request id, a
trace header, a timestamp the model regenerated — is excluded, so a retry that renumbers itself
is still one key. A name that is not a parameter of the tool is refused at decoration rather
than at the call that matters.

A tool with no `idempotency` behaves exactly as it did before this existed. Nothing is claimed,
nothing is recorded, and the retry policy is unchanged.

## The key

```python
key = idempotency_key(
    tenant="acme",
    run_id="run_1",
    tool="book",
    arguments={"flight": "BA117", "request_id": "a"},
    key_arguments=("flight",),
)
```

A SHA-256 over a canonical encoding of the tenant, the run, the tool name and the named
arguments. Canonical means: keys sorted, strings NFC-normalised, floats formatted so `2` and
`2.0` agree, booleans kept distinct from the integers they would otherwise collapse into. Two
spellings of one payload are one key.

The arguments are hashed, never stored. An idempotency record holds a digest, a tenant and an
outcome string — it is not a second copy of the payload, and `forget(tenant=...)` erases a
tenant's records outright.

The tool receives the key on `ToolContext.idempotency_key`, which is what you pass to a
downstream that has an `Idempotency-Key` header of its own. Where no key can be derived — a
named key argument the model did not send — it is `None`.

**The call id is deliberately not part of the key.** The issue's scope named it; including it
would give the two concurrent identical calls in one turn two different keys, and both would
fire. Excluding it is what makes concurrent duplicates collapse.

## The store

```python
runner = AgentRunner(
    provider=provider,
    tools=tools,
    idempotency=RedisIdempotencyStore(client, clock=clock),
    idempotency_ttl_seconds=86_400,
)
```

`MemoryIdempotencyStore` is enough for one replica and for tests. `RedisIdempotencyStore` and
`PostgresIdempotencyStore` survive a restart and are visible to every worker; both claim a key
in a single server-side operation, because a read followed by a write is a window another
replica fits through. `PostgresIdempotencyStore.ensure_schema()` is a deployment's to call.

Anything satisfying the `IdempotencyStore` protocol works. `IdempotencyStoreConformance` in
`tesserix_adk.testing` is the suite your own implementation has to pass.

## What the dispatcher does

Before the body runs, the dispatcher claims the key. If someone already recorded an outcome for
it, that outcome is returned to the agent and the body is not entered — a `TOOL_DEDUPLICATED`
event says so. If another caller holds the claim, this one waits for their answer rather than
executing alongside them.

When the body returns, its result is recorded against the key. When it fails, what happens next
depends on the declaration:

- `READ_ONLY` and `IDEMPOTENT` — the claim is released and the normal retry policy applies.
- `EFFECTFUL` — the claim is **kept**, no retry is attempted, and the run fails with
  `IndeterminateOutcomeError` naming the tool. The effect may have landed; the kit will not
  find out by doing it again.

The same applies when the key cannot be derived, or when the store cannot be reached. A store
that is down does not read as permission.

## The guarantee

**At most one side effect per key within the retention window.** This is versioned public API:
it will not weaken without a major version and a changelog entry saying so.

Read what it does not say. Not *exactly* once — a call that fails without answering leaves the
effect unknown, and the kit reports that rather than guessing. Not *ever* — a retry that arrives
after the retention window has expired sees a free key, so choose a window longer than the
longest replay you intend to support (the default is a day). And not *across tenants* — records
are tenant-scoped, which is the isolation, not a limitation of it.

The honest failure is `IndeterminateOutcomeError`: the run stops, a human or an approval path
decides, and nothing is booked twice while they do.

## Where it shows up

| Signal | Meaning |
|---|---|
| `RunEventKind.TOOL_DEDUPLICATED` | A recorded outcome answered this call |
| `RunEventKind.TOOL_INDETERMINATE` | An effect whose outcome nobody can state |
| `IndeterminateOutcomeError` | The typed error the run fails with |

`examples/tool_idempotency.py` runs all three.
