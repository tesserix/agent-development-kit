# Carrying the tenant across a hop

[`docs/tenancy.md`](tenancy.md) covers the tenant in-process: a contextvar, bound at a
boundary, read by everything below. This page covers what happens when the work leaves the
process — to a broker, a peer, a tool server, a durable workflow — where a contextvar is
worth nothing and the tenant is whatever the producer put in the message.

Left to itself every integration invents its own field. One transport says `x-tenant`,
a workflow input says `tenantId`, a tool's metadata says `org`. A shared service cannot
honour three spellings, and the one it does not know about is the one that silently runs
under the consuming worker's own tenant. So there is one contract.

## The contract

| | |
|---|---|
| Header | `adk-tenant`, lower-case |
| Payload key | `adk_tenant`, for carriers whose message is a flat string map |
| Version | `adk/1`, leading the value |
| Encoding | `adk/1 tenant=acme;user=ada;locale=en-GB`, values percent-encoded |
| Ceiling | 1024 bytes; optional fields are shed to fit and the result marked `partial` |

Percent-encoding is not decoration: without it a `user` containing `;tenant=globex`
rewrites the field beside it.

## Sending

```python
from tesserix_adk.core import carried, current_tenant

await client.post(url, headers={**headers, **carried(current_tenant())})
```

For work whose message *is* its input rather than a request with headers — a queued item,
a workflow argument — the same context goes in the payload:

```python
from tesserix_adk.core import current_tenant, in_payload

await queue.enqueue(
    WorkItem(id=job_id, tenant=tenant, payload=in_payload(current_tenant(), {"booking": "AB-1"}))
)
```

`WorkItem.tenant` already carries the tenant for the queue's own isolation. The payload
carries the *rest* of the context — the acting principal, the locale, the correlation id —
which is what an audit entry on the far side needs and what the queue's own field cannot
express.

## Receiving

```python
from tesserix_adk.core import arriving

async def handle(message):
    with arriving(message.headers, authenticated=message.peer_tenant):
        await do_the_work()          # reads the producer's tenant from the context
```

`arriving` refuses before it binds anything. `restored` and `of_payload` do the reading
without binding, for a consumer that needs the context as a value.

## What is refused, and why

Every refusal is a `TenantContextError` carrying a `reason`, so a consumer can branch on a
value rather than on message text.

| `reason` | When | Why not something softer |
|---|---|---|
| `missing` | No header, no payload key | The consumer's own tenant is not a default. A default is one typo away from being every tenant. |
| `malformed` | Unreadable, or no tenant named | Half-reading a context yields work attributed to nobody. |
| `version` | A version this build does not know | Reading unknown fields by position is how a tenant becomes a locale. Refuse, and let the deploy skew be visible. |
| `contradicted` | The context names a different tenant than the caller authenticated as | The payload never outranks the credential. Overriding silently would hide the finding; the disagreement *is* the finding. |
| `oversized` | Even the tenant alone exceeds the ceiling | Truncating is not an option — half a tenant name is a different tenant. |

`missing` and `malformed` are dead-letter cases. `contradicted` is an authorization event
worth alerting on. `version` is a deploy-skew signal.

A consumer that already holds a different tenant gets `TenantCrossingError` from
`tenant_scope`, not a silent rebind: a worker with its own tenant bound picking up
another's work is the exact failure this exists to catch.

## Durable and delayed work

The context is recorded **as input**, never taken from ambient state. That is what makes a
replay deterministic: a workflow replayed on another worker a week later reconstructs the
tenant it started under instead of inheriting the replayer's. Redelivery after a retry or
out of a dead-letter queue reads the same bytes and lands on the same tenant.

A batched or multiplexed frame carrying several tenants is several messages. Bind each one
separately; one scope over the batch is one tenant's data leaking into another's handler.

## Transports with a header ceiling

Where the context does not fit, optional fields are dropped least-load-bearing first —
`crossing`, `correlation_id`, `region`, `locale`, then `user` — and the result carries
`partial=1`. The far side sees `context.partial is True` and can tell a field that was
absent from one that was lost. The tenant is never shed.

## Proving your transport carries it

```python
from tesserix_adk.testing import TenantPropagationConformance


class TestNatsCarriesTheTenant(TenantPropagationConformance):
    def round_trip(self, headers):
        return published_and_received(headers)
```

The suite checks that the whole context survives, that header case is not load-bearing,
that a ceiling still delivers an intact tenant, and that consecutive messages on one
connection do not bleed into each other.

## Known limitations

- **Only the carriers that exist are wired.** `tesserix_adk.mcp`, `tesserix_adk.a2a` and
  `tesserix_adk.workflows` are still placeholder modules, so there is no MCP metadata
  wrapper, no A2A client middleware and no Temporal interceptor yet. The contract, the
  payload carrier and the conformance suite are here; each integration adopts them when
  that module lands, which is why the encoding is transport-agnostic.
- **Egress is explicit.** `carried(current_tenant())` is a call the producer makes. A
  transport that forgets it produces a message the consumer refuses — loudly, at the far
  end, rather than quietly at the near one.
- **The ceiling applies to payloads too.** `in_payload` uses the same encoder, so a payload
  sheds fields it did not need to. One encoding is worth more than the bytes.
