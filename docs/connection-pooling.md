# Connection pooling

A provider built per run pays a DNS lookup and a TLS handshake for every turn, and under
load the sockets outlive the runs that opened them until the process runs out of
descriptors. `ClientPool` is where connections live instead: providers borrow from it,
and it decides what may be shared with what.

```python
from tesserix_adk.models import ClientPool, PoolConfig
from tesserix_adk.models.providers import AnthropicProvider, OpenAIProvider

async with ClientPool(PoolConfig(max_connections=50, max_keepalive=10)) as clients:
    fast = OpenAIProvider("gpt-4o", pool=clients)
    careful = AnthropicProvider("claude-sonnet-5", pool=clients)
    ...
```

A provider given no pool still owns its own client, so nothing about the single-provider
case changes.

## What may share a connection

Sharing is decided by a key, and the key is the whole safety argument:

| Part of the key | Why it is part of it |
|---|---|
| provider | Two vendors are two endpoints. |
| base URL | A regional host or a gateway is a different pool. |
| credential digest | Two tenants against one endpoint must never be handed each other's connection. |
| transport settings | Timeouts and limits, since one client honours one set of them. |

The credential appears as a truncated digest, never as itself. A key is compared, logged
and used as a metric label, and a secret that reaches any of those has leaked.

## Rotation

The credential is resolved per request, not captured at construction. When it changes, the
key changes: the next request opens a pool on the new key, and the old client is *retired*
rather than closed — it stops being handed out, and it closes once the requests already on
it have finished. Nothing in flight is cut off, and nothing new goes out on the old key.

## Exhaustion is bounded

`acquire_seconds` is how long a caller waits for a free connection. When none comes free
the request fails with `PoolExhaustedError`, which is retryable: the endpoint is fine and
the process is over-subscribed. Bounding it is the point — an unbounded wait for a
connection turns a downstream slowdown into a run that queues past its own deadline.

## Forks

A pool inherited across a `fork` is half-open whatever its bookkeeping says: the
descriptors belong to the parent's event loop. Each client records the pid that opened it,
and a client from another process is discarded rather than used. The `inherited` counter
says how often that happened.

## Metrics

`pool.metrics` is a snapshot, taken rather than watched, so a reader that prints four
counters prints four counters from one moment:

| Counter | Meaning |
|---|---|
| `opened` | Clients created. |
| `reused` | Times an existing client was handed back instead. |
| `retired` | Clients replaced by rotation, whose in-flight work was left to finish. |
| `inherited` | Clients discarded because they came from another process. |
| `exhaustions` | Requests that failed because no connection came free in time. |
| `open_now` | Clients currently held. |
| `waited_seconds` | Total time spent inside provider requests waiting on the pool. |

## Lifetime

The pool is an async context manager and closing it closes everything it opened, because a
pool nobody closes is the leak this exists to fix. A provider's own `aclose()` closes only
a client it owns; one borrowed from a pool is the pool's to close.

Per-vendor ceilings go in `per_provider`, and an entry there replaces the defaults for that
provider entirely:

```python
PoolConfig(per_provider={"anthropic": PoolConfig(max_connections=8, max_keepalive=4)})
```

Runnable version: [`examples/connection_pooling.py`](../examples/connection_pooling.py).
