# State and queue adapters

`RedisStateStore` and `RedisWorkQueue` are the `StateStore` and `WorkQueue` of
[state.md](state.md) and [work-queue.md](work-queue.md), backed by a server that outlives
the process. Both conformance suites pass against them unchanged, which is the point: a
run survives a rolled pod without the loop knowing where its state lives.

```python
settings = RedisStoreSettings(dsn=SecretStr(os.environ["ADK_REDIS_DSN"]))

state = await RedisStateStore.open(redis, clock=clock, settings=settings)
queue = await RedisWorkQueue.open(redis, clock=clock, settings=settings, policy=policy)
```

`open()` runs `preflight()`. Constructing directly skips it, which is what the tests do.

## The server has to be a database

`preflight()` refuses an instance that would lose a run:

| Setting | Refused when | Because |
|---|---|---|
| `maxmemory-policy` | any `allkeys-*` or `volatile-ttl` | eviction drops a run mid-flight, and the next read says it never existed |
| `appendonly` / `save` | both off | a restart is indistinguishable from every run finishing at once |

Set `durable=False` for a queue whose work can be recreated elsewhere, and for dev. State
in production sets neither.

## What is stored where

A run is a JSON blob plus a small hash. The blob is what a worker last wrote. The hash
holds the version and the five accumulating numbers — input tokens, output tokens, cost,
iterations, cursor — so `patch_run` is `HINCRBY` and two workers patching at once both
land. A patch bumps the version, so a `put_run` holding a version the hash has moved past
is refused with `StateConflictError` carrying both numbers rather than overwriting.

Keys are length-prefixed per segment:

```
adk:state:{4:acme}:run:5:run_1
adk:state:{4:acme}:session:2:s1
```

Prefixed rather than escaped, so a tenant called `a:b` and a tenant called `a` holding an
id that starts `b:` cannot build the same key. The tenant is the cluster hash tag: a run,
its counters and its session index are one slot, so the scripts stay atomic under cluster.

The queue tags on its whole namespace (`{adk:queue}`) instead, because a claim reads
several tenants' ready sets in one script to take turns between them. **A queue therefore
lives in one slot.** Shard by running more than one namespace, not by hoping.

## Leases

The lease clock is the injected `Clock`, not the server's `TIME`. Each worker brings its
own, so a deployment runs them off NTP or accepts leases that lapse early on a drifting
node — the trade for a conformance suite that can move time.

A claim is two steps: one script moves the item out of its tenant's ready set and into the
lease set, and the item's own record is written immediately after. A crash in between
leaves an item that is leased and still says it is queued; the next `reap` corrects it.
Every settle is fenced on the lease score it saw, so a reaper that lost a race to a live
heartbeat writes nothing.

Retries, backoff and the dead letter are `QueuePolicy`'s decisions, shared with the
in-process queue. Two stores that disagree about when an item is poisonous are two
deployments with different retry semantics and one set of tests.

## Size

`max_value_bytes` (256 KiB) is checked before anything is sent. A record over it raises
`StatePersistenceError(reason="too_large")` — hold a run that size in the PostgreSQL state
store, and put a payload that size behind a claim check ([claim-check.md](claim-check.md)).
The refusal is not retried; an oversized item is not a bad connection.

## When the server is unreachable

A failover is waited out — bounded attempts, doubling with jitter, because every worker
that started together retries together. Past the budget, `StatePersistenceError` or
`QueueUnavailableError`, both retryable, and the run fails closed. An exhausted pool is
reported at once as `PoolExhaustedError`: the endpoint is fine and the process is
over-subscribed.

## Not here

**A transactional outbox.** Writing state and enqueueing work are two round trips with no
shared transaction. Enqueue after the state write and make the consumer idempotent, or use
the PostgreSQL adapters where both are one transaction.

**Schema or config management.** `preflight()` reads the configuration and refuses; it
never sets it. Importing a library must not reconfigure a production server.

## Verifying against a real server

```bash
ADK_TEST_REDIS_URL=redis://localhost:6379/9 uv run pytest tests/integration -m integration
```

The default lane excludes them and reaches no network.
