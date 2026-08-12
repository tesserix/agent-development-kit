# State and queue adapters

Two backings for the same two protocols. Pick Redis for latency and PostgreSQL when the
state change and the work it queues must commit together; the loop above them cannot tell
which it got.

| | Redis | PostgreSQL |
|---|---|---|
| Claim | Lua script, tenant ready sets | `FOR UPDATE ... SKIP LOCKED` |
| State and queue in one transaction | no | yes, via `bound()` |
| Fairness | rotation, whole queue in one slot | rotation, one row per tenant |
| Operational cost | eviction and persistence must be right | vacuum must keep up |

## Redis

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

### The server has to be a database

`preflight()` refuses an instance that would lose a run:

| Setting | Refused when | Because |
|---|---|---|
| `maxmemory-policy` | any `allkeys-*` or `volatile-ttl` | eviction drops a run mid-flight, and the next read says it never existed |
| `appendonly` / `save` | both off | a restart is indistinguishable from every run finishing at once |

Set `durable=False` for a queue whose work can be recreated elsewhere, and for dev. State
in production sets neither.

### What is stored where

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

### Leases

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

### Size

`max_value_bytes` (256 KiB) is checked before anything is sent. A record over it raises
`StatePersistenceError(reason="too_large")` — hold a run that size in the PostgreSQL state
store, and put a payload that size behind a claim check ([claim-check.md](claim-check.md)).
The refusal is not retried; an oversized item is not a bad connection.

### When the server is unreachable

A failover is waited out — bounded attempts, doubling with jitter, because every worker
that started together retries together. Past the budget, `StatePersistenceError` or
`QueueUnavailableError`, both retryable, and the run fails closed. An exhausted pool is
reported at once as `PoolExhaustedError`: the endpoint is fine and the process is
over-subscribed.

## PostgreSQL

`PostgresStateStore` and `PostgresWorkQueue` pass the same two conformance suites over a
database the deployment already runs. Anything with an async `fetch(statement, *args)` is
an executor — psycopg, asyncpg behind a two-line shim, or a pooled wrapper.

```python
settings = PostgresStoreSettings(dsn=SecretStr(os.environ["ADK_POSTGRES_DSN"]))

state = await PostgresStateStore.open(sql, clock=clock, settings=settings)
queue = await PostgresWorkQueue.open(sql, clock=clock, settings=settings, policy=policy)
```

`open()` runs `verify()`, which reads `adk_schema` and refuses a version this release was
not written for, and refuses a connection with no `statement_timeout` — a statement that
can run forever holds a pooled connection until the process dies.

### The schema is the deployment's

`EXPECTED_SCHEMA` is the shape the adapters read, published so a migration can own it. The
kit never applies it, and never alters a table. Table names come from `StateTables`, which
refuses anything that is not a plain identifier: a name is interpolated, not bound, because
a placeholder cannot name a relation.

The counters live in columns rather than the blob, so `patch_run` is `SET input_tokens =
input_tokens + $3` and two patches arriving together both land. `seq` is a `bigserial`, and
paging is by the sequence the database gave the row — nothing rewrites it, so a listing
cannot skip a run that was updated while the page was being read.

### One transaction for both

`bound(session)` returns a store or queue that runs inside a transaction the caller already
opened:

```python
async with connection.transaction():
    await state.bound(session).put_run(record)
    await queue.bound(session).enqueue(item)
```

The run and the work it queued commit together or neither does, which is the whole reason
to put both in one database. A bound adapter retries nothing: a failed statement has poisoned
the caller's transaction, and retrying inside it would only fail again.

### Claiming

A claim locks one row with `FOR UPDATE ... SKIP LOCKED`, so a worker arriving while a row is
locked steps over it rather than waiting behind it or taking it twice. Every predicate is
repeated on the locked relation. That is not redundancy: under `READ COMMITTED` PostgreSQL
re-checks the predicate against the updated row when a concurrent claim commits underneath,
and a predicate it cannot re-check — one buried in a window-function subquery, say — is a row
two workers both take.

Tenants are served in rotation. `adk_queue_turns` holds one row per tenant per queue; the
claim orders by the turn each tenant was last given and sends the served tenant to the back,
so a tenant with a backlog cannot starve the tenant beside it.

### Vacuum

`adk_work` is rewritten on every claim, heartbeat and settle, so it accumulates dead tuples
faster than any other table here. The defaults vacuum it far too rarely to keep the due-index
tight, and a bloated index turns a claim from an index scan into a sequential one under load.
`EXPECTED_SCHEMA` sets `autovacuum_vacuum_scale_factor = 0.01` and
`autovacuum_vacuum_cost_delay = 0` on it; keep those if the migration is rewritten.

Connections are the other limit. A claim is short but a worker holds one per in-flight item;
size the pool to the worker count, not to the queue depth. An exhausted pool is reported at
once as `PoolExhaustedError` rather than retried into.

## Not here

**A transactional outbox in Redis.** There, writing state and enqueueing work are two round
trips with no shared transaction: enqueue after the state write and make the consumer
idempotent, or use the PostgreSQL adapters where both are one transaction.

**Schema or config management.** `preflight()` and `verify()` read and refuse; they never
set. Importing a library must not reconfigure a production server or migrate its tables.

## Verifying against a real server

```bash
ADK_TEST_REDIS_URL=redis://localhost:6379/9 \
ADK_TEST_POSTGRES_DSN=postgresql://adk@localhost:5432/adk_test \
    uv run pytest tests/integration -m integration
```

The default lane excludes them and reaches no network. The PostgreSQL suite runs both
conformance suites plus what only a real database shows: ten workers claiming at once take
ten different items, and a transaction that fails leaves neither the run nor its work behind.
