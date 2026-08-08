# Memory adapters

Three stores, each good at one thing, behind the one `MemoryStore` a consumer binds.

| Kind | Store | Why that one |
|---|---|---|
| working | `RedisMemoryStore` | Expiry is the server's job, not a sweeper's |
| profile, episodic | `PostgresMemoryStore` | A history nobody may overwrite |
| semantic | `PgvectorMemoryStore` | A ranking belongs in the query planner |

```python
settings = MemoryStoreSettings(dsn=SecretStr(os.environ["ADK_POSTGRES_DSN"]))

store = RoutedMemoryStore(
    working=RedisMemoryStore(redis, clock=clock, settings=settings, ttl_seconds=3600),
    durable=PostgresMemoryStore(sql, clock=clock, settings=settings),
    semantic=PgvectorMemoryStore(sql, clock=clock, settings=settings, dimensions=1024),
)
```

Each store declares what it can do. A plan that needs ranking cannot be bound to a
key-value store and find out a month later, because `require_memory` checks first.

## Credentials

`MemoryStoreSettings` is the only place a connection string comes from. It is never read
from a row, and nothing here has a value that works out of the box:

```python
MemoryStoreSettings(dsn=SecretStr(""))                              # ValueError
MemoryStoreSettings(dsn=SecretStr("postgres://postgres:postgres@db/adk"))  # ValueError
```

The DSN is a `SecretStr`, so a settings object in a log or a traceback carries
`SecretStr('**********')` and not the password.

## Working memory

The key carries the whole scope — `adk:mem:<tenant>:<user>:<session>:<agent>:working:<key>` —
so two sessions under one tenant are two scratch spaces. `sliding=True` extends a key on
read, for a conversation that is still in progress rather than idle.

`append` is one server-side script. A client-side read-modify-write is two chances to lose
the other append, and the position it returns is what lets a caller notice that eviction
under `maxmemory` took the sequence: an append that says it is the first is a lost one. A
key that is gone reads as `None`, because absent is what it is.

## Profiles and episodes

Nothing is overwritten. `supersede` closes the old version in the `UPDATE`'s own predicate,
so two writers reading version 1 cannot both succeed — the loser gets `MemoryConflictError`
with the version that is actually live. `resolves=` closes named live records in the same
write, so a branch ends because somebody decided rather than because a read picked a side.

`log` inserts `ON CONFLICT (id) DO NOTHING`. A primary that fails over mid-append is
retried, and the retry commits the episode once rather than booking it twice.

A window wide enough to matter is wider than one response:

```python
page = await store.page(scope, MemoryQuery(kind=MemoryKind.EPISODIC, limit=200))
while page.cursor:
    page = await store.page(scope, query, cursor=page.cursor)
```

Keyset, not `OFFSET`: an offset re-reads every row before it in order to skip them.

## Semantic recall

The scope filter is part of the SQL predicate. A filter applied after the fetch has already
read the rows it was supposed to exclude, which for a tenant boundary is the whole point.

`metric` picks the operator — `cosine` `<=>`, `l2` `<->`, `inner` `<#>` — and must match
what the index was built for. Scores are `1 - distance`, comparable within one result set
and not across two.

```python
await store.verify()   # at startup, not on the first bad ranking
```

A collection two dimensions narrower than the embedder does not fail. It ranks badly, on
the first recall, a month later, with nobody watching. `verify()` turns that into an
`EmbeddingDimensionError` while somebody is still deploying.

## When the store is unreachable

A failover is ordinary and is waited out — bounded attempts, doubling with jitter. A store
still gone once the budget is spent raises `MemoryUnavailableError` and the run fails
closed, because an agent that silently remembers nothing looks exactly like one whose user
said nothing.

An exhausted connection pool is different and is reported at once as `PoolExhaustedError`.
The endpoint is fine and the process is over-subscribed; retrying into a full pool under a
tool fan-out is how a spike becomes an outage.

## Not here

**Schema DDL.** Tables, indexes and the `vector` extension belong to the platform's
migration repo. Importing a library must never be the thing that alters a production table.

**Erasure narrowing.** `RoutedMemoryStore.erase` refuses `kinds=` and `dry_run=`: three
servers with no shared transaction cannot promise either honestly. Use the individual
stores, or `InMemoryMemoryStore` where the two-phase protocol of `docs/erasure.md` applies.

**The knowledge-graph adapter.** A different story.

## Verifying against real servers

The unit tests check what the adapters send. Whether it is valid Lua and valid SQL is
something no fake can tell you, so the conformance suite also runs against containers:

```bash
ADK_TEST_REDIS_URL=redis://localhost:6379/9 \
ADK_TEST_POSTGRES_DSN=postgresql://adk:...@localhost/adk_test \
uv run pytest tests/integration -m integration
```

The default lane excludes them and reaches no network.
