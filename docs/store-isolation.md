# Isolation the calling code cannot forget

A store scoped by an argument is scoped by whoever remembers the argument. One missing
prefix, one dropped `WHERE`, and the query still runs — it just runs over everybody.
The worst case is a vector search, because an unfiltered similarity search does not
fail: it returns a neighbour's most relevant content, and it reads as a good answer.

`tesserix_adk.adapters.isolation` moves the tenant out of the call signature. The
partition is derived from the bound context at the moment of the call, so an unscoped
read is not something a caller can express.

## Deriving the partition

```python
from tesserix_adk.adapters import Partition
from tesserix_adk.core import tenant_scope

with tenant_scope("acme"):
    partition = Partition.current(where="pgvector.search")
    key = partition.key("adk:cache", digest)          # adk:cache:acme:<digest>
    clause, value = partition.predicate(position=1)   # ("tenant = $1", "acme")
```

`Partition` has no constructor a caller passes a tenant to, and `current()` reads
`current_tenant()`. Outside a scope it raises `MissingTenantContextError` — absence is
never read as every tenant and never filled in with a default. See
[`docs/tenancy.md`](tenancy.md) for how the context gets bound.

## Reads are checked, not just filtered

The predicate is the boundary, but it is not the only check. Every record that comes
back is verified against the bound tenant:

```python
rows = partition.only(rows, tenant_of=lambda row: row["tenant"], where="postgres.read")
```

A record under the wrong tenant raises `TenantIsolationError` and nothing is returned —
not the offending row, not the rest of the page. Dropping it quietly would be the worse
option: the row is evidence of a botched backfill or a write that bypassed the adapter,
and a read that swallows it leaves the corruption there for the next read to leak.

`Isolator` wraps the whole shape for an adapter: it resolves the partition *before*
calling the store, so a read with no context bound never reaches the database, and
checks everything that comes back.

```python
isolator = Isolator(tenant_of=lambda row: row["tenant"], where="pgvector.search")
rows = await isolator.read(lambda: connection.fetch(sql, *args))
```

## Batches

A bulk write spanning tenants is never one statement whose scope is the union of
everybody in it. Either it is refused — `partition.only(...)` raises on the first item
that is not this tenant's — or it is split:

```python
for tenant, batch in partitioned(items, tenant_of=owner).items():
    with tenant_scope(tenant, crossing="nightly backfill"):
        await store.write(batch)
```

## What each adapter actually guarantees

`ADAPTER_GUARANTEES` states the mechanism and, more usefully, the limits:

```python
>>> ADAPTER_GUARANTEES["PgvectorIndex"].statement()
'PgvectorIndex isolates by pre-filtered nearest neighbours: ... Not protected against: ...'
```

`IsolationGuarantee` refuses to be constructed with no limits. Every mechanism here has
one, and a guarantee claiming none is a guarantee nobody read.

| Mechanism | Where it is used |
|---|---|
| `KEY_PREFIX` | Redis cache and memory: `<namespace>:<tenant>:...` |
| `ROW_PREDICATE` | PostgreSQL and graph: a bound tenant argument in the `WHERE` |
| `ROW_LEVEL_SECURITY` / `SCHEMA_SEPARATION` | Available where a deployment wants the database to enforce it too |
| `PRE_FILTERED_ANN` | pgvector: the tenant filter is inside the statement, applied before ranking |

There is deliberately no post-filtered mechanism. Taking the top `k` and dropping what
is not ours returns fewer than `k` of our own rows whenever a neighbour ranks higher,
which reads as a thin answer rather than as a bug — the filter belongs in the `WHERE`.

## Erasure

An erasure that deleted rows but left embeddings, summaries and cache entries is not an
erasure. `ErasureSweep` records what each verification query counted *after* the delete:

```python
sweep = ErasureSweep(tenant="acme", remaining={"rows": 0, "embeddings": 0, "keys": 0})
assert sweep.complete
```

A sweep constructed with nothing counted is refused: verifying nothing is not evidence
that nothing remains. `outstanding` names the artefacts that still hold something.

## Limits

These are the honest edges of the mechanism:

- Anything holding the raw connection can read any key or row. This is adapter-level
  enforcement, not database-level; a deployment wanting the database to enforce it too
  uses row-level security underneath, and the guarantee then names both.
- A migration or backfill that bypasses the adapter bypasses everything here. That path
  is the main risk and needs its own checked procedure — the integrity check on read is
  what makes a bad backfill visible rather than silent.
- Pooled connections carrying session-level state (an RLS `SET`) must set and reset it
  per checkout. Nothing here does that for you.
- Recall inside a tenant's own partition still depends on the ANN index. Isolation is
  not traded for speed; recall within the partition is a separate tuning question.

## See also

- [`docs/tenancy.md`](tenancy.md) — how the tenant gets bound.
- [`examples/store_isolation.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/store_isolation.py).
