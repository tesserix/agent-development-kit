# Memory

Three products invented three memory shapes: trip context in Redis blobs, conversation
history in Postgres rows, nothing durable at all. None of it is portable, so a memory bug
fixed in one recurs unchanged in the next. `MemoryStore` is one protocol across the four
kinds of remembering an agent actually does, with the scope in every signature.

## The four kinds

| Kind | What it holds | Operations |
|---|---|---|
| `WORKING` | The current task's scratch space | `write`, `read`, `append`, `expire` |
| `PROFILE` | Durable facts about a user or tenant | `upsert`, `profile`, `supersede`, `belief`, `history` |
| `EPISODIC` | Things that happened, at a time | `log`, `episodes` |
| `SEMANTIC` | Things known, found by resemblance | `index`, `search` |

They share one record type because they share one lifecycle — written under a scope, valid
over a window, believed to a degree, traceable to a source. They do not share operations:
what working memory does (append, expire) and what semantic memory does (rank by distance)
do not collapse into a shared get/put without lying about one of them.

Kinds do not share a key space. A working key called `seat` and a profile key called `seat`
are two records, because one namespace across kinds is a scratch value quietly overwriting
a preference.

A profile fact that changes supersedes the one it replaces rather than overwriting it,
and a fact can decay out of recall without being deleted. Both are in
[beliefs.md](beliefs.md). Redaction on write and
erasure of derived artefacts are in [erasure.md](erasure.md). The Redis, PostgreSQL
and pgvector stores behind this protocol are in
[memory-adapters.md](memory-adapters.md).

## Scope is in every signature

```python
scope = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1", agent="planner")
await store.write(scope, record)
```

`tenant_id` is required, with no default and no "shared" sentinel — a default tenant is one
typo away from being every tenant. A blank one is refused too, since blank is what an
adapter's key join treats as a wildcard.

There is no unscoped overload anywhere on the surface, so a call site cannot forget one.
And the record carries its own scope: writing it under a different one raises
`MemoryScopeError` rather than filing it under whichever of the two the adapter read first.

## Records

```python
MemoryRecord(
    id="profile:seat",
    kind=MemoryKind.PROFILE,
    scope=scope,
    key="seat",
    value="aisle",
    source="onboarding-form",
    valid_from=1_760_000_000.0,
    confidence=0.9,
)
```

`source` is required. A recalled claim with no provenance is one nobody can check, and a
prompt assembled from those is one nobody can explain afterwards. `valid_from` / `valid_to`
give a record a window, which is what makes `as_of` mean something. `confidence` defaults
to certain, because a default of "probably" would quietly discount everything written by
hand.

## Capabilities are checked when the store is bound

```python
require_memory(store, MemoryNeeds(semantic=True, erasure=True))
```

An adapter with no vector index answers a semantic recall with an empty list, on every run,
without an error, and nobody notices for a month. So the adapter declares what it supports
and the consumer declares what its plan needs; the mismatch is a `CapabilityError` at bind
time naming every missing capability at once and the adapter that lacks them.

The same operations still refuse at run time for a consumer that skipped the check —
`erase` on a store that cannot erase raises rather than reporting zero rows, because zero
rows erased and cannot erase are the same number and opposite facts.

## What goes wrong is typed

| Error | When |
|---|---|
| `MemoryScopeError` | A record filed under a scope or kind that is not its own |
| `MemoryCorruptionError` | A stored record no longer validates — carries the id and the raw payload |
| `MemoryLimitError` | A value larger than the adapter declared it holds, refused at the write |
| `EmbeddingDimensionError` | An embedding that is missing, or not the collection's width |
| `CapabilityError` | An operation the adapter never declared |

A corrupt record is never dropped quietly, not from a read and not from a search. Recall
that returns what happened to survive is worse than recall that fails.

## Guarantees an adapter has to keep

- Concurrent appends are ordered and none is lost. `append` returns the position, counting
  from 1, so a caller can detect a lost write instead of assuming there wasn't one.
- A read racing an erasure sees all of the scope or none of it, never half.
- Erasure reaches every kind under the scope, and stops at the scope it was given.
- An expired working key reads as absent, not as stale.

`MemoryStoreConformance` in `tesserix_adk.testing` is where those live as executable cases.
Every adapter subclasses it:

```python
class TestRedisMemory(MemoryStoreConformance):
    def make_store(self) -> MemoryStore:
        return RedisMemoryStore(url="redis://localhost", clock=SystemClock())
```

Capability-gated cases skip themselves against a store that declares it cannot do the
thing, so an adapter is held to what it claims and not to what it does not.

`InMemoryMemoryStore` is the network-free implementation for tests and one-process
development. It passes the same suite.

## Stability contract

`MemoryStore` is public API under semver.

- **Additive only within a minor.** A method may be added; none is removed, renamed, or
  given a new required parameter.
- **One minor of notice before a removal**, with a shim that still works for that minor.
- Adding a member to the protocol means adding its case to `MemoryStoreConformance` in the
  same change, so every implementation learns about it by failing rather than by drifting.

## Not here

Concrete Redis, PostgreSQL and pgvector adapters, context-window assembly and compaction,
and contradiction and decay semantics are each their own story in this epic. This is the
shape they all have to fit.
