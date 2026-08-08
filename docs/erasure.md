# Redaction and erasure

Two halves of one promise. Redaction is what never gets stored; erasure is what leaves
once somebody asks. A deletion that reaches the row and not the embedding built from it
has kept neither.

## On the way in

Every write path — `write`, `append`, `upsert`, `supersede`, `log`, `index` — masks the
value before it stores it, using the same shape detector the run's progress stream and the
telemetry exporter use.

```python
await store.write(scope, record)   # value: {"who": "ada@example.com"}

held = await store.read(scope, "k")
held.value      # {"who": "[redacted]"}
held.redacted   # ("who",)
```

`redacted` names the paths rather than counting them, so a reader can tell a masked field
from one that was always empty. It reaches inside nested values — `"trip.contact.0"` —
because the token is never at the top level.

| Redactor | Masks |
|---|---|
| `PatternRedactor()` — the default | Emails, API-key prefixes, bearer tokens, JWTs, long hex, card numbers |
| `PatternRedactor(extra_patterns=(r"CASE-\d+",))` | The above, plus what a deployment knows about |
| `None` | Nothing. Has to be asked for rather than arrived at |

Masking is by substring, so `"filed under CASE-4471"` stores as `"filed under [redacted]"`.
A field that vanished entirely reads as a bug rather than as a decision.

## Derived artefacts

An embedding, a summary, an index entry, a cache key. Each one still says the thing the
record said, so each one registers what it came from:

```python
await store.derived(scope, Derivation(
    artefact_id="vec-1", source_id="episodic:e", adapter="vectors"
))
```

Erasure walks the registry rather than assuming that deleting rows was enough. An artefact
two scopes derived is never purged for one of them: the other tenant did not ask to be
forgotten.

A `DerivedIndex` is anything that can purge ids it is handed:

```python
class VectorIndex:
    name = "vectors"

    async def purge(self, artefact_ids: tuple[str, ...]) -> int: ...
```

It is never told what a scope or a kind is, and `purge` must be idempotent, because
erasure resumes by asking again.

## Erasing

```python
receipt = await store.erase(scope)

receipt.counts        # {"profile": 3, "episodic": 1, "semantic": 1}
receipt.records       # 5 — every version of a superseded key, not one per key
receipt.artefacts     # how many derived artefacts went with them
receipt.adapters      # the indices this erasure was responsible for
receipt.completed_at
receipt.complete      # True
```

Two phases. Records are tombstoned first and stop being readable at once — `read`,
`profile`, `belief`, `episodes`, `search` and `history` all skip them. Derived artefacts
are purged second. Nothing is deleted until both have happened, which is what makes the
operation resumable.

`kinds=` narrows it; `dry_run=True` returns accurate counts and touches nothing. A dry run
is never `complete`, because it has kept no promise to anybody. Re-running a finished
erasure returns zero counts rather than raising.

## When an index cannot be reached

```python
try:
    await store.erase(scope)
except PartialErasureError as stalled:
    stalled.adapter             # "vectors"
    stalled.receipt.complete    # False
    stalled.receipt.outstanding # ("vectors",)
```

The records stay tombstoned and out of reach, so nothing readable survives the failure.
Run `erase` again once the index is back: it resumes, and the second receipt does not
re-count what the first already removed.

## Audit

Each erasure publishes one `adk.memory.erased` event carrying record and artefact counts,
whether it completed, and which adapters are outstanding. Never a value and never a key —
an audit trail that quotes what was erased has undone the erasure.

A dry run publishes nothing. It erased nothing.

## Not here

Detecting PII by meaning rather than by shape belongs to the guardrails story. Backup and
WAL retention is infrastructure, not a store concern: this promises that nothing readable
through the kit survives an erasure.
