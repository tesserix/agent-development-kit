# Changing beliefs

A profile write that overwrites destroys the reason an agent believes what it believes.
The user was vegetarian in March and eats fish in August, and by August nothing can say
that the first was ever true, when it stopped being true, or who said so.

`supersede` writes a new version and closes the old one. Nothing is overwritten and
nothing is deleted.

```python
await store.upsert(scope, vegetarian)          # March
written = await store.supersede(scope, eats_fish)   # August

written.superseded.valid_to        # when the old belief closed
written.superseded.superseded_by   # the id of what replaced it
written.record.version             # 2
```

## Two clocks

| Field | Answers |
|---|---|
| `valid_from` / `valid_to` | When the fact was true |
| `recorded_at` | When the system learned it |

They are separate because a fact backdated to March was still acted on from August, and
an audit that cannot tell those apart cannot explain a decision that was made in between.
`valid_from` in the future is allowed: the record is on the trail immediately and is not
recalled until the time comes.

## Reading as of a time

```python
await store.profile(scope, "diet")                    # what is believed now
await store.profile(scope, "diet", as_of=march)       # what was believed in March
```

Exactly one record is live per instant, however deep the chain, because each supersession
closes the previous version at the moment the new one starts. A store that declares no
`supports_as_of` raises `CapabilityError` rather than quietly answering about now.

## When the new fact is not a restatement

A record carries a `subject` and the `predicate` aspects it speaks to. `SupersedeMatching`
— the default — supersedes only where both match exactly. Anything else branches: both
records stay live and the disagreement is surfaced.

```python
await store.supersede(scope, fact("vegetarian", predicate=("diet",)))
written = await store.supersede(scope, fact("no shellfish", predicate=("diet", "allergies")))

written.resolution        # Resolution.BRANCH
written.contradiction     # Contradiction(subject="user", holds=(both records,))
```

Nothing resolves that for you. `profile` raises `MemoryContradictionError` rather than
returning whichever record sorted first, and `belief` returns the marker for a caller that
would rather show a person than raise:

```python
held = await store.belief(scope, "diet")
held.record           # None
held.contradiction    # what to put in front of somebody
```

A branch ends when a writer says which live records it settles:

```python
await store.supersede(scope, corrected, resolves=(first.id, second.id))
```

Naming a record that is not live is a `ValueError` — a settlement of something already
settled is a caller working from a stale read. Write your own policy where the default is
wrong for the domain; `resolve` returns `SUPERSEDE`, `BRANCH` or `REJECT`, and a `REJECT`
raises `MemoryContradictionError` with the existing belief left standing.

## Two writers, one belief

```python
await store.supersede(scope, new_fact, expected_version=held.version)
```

The loser of the race gets `MemoryConflictError` carrying both the version it expected and
the version that is live, so it can re-read and decide again rather than retry blind. The
store never leaves two live records for one write and never applies one over the other.
Omitting `expected_version` takes whatever is live, for a caller with no race to lose.

## Decay

Nothing expires on its own, so an agent acts on a preference recorded a year ago as
confidently as on this morning's. A `DecayPolicy` weighs a record instead:

| Policy | Weighs by |
|---|---|
| `HalfLife(half_life_seconds, floor)` | Age, halving per half-life, zero under the floor |
| `ConfidenceFloor(minimum)` | The record's own confidence; age is not the question |

Decay changes ranking and recall eligibility. It never deletes: what a policy stops
surfacing, `history` still returns, because a fact nobody recalls now is still a fact
somebody acted on then. Deletion is erasure, which is a promise made to a different person.

A policy aggressive enough to silence a whole scope is visible rather than silent:

```python
held = await store.belief(scope, "diet")
held.record     # None
held.decayed    # the records that exist and are no longer recalled
held.weight     # what decay left of them
```

Without a policy nothing decays and every weight is 1.0.

## The trail

```python
await store.history(scope, "diet")   # every version of one key, oldest first
await store.history(scope)           # every version under the scope
```

This is what support and dispute resolution read: what was believed, from when, until what
replaced it, and what the write was made from. Erasure removes it, decay does not.

## Not here

Hard deletion is `erase`, and the erasure story owns it. Temporal edges between records
belong to the knowledge-graph story. This decides which version is live, not what the
versions mean to each other.
