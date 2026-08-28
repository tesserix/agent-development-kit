# Response caching

Every product eventually adds a cache in front of its model calls, and the two ways it
goes wrong are always the same: the key is the user's prompt, so a tool-schema change
serves an answer shaped for the old schema, and the tenant is not in the key, so one
customer is served another's answer. `CachingProvider` is a `ModelProvider` that wraps
another one, so caching is a change to where the provider is built and nothing else.

```python
from tesserix_adk.models import CachingProvider, MemoryCacheStore

model = CachingProvider(OpenAIProvider("gpt-4o"), MemoryCacheStore(), tenant="acme")
answer = await model.complete(request)
```

## The key is the whole correctness argument

An entry is served only when every determinant of the answer matches:

| Determinant | Why a change must miss |
|---|---|
| tenant | Two customers asking the same question must never share an answer. |
| model | A different model is a different answer. |
| prompt | The assembled prompt as it goes on the wire, not the user's text. |
| tool schemas | An answer shaped for the old tools is the wrong shape for the new ones. |
| output schema hash | The same for the output contract. |
| parameters | The call settings you declared. |
| prompt version | Retiring a prompt design stops serving what it produced. |
| model version | A vendor's silent upgrade is a different model. |

The tenant is also structural, not only a field: a `CachingProvider` is built for one
tenant and has no way to ask the store for another's entry.

## What may be cached at all

`CachePolicy` refuses rather than storing:

- **sampled calls** — a declared `temperature` above zero, or `n` above one. Storing a
  random draw and serving it as a fact is not caching, it is fabricating determinism;
- **anything inside `not_cacheable(...)`** — the paths the request cannot show, such as a
  personalised memory read, a side-effecting tool's result, or an approval-gated answer:

```python
with not_cacheable("read the user's own history"):
    answer = await model.complete(request)
```

A refused call is never written to the store and is reported as `CacheStatus.REFUSED`.

**Known limitation:** parameters are the ones you declare. The kit cannot see a provider's
own defaults, so a call that samples must say so in `parameters=` — treating undeclared
settings as non-deterministic would refuse everything anyone ever cached.

## Expiry and invalidation

`ttl_seconds` bounds how long an entry may be served; an expired entry is dropped rather
than kept. Retiring a prompt or a model is a key change, so nothing stale is *served* —
but the old entries are still occupying the store, so remove them:

```python
await model.forget(prompt_version="v3")   # what that design produced
await model.forget()                      # everything for this tenant, for erasure
```

## Stampedes

A cold key under concurrent load is one call. The first caller makes it and the rest wait
on the same answer, counted as `coalesced`. A failed call is not cached and does not leave
the key wedged for the next caller.

## Outages degrade, they do not fail

A store that cannot be reached is a slow run, never a broken one: the lookup failure is
counted, reported as `CacheStatus.STORE_UNAVAILABLE`, and the call goes live. A write that
fails is the same. The one thing that is *not* swallowed is `forget` — erasure that
silently failed is worse than erasure that failed loudly.

## The semantic tier

Off unless configured, because approximate matching is a correctness trade a consumer must
opt into deliberately:

```python
semantic = SemanticConfig(embedder=embedder, index=MemorySemanticIndex(), model="bge-m3",
                          threshold=0.97)
```

A near match is served only at or above the threshold, only within the tenant, and only
when it was indexed by the same embedding model — an upgraded embedder is a new vector
space, so its entries are invalidated rather than compared across. The threshold and the
embedding model are recorded on each entry, so reading one back does not depend on what
today's configuration happens to say.

## Stores

`MemoryCacheStore` holds entries in the process and nowhere else. `RedisCacheStore`
(in `tesserix_adk.adapters`) is the shared one, keyed
`<namespace>:<tenant>:<prompt version>:<model version>:<digest>` so that every purge
criterion is a key segment and erasure is one pattern rather than a scan of every value.

A shared store holds customers' answers outside the process that produced them, so:

- the model's own reasoning is **dropped before writing** (`redact_reasoning`, on by
  default) — it is sensitive, never replayed, and a cache is not a place to keep it;
- the store itself must be **encrypted at rest and in transit**. The kit cannot enforce
  that from inside a client, so it is a deployment requirement rather than a setting here;
- an erasure request runs `forget()` for the tenant, which clears the entries and the
  semantic vectors together.

## Status and metrics

Pass `observer=` to receive a `CacheOutcome` per call — status, key digest, usage saved,
refusal reason, similarity — and record it on the run's trace attributes. `metrics` totals
hits, semantic hits, misses, refusals, stores, coalesced waits, store failures and the
usage the cache saved.

Runnable version: [`examples/response_caching.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/response_caching.py).
