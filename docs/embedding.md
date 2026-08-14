# Embedding

Two costs decide whether an ingest is run again: the bill, and the holes.

The bill is paid every time a corpus is re-indexed, and almost all of it is for text that
has not changed since the last run. The holes are cheaper to create and far more expensive
to find: an embedder that answers a zero vector for the batch a provider rate-limited
produces passages that are simply never retrieved, with nothing in the index saying so.

## The shape

```python
from tesserix_adk.rag import BatchedEmbedder, MemoryEmbeddingCache

embedder = BatchedEmbedder(source, cache=MemoryEmbeddingCache())

embedded = await embedder.embed_documents([chunk.text for chunk in chunks])
embedded.vectors      # one per text, in order, repeats included
embedded.usage        # only what actually reached the provider
embedded.stats        # requested, cached, embedded, batches, retries, cache_failures

query = await embedder.embed_query("what is the escalation path?")
```

`source` is a `VectorSource`: one call, one batch, no batching or caching of its own. That
is the half an integration package writes. Everything below is the half that does not have
to be written twice per vendor.

## Paying for what changed

The cache key is `sha256` of the model name, its version, its width, the tenant and the
normalised text — nothing else. It is content-addressed, so nothing is ever invalidated: a
new model version or an edited passage simply addresses somewhere else, and the entries
nobody reaches age out of the backend on its own terms.

The text is hashed rather than carried. A key is copied into logs, dashboards and store
browsers; the corpus is not.

`normalised` composes to NFC and strips the ends, and the normalised form is what is sent
as well as what is keyed. Two spellings of one accented word are the same text to a reader
and should not be two entries with two bills, and neither should a chunk that picked up a
trailing newline from its document.

Where text is redacted before embedding, it is the redacted form that is hashed, sent and
cached. The original never reaches either the provider or the cache.

## Tenancy

By default every entry is keyed to the tenant in force, and embedding outside a tenant
scope raises `MissingTenantContextError` rather than falling back to a shared key. One
tenant's document text is not something to answer another tenant's ingest with, and a
cache is the quietest place for that to happen.

`shared=True` keys entries to `""` for a public corpus. It is a deliberate statement about
the text, made per embedder, not a default anybody inherits.

## When the provider will not answer

Only failures the kit itself calls retryable are retried, with the jitter and the
`Retry-After` handling of `RetryConfig` — the same policy the model loop uses. Past the
budget, the ingest raises `EmbeddingUnavailableError` carrying `batch` and `cursor`, the
index into the texts of the first one not embedded.

Everything that did land is written to the cache before the error is raised, so resuming
from the cursor re-sends only what failed. Nothing is substituted for a missing vector,
ever: `stats` can be wrong about the bill and be corrected, and an index with holes in it
cannot.

Cancellation propagates as `asyncio.CancelledError`. A cancelled ingest is not an outage
and must not be reported as one.

## When the shapes disagree

A vector that is not the model's declared width raises `EmbeddingDimensionError`, whether
it came from the provider or from the cache. A stale vector under a live key is the worse
of the two: a vector store compares what it is given, and a distance computed over the
overlap comes back as a rank nobody can question.

## When the cache is the thing that is down

A backend that cannot be read or written is worked around, not propagated — the vectors
can be bought again — and every such fallback is counted in `stats.cache_failures`. It is
not an error and it is not free, so it is put where somebody will see it before the
invoice does.

## Testing

`tesserix_adk.testing.FakeEmbedder` is a `VectorSource` that never reaches a network,
derives its vectors from the text so they are identical in every run, records the batches
it was given, and takes a script of failures. A retrieval test whose vectors move between
runs asserts nothing.

## Known limitations

- `MemoryEmbeddingCache` is per-process. A re-index that must be cheap across deploys
  needs a shared backend behind the `EmbeddingCache` protocol.
- The cache is read once per call, so two concurrent ingests of the same new text both
  pay for it. Coalescing that would need a lock the protocol does not have.
- `EmbeddingUnavailableError` reports the first failed batch. Where several fail, the
  cursor is that batch's, and the batches after it are already cached.
- Nothing here writes to a vector store, and nothing here chunks: see
  [`docs/chunking.md`](chunking.md).
