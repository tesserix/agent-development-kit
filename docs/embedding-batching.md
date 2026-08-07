# Embedding batching

Indexing a document embeds a few hundred chunks. Written the obvious way that is a few
hundred sequential round trips, each one paying a full request's latency and a full
request's rate-limit headroom for a single vector. `BatchingEmbedder` sits in front of a
provider and turns concurrent single-text calls into provider batches, without the caller
writing a batching loop.

```python
from tesserix_adk.models import BatchingEmbedder

async with BatchingEmbedder(provider) as embedding:
    vectors = await asyncio.gather(*(embedding.embed(chunk, model=MODEL) for chunk in chunks))
```

Each caller still asks for one text and still gets back one vector. What changed is how
many requests reached the vendor.

## Identity is the guarantee

Coalescing is only safe if a caller can never be handed a neighbour's vector, so the
answer is matched to the request rather than to a position in a list:

- every waiting caller carries the digest of its own text, and is answered by that digest;
- the provider's answer is checked for count and width before anyone is given anything —
  a short or wrong-width response is a `ModelResponseError`, never a padded or truncated
  vector;
- duplicate texts within one batch are sent once and both callers are answered;
- a cancelled caller drops out of the batch without disturbing its siblings.

The kit never substitutes a zero vector.

## What may share a batch

Batches are keyed by model, tenant and dimensionality. Two tenants are never in one batch
even for the same model, and two dimensionalities are never mixed.

## When a batch is sent

| Trigger | Why |
|---|---|
| The batch is full | The narrower of the provider's declared `max_items` and your `max_items`. |
| The batch would go past a byte ceiling | Sent as it stands rather than one item over — a batch the vendor rejects costs every item in it, not the one that tipped it. |
| The window expired | `max_wait_seconds` after the first item arrived, however full it is. A batch still filling is not held for a full one. |
| The embedder is closing | Whatever is waiting goes out; nothing is dropped. |

Ceilings come from `provider.limits(model)`, so a vendor that raises its batch size raises
yours. `BatchConfig` may narrow them, never widen them.

## Interactive requests skip the window

A query embedding has a person waiting on it, and queueing it behind a bulk flush is
latency that person sees:

```python
await embedding.embed(query, model=MODEL, interactive=True)
```

That request is sent on its own immediately and is never put in a batch.

## One bad item loses only its own caller

When a batch fails, it is bisected and re-sent until the failure is isolated to a single
text. That caller gets the provider's typed error; everyone else in the batch gets the
vector they asked for. A text longer than the model's declared item limit is refused
locally with `ContextWindowExceededError` before it can poison a batch at all.

## Metrics

| Counter | Meaning |
|---|---|
| `requests` | Texts submitted by callers. |
| `batches` | Provider calls made. |
| `deduplicated` | Callers answered from a text somebody else had already asked for. |
| `bypassed` | Interactive requests sent outside the window. |
| `isolated` | Re-sends caused by bisecting a failed batch. |
| `flushed_full` | Batches sent because they hit a ceiling. |
| `flushed_due` | Batches sent because the window expired. |

`requests` against `batches` is the compression the window is buying; `flushed_due`
dominating `flushed_full` means the window is longer than the work needs.

Runnable version: [`examples/embedding_batching.py`](../examples/embedding_batching.py).
