# Local embedding and reranking

Retrieval spends more of a turn in the small models than in the large one. Every query is
an embedding; every candidate passage is a reranking pair. On a machine without a GPU those
run on the same cores serving the request, so they are what decides whether a retrieval turn
feels interactive.

`tesserix_adk.models.onnx` runs both on the CPU: a quantized ONNX session, a fixed batch, a
version-keyed cache, and a stated throughput budget for each shipped model. A GPU changes
one argument — `device=Device.CUDA` — and nothing else.

## Why `onnxruntime` is not a dependency or an extra

It is a native wheel of a few hundred megabytes. Making it an extra means every install of
`tesserix-adk[all]` carries it, including the services that never retrieve anything. So the
kit defines two narrow protocols — `Tokenizing` and `OnnxSession` — and the operator either
supplies their own session or calls `load_session`, which imports `onnxruntime` by name and
raises `ConfigurationError` naming the install command when it is absent. This mirrors how
the kit treats llama.cpp: the heavy runtime lives outside the wheel.

## The shipped model set

Rates are texts per second, measured at the stated batch and thread count. A rate without
those two numbers cannot be reproduced, which is why they are fields on `Throughput` rather
than a sentence here.

| Model | Dimensions | Quantization | Multilingual | Rate | Batch | Threads |
|---|---|---|---|---|---|---|
| `bge-small-en-v1.5` | 384 | int8 | no | 180/s | 32 | 4 |
| `bge-base-en-v1.5` | 768 | int8 | no | 60/s | 32 | 4 |
| `multilingual-e5-small` | 384 | int8 | yes | 150/s | 32 | 4 |
| `bge-reranker-base` | — | int8 | yes | 45/s | 16 | 4 |

```python
from tesserix_adk.models import onnx_model

model = onnx_model("bge-small-en-v1.5")
model.budget.texts_per_second  # 180.0
```

An unknown name raises `ConfigurationError` listing the set, because a typo and an
unsupported model are otherwise the same error message.

## Refusing a bad artefact at startup

`verify_artefact` checks that the file exists, is not empty, and matches the digest the
model declares. It raises `ModelArtifactError` with `reason` set to `missing`, `empty` or
`digest` — a half-downloaded model should fail at startup for an operator, not at the first
query for a user.

```python
from pathlib import Path

from tesserix_adk.models import Device, load_session, onnx_model

model = onnx_model("bge-small-en-v1.5")
session = load_session(model, directory=Path("/models"), device=Device.CPU, threads=4)
```

## Embedding

`OnnxEmbeddings` satisfies `EmbeddingProvider`, so it drops into `BatchingEmbedder` and
anything else that takes one.

```python
from tesserix_adk.models import OnnxEmbeddings

embeddings = OnnxEmbeddings(session, tokenizer, model)
vectors = await embeddings.embed(["a refund question"], model=model.name)
```

The session is synchronous and CPU-bound, so it runs in a worker thread — a forward pass on
the event loop stalls every other run in the process.

`limits(model)` reports the batch the budget was measured at, the model's token ceiling and
the vector width. Asking a session for a model it does not hold raises `ConfigurationError`:
one session is one model.

### The cache

Identical text is not embedded twice. Keys are content-addressed over NFC-normalised text
together with the model's name, version and dimension, so a version bump cannot read the
previous vectors — a silently mixed index is worse than a cold one. The cache is bounded by
`cache_entries` and evicts oldest-first.

### Metrics

After each call, `metrics` reports `embedded`, `cached`, `seconds`, `texts_per_second` and
`meets_budget` — the last compared against the model's declared rate, so a service can alarm
on the machine being slower than the model was measured on.

## Reranking

`OnnxCrossEncoder` scores (query, passage) pairs and satisfies the `CrossEncoder` protocol
that `rag.CrossEncoderReranker` takes:

```python
from tesserix_adk.models import OnnxCrossEncoder
from tesserix_adk.rag import CrossEncoderReranker

reranker = CrossEncoderReranker(OnnxCrossEncoder(session, tokenizer, onnx_model("bge-reranker-base")))
```

Scoring is synchronous because the reranker already runs it off the event loop.

## Related

- [`docs/embedding-batching.md`](embedding-batching.md) — batching, limits and the provider protocol
- [`docs/reranking.md`](reranking.md) — where a cross-encoder sits in retrieval
- [`examples/local_embeddings.py`](../examples/local_embeddings.py)
