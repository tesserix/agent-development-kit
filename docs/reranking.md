# Reranking

Fused hybrid results are plausible. Plausible is not the same as good enough to cite: the
passage that actually answers the question sits fourth, and an answer built from the top
three is confidently about something adjacent.

A reranker fixes the order by reading query and passage together. It is also a second
model call per retrieval, over a candidate set someone has to bound, and the usual way it
arrives — a provider's rerank endpoint pasted into application code — is none of
substitutable, budgeted or traced.

## The shape

```python
from tesserix_adk.rag import RerankingRetriever

stage = RerankingRetriever(
    retriever,
    ModelReranker(provider, model="rerank-1"),
    candidates=50,
    top_n=8,
    timeout_seconds=5.0,
    budget=budget,
    tracer=tracer,
)

found = await stage.retrieve("how do I get my money back", scope=HANDBOOK, k=8)

found.hits[0].rerank_score   # what the reranker said
found.hits[0].score          # what fusion said, still there
found.reranked               # whether this order was computed or inherited
```

`RerankingRetriever` is a `Retriever` wrapping a `Retriever`, so it goes wherever retrieval
already goes. `NoReranking` is the default reranker for measuring the stage's own overhead
before paying a vendor for the ranking.

## The candidate set is bounded at construction

`candidates` is what the stage asks the inner retriever for; `top_n` is what it returns.
Both must be at least one and `candidates` may not be below `top_n` — a stage that cannot
produce the best N of anything is a configuration mistake, not a runtime surprise. The
call's own `k` narrows the result further but never widens the fan-out.

## Passages are data

`ModelReranker` sends the passages as a JSON payload under a system instruction that says
they are text to be scored and never instructions to follow. A candidate set containing
injection-shaped text is scored exactly like any other candidate set.

Scores come back as JSON. A reply that is prose, or JSON of the wrong shape, yields no
scores rather than an invented order — the stage then keeps the fused order and says so.
A score for a chunk id nobody retrieved is ignored: the stage reorders the hits retrieval
found, and a reranker cannot add to them.

## The order is reproducible

Ties are broken by chunk id, so replaying a retrieval gives the same ranking. A candidate
the reranker did not score keeps its fused position, behind everything that was scored —
being unranked is not evidence of being good.

Every hit keeps both numbers. `score` is fusion's, `rerank_score` is the reranker's, and
`contributions` still names the branches that found it, so a bad answer can be traced to
the step that produced it.

## What it costs, and when it is skipped

The call is recorded against the `BudgetPolicy` as one model call with its usage, like any
other. If the budget is already exhausted the reranker is not called at all: the fused
order comes back with `reranked=False`. A timeout or a failing reranker does the same.
Retrieval degrading is better than retrieval failing, but only if the caller can tell —
`reranked` is the flag, and the tracer records `adk.rerank.degraded` with the reason
(`budget`, `timeout`, `failed` or `empty`).

A reranker declared unavailable is different. `RerankingRetriever` raises `CapabilityError`
at construction rather than degrading on every call for the lifetime of the process.

## Rerankers

| Reranker | What it is |
|---|---|
| `NoReranking` | Keeps the fused order. Costs nothing, still sets `reranked`. |
| `CrossEncoderReranker` | A local cross-encoder, run off the event loop. |
| `ModelReranker` | A provider call, budgeted and traced like any other. |

`CrossEncoder` is one method, `score(pairs) -> Sequence[float]`, so a sentence-transformers
model or an in-house one drops in. Returning a different number of scores than pairs raises
`ConfigurationError`: scores that do not line up with passages cannot be attributed to any
of them.

## Known limitations

Training or fine-tuning a reranking model, and learning-to-rank from production click
signals, are out of scope. This is the stage they would be served through.
