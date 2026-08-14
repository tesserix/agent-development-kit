# Retrieval

An agent that gets half of what it asked for does not say so. It answers from what came
back, in the same tone it would use if nothing were missing.

Two things produce that half. The first is a single branch: a semantic search finds the
paraphrase and misses the booking reference `BX-7741`, and a keyword search does the
reverse. The second is a branch that failed — the vector store timed out, and the answer
reads complete because the keyword branch returned something.

## The shape

```python
from tesserix_adk.rag import Branch, HybridRetriever, IndexRetriever, RetrievalScope

retriever = HybridRetriever(
    IndexRetriever(index, branch=Branch.SEMANTIC, embedder=embedder),
    IndexRetriever(index, branch=Branch.KEYWORD),
    require=(Branch.KEYWORD,),
    timeout_seconds=2.0,
)

found = await retriever.retrieve(
    "how do I get my money back", scope=RetrievalScope(collection="handbook"), k=10
)

found.hits          # fused, best first
found.branches      # which branches answered
found.partial       # whether one did not
```

`RetrievalScope` names the collection and nothing else. The tenant is not in it, because
it is not the caller's to choose.

## Both branches, one ranking

Each branch is asked for its own `k` and the results are fused by rank, not by score:
`ReciprocalRankFusion` weighs a hit at `1 / (k0 + rank)` in each branch that found it.
Scores from a cosine distance and from `ts_rank` are in different units and comparing
them directly ranks by whichever branch happens to produce larger numbers.

`WeightedSum` is there for a corpus that has been measured, where the branches' scores
have been normalised and one branch is known to be better. Reach for it second.

Every hit carries `contributions`: one `BranchScore` per branch that found it, with the
rank it held there. `hit.found_by(Branch.KEYWORD)` answers the question a reviewer of a
bad answer actually asks — was this the exact match, or only the vector's opinion.

## The tenant predicate is not the caller's

`current_tenant()` sets `IndexQuery.tenant`, and the store applies it inside its own
query. A filter named `tenant` from the caller is refused with `SchemaViolationError`: not
merged, not overridden, not silently dropped, because each of those is a different wrong
answer to the same question.

Everything else in `filters` is pushed down alongside it. Nothing is filtered after the
fetch — a store that fetches `k` neighbours and drops the ones the caller may not read
returns fewer than `k` of their chunks, and none at all where the nearest `k` are all
somebody else's.

Caller mistakes are settled once, before any branch runs. Retrieving outside a tenant
scope, or with `k` below one, raises from the retriever rather than arriving as a branch
that happened to fail.

## When a branch does not answer

A branch that fails or exceeds `timeout_seconds` leaves `partial=True` and its name out of
`found.branches`. That is a fact about the result, and it travels with it.

`require=(Branch.KEYWORD,)` turns that into a refusal: `RetrievalDegradedError`, naming
what is missing and what answered. Use it where the missing branch changes the meaning of
the answer — an identifier lookup that quietly becomes a semantic search returns three
plausible passages about the wrong booking. A retrieval where no branch answered at all
always raises, whatever `require` says.

## Backing it with a store

`SearchIndex` is the store-side protocol: a name, `supports(branch)`, and `search(query)`.
`PgvectorIndex` is the first implementation, over one `adk_chunks` table holding both the
embedding and the `tsvector`, so both branches run in the database the product already
has. Its expected schema is `EXPECTED_CHUNK_SCHEMA`, for the migration repository to own.

A store of your own inherits `SearchIndexConformance` from `tesserix_adk.testing` and
supplies `make_index` and `branch`. The suite's corpus holds one passage of a second
tenant identical to the first tenant's, so a store that filters after the fetch fails
there rather than in production.

```python
from tesserix_adk.testing.conformance import SearchIndexConformance


class TestMyIndex(SearchIndexConformance):
    def branch(self):
        return Branch.SEMANTIC

    async def make_index(self, passages):
        return MyIndex(passages)
```
