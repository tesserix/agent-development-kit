# Citations

An answer assembled from three retrieved passages and returned as prose cannot be audited.
When it turns out to be wrong, nobody can say whether the corpus was wrong, retrieval was
wrong, or the model invented it — and the support agent holding it cannot show the customer
where the policy statement came from.

Footnote-shaped strings inside the answer text are not a fix. An answer that cites nothing
and an answer whose footnotes were invented both parse the same way: as text.

## The shape

```python
from tesserix_adk.rag import check_grounding, cite, excerpt

retrieved = cite(await retriever.retrieve(question, scope=HANDBOOK))
answer = await model_answers(question, retrieved)   # a CitedAnswer

check_grounding(answer, retrieved)                  # raises rather than returns

answer.text                                         # the claims, as prose
answer.sources(answer.claims[0])                    # what that claim rests on
excerpt(retrieved[0], document)                     # the exact characters cited
```

`cite` turns a `RetrievalResult` into citations. `check_grounding` runs before the answer
is returned to anyone. `excerpt` resolves a citation back to the span of the document
version it was made from.

## What a citation pins

| Field | Why it is there |
|---|---|
| `document_id`, `document_version` | The document may be updated between retrieval and answer. Resolving against whatever it says now shows a reader text the answer was never built from. |
| `chunk_id`, `span` | A citation to a document is a citation to forty pages. |
| `tenant` | Carried on the citation, so a result crossing a call boundary cannot lose the only thing that says who may read it. |
| `score`, `branches` | Whether the passage was an exact match or the vector's opinion — and the reranker's score where one ran. |
| `retrieved_at`, `locator` | When it was read, and where a reader goes to look. |

`cite` builds these from the chunk metadata the store carries: `version`, `start` and `end`
are required, and `uri`, `page` and `section` are used where present. A chunk missing the
version or the span raises `ConfigurationError` rather than producing a citation that
resolves to the wrong place — the ingest that wrote the chunk is where that is fixed.

## The answer is structured

`CitedAnswer` is claims and citations. Each `Claim` carries the `citation_ids` it rests on:
several citations may support one claim, and one citation may support several claims. Two
citations may not share an id, because a claim naming it would be ambiguous.

## Grounding fails closed

`check_grounding` raises, and never repairs:

- a claim resting on no citation at all → `UncitedClaimError`. Where the corpus returned
  nothing, the answer is a refusal, not an answer with the citations left off.
- a claim naming an id the answer does not carry, or a citation naming a document or a
  version this run did not retrieve → `UngroundedCitationError`, listing what is missing
  and what was available. Stripping the offending citation would leave the claim standing
  with nothing behind it, which is the exact failure this surface exists to catch.
- a citation into another tenant → `TenantCrossingError`.

## Provenance travels

`answer.provenance()` is every citation id some claim rests on. `MemoryRecord.citations`
holds them, so a summary written back to memory from retrieved content can still be asked
where it came from. A fact whose sources are gone is a claim about the corpus that the
corpus cannot answer for.

`citation_attributes(citations)` gives span attributes — counts, document ids and versions,
under `adk.citation.*`. Never document text: a tracing backend is not the corpus, and it
outlives every redaction rule the corpus has.

## When the source is gone

`CitationResolver` resolves a citation against the live corpus. A chunk erased under a
right-to-erasure request resolves to a tombstone — `erased=True` and no text — rather than
disappearing, so an audit can still see that the answer was built on something that has
since been removed.

## Known limitations

Rendering citations in a product interface is out of scope, as is corpus-level freshness
and document lifecycle management.
