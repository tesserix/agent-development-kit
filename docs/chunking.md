# Chunking

A chunk is the unit of every later decision, and no later stage can undo it.

What gets embedded, scored, reranked, quoted in a citation and pushed into the window is a
chunk. A passage split through the middle becomes two hits that are each wrong, and the
reranker cannot recover the half that was left behind. So chunking is configured per
collection, sized in the tokens of the model that will read it, and it refuses rather than
emitting a piece that will not fit.

## The shape

```python
from tesserix_adk.rag import ChunkerRegistry, ChunkingSpec, Document, tokens_via

chunkers = ChunkerRegistry(
    count=tokens_via(provider),
    collections={
        "handbook": ChunkingSpec(strategy="structural", max_tokens=512),
        "search": ChunkingSpec(strategy="fixed", max_tokens=256, overlap_tokens=32),
        "repo": ChunkingSpec(strategy="code", max_tokens=400),
    },
)

for chunk in chunkers.chunker_for("handbook").chunk(Document(id="handbook-2026", text=text)):
    ...  # chunk.text, chunk.start, chunk.end, chunk.section, chunk.tokens
```

A collection with no settings and no `default` is a `ConfigurationError`: chunking it by
guesswork indexes it differently from everything already in it, and the symptom is
retrieval quality that drifts with no deploy to blame.

## Sized by the tokeniser that will read it

`tokens_via(provider)` counts with the provider's own tokeniser. A character heuristic is
wrong by a factor of three the first time a document is not English — under the window on
one corpus, over it on another, and nothing finds out until retrieval is in production.
Any `Callable[[str], int]` works, so a deployment with a local tokeniser passes it
directly.

## The strategies

| Name | Class | Splits on |
|---|---|---|
| `fixed` | `FixedTokens` | Word boundaries, filling to the limit, with an optional overlap |
| `structural` | `Structural` | Headings, then paragraphs, then sentences, then words |
| `sentence-window` | `SentenceWindow` | Overlapping windows of whole sentences |
| `code` | `CodeAware` | Top-level definitions, then lines |

`register(name, build)` adds a deployment's own, and a registered name replaces a built-in
of the same name.

`Structural` falls to a finer boundary only where the coarser one does not fit, so a
document with no headings, no paragraphs and no sentence ends still chunks rather than
failing. `SentenceWindow` overlaps by construction: its chunks cover the document, they do
not partition it.

## A chunk knows where it came from

`chunk.text` is exactly `document.text[start:end]`, and the model refuses any chunk whose
span does not describe its text. Offsets are code points, so a document in mixed scripts
cites correctly rather than resolving to a half-character. Everything here segments by
producing cut offsets rather than pieces, so no strategy can quietly drop the whitespace
between two chunks and leave every subsequent offset wrong by one.

Ids are content-addressed from the document id and the chunk text, so re-indexing an
unchanged document writes the same ids and an edit rewrites only the chunks it touched.

## What will not divide

A run of text that cannot be cut under the limit — a base64 blob, a minified bundle — is a
`ChunkingError` naming the document and the offset. Emitting it anyway moves the failure to
a model call, where it reads as a context window error about a document nobody can name.

`overflow=Overflow.SPLIT` cuts it mid-word at the limit instead. That is right for a
document that genuinely has no boundaries and lossy for one whose boundaries the strategy
did not know about, which is why it is asked for rather than assumed.

## Known limitations

- **A heading whose section overflows becomes its own chunk.** Structural splitting cuts
  at the heading, so where the section beneath it does not fit, the heading line is the
  first piece. Its text is carried on every following chunk's `section` regardless, so
  nothing is lost — but a collection where this is common wants a larger `max_tokens`.

- **`CodeAware` has no parser.** Column zero is where a top-level definition starts in
  every language that indents its bodies, which keeps whole functions together without a
  grammar per language. A language that does not indent is chunked by lines.

- **Sentence segmentation is regex-based.** It ends a sentence at terminal punctuation, so
  `Dr.` and `e.g.` end one. The cost is a slightly early boundary, not a lost span.

- **The counter is called several times per chunk.** Fitting is an exponential-then-binary
  search rather than one unit at a time, so a large document costs a logarithmic number of
  counts per chunk. A counter that makes a network call per invocation is still the wrong
  counter.

- **Loading is somebody else's job.** A `Document` is text that has already been extracted.
  PDF, HTML and office loaders are not in the kit.

A runnable walkthrough: [`examples/chunking.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/chunking.py).
