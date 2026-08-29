# Content-typed compression at admission

Tool output is the largest and least information-dense thing in an agent prompt. A directory
listing, a hundred-row query result, a stack trace and a paragraph of prose all arrive as
opaque strings and are all paid for at full token price on every subsequent turn, because
the prompt is rebuilt each time.

Most of what that costs is repeated keys, alignment whitespace, boilerplate framing and
identical rows — none of which the model was using. Removing it generically does not work: a
summariser applied to JSON destroys the field names needed to write the next query, and
applied to code it destroys the identifiers. So compression is dispatched on what the
content actually is.

## Routing

```python
from tesserix_adk.memory import ContentRouter

router = ContentRouter(threshold_tokens=256)
admitted = router.admit(tool_output, budget_tokens=4_000, untrusted=True)

print(admitted.compressor, admitted.ratio, admitted.saved_tokens)
messages.append(user(admitted.content))
```

`admit` returns a `Compressed` carrying the content to admit, the `ContentKind` it was
classified as, which compressor produced it, the sizes before and after, and — where nothing
was compressed — why.

| Kind | Compressor | What it removes | What it keeps |
|---|---|---|---|
| `JSON` | `StructuredCompressor` | one copy of every key per row, whitespace, repeated rows | every distinct field name and value |
| `TABULAR` | `TabularCompressor` | alignment padding, repeated rows | the columns and their order |
| `CODE` | `CodeCompressor` | function bodies | signatures, annotations, decorators, every `raise` |
| `PROSE` | `ProseCompressor` | repeated sentences, run-on whitespace | the first occurrence of each sentence |
| `UNKNOWN` | `PassThrough` | nothing | everything |

### The structured form

500 rows of `{"id": …, "region": "apac", "status": "active", "name": …}` become:

```json
{"fields":["id","name"],"rows":[[0,"host-000"], …],"shared":{"region":"apac","status":"active"}}
```

Consecutive identical rows are folded and counted under `repeats`. This is a re-encoding,
not a summary: nothing distinct is dropped, so the model can still write the next query
from it.

## What it declines to do

The fallback is the design, not the edge case. A wrong compressor silently destroying
content is worse than paying full price for it, so:

* content whose type cannot be established with confidence is admitted whole — anything that
  half-looks like a type it does not parse as is `UNKNOWN`, and `reason` says so;
* a compressor that raises falls back to pass-through and never propagates;
* a compressor that expands its input is discarded and the original admitted;
* content at or below `threshold_tokens` skips the router entirely, since classifying a
  two-line result costs more than compressing it saves.

## Guarantees

* **Offline and pure.** No network call, no model call, no state.
* **Deterministic.** The same bytes in produce the same bytes out, or the provider's prefix
  cache cannot hold across turns.
* **Content is never executed.** Classification parses — `json.loads`, `ast.parse` — and
  never follows a reference inside the content.
* **Compression is not sanitisation.** `Compressed.untrusted` carries the trust of what came
  in, unchanged. Compressed tool output is exactly as untrusted as the tool output was.

## Known limitations

* `CodeCompressor` understands Python. Other languages are classified as `PROSE` or
  `UNKNOWN` and are not body-elided.
* `budget_tokens` is advisory. Only `ProseCompressor` truncates to reach it; the structured
  and tabular compressors return what they have rather than drop a value.
* `estimate_tokens` is four characters to a token, used to compare a before against an
  after. Pass the provider's own counter as `count` where the ratio is reported as money.
* `ContentRouter` by itself is lossy. Wrap it in the
  [reversible router](reversible-compression.md) when the model must be able to redeem an
  original through a run- and tenant-scoped handle.
* Admission does not move a conversation's cache boundary. Use
  [FrozenPrefix](frozen-prefix.md) to compress only newly admitted content and detect drift
  in bytes the provider may already have cached.
