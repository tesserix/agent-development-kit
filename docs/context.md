# The context window

`ContextWindow` decides two things a retrieval loop otherwise gets wrong: what is worth
sending at all, and what leaves when there is no more room.

```python
window = ContextWindow(limit_tokens=4096)
window.admit(Segment(text=page, layer=PromptLayer.RETRIEVED, key="p12", score=0.9))
evicted = window.fit()
prompt = assemble_prompt(agent, question, retrieved=window.texts(PromptLayer.RETRIEVED))
```

Worked through with no network: `examples/context_window.py`.

## Admission is keyed

A segment carries a `key` — a chunk id, a document hash, whatever identifies the *content*.
A segment whose key is already held is refused, and `admit` returns `False` to say so.

The largest waste in a retrieval loop is re-injecting a chunk the model already has. On a
GPU that is a rounding error. On CPU it is seconds of prefill, on every turn, for content
that changes nothing about the answer.

Keys match across layers, not within one: the same page pinned and then retrieved is still
the same page, and it is sent once. `key=None` means never deduplicated — two turns of
conversation can read identically and both still happened.

## Eviction is ordered, not incidental

`fit()` evicts until what is held fits `limit_tokens`, and returns what left so a caller
can log it or re-rank it for a later turn.

| Order | What goes | Which first |
|---|---|---|
| 1 | `CONVERSATION` | Oldest first — the turn furthest from the question. |
| 2 | `RETRIEVED` | Lowest score first — it was ranked for this turn; believe the ranking. |
| — | `SYSTEM`, `TOOLS`, `PINNED` | Never. |

The prefix is never evicted, even where dropping it would free the most tokens fastest.
Trimming it invalidates the cacheable prefix (see [`run-loop.md`](run-loop.md#the-prefix-and-why-the-order-is-an-invariant))
and every downstream turn pays the refill — a cost dressed as a saving. A prefix that
cannot fit alone raises `ContextWindowExceededError` instead, because the honest answer
there is that the deployment is misconfigured, not that some of the case file should go.

Eviction frees the key. A chunk dropped for room is no longer content the model has, so
the same chunk is admissible again on a later turn.

## Counting

`tokens` is measured by the tokenizer the window was given. The default is
`approximate_tokens` — four characters to a token, fine for a log line and wrong for a
boundary check. Pass the server's own where the limit has to be exact:

```python
ContextWindow(limit_tokens=4096, tokenizer=tokenizer.count)
```

## Known limitations

- Eviction drops segments whole. Summarisation-based compaction — replacing three turns
  with a summary rather than deleting them — is M2, not this.
- `score` is taken as given. The window does not re-rank, and a retriever that scores
  everything `0.0` gets arbitrary-but-stable eviction order within the retrieved layer.
- Deduplication is by key, not by content. Two ids for the same paragraph are two
  segments; give the chunker a stable content hash if that matters.
