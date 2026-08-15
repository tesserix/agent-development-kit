# The frozen prefix

Compression and prefix caching pull in opposite directions, and the conflict is silent.
Recompressing the whole context each turn produces slightly different bytes each time, which
invalidates the provider's cached prefix and forces a full prefill. On CPU inference the
trade is catastrophic: the compression saves perhaps a third of the tokens and the cache
miss costs tens of seconds of prefill on all of them. Nothing errors. The latency graph
moves, and a week later somebody asks why.

So there is a boundary. Everything above it is frozen and byte-identical from turn to turn.
Only content that arrived since is eligible for anything, and it is compressed once, on
arrival, after which it becomes frozen history itself.

## Holding the line

```python
from tesserix_adk.memory import FrozenPrefix, compressed

prefix = FrozenPrefix()

for index in prefix.compressible(messages):
    admitted = router.admit(text_of(messages[index]), budget_tokens=budget)
    messages[index] = compressed(rewritten(messages[index], admitted), by=admitted.compressor)

prefix.verify(messages)
await model.complete(messages)
prefix.advance(messages, turn=turn)
```

`compressible` returns only live indices that carry no compression mark, so a message
already shrunk on an earlier turn is never shrunk again. `live` gives the messages
themselves where that is more convenient. An empty live zone means the prompt is
byte-identical to the last one — the best case, not an error.

`advance` is idempotent per turn: an advance for a turn at or before the one already
recorded is ignored, so two concurrent turns cannot move the boundary twice.

## The assertion

```python
try:
    prefix.verify(messages)
except PrefixDriftError as drift:
    log.error("prefix drift", layer=drift.layer, position=drift.position)
```

`verify` compares the frozen region against the digests taken when it was frozen and raises
`PrefixDriftError` if anything was rewritten, reordered or lost. The error names the layer —
the `adk.section` metadata value, or the role where there is none — so the report says
`'instructions'` rather than `index 0`.

Put it in a test. A silent doubling of prefill latency should fail a build rather than reach
production:

```python
def test_the_system_prompt_does_not_move() -> None:
    prefix.advance(assemble(turn_one), turn=1)
    prefix.verify(assemble(turn_two))
```

## Giving the prefix up

An eviction or a redeployed instruction block genuinely invalidates the cache. That is
recorded rather than hidden:

```python
prefix.reset("eviction removed frozen content")
prefix.boundary.resets  # never decremented
```

`PrefixBoundary.attributes()` reports `adk.frozen_prefix_position` and
`adk.frozen_prefix_resets` for a span or a metric, worth watching beside the cache hit ratio
they exist to protect. Positions and counts only — no prompt content reaches a metric.

## Known limitations

* The boundary counts messages, not tokens. A provider that caches at a token granularity
  finer than a message may keep more than the position claims; it will not keep less.
* A `FrozenPrefix` is per context and holds nothing global, so it is the caller's job to
  keep one per conversation. That is also why a position never crosses a tenant.
* Digests cover the message content and role, not its metadata, so annotating a frozen
  message is not drift. Rewriting what the model reads is.
