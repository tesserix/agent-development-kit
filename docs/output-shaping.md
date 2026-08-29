# Output shaping

Input compression is the well-known half of the cost problem and the smaller one. Output is
billed at several times the input rate on reasoning-capable models, and a large share of what
is generated is preamble, restated context and narration of work nobody asked to see. A turn
that merely resumes after a successful tool result rarely needs full reasoning effort, yet
most gateways send the same setting for every call in a run.

Two levers, both clamp-only.

```python
from tesserix_adk.models import Effort, Shaping

shaping = Shaping(
    enabled=True,
    baseline=Effort.HIGH,
    resumption=Effort.LOW,
    terseness="Answer directly. No preamble.",
)
```

Both are off until an agent turns them on.

## Clamping effort

```python
shaped = shaping.shape(request, effort=Effort.HIGH, final=is_last_turn)
body |= provider_effort(shaped.effort, provider=provider.provider_name)
```

`shape` reads the request and returns a decision; it never modifies the request, so the
cacheable prefix is exactly what the assembler produced. `shaped.effort` is never above what
the caller asked for — the clamp is a `min`, and a policy whose `resumption` exceeds its
`baseline` raises `ConfigurationError` at construction rather than at call time.

What is not clamped, and why:

| Turn | Effort | Why |
|---|---|---|
| New question | as asked | the deliberation is the answer |
| Tool resumption | clamped to `resumption` | the work was done by the tool |
| Error recovery | as asked | failures are where deliberation pays |
| Final turn (`final=True`) | as asked | the saving is small; what it degrades is what the user reads |
| Structured output | as asked | the schema requires completeness |

A turn is classified from what the conversation ends with. Mark a failed tool result with
`errored` so the recovery is not clamped:

```python
messages.append(errored(result) if outcome.failed else result)
```

`provider_effort` maps the one expression of effort onto each provider's own parameter, and
returns nothing for a provider that has none — a local llama.cpp build has no effort setting,
and that is not an error.

`shaped.attributes()` gives the model call record what was applied and what was asked for.
Settings and outcomes only; no prompt content.

## Steering terseness

```python
request = shaping.steer(request)
```

The instruction is appended after everything already in the system prompt. Nothing existing is
rewritten, because a rewrite invalidates the cached prefix and gives back more than the shorter
output saves. The suffix is byte-stable and applied idempotently, so enabling it costs exactly
one prefix invalidation and none afterwards. Nothing below the system prompt is touched; where
there is no system message, one is added.

## Known limitations

* Classification reads the tail of the conversation only. An agent that interleaves its own
  scratch messages after a tool result should call `shape` with the turn it knows it is on.
* The effort vocabulary is the kit's four levels. A provider offering a token budget rather
  than a level is mapped at its own adapter; `provider_effort` returns nothing for it here.
* Steering alone measures nothing. Use [savings accounting](savings-accounting.md) with a
  stable control holdout to label input savings as measured and output savings as estimated.
