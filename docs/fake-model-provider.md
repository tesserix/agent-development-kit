# A model provider you can script

A suite that reaches a real provider is slow, costs money, and fails on somebody else's
outage. `FakeModelProvider` answers from a script instead: no vendor SDK, no network, and
exactly the tokens the script names.

```python
from tesserix_adk.testing import FakeModelProvider, Fault, ScriptedTurn

provider = FakeModelProvider(
    ScriptedTurn.calling("lookup_charge", {"id": "ch_1"}, input_tokens=40, output_tokens=8),
    ScriptedTurn.failing(Fault.RATE_LIMIT),
    ScriptedTurn.saying("refunded", input_tokens=60, output_tokens=12),
)
```

## Writing a script

| Builder | The turn it stands for |
|---|---|
| `ScriptedTurn.saying(text, ...)` | An answer in prose |
| `ScriptedTurn.calling(name, arguments, ...)` | A request to run a tool |
| `ScriptedTurn.returning(payload, ...)` | A structured answer, serialised as the content |
| `ScriptedTurn.failing(fault, payload=...)` | A failure, raised in the kit's own vocabulary |

Every builder takes `input_tokens`, `output_tokens` and `cost`, and they are reported
back verbatim. An assertion about a budget compared against an estimate is an assertion
that flakes.

## Faults

`Fault` covers what a provider really does: `TIMEOUT`, `RATE_LIMIT`, `TRANSPORT` and
`MALFORMED`. Each is raised as the kit's own error — `ProviderTimeoutError`,
`RateLimitError`, `ProviderError`, `ModelResponseError` — so the retry and degradation
paths are exercised against the types they will really see, and `payload=` keeps the raw
body a failure report needs.

A payload that violates the requested schema is **returned, not raised**. What an invalid
answer means is the runtime's decision; a fake that refuses to produce one hides the
repair path.

## Strict by default

An unscripted call raises `ScriptExhaustedError` naming how many calls were made against
how many turns were written. A fake that answers forever lets a runaway loop pass its
test and arrive as a bill. `strict=False` returns an empty end-of-turn reply instead, for
a test that is about something else.

`remaining` counts the turns the run never reached: above zero, the run stopped earlier
than the script expected, which is the failure a passing assertion can otherwise hide.

## The call log

`requests` holds every `ModelRequest` the runtime assembled, in order, and `calls` counts
them. That is where a test asserts what the prompt actually contained — the system text,
the tools offered, the history that survived compaction — rather than what it was meant
to.

## Capabilities

`capabilities=` declares whatever the test needs, including a capability the fake lacks.
Refusal is a path worth testing, and it needs a provider that refuses.

## Fixtures

With `pytest_plugins = ["tesserix_adk.testing.pytest_plugin"]`:

- `fake_model` — one empty strict fake, scripted by the test with `.script(...)`.
- `fake_model_factory` — builds one fake per run.

Concurrent runs must not share a fake: they consume each other's turns and both tests
then assert about a conversation neither of them had. `FakeModelProvider.factory(*turns)`
hands each run its own on the same script.

## Known limitations

- Latency is not simulated. A test about timeouts scripts `Fault.TIMEOUT` rather than
  waiting for one.
- Streaming replays an already-decided response; it does not model a vendor's chunking.

## Related

- [`docs/testing.md`](testing.md) — the rest of the fakes.
- [`docs/cassettes.md`](cassettes.md) — recording a real provider instead of scripting one.
