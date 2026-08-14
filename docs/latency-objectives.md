# Latency objectives

"Kit overhead under twenty milliseconds" is easy to state and says almost nothing. Against
a CPU model call of five to thirty seconds, kit overhead is a rounding error. Three numbers
decide whether a product built on CPU inference feels usable:

- **Time to first token** — how long the user stares at nothing.
- **Sustained tokens per second** — the rate they read at once text starts arriving.
- **Prompt-cache hit ratio** — prefill is where CPU latency goes, so this belongs beside
  the other two rather than in a different dashboard.

`tesserix_adk.observability.latency` records all three per run, and the benchmark suite
tracks them against a committed baseline so a regression fails CI rather than arriving as
a support ticket.

## Recording a run

```python
from tesserix_adk.observability import CacheHits, RunTimer

timer = RunTimer(streaming=True, cold=False)
stream = runner.stream(agent, asked, tenant="acme")
async with stream:
    async for event in stream:
        timer.first_token()

result = await stream
report = timer.finished(
    output_tokens=result.usage.output_tokens,
    hits=CacheHits(input_tokens=812, cached_tokens=690),
)
report.emit(meter, model="llama-3.1-8b-instruct")
print(report.render())
```

```
warm stream: first token 0.412s, 18.3 tok/s, 11.640s total, cache 85%
```

`first_token()` is safe to call on every event; only the first one counts. The sustained
rate excludes the wait for that first token, because a slow prefill averaged into the
decode rate flatters a run that felt slow.

## Unknown is not zero

A provider that reports no cached-token count leaves `CacheHits.cached_tokens` as `None`,
and `ratio` is then `None` too. It is not zero and it is not one — both of those are
answers, and an unknown that reads as an answer is how a cache that quietly stopped working
goes unnoticed for a quarter. `emit` counts nothing for a number the run does not have.

## Cold and warm, streamed and blocking

These are dimensions on every metric, and separate scenarios in the suite. A cold start
averaged in with warm runs hides both: the cold number looks fine and the warm number looks
bad. A blocking run has no first token to time at all, so `time_to_first_token` is `None`
rather than a made-up equivalent.

## Metrics, not only spans

Every number is emitted through `Meter` as well as being attached to the span. Traces are
sampled; a latency percentile computed from whatever the sampler happened to keep is
precise-looking and wrong. `attributes()` carries durations and counts only — never a token
of content — for the reason `docs/tracing.md` gives.

### Sampling rate versus audit completeness

Full trace coverage of every production run and a strict tracing-overhead ceiling cannot
both hold at volume. They are different pipelines and the kit keeps them apart:

| | Traces | Audit |
|---|---|---|
| Coverage | Sampled, and dropped first under load | Complete, every decision |
| Purpose | Where did the time go? | What did the agent do, and what did it refuse? |
| Under pressure | Sample rate falls | Back-pressure, never loss — see `docs/audit.md` |

Latency lives in metrics and traces. "Did this agent refuse the transfer?" is an audit
question, and no sampling rate is allowed to affect the answer.

## What the benchmark measures, and what it models

A shared CI runner cannot measure CPU inference reproducibly: its cores are contended and
its neighbours are invisible. So `benchmarks/suite.py` measures what is measurable there —
the kit's own time, and the token counts — and models the model's time from a declared
profile for the documented target CPU:

```python
TARGET_CPU = CpuProfile(prefill=420.0, decode=18.0)
```

Eight contemporary x86 cores, no GPU, an int8-quantized small model. Stating that plainly
matters more than pretending the runner is the target: the number that moves under a change
is the *uncached token count*, and that is measured, not modelled. Break prefix stability
and the uncached count rises, the modelled first token rises with it, and the hit ratio
falls.

Three scenarios, kept apart:

| Scenario | What it holds |
|---|---|
| `first-token-cold` | First token with nothing cached — the worst case a user sees |
| `first-token-warm` | A fresh turn on a stable prefix, which is what a conversation is |
| `sustained-stream` | A two-hundred-token answer, for the rate a reader perceives |

A change that puts something volatile near the front of the prompt — a timestamp, a
reordered tool declaration — fails the gate naming both numbers that moved:

```
first-token-warm time_to_first_token: regressed (+58.4%, measured 0.0749)
first-token-warm cache_hit_ratio: regressed (-38.2%, measured 0.3469)
```

Record a new baseline with `make bench-record`, which writes only the deterministic metrics
— token counts, peak bytes, and these three — because wall-clock percentiles from one
laptop are not a baseline anyone else's machine can be judged against. See
`docs/benchmarks.md` for the harness itself and `docs/cpu-inference.md` for the server
these objectives are stated against.
