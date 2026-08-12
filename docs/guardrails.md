# Guardrails — the order checks run in, and what happens when one cannot answer

A safety check written inline in application code is invisible in the agent's definition:
the same agent is guarded in one product and unguarded in another, and nobody can tell by
reading it. Worse, an inline check that raises is usually swallowed, so the run carries on
with nothing checking it — an unavailable guard silently becomes a permissive one.

So a check is a `Guard`, its answer is a `GuardResult`, and the order they are asked in is
a `GuardrailPipeline` that the run loop applies at both ends of a run.

```python
class NoCardNumbers(Guard):
    name = "no_card_numbers"

    async def check_input(self, content: str) -> GuardResult:
        if _looks_like_a_card(content):
            return GuardResult.blocked(code="pii_detected", detail="one card number")
        return GuardResult.allow()

pipeline = GuardrailPipeline((NoCardNumbers(), NoPromptLeak()))
checked = await pipeline.check_input("…")
```

`Guard` answers allow on both stages, so a check that is about one overrides one method.
The protocol requires both, so a pipeline can be told what a guard covers without calling
it.

## What a guard may say

| | |
|---|---|
| `GuardResult.allow()` | Nothing to object to. |
| `GuardResult.redacted(content, code=…)` | Continue, on this content instead. The guards after it see the redacted version, and it is what comes back. |
| `GuardResult.blocked(code=…, detail=…)` | Stop. The content does not continue in any form. |

A block needs a `code`. A caller matching on why must match on something stable, and a
sentence gets reworded. `detail` is a short explanation that is safe to log — never the
offending content, which is the one thing an error carrying it would spread into every log
that catches it.

The order guards are asked in is the order they were declared, always: never dependent on
registration timing. The first block ends the pipeline, so where two guards disagree the
more restrictive verdict is what the run acts on, deterministically.

## Failing closed

A guard that raises, that does not answer within `timeout_seconds`, or that answers with
something that is not a `GuardResult` raises `GuardrailEvaluationError` with the guard, the
stage and a `reason` of `raised`, `timeout` or `unreadable`. The content does not continue:
a guard that is down is not a guard that consented, and the guards after it are not asked.

`GuardrailViolationError` and `GuardrailEvaluationError` share a `GuardrailError` base —
catch the base to stop either way, catch the subclass to tell a decision from an outage.
Neither is retryable.

Cancelling a check is not a verdict. `asyncio.CancelledError` propagates rather than being
recorded as a refusal, because the caller withdrew the question.

## In a run

An agent declares guard names in order; the runner is given the guards by name.

```python
agent = Agent(name="triage", model="…", guardrails=("no_card_numbers", "no_prompt_leak"))
runner = AgentRunner(provider=provider, guardrails={"no_card_numbers": …, "no_prompt_leak": …})
```

Every run then passes its input through the pipeline once, before the prompt is assembled,
and every model response through it before the answer is used — with no per-agent wiring
and no path to the provider that skips it. A redaction on the way in is what the model is
sent; a redaction on the way out is what enters the conversation and what the run returns.
A block or an evaluation failure ends the run as `FAILED` with a `guardrail_refusal` event
naming the guard; a redaction records `guardrail_redaction`. Each verdict is also a
`GuardrailDecision` progress event carrying the guard, the stage and what it decided —
never the content.

An agent declaring no guards is unguarded, and that is visible in its definition rather
than in the absence of a call somewhere else.

## A streamed answer

`check_stream` buffers the whole answer, checks it, and then hands it on as one piece.
That costs the latency streaming was for, and the alternative is emitting the first half of
something a guard was about to block — which is the failure the guard exists to prevent. A
consumer that wants tokens on screen sooner runs the model stream to the caller and the
guard over the assembled answer, and accepts that it is showing unchecked text.

## Known limitations

- Guards see text. Structured output is checked on its serialised form; a guard that needs
  the parsed object parses it itself.
- Tool arguments and tool results are not on this path — they have their own boundary
  (`docs/tool-results.md`).
- A delegated run inherits its caller's guards and runs them in the caller's order, so
  handing work to a sub-agent is not a way around one; see `docs/delegation.md`.
- `timeout_seconds` is per guard, not per pipeline. A run's own deadline still bounds the
  whole thing, and inside the loop that deadline is what applies.
