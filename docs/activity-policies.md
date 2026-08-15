# Activity policies

A reasoning call that streams for nine minutes is healthy. A tool that charges a card is not
healthy to repeat. One set of retry and timeout numbers cannot serve both, so the kit
declares them per activity class and derives the rest from what the tool itself said.

## The classes

| Class | Timeout | Heartbeat | Attempts | Why |
|---|---|---|---|---|
| `MODEL` | 900s | 60s | 3 | Slow, streams, worth retrying on a provider blip |
| `TOOL` | 60s | none | 1 | Assumed to change the world until it says otherwise |
| `RETRIEVAL` | 30s | none | 3 | A read; repeating it costs latency and nothing else |
| `MEMORY` | 15s | none | 3 | A read |

These are values, not contract. `DEFAULT_ACTIVITY_POLICIES` may be retuned in a patch
release; `ActivityPolicy` is the surface.

## What is never retried

Two questions stay separate: whether a failure is a fault at all, and how long to wait
before trying again. `NEVER_RETRYABLE` answers the first, and no consumer override changes
it — retrying a guardrail block is asking the guardrail until it gives in, and retrying a
budget breach is spending past the ceiling one attempt at a time.

```python
from tesserix_adk.core import ActivityPolicy, GuardrailViolationError

ActivityPolicy(retry=RetryConfig(max_attempts=9)).retryable(GuardrailViolationError("no"))
# False
```

Anything outside the kit's error hierarchy is left alone: the kit does not know what
repeating it repeats.

## How many attempts a tool gets

`attempts_for_tool` reads what the tool declared. A tool declared `EFFECTFUL` gets one
attempt, because the second one books a second seat. A tool that declared nothing is treated
as effectful — assuming otherwise is assuming in the expensive direction. Supply an
idempotency key and the attempts come back, because the second call then lands on the first
call's result rather than beside it.

```python
attempts_for_tool(charge)                 # 1
attempts_for_tool(charge, keyed=True)     # the policy's attempts
```

## Windows follow the tool, not the worker

A global heartbeat constant kills exactly the calls it was meant to protect.
`activity_policy_for(cls, timeout=...)` takes the timeout the tool declared and fits a
heartbeat window inside it — a tenth of the timeout, never under 5s (a slow first token is
not a dead worker) and never over 60s (past that a dead worker goes unnoticed either way).
A policy a consumer already tuned keeps its retry settings; only the windows are derived.

An `ActivityPolicy` whose heartbeat window is wider than the activity is refused at
construction. Nothing could ever breach it, so it is not a timeout.

## Heartbeats carry counts

`Heartbeater` beats three times per window, so one lost beat is not read as a death, and
never once per token, which is a write per token.

```python
beating = Heartbeater(beats.append, policy=policy, clock=clock, step="step-3")
await beating.chunk(tokens=10)
```

A `Heartbeat` carries the step, a token count, a chunk count and a time — and nothing else;
`is_result` is permanently `False`. A heartbeat carrying the partial text is read as an
answer by the first consumer in a hurry, and half a sentence rendered as a reply is worse
than a spinner.

## Jitter is drawn inside the activity

Never on the workflow path: two replays that draw their own delays are two different
histories. `random.uniform` and `random.gauss` are named by the ADK-W002 replay guard for
exactly this reason. `ActivityAttempts` draws the jitter itself, and cuts the wait to what
is left of `start_to_close_seconds` rather than scheduling one the activity cannot afford —
a wait that does not fit is reported as `truncated`, because dropping it silently leaves an
operator reading a schedule the run never followed.

## The ceiling is charged before the attempt

Each attempt is recorded against the run's budget *before* it is made, so a retry the
allowance cannot cover never happens and the ceiling is never widened to let one more
through. The resulting `BudgetExceededError` reports how many attempts were spent reaching
it, under `details["attempts"]`.

## Retry storms across a fan-out

Ten branches each retrying three times is thirty calls at a provider that is already
struggling. `AttemptBudget` is one pool of retries shared by every branch; it caps the total
and never refuses a first attempt, which is the work itself rather than a retry.

```python
pool = AttemptBudget(4)
ActivityAttempts(policy, clock=clock, shared=pool)  # per branch, one pool
```
