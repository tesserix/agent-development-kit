# When a vendor will not answer

A rate-limited vendor should not end a run that another vendor could finish. But a fallback
is a second bill and, if it happens after a tool ran, possibly a second side effect — so it
is narrow on purpose, and it is never silent. Nobody should have to guess afterwards which
model answered or why the invoice has two calls on it for one question.

## The chain is the routing order

There is no separate fallback configuration. The chain is the eligible candidates of the
rule that already matched, chosen one first:

```toml
[[rules]]
task_class = "cheap"

  [[rules.candidates]]
  provider = "openai"
  model = "gpt-4o-mini"
  capabilities = { tool_calling = true, streaming = true, context_window_tokens = 128000 }

  [[rules.candidates]]
  provider = "anthropic"
  model = "claude-haiku-4-5"
  capabilities = { tool_calling = true, streaming = true, context_window_tokens = 200000 }
```

`cheap` prefers `gpt-4o-mini` and falls back to `claude-haiku-4-5`. A fallback order invented
apart from the routing order is a second opinion on the same question, and the two drift.

Two things follow from building it this way. A candidate the router rejected for missing a
capability is not in the chain either, so falling down it cannot quietly lose structured
output or tool calling. And a **pinned** model has a chain of one: naming the model was the
point, and answering from somewhere else answers a different question.

## The vendor's own retries come first

The chain moves only once this vendor's `RetryConfig` is spent. Leaving on the first 429
gives up an allowance that was about to clear, and costs a whole extra vendor relationship
to save a two-second backoff.

## Which failures are worth another vendor

| Failure | Another vendor? | Why |
|---|---|---|
| `RateLimitError` | yes | Another vendor's allowance is a different allowance — including a spent quota, which no amount of waiting clears. |
| `ProviderUnavailableError` | yes | This vendor's capacity, not the request's. |
| `ProviderTimeoutError` | yes | This vendor's queue. |
| `AuthenticationError` | no | A second vendor will not fix the first one's key. |
| `InvalidRequestError`, `ContextWindowExceededError` | no | The request is wrong; it will be wrong there too. |
| `ContentFilteredError` | no | Shopping a refused prompt around vendors is not a retry strategy. |
| `CapabilityError`, `ModelResponseError` | no | Nothing about the next vendor changes the answer. |
| `BudgetExceededError` | no | A chain is not a way past a ceiling. |
| anything unmapped | no | Opening a second bill on a failure nobody has classified is how one broken deployment becomes two. |

## A stream that already emitted is yours to restart

Once tokens have reached the consumer, the kit does not restart underneath it — that shows
one question two answers. `StreamInterruptedError` carries what was emitted and the decision
to start again is the caller's, explicitly.

## A side effect that must not happen twice

Falling back replays the tool results already recorded rather than invoking the tools again.
That is sound only where invoking once was the whole story, so it is allowed only for tools
the agent declares idempotent:

```python
Agent(
    name="billing",
    instructions="Handle the request.",
    tools=("charge", "lookup"),
    idempotent_tools=("lookup",),
)
```

A run that has already called `charge` fails closed with `FallbackUnsafeError`, naming the
tool. The tool is not called a second time. Fail-closed is the only defensible default here:
a duplicated charge cannot be withdrawn by an apology in a log.

## Everything is on the record

Each attempt records `ATTEMPT_FAILED` with the model and the error class; each move records
`MODEL_FELL_BACK` naming the model taken up and the failure that caused it. The run's `model`
field is the model that actually answered, not the one it started at. When every candidate
refuses, `FallbackExhaustedError` names them all with their reasons — the last refusal alone
is not the story.

Failed attempts still reserve against the run's budget. A chain that did not charge for its
failures would be a way to spend past a ceiling.

## Limits the chain does not evade

- **Cancellation** is checked between candidates. A chain that runs on after the caller let
  go bills them for changing their mind.
- **A candidate that cannot hold the prompt** is skipped with the reason recorded, rather
  than called for a certain `ContextWindowExceededError`.
- **A candidate this runner has no provider for** is skipped and named in the failure.
- **A candidate already tried** is never returned to. That is a loop, not a fallback.

## Deliberately not here

- No health tracking or circuit breaking across runs — a fallback decides this run only.
- No quality comparison between the primary's answer and the fallback's. Measure that on a
  versioned evaluation suite; answering it here would make every live run pay for two.
- The in-process fallback attempt sequence is not itself a durable journal. Put model calls
  behind the [durable workflow](durable-runs.md) activity boundary when the sequence must
  survive a worker restart.

See also [`docs/routing.md`](routing.md) for the table the chain comes from and
[`docs/resilience.md`](resilience.md) for the per-vendor retry policy it waits on.

## The boundary the chain may not cross

Every link has passed the capability floor, and — where a boundary is declared — sits
inside the same one as the chosen model. A chain spent with an out-of-boundary
alternative left fails the run closed rather than promoting it. See
[`trust-boundary.md`](trust-boundary.md).
