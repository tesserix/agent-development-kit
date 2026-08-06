# Failures, waits and allowances

Three vendors have three vocabularies for the same half-dozen events. A caller that
branches on `rate_limit_error` versus `rate_limit_exceeded` versus `RESOURCE_EXHAUSTED`
has written three error handlers and will write a fourth for the next endpoint. This is
the one taxonomy they all land in, what each one carries, and the two things the kit does
in front of the vendor rather than after it: shaping the calls, and bounding the waits.

## One taxonomy

Every adapter raises the same types, and each already knows whether it is worth another
attempt. Nothing above the provider layer reads a vendor string.

| Error | Raised for | `retryable` |
|---|---|---|
| `RateLimitError` | A rate limit, however the vendor spells it | Yes |
| `RateLimitError(quota=True)` | A spent allowance — `insufficient_quota`, hard billing limit | **No** |
| `AuthenticationError` | A rejected or unpermitted key | No |
| `ContentFilteredError` | A prompt or answer the vendor would not process | No |
| `ContextWindowExceededError` | A request longer than the model's window | No |
| `InvalidRequestError` | A malformed request, an unknown model, a 4xx the vendor rejected | No |
| `ProviderTimeoutError` | A wait that ran out, ours or the vendor's own 408 | Yes |
| `ProviderUnavailableError` | 502/503/504, Anthropic's 529, an unreachable host | Yes |
| `ProviderError` | Anything nobody has mapped | Follows the status |

`RetryPlan` asks the error and nothing else:

```python
plan = RetryPlan(RetryConfig(max_attempts=3))
plan.retryable(failure)                      # the type decides
plan.delay_for(1, retry_after=failure.retry_after)
```

The default is deliberate rather than absent. An unrecognised code becomes a plain
`ProviderError` whose retryability follows its HTTP status, so a 500 is tried again and a
418 is not. Guessing that an unknown failure is transient is how a broken deployment
becomes a burst of identical calls.

A quota is not a rate limit. A rate clears by waiting; an allowance clears when somebody
pays. Retrying the second is the same call every time, so it is classified as the
configuration failure it is — still a `RateLimitError`, so a handler catching rate limits
still sees it, but `retryable` is false and `quota` is true.

### What a failure carries

`provider`, `model`, `request_id`, `status`, `retry_after`, and `details["code"]` — the
vendor's own word for the event, because that is what a support ticket is answered
against.

### What it does not carry

The vendor's free-text message is dropped. A 400 body quotes the request that caused it,
and the request body is the prompt; copied into an exception, that is prompt content in
every log line the exception reaches. The status, the code and the request id are enough
to read the failure without it.

An operator debugging against a deployment whose prompts they are already entitled to read
can ask for it back, per provider:

```python
OpenAIProvider("gpt-4o", redact_vendor_messages=False)
```

## Bounding the waits

Connecting and generating get separate budgets. One number for both means either a dead
host is waited on for a minute, or a long answer is cut off at the length of a reasonable
connect.

| Phase | Default | Why |
|---|---|---|
| `connect` | 10s | A host either accepts a connection or it does not |
| `read` | 60s | This one is the model thinking, including between stream frames |
| `write` | 30s | Sending the request body |
| `pool` | 10s | Waiting for a free connection |

`timeout` moves the read budget and `connect_timeout` moves the connect budget; the
defaults are `PHASE_DEFAULTS`, and a provider's own are on `provider.timeouts`.

```python
OpenAIProvider("gpt-4o", timeout=180.0, connect_timeout=3.0)
```

Whichever wait ran out is named on the error as `details["phase"]`. httpx reports a dead
host and a slow model as the same exception; they are different operational events, so an
operator reads one graph rather than one counter.

## Shaping the calls

A key's allowance belongs to the key, not to the process holding it. Twenty concurrent
runs sharing one key each get a twentieth of the limit, discover that as 429s, and retry
into the same wall together. `RateLimiter` is the one place that knows the whole
allowance, so calls are spaced before they are sent rather than rejected after.

```python
shared = RateLimiter(requests_per_minute=600, tokens_per_minute=90_000)
fast = OpenAIProvider("gpt-4o", limiter=shared)
cheap = OpenAIProvider("gpt-4o-mini", limiter=shared)
```

One limiter across every provider holding one key is the point. Separate limiters each
believe they have the whole allowance.

Two buckets, because every vendor meters both requests and tokens; a call waits for
whichever is short. The token cost is the provider's own estimate of the request, taken
before it is sent. Both refill continuously rather than on the minute — a window that
resets on the minute is a stampede on the minute.

`burst` is the fraction of a minute's allowance that may go out at once. One matches what
the vendor's own window allows; a smaller number spreads a burst that would otherwise
arrive as one. A limit that is not above zero is refused at construction, and a single
request larger than the whole token allowance is refused rather than waited on, because no
amount of waiting would make room for it.

A caller cancelled while waiting spends nothing: a cancelled call never reaches the
vendor, so it never used the allowance.

Runnable: [`examples/resilience.py`](../examples/resilience.py).
