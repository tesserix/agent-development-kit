# Surviving credential rotation without restarting the run

A run that lasts longer than its token has two conventional escapes, and both are
bad. Lengthening the TTL trades away the reason for short-lived credentials in the
first place. Retrying the whole run re-executes every side effect the run already
committed — a second payment, a second email, a second ticket.

`RunCredentials` takes the third option: refresh in place, and re-derive the
caller's authority while doing it. The run keeps its position. Nothing already done
is done again.

## What it is

```python
from tesserix_adk.runtime import RefreshPolicy, RunCredentials

credentials = RunCredentials(
    broker,
    identity=identity,
    clock=clock,
    policy=RefreshPolicy(skew_seconds=30.0),
    reauthorise=directory,
)

credential = await credentials.for_call(
    audience="https://payments.internal", needs=("payments:read",), run_id=run_id
)
```

`broker` is anything satisfying `tesserix_adk.core.ExpiringCredentialSource` — a
mint whose credentials say when they stop working. `reauthorise` is optional; where
it is given, it is asked on every mint whether the caller still holds what they held.

## Refreshing before the far side notices

A credential is replaced once it is inside `skew_seconds` of expiry, not once it has
expired. The window exists because the clock here and the clock at the far side are
not the same clock: a credential that is fresh by a second here can be stale by a
second there, and is rejected. Widening the skew costs an earlier mint. Narrowing it
costs a rejected call.

Skew is not a guess that has to be right, because a downstream rejection is also
handled — `RunCredentials.call` treats `CredentialExpiredError`, and an
`McpAuthError` whose reason is `EXPIRED`, as a signal to mint and try once more.

## One refresh, however wide the fan-out

Refresh is single-flight per audience. Six tool calls dispatched concurrently inside
the skew window produce one mint, and all six resume on the same credential. Two
different audiences refresh in parallel, since they are unrelated mints.

Cancellation during a refresh leaves nothing behind: no half-written cache entry, no
credential attributed to a call that never happened.

## Refresh is not renewal

A refreshed credential is not the old credential with a later expiry. Each mint goes
through the caller's *current* authority:

- `reauthorise` is asked what the caller holds now.
- What comes back is intersected with what the run already held, so authority can
  only ever narrow. A directory that answers with more scopes than the run started
  with does not widen the run.
- The call's `needs` are checked against that, so a call cannot use a refresh to
  reach past what it could reach before.

Where the caller's grant has been withdrawn — `reauthorise` raises
`AuthorisationError`, or the principal's own expiry has passed — the run halts with
`AuthorityRevokedError`. Held credentials are dropped, `halted` latches true, and
every later call raises rather than proceeding on the credential that is still
technically in hand. The kit does not return a partial result as though it were
complete.

## Suspension

`suspend()` clears every held credential. A run waiting on a human approval gate may
resume days later, and a token carried across that gap is a token that outlived the
window it was minted for. On resume the authority is re-derived through the same
path as any other refresh — which is also what makes a durable run legible: the
authority it acts under is the one recorded for it, not one inherited from a session
that no longer exists.

## Idempotency

The retry after a reactive refresh reuses the original idempotency key, so a
refreshed call cannot duplicate a side effect.

A call made with no key is not retried at all. If it is rejected for expiry, it was
already in flight when the credential lapsed and may well have landed; the kit
raises `CredentialExpiredError` with `outcome="unknown"` and `retryable` false. That
is an honest answer, and it is deliberately not the same as "it failed".

## Transient failure

A mint that cannot be reached is retried with the policy's `RetryConfig`, delayed
with full jitter — many runs sharing one principal all refresh at roughly the same
moment, and unjittered backoff would have them refresh in unison. Once the attempts
are exhausted, the call raises `CredentialExpiredError` with `outcome="not_started"`
and `retryable` true. The run is *not* halted: nothing was refused, something was
unreachable, and the same call is worth making again.

| Error | Meaning | Run continues |
|---|---|---|
| `CredentialExpiredError(outcome="not_started")` | The mint was unreachable; the call was not made | Yes |
| `CredentialExpiredError(outcome="unknown")` | A keyless call was in flight when the credential lapsed | Yes, but the effect is unknown |
| `AuthorityRevokedError` | The caller's grant is gone | No |

## Limitations

- The skew window is per policy, not per audience. A far side with unusually bad
  clock discipline needs its own `RunCredentials`.
- `reauthorise` is asked on every mint, not on every call. Between two mints, a
  revocation is not seen until the credential is next replaced. Short lifetimes are
  what bound that gap; `invalidate(audience)` closes it immediately where something
  else has learned of the revocation.
- Nothing here refreshes a credential the run never asked for. A credential held by
  a long-lived connection outside `RunCredentials` is that connection's problem —
  see [`docs/connection-pooling.md`](connection-pooling.md).

## See also

- [`docs/tool-credentials.md`](tool-credentials.md) — minting and scope derivation.
- [`docs/agent-identity.md`](agent-identity.md) — how effective scopes are resolved.
- [`examples/credential_refresh.py`](../examples/credential_refresh.py).
