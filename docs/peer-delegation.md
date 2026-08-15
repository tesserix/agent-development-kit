# Calling a peer agent

An agent that authenticates to its peer as itself hands that peer its own access, and the
peer has no idea whose work it is doing. A peer with broader reach is then the route to
doing what the caller could not: call it, and it will.

So a peer call carries a **delegation** — the original principal, the chain the work came
through, and a strict subset of what the caller itself holds. This is the wire-level
counterpart to [`docs/delegation.md`](delegation.md), which bounds delegation *inside* one
process.

```python
from tesserix_adk.a2a import AgentCard, DelegationClaims, PeerDelegator, PeerVerifier

payments = AgentCard(
    agent="payments-agent",
    audience="https://payments.peer",
    declared=("itinerary:read", "payments:write"),
    accepted_issuers=("desk",),
)

delegation = await delegator.delegate(
    identity=identity, peer=payments, needs=("itinerary:read",), run_id=run_id
)
await transport.post(url, headers=delegation.headers(), json={"meta": delegation.meta()})
```

On the far side:

```python
identity = PeerVerifier(card, clock=clock).accept(DelegationClaims.from_meta(request.meta))
```

`accept` either returns the identity the peer may run under, or raises. There is no third
outcome, and in particular no path where the peer proceeds on its own service identity.

## Narrowing happens twice

| Side | Cuts to |
|---|---|
| caller, in `delegate` | what the call needs ∩ what the caller holds ∩ what the card declares |
| peer, in `accept` | what arrived ∩ what the card declares |

The second cut is what makes the first one safe to be wrong. A caller that sends more than
it should — a bug, an old build, a peer that is not this kit — still cannot widen anything
on arrival.

A caller holding `itinerary:read` invoking a peer that declares `itinerary:read` and
`payments:write` produces a peer whose effective scopes are `itinerary:read` only. The
peer's payment tool is refused with `AuthorisationError` at dispatch, and the chain naming
the original principal is on both runs' spans.

## What travels

| Rendering | Carries |
|---|---|
| `delegation.headers()` | the credential, minted for the peer's audience |
| `delegation.meta()` | issuer, subject, tenant, audience, scopes, expiry, run, chain |
| `delegation.span_attributes()` | the same identifiers, for the calling run's span |

No token material appears in `meta()` or `span_attributes()`. The chain is identifiers
only — agent names, versions and scope names — never payloads.

## The wire contract

`meta()` and `DelegationClaims.from_meta()` are the documented contract, so a peer that is
not this kit can verify and honour a delegation. Every key is prefixed
`tesserix/adk/delegation/`:

| Key | Value |
|---|---|
| `issuer` | the calling agent |
| `subject`, `tenant` | who the work is for |
| `audience` | the peer this was minted for, checked on arrival |
| `scopes` | space-separated |
| `expires` | seconds on the same clock as the credential |
| `run` | the calling run |
| `chain` | `agent@version=scope+scope>agent@version=…`, outermost first |

Metadata that is absent, incomplete or malformed raises `AuthorisationError` before any
model call — a peer that cannot read the delegation refuses, rather than carrying on with
whatever parsed.

## Bounds

`MAX_CHAIN_DEPTH` is 8. A ninth hop raises `DelegationLimitError(reason="depth")`, and a
hop through an agent the work has already been through raises `reason="cycle"` — a cycle
is refused where it would be created, so the refusal names the loop rather than a ceiling.
Together they bound what deep composition costs to carry on every request.

## Known limitations

- The kit mints and verifies claims; it does not sign them. A delegation is trusted
  because the credential presented alongside it was minted for that audience — put the
  claims inside a signed token where the transport between peers is not itself trusted.
- The delegation's expiry is the delegator's `ttl_seconds`, not the credential's own. Set
  it no longer than the credential ttl.
- **A peer's output is untrusted input however well the peer authenticated.** Delegation
  says who is asking; it says nothing about what comes back. Run returned content through
  the same guardrails as any other external text.
- Agent card publication and discovery, typed peer invocation, and trace context
  propagation across hops are not here — they belong to the A2A and observability work.

## See also

- [`docs/delegation.md`](delegation.md) — delegation within one run, and what a child holds
- [`docs/agent-identity.md`](agent-identity.md) — where the caller's effective scopes come from
- [`docs/tool-credentials.md`](tool-credentials.md) — minting the credential this presents
- [`docs/mcp-credentials.md`](mcp-credentials.md) — the same idea for tool servers
