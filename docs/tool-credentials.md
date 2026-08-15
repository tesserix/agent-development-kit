# Tool credentials

A static key shared by every agent and every tenant is one value that grants everything,
forever, to whoever holds it. When it leaks, the blast radius is every downstream call the
kit can make; and the downstream audit log, full of that one key's requests, cannot say
which run made any of them. Unbounded in scope, in time and in blame at once.

A tool call instead carries a credential minted for one audience, holding only what that
tool needs intersected with what the run holds, expiring in minutes, and naming the run
that made it.

```python
from tesserix_adk.tools import CachingCredentials, CredentialBroker, ExchangedCredentials

broker = CredentialBroker(
    CachingCredentials(ExchangedCredentials(exchange, clock=clock), clock=clock),
    clock=clock,
)

credential = await broker.for_tool(
    identity=identity,          # what the run holds, resolved once at its start
    audience="https://payments.internal",
    needs=("payments:read",),
    run_id=run.id,
)
response = await http.get(url, headers=credential.headers())
```

## A tool cannot ask for more than the run holds

`for_tool` intersects what the tool needs with the run's effective scopes and refuses
anything outside it with `AuthorisationError` — before the mint, so the request is never
made. There is nothing to read back from an audit log and nothing to explain. The scope
set itself comes from [`agent-identity.md`](agent-identity.md); this is the same authority,
carried one hop further.

Asking for no scopes at all is refused too. A credential naming nothing is a request for
whatever the audience is willing to give.

## Lifetime

| | |
|---|---|
| `DEFAULT_TTL_SECONDS` | 300 — long enough that a tool call is not spent at the token endpoint |
| `MAX_TTL_SECONDS` | 3600 — the hard ceiling; a consumer cannot configure past it |
| `EXPIRY_SKEW_SECONDS` | 30 — a credential is spent early, so a fast clock downstream does not reject one this process still considers live |

The lifetime that expires is the one actually granted, not the one asked for: an endpoint
handing back an hour when asked for five minutes is honoured only up to the ceiling.

## Attribution

Every credential carries the run id, agent and agent version, and the tenant and subject
where they are known:

```
X-Tesserix-Run: run_1
X-Tesserix-Agent: desk
X-Tesserix-Agent-Version: 2.1.0
X-Tesserix-Tenant: acme
X-Tesserix-Subject: ada
```

A downstream audit record now ties to one run of one revision of one agent, which is the
difference between "something in the platform called this" and a specific thing to look at.

## Caching, fan-out and retries

`CachingCredentials` keys on tenant, subject, audience and scopes, so nothing is ever
served across a tenant or a caller. Within that key:

- A fan-out asking for the same credential shares a single mint — six concurrent tool calls
  are one request to the token endpoint, not six.
- A retry after a partial failure gets the credential the first attempt used. A
  non-idempotent downstream call authenticated twice as two different callers is a
  duplicate side effect that looks legitimate from both sides.
- A cancelled mint leaves nothing cached.
- `invalidate_all()` drops everything for a revocation nobody wants to wait out.

## Failure has no fallback

An endpoint that is unreachable, or that refuses the requested scopes, raises
`CredentialError` and the tool call fails. The kit does not retry with a broader
credential, does not fall back to a static key, and does not fabricate a tool result. A
tool that quietly authenticates with a shared key when its scoped mint failed has defeated
the point of scoping it.

## The static-key exception

Some third parties only accept a long-lived key. That is modelled explicitly rather than
worked around:

```python
StaticCredentials(
    secrets,
    keys={"https://legacy.example.com": SecretRef(name="{tenant}-legacy-key")},
    clock=clock,
)
```

The key is still a [`SecretRef`](secrets.md) resolved at the point of use, still
per-audience, and still per-tenant where the reference is templated. An audience with no
documented key is refused, so the exception cannot spread by omission — adding one is a
diff somebody reviews.

Every credential this mints is flagged `long_lived`, so the remaining exceptions can be
counted and driven down rather than being invisible.

## Writing an exchange

`TokenExchange` is one method, so the kit takes no HTTP dependency and a deployment binds
its own — RFC 8693 token exchange, GCP workload identity, an internal minting service:

```python
class WorkloadIdentity:
    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        response = await self._client.post(
            self._endpoint,
            json={
                "audience": request.audience,
                "scope": " ".join(sorted(request.scopes)),
                "lifetime": int(request.ttl_seconds),
            },
            headers=request.attribution.headers(),
        )
        body = response.json()
        return body["access_token"], float(body["expires_in"])
```

Raise `CredentialError` for a refusal you can describe; anything else the kit wraps into
one naming the audience.

## Known limitations

- No refresh mid-call. A credential minted for a long-running tool call can expire while
  that call is in flight; the skew window covers seconds, not minutes. Refresh during a
  long run is separate work.
- A revealed token is an ordinary string in the headers dictionary. `headers()` is the one
  place the value appears — do not log the result.
- Scopes are opaque strings with no hierarchy, matching `agent-identity.md`. An endpoint
  that understands `payments:*` is doing so on its own.
- The kit does not verify that the audience it minted for is the host it then called.

## See also

- [`agent-identity.md`](agent-identity.md) — where the run's scopes come from
- [`secrets.md`](secrets.md) — how the static-key exception resolves its key
- [`tool-allowlists.md`](tool-allowlists.md) — whether the tool may run at all
