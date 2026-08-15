# MCP credentials and scope propagation

A configured MCP server is one connection shared by every tenant in the process. The
authority therefore travels with the **call**, never with the connection: a credential is
minted per call, for one tenant, one audience and one narrowed set of scopes.

```python
from tesserix_adk.mcp import McpAuthorizer, McpServerAuth

authorizer = McpAuthorizer(
    broker,  # anything that mints per-call credentials
    servers={
        "bookings-mcp": McpServerAuth(
            server="bookings-mcp",
            audience="https://bookings.mcp",
            scopes=("bookings:read", "bookings:write"),
        )
    },
)

call = await authorizer.authorise(
    server="bookings-mcp", identity=identity, needs=("bookings:read",), run_id=run_id
)
await transport.post(url, headers=call.headers(), json={"_meta": call.meta(), ...})
```

## What the server is handed

| Rendering | Carries | Used for |
|---|---|---|
| `call.headers()` | `Authorization: Bearer …` plus run attribution | the transport, and nothing else |
| `call.meta()` | tenant, subject, run, agent, granted scopes | the MCP request's `_meta`, under `tesserix/adk/` |
| `call.span_attributes()` | server, tenant, granted scopes | telemetry |

Only `headers()` renders the credential. `meta()` and `span_attributes()` are copied into
places a token must not reach — logs, traces, and in some servers the tool result itself —
so they are built from the shape of the authority rather than the authority.

## Narrowing

What a server receives is the intersection of three sets:

```
run's effective scopes  ∩  what the call needs  ∩  the server's declared allowlist
```

`McpServerAuth.scopes` is an allowlist, not a request. A server configured for
`bookings:read` never receives `bookings:admin`, whatever the run holds and whatever the
server asks for. `narrowed_for()` answers the same question without minting anything, for
the negotiation handshake:

- a server asking for elevated scopes during initialisation gets the same intersection —
  negotiation moves the grant down or leaves it alone, never up;
- a server-initiated capability change carries no implied authority: a capability nobody
  configured narrows to the empty set, and a call presenting nothing is refused here
  rather than at the server as an opaque `403`.

## Long-lived sessions

stdio and streaming transports hold a session open for far longer than any one caller's
authority is good for. Attach the credential per request, not per session — `authorise()`
is called for each tool call and returns a credential whose lifetime is the call's, not
the connection's.

Pooled connections are leased through `ServerSessions`, keyed by `(server, tenant,
subject)`. A lease refuses to serve anyone else:

```python
lease = sessions.lease(server="bookings-mcp", identity=identity)
lease.check(identity)  # McpAuthError if this is another tenant's connection
```

That check is the one that matters under fan-out: a scoped per-call credential does not
help if the connection it travelled over is already authenticated as somebody else.

## Failures

`McpAuthError` carries a `reason`, because the three refusals need different fixes:

| `McpAuthReason` | Means | Fix |
|---|---|---|
| `UNAUTHENTICATED` | no usable credential was presented | configuration or the token endpoint |
| `INSUFFICIENT_SCOPE` | the credential was valid and too narrow | the grant, or the server's allowlist |
| `EXPIRED` | it was valid and is not any more | re-mint and retry — the only retryable one |

`McpAuthError.from_status(status, server=…, scopes=…, description=…)` classifies a
server's response. The description is the only place an expiry is distinguishable from a
credential that was never valid.

Two failures stay distinct rather than being collapsed:

- `AuthorisationError` — the kit refusing before the call, because the run does not hold
  the scope. Re-minting will not help.
- `McpAuthError` — the far side refusing, or the credential not being minted at all.

When minting fails, `authorise()` raises and returns nothing. There is no half-initialised
call for a session to fall back on, and no stale token to reuse.

## Known limitations

- The kit ships no MCP transport. `AuthorisedCall` renders headers and `_meta`; sending
  them is the client's job.
- `CredentialSource` and `CallCredential` are structural protocols declared here rather
  than imported from `tesserix_adk.tools`: the two packages are siblings in the layering,
  so this one states what it needs. `tesserix_adk.tools.CredentialBroker` satisfies both.
- Single-flight per tenant comes from the credential source, not from this package — use
  `CachingCredentials` under the broker, as in the example.

## See also

- [`docs/tool-credentials.md`](tool-credentials.md) — minting the credentials this uses
- [`docs/tenancy.md`](tenancy.md) — what a tenant boundary means elsewhere in the kit
- [`docs/agent-identity.md`](agent-identity.md) — where the effective scopes come from
