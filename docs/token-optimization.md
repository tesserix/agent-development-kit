# Token optimiser routing

RTK and Headroom save tokens at different boundaries. The ADK chooses between them from
declared origin, never by guessing what arbitrary text resembles.

| Input | Decision | Why |
|---|---|---|
| `git`, `kubectl`, tests, logs, and other supported CLI commands | RTK | RTK filters while the command runs and understands its output format. |
| JSON and API responses | Headroom | The response already exists and benefits from structured compression plus retrieval. |
| MCP tool responses | Headroom | Compression and the retrieval hash stay on the MCP path. |
| RAG and conversation history | Headroom | These are context inputs, not commands. |
| Gateway and multi-agent context | Headroom | One shared policy can meter compression across callers. |
| Unknown origin or unsupported command | unchanged | A guessed optimiser can destroy meaning or cross the wrong trust boundary. |

Kubernetes has both cases. A local `kubectl get pods` command goes through RTK; JSON from a
Kubernetes API received at the gateway goes through Headroom. The platform does not choose
one product globally.

## Wire it once per run

`HeadroomMcpOptimizer` accepts any MCP session with `call_tool(name, arguments)`. It uses
Headroom's `headroom_compress` and `headroom_retrieve` tools and validates their JSON
responses before exposing them to the run.

```python
from tesserix_adk.memory import (
    HeadroomMcpOptimizer,
    OptimizationChannel,
    TokenOptimizer,
)

headroom = HeadroomMcpOptimizer(mcp_session, tenant="acme", run_id="run-42")
tokens = TokenOptimizer(headroom=headroom)

# Plan before execution. Execute argv directly; never join it into a shell string.
plan = tokens.plan_command(("kubectl", "get", "pods", "-o", "wide"))
assert plan.argv == ("rtk", "kubectl", "get", "pods", "-o", "wide")

# Compress after a structured boundary. Permission is explicit because this may send
# tenant content to another process or service.
result = await tokens.optimize(
    api_response,
    channel=OptimizationChannel.JSON,
    tenant="acme",
    run_id="run-42",
    headroom_allowed=True,
    untrusted=True,
)

if result.handle:
    original = await tokens.retrieve(result.handle, tenant="acme", run_id="run-42")
```

Construct one `HeadroomMcpOptimizer` per tenant and run. The Headroom MCP wire format uses a
retrieval hash but does not carry tenant or run identity; binding the adapter is what stops
a hash from another run being redeemed through the same object. A mismatch fails before an
MCP call is made.

`headroom_allowed` defaults to false. A deployment with an in-cluster Headroom server may
set it at its trusted gateway boundary; an application using an external service can apply
its data-residency and content policy first. Compression preserves the `untrusted` label and
is not sanitisation.

## Failure and observability

Headroom compression failure returns the original content with `backend="none"` and a
reason, so an optimiser outage costs tokens rather than availability. Retrieval fails
closed because returning guessed or cross-scope content would be a correctness and tenancy
failure. RTK is returned as argv only; process execution, timeout, and exit status remain
the command tool's responsibility.

Every `OptimizationResult` records original, optimised, and saved token counts, transforms,
the backend, decision reason, retrieval handle, and trust label. Feed those fields into the
same run span that already holds model usage and cost. No prompt or tool payload needs to be
logged.

## Relationship to the rest of the ADK

The optimiser sits before the existing run path:

```text
agent / LangGraph / custom SDK
        |
        +-- command ----------------> TokenOptimizer.plan_command -> RTK
        |
        +-- JSON/MCP/RAG/context ----> TokenOptimizer.optimize ----> Headroom MCP
                                                                  |
                                                                  v
AgentRunner -> auth/policy/budget -> routing/cache/rate limit -> provider
                                               |                   |
                                               +---- usage/cost/trace
```

`ContentRouter` remains the offline, deterministic local compressor.
`ReversibleRouter` remains the local tenant/run-scoped alternative when sending content to
Headroom is not allowed. `CachingProvider` caches deterministic model responses. The model
router, provider fallbacks, budgets, rate limiters, and spend metrics remain in their
existing boundaries; token optimisation does not bypass any of them.

The architectural decision and rollback path are recorded in
[`ADR 0001`](adr/0001-token-optimizer-routing.md).
