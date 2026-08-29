# Google ADK bridge

Tesserix ADK and Google Agent Development Kit are complementary, not interchangeable.
Tesserix supplies a provider-neutral, policy-enforced runner and typed production
boundaries. Google ADK supplies its own agent composition and runtime ecosystem. The
bridge connects them over the official Agent2Agent (A2A) 1.x protocol, so neither runtime
imports the other's internal agent model.

| Concern | Tesserix ADK owns | Google ADK owns | Shared boundary |
|---|---|---|---|
| Local agent execution | Provider, tools, guardrails, budgets, tenant context, telemetry | Google agent and workflow execution | Official A2A task |
| Discovery | Reviewed official Agent Card | Card resolution and remote-agent composition | Official A2A Agent Card |
| Identity | Verified core `Principal` passed by server/gateway | Client credential and request metadata configuration | Authenticated A2A transport context |
| Durability | Tesserix run/store composition chosen by application | Google session/event services | Deployment-selected official A2A `TaskStore` |

The model behind the Tesserix agent can be OpenAI, Anthropic, Gemini, Groq, xAI/Grok,
OpenRouter, a local model, or any conforming `ModelProvider`. That choice does not change
the A2A contract Google ADK consumes.

## Install the reviewed integration

From a source checkout:

```bash
uv sync --frozen --extra google-adk
```

The extra includes Google ADK's A2A support and the official A2A HTTP server runtime.
Neither SDK enters the base installation. For an application dependency, select an exact
reviewed tag as described in [Keep agents current safely](keeping-current.md); do not
depend on the moving `main` branch.

The compatibility floor is Google ADK 2.8. The lockfile currently verifies Google ADK
2.8.0 with `a2a-sdk` 1.1.2 on Python 3.12 and 3.13. Later compatible versions remain
installable so the dedicated `google-adk` CI leg detects upstream changes.

## Try the offline assembly

This example builds a real official card, Starlette routes, Tesserix executor, and Google
`RemoteA2aAgent` without opening a socket or calling a model:

```bash
uv run --frozen --extra google-adk python examples/google_adk_a2a_bridge.py
```

Read the complete
[bridge example](https://github.com/tesserix/agent-development-kit/blob/main/examples/google_adk_a2a_bridge.py)
before replacing its scripted provider. Its `InMemoryTaskStore` is deliberately local
only.

## Step 1: build the Tesserix agent

Create an `AgentDefinition`, `ModelProvider`, tools, and `AgentRunner` exactly as for any
other Tesserix agent. Start with [Build a custom agent](custom-agent.md) and choose a
provider through [Provider recipes](provider-recipes.md). The bridge receives the complete
runner, so its allowlists, approvals, guardrails, budgets, cancellation, and telemetry
remain active.

The definition is also the source for reviewed public card metadata. Private instructions,
model identifiers, evaluation paths, and owner contact details are not copied into the
card.

## Step 2: publish the official card

```python
from tesserix_adk.adapters import A2AInterface, A2ASkill, a2a_card_for

card = a2a_card_for(
    definition,
    description="Plans an itinerary from traveller constraints.",
    provider_url="https://agents.example.com",
    documentation_url="https://agents.example.com/docs/trip-planner",
    interfaces=(
        A2AInterface(
            url="https://agents.example.com/a2a/trip-planner",
            protocol_binding="JSONRPC",
        ),
    ),
    skills=(
        A2ASkill(
            id="plan-trip",
            name="Plan a trip",
            description="Creates a day-by-day itinerary.",
            tags=("travel", "planning"),
        ),
    ),
)
```

Advertise only behavior the mounted server actually provides. The bridge returns one
final artifact and therefore does not advertise partial streaming by itself.

## Step 3: bind verified identity

Authentication belongs before the executor. Server or gateway middleware validates the
credential, issuer, audience, expiry, and algorithm, authorizes the requested agent, then
places a core `Principal` in official server context. The resolver copies that verified
object; it never reads tenant, user, role, or scopes from A2A message metadata.

```python
from a2a.server.agent_execution import RequestContext
from tesserix_adk.core import Principal


async def resolve_principal(context: RequestContext) -> Principal:
    principal = context.call_context.state.get("principal")
    if not isinstance(principal, Principal):
        raise PermissionError("verified A2A principal is required")
    return principal
```

The resolver runs for execution and cancellation. During an active run, cancellation must
resolve to the original tenant and subject. A mismatch returns the official task-not-found
error so a caller cannot enumerate another tenant's task.

For Starlette or FastAPI, use authentication middleware plus a custom official
`ServerCallContextBuilder` to copy the already-verified principal into
`ServerCallContext.state`. The offline example contains the fail-closed builder. Merely
publishing `A2ABearerSecurity` on a card does not perform authentication.

## Step 4: mount the official server

```python
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette
from tesserix_adk.adapters import a2a_agent_executor

executor = a2a_agent_executor(
    runner,
    definition,
    resolve=resolve_principal,
)
task_store = InMemoryTaskStore()  # local development only
handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=task_store,
    agent_card=card,
)
app = Starlette(
    routes=[
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(
            handler,
            rpc_url="/a2a/trip-planner",
            context_builder=authenticated_context_builder,
        ),
    ]
)
```

In production, inject an official durable `TaskStore` whose owner resolver scopes every
read and write to authenticated server context. The official request handler and store
own duplicate delivery, task lookup, stored messages/artifacts, subscriptions, and
resubscription. The Tesserix executor owns only translation into one run and its active
cancellation token.

A persistent task store does not restart a model call killed with the process. Pair it
with a durable runner or a reconciliation policy that marks or safely resumes tasks left
in `working` after a crash.

## Step 5: connect from Google ADK

```python
from google.adk.agents import LlmAgent
from tesserix_adk.adapters import google_adk_remote_agent

trip_planner = google_adk_remote_agent(
    name="tesserix_trip_planner",
    description="Plans trips through the Tesserix service.",
    agent_card="https://agents.example.com/.well-known/agent-card.json",
    timeout_seconds=30.0,
)

root_agent = LlmAgent(
    name="travel_coordinator",
    model="gemini-2.5-flash",
    instruction="Delegate itinerary requests to the trip planner.",
    sub_agents=[trip_planner],
)
```

The helper deliberately selects Google ADK's current A2A implementation with
`use_legacy=False` and never accepts or stores a token. Configure authentication through
Google ADK's credential configuration, request interceptors, or a reviewed official A2A
client factory. If those advanced parameters are needed, construct Google's
`RemoteA2aAgent` directly and keep `use_legacy=False`; the Tesserix server contract is the
same.

When a card is fetched from a URL, allowlist the expected HTTPS origin. Google ADK 2.8
also checks that RPC targets advertised by a remotely fetched card stay on the card's
origin. A registry or gateway may impose a stronger signature, issuer, endpoint, or
tenant-entitlement policy.

## Gateways and registries

The serving route can sit behind an API gateway, A2A gateway, or service mesh. The gateway
may terminate authentication, rate limits, policy, and telemetry, but it must pass a
verified principal through trusted server context rather than editable headers accepted
from the internet.

For discovery, implement the vendor-neutral `A2ARegistry.resolve(name)` protocol and call
`a2a_client_from_registry`. For a non-standard gateway binding, register the official
SDK's `TransportProducer` with `a2a_client_factory` and advertise the same binding in
`A2AInterface`. Google-specific custom client factories can be supplied directly to
Google's `RemoteA2aAgent` when its default JSON-RPC transport is not enough.

## Limits and failure behavior

| Scenario | Result |
|---|---|
| Missing, invalid, expired, or unavailable identity | Task rejected; runner is not called |
| Cancellation by another tenant or subject | Non-enumerating task-not-found error; run continues |
| Non-user role, binary/data part, empty text, or input over 64 KiB | Task rejected |
| Final artifact over 1 MiB | Task failed; oversized content is not emitted |
| Model, tool, guardrail, or runner failure | Generic failed task with stable code; internal error text stays server-side |
| Cancellation racing completion | Exactly one terminal state wins |
| Task-store outage | Official handler fails; there is no unsafe in-memory fallback |
| Duplicate send or resubscription | Selected official request handler and `TaskStore` decide |
| Process crash during a model call | Stored history survives only with a durable store; execution needs reconciliation |

Current bridge input is text-only. Output is one buffered `text/plain` or
`application/json` artifact. Partial answer streaming, `input-required`, `auth-required`,
automatic workflow resumption, push delivery/webhook verification, and extended-card
authorization endpoints are outside the bridge. A deployment may add them through the
official SDK only after testing and advertising the exact behavior.

## Related

- [Official A2A interoperability](a2a.md)
- [Integrations and gateways](integrations.md)
- [Authentication and threat boundaries](threat-model.md)
- [Tenant propagation](tenant-propagation.md)
- [Keep agents current safely](keeping-current.md)
- [ADR 0003](adr/0003-google-adk-a2a-bridge.md)
