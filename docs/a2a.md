# Official Agent2Agent interoperability

Tesserix ADK integrates with the official Agent2Agent (A2A) 1.x SDK through an optional
adapter. It intentionally does not relabel Tesserix's existing peer protocol as official
A2A.

```bash
uv sync --frozen --extra a2a
```

That command prepares a source checkout. Applications should add the `a2a` extra to the
exact tagged artifact selected through [Keep agents current safely](keeping-current.md).

## Two agent protocols

| Surface | Purpose | Wire contract |
|---|---|---|
| `tesserix_adk.adapters.a2a` | Official cards, clients, registries, and custom bindings | Official `a2a-sdk` 1.x protobuf types |
| `tesserix_adk.a2a` | Tesserix typed discovery, delegation, peer calls, and trust containment | Tesserix peer models and transports |

They can coexist in one application, but a card or client from one is not accepted by the
other without an explicit bridge.

## Current support

| Official A2A capability | Status |
|---|---|
| Generate an Agent Card | Supported |
| `supportedInterfaces`, protocol version, capabilities, modes, provider, and skills | Supported |
| Bearer security scheme and requirement metadata | Supported |
| Official client factory and interface negotiation | Supported |
| Standard SDK protocol bindings | Supported when the installed official SDK supplies the transport |
| Custom gateway protocol binding | Supported through `ClientFactory.register` |
| Custom registry | Supported through the vendor-neutral `A2ARegistry` protocol |
| Card verification policy | Supported through a caller-supplied callback |
| Agent-name substitution protection | Supported |
| Serve the card over HTTP | Supported through official HTTP server routes; application mounts them |
| Official A2A `AgentExecutor` | Supported through `a2a_agent_executor` |
| Runtime-to-A2A task creation and terminal state mapping | Supported |
| Text input and final text/JSON artifact | Supported with byte bounds |
| Durable task/message/artifact persistence | Deployment injects an official `TaskStore` |
| Task cancellation | Supported with fresh principal authorization and one terminal winner |
| Duplicate delivery, subscriptions, and resubscription | Official request handler and `TaskStore` own them |
| Partial answer streaming and input/auth-required continuation | Not implemented by the bridge |
| Push-notification delivery and webhook verification | Deployment responsibility |
| Extended-card authorization endpoint | Deployment responsibility |

The distinction matters: setting `streaming=True` or `push_notifications=True` on a card
advertises a capability. The bridge buffers one final artifact and does not implement push
delivery. Advertise only behavior the deployment actually provides and tests.

## Publish an official Agent Card

Start with a reviewed `AgentDefinition`. The adapter copies explicit public metadata and
does not publish private instructions, model choice, evaluation paths, or the owner's
on-call address.

```python
from tesserix_adk import Agent
from tesserix_adk.adapters import (
    A2ABearerSecurity,
    A2AInterface,
    A2ASkill,
    a2a_card_for,
)
from tesserix_adk.core import AgentDefinition, Owner

definition = AgentDefinition(
    agent=Agent(
        name="trip-planner",
        version="1.0.0",
        instructions="Private operating instructions.",
        task_class="planning",
        free_text=True,
        scopes=("trips:read",),
    ),
    owner=Owner(
        team="Travel Platform",
        contact="https://agents.example.com/support",
        service="trip-planner-api",
    ),
    evaluation_suite="evals/trip-planner.jsonl",
)

card = a2a_card_for(
    definition,
    description="Plans an itinerary from a traveller's constraints.",
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
            examples=("Plan three accessible days in Melbourne.",),
        ),
    ),
    streaming=True,
    security=A2ABearerSecurity(scopes=("trips:read",)),
)
```

The result is an official SDK `AgentCard` protobuf. Serve its official JSON form through
the application's A2A discovery endpoint with cache headers and a bounded response. Card
generation refuses missing descriptions, interfaces, skills, media modes, non-absolute
URLs, and duplicate skill identifiers before publication.

An A2A skill is an outcome a peer may request, not an internal tool export. Publishing
tool names and arguments would couple callers to implementation details and can expose
privileged operations.

## Create an official client

```python
from tesserix_adk.adapters import a2a_client_factory

factory = a2a_client_factory(
    protocol_bindings=("JSONRPC", "HTTP+JSON"),
    streaming=True,
    polling=False,
    accepted_output_modes=("text/plain", "application/json"),
)

client = factory.create(card)
```

Use official SDK interceptors to add credentials, correlation, and policy at the request
boundary. Applications needing a custom HTTP client can construct the official SDK
`ClientFactory` directly; the ADK helper keeps third-party HTTP types out of its public
contract. Do not put access tokens on the card.

## Resolve through any registry

```python
from a2a.types import AgentCard
from tesserix_adk.adapters import a2a_client_from_registry


class CompanyRegistry:
    async def resolve(self, name: str) -> AgentCard:
        return await trusted_registry_lookup(name)


client = await a2a_client_from_registry(
    CompanyRegistry(),
    "trip-planner",
    factory=factory,
    verify=verify_company_card,
)
```

The helper requires the returned card name to equal the requested name. The verifier is
the place to enforce the registry's stronger guarantees, for example:

- issuer or signature trust;
- card expiry and allowed protocol versions;
- endpoint scheme, host, port, and network allowlists;
- provider organization and skill policy;
- a pinned card fingerprint or approved revision;
- tenant entitlement to discover or call that agent.

A registry returning a syntactically valid card is not proof that the card is authorized
for the caller.

## Connect a custom gateway binding

Official A2A clients negotiate the labels in `supportedInterfaces`. A company gateway
can register its own label:

```python
factory = a2a_client_factory(
    protocol_bindings=("COMPANY-A2A-GATEWAY",),
    transports={
        "COMPANY-A2A-GATEWAY": company_transport_producer,
    },
)
```

`company_transport_producer` follows the official SDK's `TransportProducer` contract
and returns a `ClientTransport`. Put authentication, tenant routing, retries, telemetry,
and connection lifecycle in that transport. The label is automatically included in the
client's supported bindings.

The card must advertise the same binding:

```python
A2AInterface(
    url="https://gateway.example.com/agents/trip-planner",
    protocol_binding="COMPANY-A2A-GATEWAY",
)
```

This works with a Tesserix gateway, another vendor's gateway, or an in-house registry
because the adapter does not prescribe discovery storage or gateway routing.

## Authentication and authorization

`A2ABearerSecurity` writes an official card scheme and requirement. It does not inspect
a request, validate a token, or authorize a task.

The serving gateway or server must:

1. authenticate every request and bind a verified principal;
2. derive tenant from authenticated routing, not from an untrusted task payload;
3. authorize agent, skill, task, context, message, artifact, and cancellation separately;
4. prevent one tenant from reading or subscribing to another tenant's identifiers;
5. attenuate authority on delegation and reject loops or excess depth;
6. enforce size, rate, concurrency, deadline, and spend limits;
7. validate callback URLs and sign push notifications;
8. redact task content and credentials from logs and traces;
9. keep an audit record of task-state and authorization decisions.

Agent Card security metadata should be treated like OpenAPI security metadata: it tells a
client what credential is expected, while enforcement remains server code.

## Serve a Tesserix runner through official A2A

The bridge implements the official SDK's `AgentExecutor`; the application still owns the
HTTP process, authentication middleware, request handler, and task store.

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
    max_input_bytes=64 * 1024,
    max_output_bytes=1024 * 1024,
)
handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),  # local development only
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

`resolve_principal` is an async `A2APrincipalResolver`. It receives the official
`RequestContext` and must return a core `Principal` derived from authenticated server or
gateway context. It is called for execution and cancellation. A resolver failure rejects
execution before the runner is called; cancellation by a different tenant or subject gets
the non-enumerating official task-not-found error.

The bridge accepts user-role text parts only, joins them with newlines, and applies the
UTF-8 byte limits shown above. It maps submitted, working, completed, failed, rejected,
and cancelled outcomes. A completed run emits exactly one final `text/plain` or
`application/json` artifact. Internal exception text is never returned to the peer.

Use an official durable, tenant-scoped `TaskStore` in production. The official handler and
store own duplicate delivery, task lookup, stored messages and artifacts, subscriptions,
and resubscription. Persistent task state alone cannot restart a model call killed by a
process crash; add a durable runner or a reconciliation policy for tasks left working.

The bridge does not implement partial answer streaming, input-required or auth-required
continuation, push delivery/webhook verification, automatic workflow resumption, or an
extended-card authorization endpoint. Those remain deployment features and must not be
advertised until implemented and tested there.

For a complete identity builder and a Google `RemoteA2aAgent` consumer, follow the
[Google ADK bridge](google-adk.md). The offline HTTP integration test exercises the
official client, routes, request handler, task store, executor, runner, and final artifact
without network access.

## Related

- [Integrations and gateways](integrations.md)
- [Google ADK bridge](google-adk.md)
- [Tesserix agent cards](agent-cards.md)
- [Tesserix peer discovery](peer-discovery.md)
- [Tesserix peer invocation](peer-invocation.md)
- [Peer output trust boundary](peer-output.md)
- [Tenant propagation](tenant-propagation.md)
- [Trace propagation](trace-propagation.md)
