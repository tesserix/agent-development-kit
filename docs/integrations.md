# Integrations and gateways

Tesserix Agent Development Kit integrates at typed boundaries. The application chooses concrete
implementations at startup; an agent declaration does not import a database client,
gateway SDK, or vendor model object.

## Installation profiles

| Extra | Adds | Use it for |
|---|---|---|
| `a2a` | Official `a2a-sdk[http-server]` | Official A2A Agent Cards, clients, and server bridge |
| `google-adk` | Google Agent Development Kit with A2A support | Import Google tools/agents or connect either runtime through official A2A |
| `mcp` | MCP SDK and JSON Schema validation | MCP clients, servers, and gateway routing |
| `temporal` | Temporal Python SDK | Durable workflow-engine integration |
| `graphiti` | Graphiti core | Temporal graph memory |
| `redis` | Redis client | Cache, state, checkpoints, leases, ledgers, idempotency |
| `postgres` | psycopg, binary driver, and pool | State, work queues, checkpoints, ledgers, outbox, pgvector |
| `all` | Union of every extra above | Integration development and compatibility testing |

```bash
uv sync --frozen --extra google-adk --extra mcp --extra postgres --extra redis --extra temporal
```

That command prepares a source checkout. Applications should add only the required extra
names to the exact tagged artifact selected through
[Keep agents current safely](keeping-current.md).

Importing the base package does not eagerly import any optional SDK. Reaching an
uninstalled integration raises `MissingExtraError` with the install command.

Use [Framework interoperability](framework-interop.md) to decide whether an existing
asset should be imported as a tool, wrapped as a tool or sub-agent, exported as a direct
callable or MCP capability, or connected through official A2A task delegation.

## Model gateways

There are three supported gateway shapes:

| Gateway contract | Connect with |
|---|---|
| Preserves OpenAI Chat Completions | `OpenAICompatibleProvider`, a preset, base URL, headers, and key variable |
| Preserves a native OpenAI, Anthropic, or Gemini contract | The native adapter with an overridden `base_url` |
| Changes request, response, authentication, or streaming semantics | A dedicated `ModelProvider` adapter |

The compatible adapter protects `Authorization` and `Content-Type` from static-header
overrides. It supports provider-specific completion paths, custom routing/attribution
headers, and an injected HTTP transport. See [Provider recipes](provider-recipes.md).

Do not use static headers for secrets or caller authority. Those must be resolved per
request from verified identity and a secret or credential provider.

## MCP

MCP connects agents to tools, not to other agent task lifecycles. The kit provides:

- stdio and streamable HTTP transports;
- local validation of remotely advertised tool schemas;
- deterministic namespacing and surface pinning;
- per-call tenant/caller authority and scoped credentials;
- timeouts, circuit breaking, required/optional server policy, and bounded payloads;
- exporting an allowlisted set of kit tools through an MCP server;
- AgentGateway routing through operator-approved routes.

Start with [MCP client](mcp-client.md), [MCP transports](mcp-transports.md), and
[AgentGateway](agentgateway.md).

The Agentic Registry and AgentGateway pattern keeps the registry in the control plane:
routes are reconciled into the gateway, while a run uses a pinned tool surface. A registry
outage therefore does not alter an in-flight run. Direct fallback to an MCP server is not
automatic because it could bypass gateway authentication, rate limits, and policy.

## Official Agent2Agent (A2A)

Install `tesserix-adk[a2a]` to use the official A2A SDK types. The adapter supports:

- generating official A2A 1.x Agent Cards from reviewed definitions and explicit public
  skills;
- standard or custom protocol bindings in `supportedInterfaces`;
- official client construction and transport negotiation;
- custom gateway transports registered through the official client factory;
- a vendor-neutral `A2ARegistry.resolve(name)` protocol;
- an optional card-verification callback and agent-name substitution protection;
- an official `AgentExecutor` backed by `AgentRunner` with bounded text input and output;
- verified principal binding for execution and cancellation;
- submitted, working, completed, failed, rejected, and cancelled task mapping.

The application mounts the official request handler and routes and injects the official
`TaskStore`; the adapter does not take over the process or persistence. See
[Official A2A interoperability](a2a.md) for the exact support matrix and security
responsibilities.

## Google Agent Development Kit

Install `tesserix-adk[google-adk]` for the tested Google Agent Development Kit 2.x
integration. Import same-process `FunctionTool` definitions through
`import_google_adk_toolset`, wrap a `BaseAgent` through `wrap_google_adk_agent` and an
application-owned invoker, or create a non-legacy Google `RemoteA2aAgent` from a Tesserix
official card or card URL.

These are framework boundaries, not model-provider coupling: the served Tesserix runner
can continue using Groq, xAI/Grok, OpenRouter, Gemini, OpenAI, Anthropic, a local model, or
any other conforming provider.

The helper stores no token. Configure Google-side credentials and request interceptors in
Google Agent Development Kit, authenticate the server or gateway independently, and resolve only a verified
core `Principal` into the Tesserix runner. Follow the [Google Agent Development Kit bridge](google-adk.md)
for the complete server and client sequence, limitations, and failure behavior.
The framework-neutral [interoperability guide](framework-interop.md) also covers generic
tool and agent adapters, export descriptors, authenticated MCP export, and official A2A
export.

## Tesserix typed peer protocol

`tesserix_adk.a2a` predates and is richer than the official adapter. It supplies typed
cards, tenant-aware discovery, delegation chains, scope attenuation, peer invocation,
streaming progress, output containment, trust policy, and peer-as-tool composition.

It is not wire-compatible with official A2A merely because both concern agents. Use
`tesserix_adk.adapters.a2a` for official A2A types and `tesserix_adk.a2a` only when
both peers intentionally adopt the Tesserix protocol.

## Registries

Registries remain replaceable control-plane components.

For official A2A, implement one method:

```python
from a2a.types import AgentCard


class CompanyRegistry:
    async def resolve(self, name: str) -> AgentCard:
        """Return the trusted card registered under this exact name."""
        ...
```

Then resolve through `a2a_client_from_registry`. The helper refuses a card whose
`card.name` differs from the requested name. Supply `verify=` to enforce signatures,
issuer policy, endpoint allowlists, expiry, or an organization-specific trust chain.

The Tesserix peer stack separately supports static and registry-backed discovery,
capability/version matching, fingerprints, positive/negative caching, and bounded stale
use during registry outages. See [Peer discovery](peer-discovery.md).

## Data and durability

| Concern | In-process/reference | Durable options |
|---|---|---|
| Memory | In-memory memory stores | Redis, PostgreSQL, pgvector, Graphiti |
| State | In-memory implementations in tests | Redis and PostgreSQL |
| Checkpoints and leases | Test fakes | Redis and PostgreSQL |
| Idempotency | In-memory/test implementations | Redis and PostgreSQL |
| Work queue | Test implementations | PostgreSQL `SKIP LOCKED` queue |
| Event delivery | In-memory publishers | NATS JetStream-shaped adapters and transactional outbox |
| Workflow replay | Deterministic workflow primitives | Temporal integration extra |
| Cache and ledgers | In-memory references | Redis and PostgreSQL |

Adapters validate schema/version expectations at startup and carry tenant scope in every
operation. The database, network policy, backups, failover, capacity, credentials, and
disaster-recovery plan remain deployment responsibilities.

Read [State adapters](state-adapters.md), [Checkpointing](checkpointing.md),
[Transactional outbox](outbox.md), and [Idempotency](idempotency.md).

## Custom integration contracts

Prefer the narrowest existing protocol:

| New integration | Implement or register |
|---|---|
| Model API | `ModelProvider` plus `ModelProviderConformance` |
| Secret manager | `SecretProvider` |
| Memory backend | Memory protocol plus `MemoryStoreConformance` |
| State/checkpoint/lease/queue backend | Corresponding protocol and conformance suite |
| Official A2A registry | `A2ARegistry` |
| Official A2A gateway binding | Official SDK `TransportProducer` through `a2a_client_factory` |
| Official A2A server execution | `A2APrincipalResolver` plus `a2a_agent_executor` |
| Tesserix peer transport | `PeerTransport` or `StreamingPeerTransport` |
| MCP route | `McpTransport` or AgentGateway configuration |

`mypy --strict` checks signatures, runtime construction checks required members, and
the behavioral conformance suites check semantics the type system cannot express.

## Boundary checklist

Before enabling any remote integration:

- authenticate the transport peer and derive tenant/caller identity from that credential;
- authorize the exact model, tool, agent skill, task, and stored object;
- use an endpoint allowlist and prevent redirects to untrusted networks;
- set connect, read, operation, and overall run deadlines;
- bound request, response, schema, stream, and card sizes;
- retry only typed transient failures and preserve idempotency;
- redact secrets, prompts, tool arguments, and returned content from telemetry;
- verify cards and registry results before following endpoints;
- test registry, gateway, provider, and store outages independently;
- close pools and sessions on shutdown;
- document which side owns rate limits, persistence, cancellation, and recovery.

The [Threat model](threat-model.md) states what the kit does and does not enforce.
