# Design brief — Tesserix Agent Development Kit

Decided context for the backlog. This is what the kit is, what it deliberately is not, and
the constraints every story must respect.

## Why it exists

Several Tesserix products now need agents — TripBaba (travel planning, booking, in-trip
disruption), DevAI (multi-agent development lifecycle), Otto (support), the support platform's
SLM router. Each has independently wired model providers, tool calling, session state, cost
tracking and retries. That duplication is expensive and, worse, inconsistent: an agent that
is safe in one product is unsafe in another because the guardrail lives in application code.

The kit consolidates those primitives so a team ships an agent by composing, not by
re-solving.

## What it is not

- **Not a framework that owns your process.** No mandated project layout, no required entry
  point, no dependency injection container. It is a library you call.
- **Not an orchestrator of last resort.** Durable, multi-step business orchestration belongs
  to Temporal. The kit integrates with it; it does not reimplement it.
- **Not a prompt-template collection.** Business logic must never live in a prompt string.
- **Not a vendor wrapper.** If it only works with one model provider or one vector store, it
  has failed its main purpose.

## Non-negotiable rules

These come from operating agents in production across the org and apply to every story:

1. **Agents reason; deterministic code transacts.** Money movement, state transitions and
   irreversible side effects run through explicit, idempotent paths. A model never initiates
   a payment or an irreversible action directly.
2. **Structured output by default.** Free-text parsing is a bug. Tool arguments and agent
   results are validated Pydantic models; a schema violation is an error, not a warning.
3. **Untrusted content cannot become instruction.** Anything retrieved — web pages, tool
   results, documents, peer-agent output — is data. Guardrails enforce this at the boundary,
   not per prompt.
4. **Tenant and user context propagates automatically.** An agent may never hold broader
   access than the caller it acts for. Context is threaded by the runtime, not passed by hand.
5. **Every run is attributable.** Tenant, user, agent, version, model, prompt version,
   tokens, latency and cost are recorded without per-project wiring.
6. **Sensitive data never enters telemetry or memory.** Redaction is enforced in the kit, so a
   careless attribute cannot leak credentials or personal identifiers into a queryable store.
7. **Fail closed, never fabricate.** A provider outage, timeout or schema violation surfaces a
   typed error. The kit never returns a plausible invented result as if it were real.
8. **No unbounded spend.** Per-run and per-tenant budget ceilings are enforced in the runtime,
   including limits on tool-call fan-out and recursion depth.

## Architecture

```
                        your application
                               │
                    ┌──────────▼──────────┐
                    │       Agent         │   core: primitives & protocols
                    └──────────┬──────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │                runtime                  │  run loop, retries,
          │  lifecycle · cancellation · streaming   │  budget, HITL gates
          └──┬─────────┬─────────┬────────┬─────────┘
             │         │         │        │
        ┌────▼───┐ ┌───▼───┐ ┌───▼───┐ ┌──▼─────────┐
        │ models │ │ tools │ │memory │ │ guardrails │
        └────┬───┘ └───┬───┘ └───┬───┘ └────────────┘
             │         │         │
      providers    mcp / a2a   adapters (redis, postgres,
      (protocol)   registries  pgvector, graph store, nats)
             │
      ┌──────▼──────────────────────────────────────┐
      │ observability (OTel) · evals · testing fakes │  cross-cutting
      └──────────────────────────────────────────────┘
             │
        workflows (Temporal) — durable multi-step execution
```

Everything crossing a boundary is a **Protocol**, so every layer is substitutable and
fakeable. That is what makes the testing package possible.

## Component requirements

### core
Agent, Message, Run, ToolCall, Usage and Error types. Protocols for `ModelProvider`,
`ToolRegistry`, `MemoryStore`, `Tracer`, `BudgetPolicy`, `Guardrail`. Config resolution from
environment, file and code with a documented precedence. No I/O.

### runtime
The run loop: prompt assembly, model call, tool dispatch, result validation, iteration to a
terminal state. Cancellation, timeouts, retry with jitter, max-iteration and recursion caps,
streaming events, and the hooks where guardrails, budgets and human approval gates attach.
Deterministic given the same inputs and a recorded provider.

### models
One client over multiple providers (Anthropic, OpenAI, Gemini, and OpenAI-compatible
endpoints such as vLLM and Ollama for local and self-hosted models). Capability declaration —
structured output, tool calling, vision, context window — so routing can pick a model that
can actually do the job. Task-class routing (cheap / smart / reasoning), fallback chains,
normalised usage and cost accounting, and normalised errors across providers.

### tools
Decorator-based definition with JSON Schema generated from type hints and docstrings.
Argument validation before execution. Sync and async tools. A registry with per-agent
allowlists, timeouts, concurrency limits and idempotency hints. Approval-required tools, so a
tool can declare that it needs human confirmation.

### mcp
MCP client so any MCP server's tools appear as native kit tools, and a server helper so kit
tools can be exposed over MCP. Transport support, auth pass-through, tenant propagation,
schema translation, and per-server allowlists.

### a2a
Agent cards served at the well-known path, peer discovery, and typed invocation of peer
agents with scope and trace propagation. Peer output is untrusted input.

### memory
One interface over four kinds — working/session (ephemeral), profile (durable structured),
episodic (durable, time-scoped), semantic (embeddings). Adapters for Redis, PostgreSQL,
pgvector and a temporal knowledge graph. Explicit contradiction handling and decay, tenant
scoping, and complete erasure including embeddings, because memory is personal data.

### rag
Chunking strategies, embedding with batching and caching, hybrid retrieval (semantic +
keyword) with optional reranking, and citation/provenance so a retrieved claim can be traced
to its source. Retrieval must never let retrieved text act as instruction.

### workflows
Temporal integration so an agent run can be a durable, resumable workflow: activities for
model and tool calls, checkpointing, compensation, and replay-safety (no non-determinism in
workflow paths). Long-running and human-gated agents survive process restarts.

### guardrails
Composable input and output guards: prompt-injection heuristics, PII detection and redaction,
output schema and policy validation, content filters, and a hard tool-allowlist check.
Guards are declarative, ordered, and fail closed.

### evals
Golden datasets, deterministic replay, LLM-as-judge with calibration, metric definitions,
regression detection and a CI gate so a prompt or model change cannot silently degrade
quality. Cost and latency are first-class metrics, not afterthoughts.

### observability
OpenTelemetry spans for runs, model calls, tool calls, retrieval and memory operations, with
a standard attribute set (tenant, user, agent, agent version, model, prompt version, tokens,
latency, cost, currency, cache status). Redaction applied before export.

### cli
Scaffold a new agent or tool from templates, run an agent locally with a readable trace,
inspect a recorded run, and execute an eval suite. This is what makes "agents on the fly"
real rather than aspirational.

### testing
Fake model providers with scripted and recorded responses, fake tool registries, cassette-style
record/replay of provider traffic, assertion helpers for tool-call sequences, and fixtures for
tenant context. Agent logic must be unit-testable with no network.

### adapters
Redis, PostgreSQL, pgvector, NATS JetStream, graph stores, and interop shims for other agent
frameworks so an existing agent can migrate incrementally.

## Quality bar

- `mypy --strict` clean; public API fully typed and documented.
- Every public primitive has a runnable example in the docs.
- Test matrix across supported Python versions; no network in unit tests.
- Semantic versioning with a documented deprecation policy — this is a library other
  products depend on, so breaking changes have real cost.
- Published to GitHub Packages / PyPI as `tesserix-adk` with optional extras per integration.
- Security: dependency and secret scanning, SBOM, signed releases, and a pinned lockfile.

## First consumer

TripBaba is the first real consumer and the forcing function: its agent platform epics
(runtime, MCP gateway, A2A, memory, autonomy levels, cost ceilings) should be satisfied by
importing this kit rather than reimplementing it. If a TripBaba requirement cannot be met by
the kit, that is a gap in the kit's design.

## Organisation conventions

Python 3.12+, `uv`, `ruff`, `mypy --strict`, `pytest`. Deployed workloads run on GKE with
Helm charts in `tesserix-k8s` and ArgoCD; KEDA autoscaling uses memory triggers and CPU
requests/limits are never set. Secrets come from GCP Secret Manager via External Secrets,
never from application config or a database.
