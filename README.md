# Agent Development Kit

**A reusable Python toolkit for building production AI agents on the fly.**

Every Tesserix product that needs an agent currently re-solves the same problems: wiring a
model provider, defining tools, threading tenant and auth context, tracking token cost,
persisting session state, retrying a flaky provider, and proving the thing works. This kit
exists so that work is done once, correctly, and imported.

The design goal is a **thin, composable, typed** library — not a framework that owns your
process. You should be able to build a useful agent in twenty lines, and a durable
multi-agent system without abandoning the same primitives.

## Design principles

1. **Composable over configurable.** Small primitives that combine, not one god-object with
   fifty keyword arguments.
2. **Typed and predictable.** Pydantic models at every boundary. Structured output is the
   default, not an add-on.
3. **Provider-agnostic.** Models, memory stores, vector stores, queues and graph databases
   sit behind protocols. Swapping OpenAI for Anthropic, or pgvector for a graph store, is a
   config change — never a rewrite.
4. **Deterministic where it matters.** Agents reason; code transacts. Money, state
   transitions and side effects run through explicit, idempotent, testable paths.
5. **Observable by default.** Every agent run, tool call and model call is a span with token
   and cost attributes, with no per-project wiring.
6. **Testable without a network.** Fakes and recorded fixtures ship as a first-class
   package, so agent logic is unit-testable in CI.
7. **Async-first**, with sync wrappers where they genuinely help.

## Packages

One distribution, `tesserix-adk`, with optional extras per integration so a small agent
stays a small dependency.

```bash
uv add tesserix-adk                    # pydantic, httpx, opentelemetry-api — nothing else
uv add 'tesserix-adk[redis,postgres]'  # add only the integrations you use
```

Extras: `mcp`, `temporal`, `graphiti`, `redis`, `postgres`, and `all` as their union.
Reaching an integration you have not installed raises `MissingExtraError` naming the extra
and the install command. The rule and its tests are in
[`docs/contributing.md`](docs/contributing.md#extras-and-the-base-footprint).

| Subpackage | Responsibility |
|---|---|
| `core` | Agent and message primitives, protocols, errors, config |
| `runtime` | Run loop, lifecycle, cancellation, retries, streaming |
| `models` | Provider-agnostic LLM client, routing, structured output, fallback |
| `tools` | Tool definition, schema generation, validation, registry, allowlists |
| `mcp` | MCP client and server integration |
| `a2a` | Agent-to-agent cards, discovery, invocation |
| `memory` | Working, profile, episodic and semantic memory behind one interface |
| `rag` | Chunking, embedding, hybrid retrieval, reranking |
| `workflows` | Durable orchestration (Temporal), checkpointing, sagas |
| `guardrails` | Prompt-injection defence, PII redaction, output validation |
| `evals` | Golden sets, judges, regression harness, CI gating |
| `observability` | OpenTelemetry tracing, token and cost accounting |
| `cli` | Scaffolding, run, inspect, eval |
| `testing` | Fakes, recorded fixtures, deterministic replay |
| `adapters` | Redis, PostgreSQL, NATS, graph stores, framework interop |

```python
from tesserix_adk import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    ...

agent = Agent(model="anthropic:claude-sonnet-5", tools=[get_weather])
result = await agent.run("What should I pack for Kyoto tomorrow?")
```

A runnable version of the seams above, needing no network and no credentials, is
[`examples/getting_started.py`](examples/getting_started.py) — the same file every
release installs from the index and runs, once per extra.

## Status

Planning repository. The implementation plan lives as GitHub issues on the project board;
code lands here as the kit is built.

| Resource | Where |
|---|---|
| Design brief & architecture | [`docs/design-brief.md`](docs/design-brief.md) |
| Core primitives — agents, messages, runs, usage, errors | [`docs/primitives.md`](docs/primitives.md) |
| Boundary models — strictness, extras, sensitive fields, field changes | [`docs/models.md`](docs/models.md) |
| Schemas — generation, docstrings, provider dialects, the schema hash | [`docs/schemas.md`](docs/schemas.md) |
| Tools — the `@tool` decorator, import-time refusals, argument validation, injected context, per-agent allowlists and ceilings | [`docs/tools.md`](docs/tools.md) |
| Tool results — the untrusted-data envelope, return-type enforcement, injection heuristics, suspicion policy, conformance fixtures | [`docs/tool-results.md`](docs/tool-results.md) |
| Tool errors — failure vs refusal, reason codes, declarative exception mapping, what the loop retries | [`docs/tool-errors.md`](docs/tool-errors.md) |
| Tool approval — declaring a gate on the tool, redacted summaries, grants bound to one payload, what a denial means | [`docs/tool-approval.md`](docs/tool-approval.md) |
| Tool idempotency — declaring what a repeat would do, key derivation, durable stores, the at-most-once guarantee | [`docs/tool-idempotency.md`](docs/tool-idempotency.md) |
| Memory — the four kinds, scope in every signature, capabilities checked at bind time, adapter guarantees | [`docs/memory.md`](docs/memory.md) |
| Changing beliefs — supersession, bitemporal reads, contradictions, decay | [`docs/beliefs.md`](docs/beliefs.md) |
| Redaction and erasure — masking on write, derived artefacts, two-phase erasure, receipts | [`docs/erasure.md`](docs/erasure.md) |
| Context assembly — the plan, the provider's budget, pinning, compaction strategies, failing closed | [`docs/context-assembly.md`](docs/context-assembly.md) |
| Providers — the protocol, capability declaration, conformance | [`docs/providers.md`](docs/providers.md) |
| CPU inference — llama.cpp, GGUF quantization, fitting a model, tuning | [`docs/cpu-inference.md`](docs/cpu-inference.md) |
| Resilience — the error taxonomy, redaction, phase timeouts, rate limiting | [`docs/resilience.md`](docs/resilience.md) |
| Routing — task classes, the routing table, precedence, entitlements | [`docs/routing.md`](docs/routing.md) |
| Trust boundaries — fail-closed fallback, recorded rationale | [`docs/trust-boundary.md`](docs/trust-boundary.md) |
| Fallback — the chain, eligible failures, side-effect safety, attribution | [`docs/fallback.md`](docs/fallback.md) |
| Usage and cost — normalised counts, decimal money, dated prices, confidence | [`docs/cost.md`](docs/cost.md) |
| Budgets — the limits vocabulary, scope precedence, tenant ledgers, mid-run enforcement | [`docs/budget.md`](docs/budget.md) |
| Cost attribution — chargeback dimensions, metric cardinality, redaction, reconciliation | [`docs/cost-attribution.md`](docs/cost-attribution.md) |
| Estimation — pre-flight cost, confidence, refusal, calibration | [`docs/estimation.md`](docs/estimation.md) |
| The spend ledger — shared ceilings, windows, leases, sharding, coalescing | [`docs/ledger.md`](docs/ledger.md) |
| Structured output — declared answer shapes, provider fallback, violations | [`docs/structured-output.md`](docs/structured-output.md) |
| Repair — bounded correction, what goes back, what it costs | [`docs/repair.md`](docs/repair.md) |
| Typing — the answer type, the escape-hatch policy, third-party boundaries | [`docs/typing.md`](docs/typing.md) |
| The agent definition — owner, evaluation suite, revision, pinning | [`docs/agent-definition.md`](docs/agent-definition.md) |
| The run loop — assembly, dispatch, terminal states, cancellation | [`docs/run-loop.md`](docs/run-loop.md) |
| Watching a run — typed progress events, consumption patterns, provisional output, cancellation, backpressure | [`docs/run-progress.md`](docs/run-progress.md) |
| Tool concurrency — bounded fan-out, per-tool and per-tenant lanes, per-call failure | [`docs/tool-concurrency.md`](docs/tool-concurrency.md) |
| Connection pooling — client reuse across runs, credential rotation, bounded exhaustion | [`docs/connection-pooling.md`](docs/connection-pooling.md) |
| Embedding batching — coalescing window, identity guarantees, per-item error isolation | [`docs/embedding-batching.md`](docs/embedding-batching.md) |
| Response caching — key determinants, what is refused, tenant isolation, semantic tier | [`docs/response-caching.md`](docs/response-caching.md) |
| Transports — SSE framing, websocket control channel, reconnection, the boundary | [`docs/transports.md`](docs/transports.md) |
| Async and sync — the sync surface, the running-loop refusal, worker pools, stall detection | [`docs/async-and-sync.md`](docs/async-and-sync.md) |
| The context window — keyed admission, eviction order, counting | [`docs/context.md`](docs/context.md) |
| Benchmarks — the gated metrics, variance controls, updating a baseline | [`docs/benchmarks.md`](docs/benchmarks.md) |
| Determinism, cassettes & replay | [`docs/determinism.md`](docs/determinism.md) |
| Backlog conventions | [`docs/backlog.md`](docs/backlog.md) |
| Versioning & deprecations | [`docs/versioning.md`](docs/versioning.md) |
| Releasing & index policy | [`docs/releasing.md`](docs/releasing.md) |
| Stability & the alpha channel | [`docs/stability.md`](docs/stability.md) |
| Reporting a vulnerability | [`SECURITY.md`](SECURITY.md) |
| Threat model — and what the kit does not defend against | [`docs/threat-model.md`](docs/threat-model.md) |
| Dependencies, admission gate & update policy | [`docs/dependencies.md`](docs/dependencies.md) |
| Security scanning, licences & SBOM | [`docs/security.md`](docs/security.md) |
| Verifying a release | [`docs/verifying.md`](docs/verifying.md) |
| Issue template | [`.github/ISSUE_TEMPLATE/engineering-story.md`](.github/ISSUE_TEMPLATE/engineering-story.md) |

## A note on the name

Google publishes its own "Agent Development Kit" and owns `google-adk` on PyPI, so **"ADK"
is not a distinctive name**. This repository keeps `agent-development-kit` as the internal
repo name, but the published distribution is `tesserix-adk` (verified available) with the
import namespace `tesserix_adk`. If the kit is ever open-sourced or promoted externally, it
should be given a distinct product name rather than competing on a generic acronym.

## Intended stack

Python 3.12+ · Pydantic v2 · `asyncio` · `httpx` · OpenTelemetry · `uv` for dependency
management · `ruff` + `mypy --strict` · `pytest` + `pytest-asyncio`.

Integrations are optional extras: Temporal, NATS JetStream, Redis, PostgreSQL/pgvector,
MCP, and a graph store for episodic memory.
