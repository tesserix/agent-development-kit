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
| Memory admission — what may persist, provenance on every fact, re-judged on the way back in | [`docs/memory-admission.md`](docs/memory-admission.md) |
| Content compression — routing admitted content to a compressor that understands it, and declining where nothing does | [`docs/content-compression.md`](docs/content-compression.md) |
| Reversible compression — a retrieval handle on compressed content, scoped to one run, audited on the way back | [`docs/reversible-compression.md`](docs/reversible-compression.md) |
| Frozen prefix — the boundary compression may not cross, so token savings do not cost the prefix cache | [`docs/frozen-prefix.md`](docs/frozen-prefix.md) |
| Memory adapters — Redis, PostgreSQL and pgvector, settings, retries, paging | [`docs/memory-adapters.md`](docs/memory-adapters.md) |
| Graph memory — relations over a temporal graph, metered extraction, tenant ceilings, queued writes | [`docs/graph-memory.md`](docs/graph-memory.md) |
| Sandboxed code execution — what the code cannot reach, ceilings, artifacts, stronger backends | [`docs/sandbox.md`](docs/sandbox.md) |
| Claim-check tool results — heads and handles, thresholds, handle scope, reading back | [`docs/claim-check.md`](docs/claim-check.md) |
| Document and audio intake — OCR and transcription on the CPU, cited by page and timestamp | [`docs/document-intake.md`](docs/document-intake.md) |
| Context assembly — the plan, the provider's budget, pinning, compaction strategies, failing closed | [`docs/context-assembly.md`](docs/context-assembly.md) |
| Providers — the protocol, capability declaration, conformance | [`docs/providers.md`](docs/providers.md) |
| CPU inference — llama.cpp, GGUF quantization, fitting a model, tuning | [`docs/cpu-inference.md`](docs/cpu-inference.md) |
| Resilience — the error taxonomy, redaction, phase timeouts, rate limiting | [`docs/resilience.md`](docs/resilience.md) |
| Routing — task classes, the routing table, precedence, entitlements | [`docs/routing.md`](docs/routing.md) |
| Output shaping — effort clamped on resumption turns, terseness steered by suffix only | [`docs/output-shaping.md`](docs/output-shaping.md) |
| Trust boundaries — fail-closed fallback, recorded rationale | [`docs/trust-boundary.md`](docs/trust-boundary.md) |
| Fallback — the chain, eligible failures, side-effect safety, attribution | [`docs/fallback.md`](docs/fallback.md) |
| Usage and cost — normalised counts, decimal money, dated prices, confidence | [`docs/cost.md`](docs/cost.md) |
| Budgets — the limits vocabulary, scope precedence, tenant ledgers, mid-run enforcement | [`docs/budget.md`](docs/budget.md) |
| Cost attribution — chargeback dimensions, metric cardinality, redaction, reconciliation | [`docs/cost-attribution.md`](docs/cost-attribution.md) |
| Multi-agent tracing — one trace across processes, per-participant cost, totals that name what is missing | [`docs/multi-agent-trace.md`](docs/multi-agent-trace.md) |
| Tenancy — context-carried tenant identity, refusal on absence, declared crossings | [`docs/tenancy.md`](docs/tenancy.md) |
| Tenant propagation — one wire contract across queues, peers and workflows | [`docs/tenant-propagation.md`](docs/tenant-propagation.md) |
| Per-tenant configuration — entitlements as data, resolved once, failing closed | [`docs/tenant-config.md`](docs/tenant-config.md) |
| Secrets — config holds a reference, the value is fetched at the point of use, cached with a ttl and never logged | [`docs/secrets.md`](docs/secrets.md) |
| Chunking — token-sized pieces, strategies per collection, spans a citation can quote | [`docs/chunking.md`](docs/chunking.md) |
| Embedding — batching, a content-addressed cache per tenant, and no substituted vectors | [`docs/embedding.md`](docs/embedding.md) |
| Retrieval — both branches, fused by rank, with the tenant predicate inside the query | [`docs/retrieval.md`](docs/retrieval.md) |
| Reranking — a bounded candidate set, a budgeted call, and the fused order when it fails | [`docs/reranking.md`](docs/reranking.md) |
| Citations — an answer that resolves back to the version and span it was built from | [`docs/citations.md`](docs/citations.md) |
| Local embedding and reranking — quantized ONNX on the CPU, with a stated throughput budget | [`docs/local-embeddings.md`](docs/local-embeddings.md) |
| Conversation compaction — older turns folded away, every source they cited carried through | [`docs/conversation-compaction.md`](docs/conversation-compaction.md) |
| Corpus poisoning — retrieved content held as data, fenced, screened and never obeyed | [`docs/corpus-poisoning.md`](docs/corpus-poisoning.md) |
| Estimation — pre-flight cost, confidence, refusal, calibration | [`docs/estimation.md`](docs/estimation.md) |
| The spend ledger — shared ceilings, windows, leases, sharding, coalescing | [`docs/ledger.md`](docs/ledger.md) |
| Structured output — declared answer shapes, provider fallback, violations | [`docs/structured-output.md`](docs/structured-output.md) |
| Repair — bounded correction, what goes back, what it costs | [`docs/repair.md`](docs/repair.md) |
| Typing — the answer type, the escape-hatch policy, third-party boundaries | [`docs/typing.md`](docs/typing.md) |
| The agent definition — owner, evaluation suite, revision, pinning | [`docs/agent-definition.md`](docs/agent-definition.md) |
| Prompt registry — versioned prompt text, aliases resolved to versions, attribution on every span | [`docs/prompt-registry.md`](docs/prompt-registry.md) |
| Prompt templates — declared typed variables, no empty substitutions, retrieved text kept as data | [`docs/prompt-templates.md`](docs/prompt-templates.md) |
| Prompt lint — the rules that belong in code, suppression with a recorded reason, running it in CI | [`docs/prompt-lint.md`](docs/prompt-lint.md) |
| Prompt diff and rollback — what moved between two versions, and repointing an alias safely | [`docs/prompt-rollback.md`](docs/prompt-rollback.md) |
| Prompt gate — per-metric regression thresholds, cost as a gate, promotion tied to a digest | [`docs/prompt-gate.md`](docs/prompt-gate.md) |
| Choosing an execution model — the durability ladder, the decision table, what is not a reason to reach for a workflow | [`docs/execution-model.md`](docs/execution-model.md) |
| Durable runs — a run as a workflow, every call an activity, nothing paid for twice after a restart | [`docs/durable-runs.md`](docs/durable-runs.md) |
| Replay safety — the calls a workflow cannot make, caught in CI rather than on a resumed run | [`docs/replay-safety.md`](docs/replay-safety.md) |
| The run loop — assembly, dispatch, terminal states, cancellation | [`docs/run-loop.md`](docs/run-loop.md) |
| State — versioned writes, patches that commute, cursor paging, session lifetime | [`docs/state.md`](docs/state.md) |
| Checkpointing — frontiers at unambiguous boundaries, resume, calls nobody can decide | [`docs/checkpointing.md`](docs/checkpointing.md) |
| Resuming a run — one worker at a time behind a fenced lease, durable stores, continue-as-new, `adk run --resume` | [`docs/checkpointing.md`](docs/checkpointing.md#one-worker-at-a-time) |
| Compensation — pairing a reversal with the work it undoes, unwinding a failed run, what can never come back | [`docs/compensation.md`](docs/compensation.md) |
| Activity policies — retries and heartbeats per activity class, what is never retried, jitter inside the activity | [`docs/activity-policies.md`](docs/activity-policies.md) |
| Suspension — stopping for a multi-day decision holding nothing, single-use tokens, what re-checks on resume | [`docs/suspension.md`](docs/suspension.md) |
| Work queues — leases, reaper, capped attempts, dead letter, per-tenant fairness | [`docs/work-queue.md`](docs/work-queue.md) |
| State and queue adapters — Redis and PostgreSQL backings, `SKIP LOCKED` claims, one transaction for state and work | [`docs/state-adapters.md`](docs/state-adapters.md) |
| Autonomy — grants with ceilings and expiry, fail-closed levels, reports enforced, instant revocation, nobody grants themselves | [`docs/autonomy.md`](docs/autonomy.md) |
| Audit — one record per autonomous decision, refusals as visible as executions, digests not payloads, erasure that keeps the decision | [`docs/audit.md`](docs/audit.md) |
| Delegation — depth, fan-out and run ceilings, scope narrowing, escalation refused, expiry, and the guards a sub-run inherits | [`docs/delegation.md`](docs/delegation.md) |
| Planning — typed plans, a planner that cannot dispatch, validation before the first step, bounded replanning, resume | [`docs/planning.md`](docs/planning.md) |
| Parallel fan-out — a concurrency cap somebody chose, per-branch spend, declared-order results, aggregates that refuse rather than go partial | [`docs/parallel.md`](docs/parallel.md) |
| Dispatch — declared dependencies, derived schedule, cycles refused at construction, contained failure | [`docs/dispatch.md`](docs/dispatch.md) |
| Escalation ladder — the measured bar for each step from one agent to many, roles that are not services, reasons that need no measurement | [`docs/escalation-ladder.md`](docs/escalation-ladder.md) |
| Guardrails — declared order, redaction that carries, failing closed on a guard that cannot answer | [`docs/guardrails.md`](docs/guardrails.md) |
| Prompt injection — trust that follows the origin, an envelope the payload cannot close, the three things untrusted content may never change | [`docs/prompt-injection.md`](docs/prompt-injection.md) |
| Output validation — the declared type or a typed error, rules decided in code, abstention as an answer, bounded repair | [`docs/output-validation.md`](docs/output-validation.md) |
| PII and content policy — one set of detectors on every path, a placeholder that keeps the subject, one transcript at two tenants' bars | [`docs/pii-and-content-policy.md`](docs/pii-and-content-policy.md) |
| Acting for a caller — declared scopes intersected with the caller's once, refused at dispatch, delegation that only narrows | [`docs/agent-identity.md`](docs/agent-identity.md) |
| Tool credentials — scoped, short-lived, minted per audience and caller, carrying run attribution downstream | [`docs/tool-credentials.md`](docs/tool-credentials.md) |
| MCP credentials — authority on the call rather than the connection, narrowed to the server's allowlist, no token in the metadata | [`docs/mcp-credentials.md`](docs/mcp-credentials.md) |
| Calling a peer agent — the caller's principal on the wire, narrowed on both sides, a chain that cannot loop | [`docs/peer-delegation.md`](docs/peer-delegation.md) |
| Credential refresh — one mint inside the skew window, authority re-derived rather than renewed, a revoked caller that halts | [`docs/credential-refresh.md`](docs/credential-refresh.md) |
| Store isolation — the partition derived from context rather than passed, a mismatched row withheld, guarantees stated with their limits | [`docs/store-isolation.md`](docs/store-isolation.md) |
| Proving isolation — two tenants seeded with confusable data, markers that survive summarising, surfaces nobody read counted as a failure | [`docs/isolation-suite.md`](docs/isolation-suite.md) |
| Spans without wiring — a run-rooted trace the run emits itself, retries as siblings, a collector outage that costs telemetry rather than the run | [`docs/auto-instrumentation.md`](docs/auto-instrumentation.md) |
| Redaction before export — a payload allowlist off by default, nested tool arguments walked, a detector outage that costs an attribute rather than the span | [`docs/export-redaction.md`](docs/export-redaction.md) |
| One set of attribute names — a versioned telemetry convention, unmeasured numbers stated rather than guessed, cardinality declared per name | [`docs/telemetry-convention.md`](docs/telemetry-convention.md) |
| Spend you can alert on — a versioned pricing table on every figure, unreported usage counted as unknown rather than zero, a replay that does not double count | [`docs/spend-metrics.md`](docs/spend-metrics.md) |
| Debugging without a collector — a run drawn as a tree, a failure never drawn as a tidy one, a trace file redacted before it is shared | [`docs/local-trace-view.md`](docs/local-trace-view.md) |
| One trace across hops — W3C headers a non-kit peer already reads, identifiers a replay rebuilds rather than duplicates, a broken hop linked back rather than dropped | [`docs/trace-propagation.md`](docs/trace-propagation.md) |
| Tool allowlists — agent, tenant and caller intersected once, enforced at dispatch, delegation that only narrows | [`docs/tool-allowlists.md`](docs/tool-allowlists.md) |
| Testing a guard — an attack corpus with a benign control set, recall and false positives together, per-guard bars that ratchet | [`docs/guard-testing.md`](docs/guard-testing.md) |
| A model provider you can script — exact token counts, injectable provider faults, an unscripted call that fails rather than answers forever | [`docs/fake-model-provider.md`](docs/fake-model-provider.md) |
| Recorded provider traffic — three modes with replay the default, a miss that names the diverging field, chunk boundaries kept, a credential that refuses to be written | [`docs/cassettes.md`](docs/cassettes.md) |
| Watching a run — typed progress events, consumption patterns, provisional output, cancellation, backpressure | [`docs/run-progress.md`](docs/run-progress.md) |
| Tool concurrency — bounded fan-out, per-tool and per-tenant lanes, per-call failure | [`docs/tool-concurrency.md`](docs/tool-concurrency.md) |
| Connection pooling — client reuse across runs, credential rotation, bounded exhaustion | [`docs/connection-pooling.md`](docs/connection-pooling.md) |
| Embedding batching — coalescing window, identity guarantees, per-item error isolation | [`docs/embedding-batching.md`](docs/embedding-batching.md) |
| Response caching — key determinants, what is refused, tenant isolation, semantic tier | [`docs/response-caching.md`](docs/response-caching.md) |
| Transports — SSE framing, websocket control channel, reconnection, the boundary | [`docs/transports.md`](docs/transports.md) |
| Async and sync — the sync surface, the running-loop refusal, worker pools, stall detection | [`docs/async-and-sync.md`](docs/async-and-sync.md) |
| The context window — keyed admission, eviction order, counting | [`docs/context.md`](docs/context.md) |
| Benchmarks — the gated metrics, variance controls, updating a baseline | [`docs/benchmarks.md`](docs/benchmarks.md) |
| Latency objectives — first token, sustained rate, cache hit ratio, sampling versus audit | [`docs/latency-objectives.md`](docs/latency-objectives.md) |
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
