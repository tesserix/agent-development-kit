# Capability map

This map is the task-oriented index to the complete documentation set. The curated
navigation keeps the common path short; this page makes every implemented capability and
design record reachable without knowing its name in advance. For exact signatures, use the
[generated API reference](reference/index.md). For executable code, use the
[runnable cookbook](cookbook/index.md).

## Agent authoring and execution

- [Agent identity](agent-identity.md) — bind a run to the caller it acts for.
- [Activity policies](activity-policies.md) — bound retries, timeouts, and activity behavior.
- [Audit](audit.md) — record security- and operation-relevant decisions.
- [Autonomy](autonomy.md) — declare how independently an agent may act.
- [Delegation](delegation.md) and [peer delegation](peer-delegation.md) — narrow authority
  across child and remote work.
- [Dispatch](dispatch.md) — derive an execution schedule from declared dependencies.
- [Estimation](estimation.md) — estimate a run before admitting live spend.
- [Execution models](execution-model.md) — choose in-process, queued, or durable execution.
- [Parallel fan-out](parallel.md) — aggregate concurrent branches without hiding failures.
- [Planning](planning.md) — separate model-authored plans from deterministic execution.
- [Suspension](suspension.md) — persist approval waits and resume them safely.
- [Compensation](compensation.md) — account for effects that need deterministic reversal.
- [Escalation ladder](escalation-ladder.md) — choose the smallest agent architecture that
  fits the task.

## Prompts, context, and outputs

- [Context assembly](context-assembly.md) and [conversation compaction](conversation-compaction.md).
- [Content-typed compression](content-compression.md), [reversible compression](reversible-compression.md),
  and the [frozen prefix](frozen-prefix.md).
- [Output validation](output-validation.md) and [output shaping](output-shaping.md).
- [Prompt registry](prompt-registry.md), [templates](prompt-templates.md),
  [linting](prompt-lint.md), [evaluation gate](prompt-gate.md), and
  [rollback](prompt-rollback.md).
- [Response caching](response-caching.md) and [token-optimizer routing](token-optimization.md).
- [Savings accounting](savings-accounting.md) — distinguish measured input savings from
  estimated output savings.

## Knowledge, memory, and retrieval

- [Memory admission and provenance](memory-admission.md) and [graph memory](graph-memory.md).
- [Document and audio intake](document-intake.md).
- [Chunking](chunking.md), [embedding](embedding.md), [embedding batching](embedding-batching.md),
  and [local embedding and reranking](local-embeddings.md).
- [Reranking](reranking.md), [citations](citations.md), and
  [corpus-poisoning defenses](corpus-poisoning.md).

## Tools, code, and remote boundaries

- [Tool approval](tool-approval.md), [concurrency](tool-concurrency.md),
  [credentials](tool-credentials.md), and [typed errors](tool-errors.md).
- [Code intelligence](code-intelligence.md) and [sandboxed code execution](sandbox.md).
- [Credential rotation](credential-refresh.md) and [secret resolution](secrets.md).
- [PII redaction and content policy](pii-and-content-policy.md).
- MCP [authentication context](mcp-auth-context.md), [credentials](mcp-credentials.md),
  [resilience](mcp-resilience.md), [server export](mcp-server.md), and
  [surface pinning](mcp-tool-surface.md).

## State, events, and recovery

- [Event contracts](event-contract.md) and [JetStream delivery](jetstream-events.md).
- [Spend ledger](ledger.md) and [versioned persisted state](state-versioning.md).
- [Durable runs](durable-runs.md), [checkpointing](checkpointing.md),
  [transactional outbox](outbox.md), and [dead-letter replay](dead-letters.md).

## Testing, evaluation, and observability

- [Determinism and replay](determinism.md), [provider cassettes](cassettes.md), and the
  [scripted fake provider](fake-model-provider.md).
- [Benchmarks](benchmarks.md), [compression gates](compression-gate.md), and the
  [calibrated evaluation judge](eval-judge.md).
- [Latency objectives](latency-objectives.md), [local trace view](local-trace-view.md), and
  [spend and performance metrics](spend-metrics.md).

## Developer workflows

- [Command-line guide](cli.md) for the installed `tesserix-adk` command.
- [Runnable cookbook](cookbook/index.md) for offline examples mapped to every public symbol.
- [Testing guide](testing.md), [reference agents](reference-agents.md), and
  [migration guide](migration.md).

## Project records

- [Backlog and issue conventions](backlog.md).
- [ADR 0001: token-optimizer routing](adr/0001-token-optimizer-routing.md).
- [ADR 0002: AgentGateway as the MCP data-plane boundary](adr/0002-agentgateway-mcp-router.md).
- [RFC 0001: package layout and boundaries](rfcs/0001-package-layout.md).
