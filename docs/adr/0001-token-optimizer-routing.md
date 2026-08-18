# ADR 0001: Route token optimisation by content origin

## Context

The kit already has deterministic content compression, reversible claim checks, context
assembly, response caching, rate limiting, model routing, and provider observability. RTK
and Headroom solve different parts of the remaining problem: RTK filters a command while
it runs, while Headroom compresses content that has already crossed a tool, retrieval, or
gateway boundary.

The initial sizing assumption is 100 peak optimisation decisions per second per process,
4–64 KiB typical inputs with caller-enforced upper bounds, 10× traffic growth, 99.9% run
availability, and no optimiser-owned durable data in the ADK at 12 or 36 months. Local
selection must stay below 1 ms p99. RTK execution and Headroom MCP calls are dependency
latency and keep their caller deadlines.

The assets are tenant prompts, retrieved documents, tool results, and command output. The
threats are an unauthenticated source controlling tool output, an authenticated caller from
another tenant, and a compromised optimiser. The trust boundary is crossed whenever
content is sent to Headroom or a retrieval hash is redeemed.

## Decision

Add one explicit policy router in the library:

- shell and CLI commands use RTK only when their executable is allowlisted;
- JSON, API, MCP, RAG, conversation, gateway, and multi-agent context use Headroom only
  when the caller explicitly permits that content to cross the Headroom boundary;
- unknown content and unsupported commands pass through unchanged;
- RTK is planned as an argv tuple and never through a shell;
- a Headroom MCP optimiser is bound to exactly one tenant and run;
- compression failure degrades to original content, while retrieval failure and scope
  mismatch fail closed;
- every result names the backend, reason, token counts, savings, transforms, and handle.

The ADK does not deploy a gateway. A Solo Agent Gateway or another host owns authentication,
rate limits, model credentials, and the lifecycle of its Headroom MCP session. The existing
`AgentRunner` continues to own model routing, budgets, retries, and run records.

## Alternatives considered

Always use Headroom was rejected because it sends local CLI output across an unnecessary
boundary and ignores RTK's command-aware filters. Always use RTK was rejected because it
does not handle shared RAG, conversation, MCP, or gateway context. Guessing from content was
rejected because the same JSON bytes can be a local `kubectl` result or tenant-sensitive API
data; origin is part of the contract, not a classifier prediction.

## Consequences

Callers declare origin and external-processing permission. The policy is deterministic and
testable, but cannot rescue a caller that labels content incorrectly. A Headroom session
must not be shared across tenant/run scopes. RTK absence is visible when the returned argv
is executed; Headroom outages are visible in the optimisation result and do not end a run.

Rollout is additive: consumers opt into the router and can compare recorded savings before
making it the default. Rollback removes the router from wiring and restores the original
command/content path; no data migration is required. Cost is one optional MCP call for a
Headroom-eligible item and no new ADK infrastructure.
