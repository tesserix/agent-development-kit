# Public-readiness review

Review date: 2026-08-28. Scope: package/runtime architecture, provider boundary, official
A2A integration, MCP and gateway seams, security/release controls, testing, examples, and
the public documentation path.

## Executive decision

The repository and documentation site are public and suitable for technical review. The
core library is substantially implemented and heavily gated; it is not a planning-only
repository. Branch, tag, security, Pages, and CODEOWNERS settings remain hosted controls
that maintainers must periodically audit for drift.

It should be presented as a **pre-1.0 production-oriented library**, not as:

- a hosted agent platform;
- a drop-in replacement for another product named ADK;
- a turnkey, process-owning official A2A deployment;
- a universal adapter for every model API;
- a substitute for gateway authentication, infrastructure isolation, or operations.

Production adoption should be approved per subpackage and integration. The official A2A
adapter now includes an `AgentRunner` executor bridge and Google ADK interoperability,
while the embedding service still owns HTTP hosting, authentication, durable task state,
crash recovery, subscriptions, and any push delivery it advertises.

## What is strong today

| Area | Assessment | Evidence |
|---|---|---|
| Core runtime | Ready with pre-1.0 stability limits | Typed agent/run contracts, bounded loop, cancellation, deadlines, retry policy, budgets, structured output, tools, approvals, idempotency, checkpoints |
| Provider abstraction | Ready | Runtime-facing `ModelProvider`, explicit capabilities, normalized errors/usage/streams, conformance suite |
| Hosted portability | Ready for supported wire contracts | Native OpenAI/Anthropic/Gemini plus Groq, xAI/Grok, and OpenRouter compatible presets |
| Self-hosting | Ready with deployment-owned operations | vLLM, Ollama, TGI, llama.cpp, explicit capabilities and timeouts |
| Tool safety | Ready with correct deployment wiring | Typed schemas, default-deny allowlists, scopes, approvals, concurrency, timeouts, result boundary, idempotency |
| Tenant boundary | Strong library controls; infrastructure still matters | Required tenant per run, ambient scoped context, propagation contract, store/isolation tests |
| MCP | Broad alpha integration | Clients, servers, transports, credential context, resilience, surface pinning, AgentGateway routing |
| Tesserix peer protocol | Broad alpha integration | Typed discovery, delegation, invocation, trust containment, registry caching |
| Official A2A | Server bridge ready with deployment-owned operations | Official 1.x cards, clients, registry/custom transport seams, bounded runner executor, cancellation, HTTP round trip, and Google ADK 2.8 card compatibility |
| Durability | Broad primitives and adapters | State, queues, checkpoints, leases, outbox, events, replay-safe workflow primitives |
| Testing | Strong | Strict fake provider, network blocking, HTTP replay, conformance suites, isolation, evals, release gates |
| Supply chain | Strong controls, subject to configuration | Frozen lock, dependency admissions, advisory/licence/secret scans, build provenance and trusted publishing workflows |
| Public onboarding | Addressed in this review | Lean README, quickstart, provider recipes, integration/A2A/testing guides, strict docs build |

At audit start, `make check` passed 8,518 tests with 99.16% coverage, strict Ruff and
mypy, import boundaries, API/event/release gates, and dependency policy. The offline
getting-started example and package build also passed. Final verification after the
review's changes is recorded at the end of this page.

## Gaps that must stay explicit

### Priority 0: required before claiming a turnkey official A2A deployment

1. Inject a durable, tenant-scoped official `TaskStore` and reconcile tasks left working
   after a process crash.
2. Authenticate every route and authorize task, context, message, artifact, subscription,
   and cancellation objects in server/gateway context.
3. Add signed push notifications, callback validation, retries, and dead letters only if
   the deployment advertises push support.
4. Run deployment-specific conformance and fault tests against its selected gateway,
   task store, and at least one additional external A2A implementation.

The bridge now maps bounded text requests, final artifacts, terminal failures, and
cancellation. Card security metadata remains descriptive and cannot replace enforcement.

### Priority 0: required for the public GitHub launch

1. Keep the public repository description, homepage, topics, social preview, and default
   branch protection current.
2. Keep GitHub Pages on **GitHub Actions** and monitor documentation deployments.
3. Keep private vulnerability reporting, secret scanning, push protection, Dependabot
   alerts, and code scanning enabled where the organization plan permits them.
4. Audit required checks and reviewed-pull-request enforcement for `main` after workflow
   or ownership changes.
5. Verify the PyPI project, trusted publisher, release environments, and package ownership
   are suitable for public contributors.
6. Keep the organization-owned `support@tesserix.app` conduct-reporting rota private,
   monitored, and current.

### Priority 1: reliability and adaptability

- Add dedicated adapters for Azure OpenAI, Amazon Bedrock, Vertex AI, and any other
  non-compatible API that is a supported product requirement.
- Add scheduled credentialed integration tests against real provider sandboxes. Keep the
  default suite offline and replay-first.
- Automate model-catalogue freshness and require a reviewed effective date for price,
  context, and capability changes.
- Add fault-injection and soak tests against real Redis, PostgreSQL, NATS, Temporal, MCP,
  gateway, and A2A deployments, including restarts and partial network failure.
- Define supported production SLO profiles for provider latency, queue lag, registry
  staleness, checkpoint recovery, and trace loss.
- Add a standard signed-card verifier or document the one trust service every deployment
  must use.
- Decide whether to publish a unified console entry point. The current module CLIs work,
  but the package intentionally installs no global `adk` binary.
- Reduce the very broad pre-1.0 public API before 1.0. Keep new experimental seams out of
  stable re-export modules until consumers need them.
- Continue reconciling legacy deep-reference pages with implementation. The curated
  public path is current, while older issue-era wording should not be treated as a
  feature-status database.

### Priority 2: ecosystem usability

- Publish framework-specific examples for FastAPI/ASGI lifecycle, Kubernetes shutdown,
  background workers, and serverless constraints.
- Publish a complete sample combining a model gateway, MCP, official A2A discovery,
  durable state, telemetry, and policy without embedding real infrastructure credentials.
- Add generated API reference pages alongside the existing API-surface compatibility
  snapshot.
- Add migration guides from common agent frameworks after the public API is closer to
  1.0.
- Consider TypeScript/Java/Go clients only for cross-language protocol needs; do not copy
  the Python runtime merely for ecosystem parity.

## Failure scenarios reviewed

| Scenario | Existing defense | Remaining deployment responsibility |
|---|---|---|
| Provider is slow or unavailable | Phase/run deadlines, normalized transient errors, opt-in jittered retry, fallback rules | Provider SLO, quota, regional strategy |
| Provider claims success with an error body | Compatible adapter refuses it | Vendor-specific regression recordings |
| Provider omits usage or tool IDs | Preset reconciliation and estimated-count attribution | Validate every deployed model/server version |
| Model lacks a required feature | Explicit capability gate before request | Keep capability records current |
| Tool call is duplicated | Idempotency declaration/store and stable keys | Durable shared store and downstream idempotency |
| Process dies mid-effect | Checkpoint frontier and indeterminate disposition | Recovery runbook and status/read-back APIs |
| Event is delivered twice | Idempotent consumer and transactional outbox | Broker retention and dead-letter operations |
| Registry is unavailable | Tesserix peer cache/stale policy; official registry is injectable | Official registry caching/availability policy in application |
| Registry substitutes an agent | Exact card-name check plus optional verifier | Signature, endpoint, issuer, tenant policy |
| Gateway is bypassed | No automatic direct fallback in AgentGateway route | Network policy and service identity |
| A2A card says Bearer | Official security metadata emitted | Actual token validation and per-object authorization |
| One tenant guesses another object ID | Tenant-scoped runtime/store contracts and isolation suite | Database RLS/partitioning and gateway enforcement |
| Tool or peer returns instructions | Untrusted result/peer boundaries and guardrails | Domain-specific allowlists and human approval |
| Credential rotates | Secret resolved at use time; pools can retire old key | Secret-manager availability and rotation procedure |
| Telemetry backend fails | Observation path designed to fail open and redact | Buffering, sampling, alerting, retention |
| Docs drift | Link test and strict MkDocs build | Executable snippet and owner review expansion |

## Comparison position

The public site linked during this review represents the broader “Agent Development Kit”
category. Tesserix ADK should not compete on the generic acronym alone. Its credible
differentiation is:

- explicit production policy and tenant boundaries;
- infrastructure and protocol substitutability;
- conservative provider capability handling;
- deterministic tests and compatibility gates;
- clear separation of MCP, official A2A, and the richer Tesserix peer protocol;
- a lean base dependency graph.

It is not a fork of Google's ADK and does not promise source compatibility with it. Users
should choose Tesserix ADK when those production controls and replaceable boundaries are
the requirement, not because both projects share three initials.

## Public launch checklist

- [x] Accurate root README and five-minute path
- [x] Step-by-step custom agent guide
- [x] Groq, xAI/Grok, OpenRouter, native, self-hosted, and gateway recipes
- [x] Official A2A support/limitation matrix
- [x] Integration and registry guidance
- [x] Apache-2.0 licence file
- [x] Root contribution guide and pull-request template
- [x] Strict local documentation build and link test
- [x] GitHub Pages workflow in source
- [x] GitHub Actions pinned to reviewed commit SHAs
- [x] Repository made public
- [x] Pages source enabled and deployed
- [x] Public security settings verified
- [x] Organization-owned conduct-report contact chosen and published privately
- [ ] PyPI/trusted-publisher ownership verified
- [ ] First external clean-room onboarding completed

## Verification record

Final verification for the Google ADK/A2A bridge on 2026-08-28 used Python 3.13.15:

- `make check`: passed lint, formatting, import-boundary, strict typing, dependency,
  admission, disclosure, API, event, replay, deprecation, release-note, documentation, and
  complete test gates; 9,148 tests passed, 2 were skipped, 125 were deselected by the
  coverage profile, and total coverage was 98.97%.
- `make audit`, `make secrets`, `make licences`, `make deps`, and `make admissions-check`
  passed with no locked advisories, credential-shaped values, licence-policy violations,
  dependency-policy violations, or unrecorded consumer dependencies.
- `uv lock --check` passed for the 153-package all-extras development graph. The inventory
  now follows dependency-selected extras, so the official A2A HTTP server and psycopg
  binary/pool packages carry their true profiles and package-specific licence decisions.
- `uv build` produced the source archive and universal Python wheel. The wheel contains
  the typed-package marker, A2A executor, and Google ADK helper; the source archive also
  contains the Code of Conduct.
- The offline official HTTP round trip passed through the official client, Starlette
  route, request handler, task store, Tesserix executor, runner, and final artifact. Google
  ADK 2.8.0 accepted the same official card through its current non-legacy A2A path.

These source and artifact checks do not replace the unchecked PyPI or external clean-room
actions in the launch checklist.
