# Backlog conventions

Same model as the other tesserix planning repositories (`tesserix/hms`, `tesserix/TripBaba`):
every work item is a GitHub issue in this repository written to one engineering-story
template, classified by labels, and tracked on a single org-level project board.

**Board:** [Agent Development Kit](https://github.com/orgs/tesserix/projects/14) (private).

## Issue template

All issues use [`.github/ISSUE_TEMPLATE/engineering-story.md`](https://github.com/tesserix/agent-development-kit/blob/main/.github/ISSUE_TEMPLATE/engineering-story.md):
Situation → Task (story + in/out of scope) → Acceptance Criteria (primary, failure, edge
cases) → Engineering Guardrails → Result → PR Evidence → Definition of Done → Exceptions.

Because this is a **library other products depend on**, two rules apply beyond the usual:

- **The failure scenario is mandatory and must be real** — provider outage or timeout, schema
  violation, tool raising, budget exceeded, cancellation mid-run, a retry causing a duplicate
  side effect, an injection attempt in retrieved content.
- **Every public surface carries four obligations**: full typing (`mypy --strict`), a
  docstring, a runnable example, and a test that needs no network.

## Board fields

| Field | Values |
|---|---|
| **Status** | Backlog · Ready · In Progress · In Review · Blocked · Test · Security Review · Done |
| **Priority** | P0 Kit critical · P1 v1 core · P2 Expansion · P3 Ecosystem / Later |
| **MVP** | M0–M6 (see below) |
| **Epic** | one of 28 epics, mirrored as an `epic:` label |
| **Product** | the subpackage, mirrored as a `pkg:` label |

| Milestone | Scope |
|---|---|
| M0 | Design & Foundations |
| M1 | Core Runtime & Single Agent |
| M2 | Tools, MCP & Memory |
| M3 | Multi-Agent & Durable Orchestration |
| M4 | Guardrails, Evals & Observability |
| M5 | DX, Docs & 1.0 Release |
| M6 | Ecosystem & Adapters |

Priority and MVP must agree: a P0 sits in M0–M1, a P1 in M1–M2.

## Labels

| Prefix | Meaning |
|---|---|
| `type:` | `feature`, `planning` (RFC/ADR), `spike` (time-boxed research with a decision as output) |
| `pkg:` | the subpackage: `core`, `runtime`, `models`, `tools`, `mcp`, `a2a`, `memory`, `rag`, `workflows`, `guardrails`, `evals`, `observability`, `cli`, `testing`, `adapters` |
| `team:` | `core`, `ai`, `platform`, `devex`, `docs`, `qa`, `identity-security`, `sre` |
| `epic:` | the 28 epics, matching the Epic board field |
| `area:` | `core`, `integration`, `docs`, `testing`, `infra`, `devex`, `perf` |

Every issue carries exactly one `type:`, one `pkg:`, one `team:` and one `epic:` label, plus
an optional `area:`.

## Epics

Foundations — Kit Architecture & Packaging · Release Engineering & CI ·
Security & Supply Chain · Documentation & Examples

Runtime — Agent Runtime & Lifecycle · Structured Output & Type Safety ·
Streaming & Transport · Performance & Concurrency

Models — Model Gateway & Providers · Prompt & Template Registry · Cost Quota & Budgets

Tools & interop — Tools & Tool Registry · MCP Integration · A2A Interop ·
Framework Adapters & Interop

Knowledge — Memory & Context · RAG & Retrieval · State & Persistence Adapters ·
Eventing & Messaging

Orchestration — Workflow Orchestration & Durability · Multi-Agent Patterns ·
Human-in-the-Loop & Autonomy

Safety & operability — Guardrails & Safety · Auth Identity & Secrets ·
Multi-Tenancy & Isolation · Observability & Tracing

Quality & DX — Evaluation & Testing Harness · Developer Experience & CLI

## Non-negotiables every relevant issue must respect

Taken from [`design-brief.md`](design-brief.md):

- Agents reason; deterministic code transacts. A model never initiates a payment or an
  irreversible action directly.
- Structured output by default — a schema violation is an error, not a warning.
- Untrusted content never becomes instruction, enforced at the boundary rather than per prompt.
- Tenant and user context propagates automatically; an agent never holds broader access than
  the caller it acts for.
- Every run is attributable without per-project wiring.
- Sensitive data never enters telemetry or memory.
- Fail closed, never fabricate.
- No unbounded spend — per-run and per-tenant ceilings, plus fan-out and recursion caps.
- Nothing hard-wires a single vendor; providers and stores sit behind protocols.
