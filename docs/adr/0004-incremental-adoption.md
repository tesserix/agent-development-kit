# ADR 0004: incremental adoption with explicit escape hatches

- Status: accepted
- Date: 2026-08-29
- Owners: Tesserix maintainers

## Context

Existing agents already own providers, tools, orchestration, memory, identity and
telemetry. Requiring them to replace all of those in one release creates a large blast
radius, makes regressions impossible to attribute and encourages teams to fork internal
kit code when one integration is missing.

The architecture designs reviewed for this project consistently separate authoring from
execution, effects from outcomes, identity from credentials and typed delegation from
tool calls. A migration must preserve those boundaries even while another framework
remains in charge.

## Decision

Adoption proceeds through five independently useful, reversible stages:

1. observe the existing path without changing decisions;
2. place the model client behind `ModelProvider`;
3. place tools behind typed registry views;
4. move the control loop to `AgentRunner`;
5. add ordered guardrails and tenant-scoped memory.

Each stage publishes before/after evidence for quality, cost, latency, reliability and
safety, and retains a tested rollback. A consumer may stop at any public boundary.

Escape hatches are allowed only through public protocols or adapters. They preserve
authenticated principal, tenant/user, scopes, deadline, cancellation, shared budget,
trace, idempotency and typed outcomes. They never use a private module, widen authority,
put credentials in context, or bypass schema/guardrail checks.

MCP remains capability invocation. Official A2A remains independently addressable task
delegation. A migration may use either, but it does not relabel one as the other.

## Consequences

- Consumers gain value before replacing orchestration or personal-data stores.
- Regressions are attributable to one changed boundary and rollback does not require data
  deletion.
- The kit must keep generic provider, tool, agent, MCP, A2A and registry boundaries small,
  typed and tested.
- Some migrations intentionally retain duplicate infrastructure during a bounded canary
  or dual-read window, increasing temporary operating cost.
- Internal imports and local monkey-patches are explicitly unsupported even when faster in
  the first week; they remove the compatibility and security properties this decision is
  intended to preserve.

## Rejected alternatives

**Flag-day rewrite.** It couples provider, tool, orchestration and data changes, makes
comparison ambiguous and expands rollback risk.

**Framework-specific fork.** It fixes one consumer by creating a permanent security and
compatibility branch outside the public conformance suite.

**A generic untyped callback everywhere.** It appears adaptable but loses schema,
authority, cancellation, usage and failure semantics at precisely the boundaries that need
them most.

## Verification

The migration guide is tested for every stage's immediate value, evidence, rollback,
escape hatch, compatibility policy and decision checklist. The legacy-provider example is
executed in CI. Provider/tool/agent interoperability and official A2A/MCP paths have
offline behavioral tests.
