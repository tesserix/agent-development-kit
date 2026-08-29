# ADR 0005: add structured input without reinterpreting `Agent`

- Status: accepted
- Date: 2026-08-29
- Owners: Tesserix maintainers
- Tracks: GitHub issue #161

## Context

Release 0.52.0 defines `Agent[OutputT]`, `AgentDefinition[OutputT]`, and string input on
`AgentRunner.run`, `stream`, and their synchronous variants. Changing those classes to
`Agent[InputT, OutputT]` changes what an existing `Agent[TripPlan]` annotation means even
when runtime calls still happen to work. The release compatibility gate found 23 changed
public aliases or signatures, all without a completed deprecation window.

Applications also need a statically checked Pydantic request boundary. That request must
be validated before policy or provider work and must have a serializable JSON Schema for
registry, function, and MCP exports.

## Decision

Keep the released contracts unchanged and add separately named contracts:

- `TypedAgent[InputT, OutputT]` extends `Agent[OutputT]` with `input_type` and canonical
  request rendering;
- `TypedAgentDefinition[InputT, OutputT]` records the input schema in its reviewed
  revision;
- `run_typed`, `stream_typed`, `run_typed_sync`, and `stream_typed_sync` accept the typed
  request while delegating to the same private execution loop;
- `estimate_run_typed`, `PromptDefinition.instruct_typed`, `load_typed_config`, and
  `resolve_typed_config` are additive companions where narrowing an existing parameter
  would break static consumers.

No compatibility allowlist or release-check exception is added. Issue #161 stays open for
any future unification. Such a change must first publish a deprecation path and retain the
old surface for its declared window; a major version remains the clean fallback.

The runtime target is unchanged: no additional network hop, provider call, retry, or tool
dispatch. Typed input adds one bounded validation and canonical JSON serialization before
the existing input hook and guardrail boundary. Invalid input makes zero provider calls.

## Consequences

- Existing `Agent[OutputT]` consumers retain their static and runtime meaning.
- New consumers get strict input and output inference under mypy and Pyright.
- Providers, Groq/xAI/OpenRouter presets, custom gateways, tools, A2A, budgets, identity,
  tracing, cancellation, and terminal outcomes remain shared rather than forked.
- Function and MCP exports require a typed agent because their descriptors need a request
  schema. Official A2A text ingress continues to accept `AgentDefinition[OutputT]`.
- The public vocabulary is larger during the compatibility window, but each name has one
  unambiguous contract.

When a provider, gateway, registry, or policy dependency is unavailable, behavior is the
same on both input surfaces: the existing typed failure and fail-closed policy apply. A
process crash after input rendering is recovered from the same checkpoints and
idempotency records as a text-input run.

## Migration and rollback

Migrate one call site by replacing its local declaration with `TypedAgent`, passing its
Pydantic request to `run_typed`, and running the same evaluation, cost, latency, and safety
suite. Roll back by restoring the application's prior text serialization and `run` call;
no provider, tool, checkpoint, registry, or persisted run format has to change.

## Rejected alternatives

**Reinterpret `Agent[T]` immediately.** Existing annotations silently acquire a different
meaning and the compatibility guard correctly blocks the release.

**Weaken the release snapshot.** This makes CI green by deleting the evidence of the
break; it gives consumers no migration path.

**Accept `object` on `run`.** Runtime validation could work, but static input errors would
move from the authoring line to production and every string-input consumer would receive
a weaker contract.

**Duplicate the runtime.** A second loop would drift on guardrails, budgets, identity,
telemetry, cancellation, and recovery. The additive methods therefore converge before
any execution decision.

## Verification

- strict mypy and strict Pyright prove both `Agent[TripPlan]` plus `run`, and
  `TypedAgent[TripRequest, TripPlan]` plus `run_typed`, infer `Run[TripPlan]`;
- runtime tests prove canonical input rendering, pre-provider mismatch refusal, sync and
  streaming parity, and typed export behavior;
- the release check compares 0.53.0 with 0.52.0 and permits no removed or changed released
  signature.
