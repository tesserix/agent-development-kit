# ADR 0003 — Google ADK interoperability through official A2A

**Status:** Accepted

**Date:** 2026-08-28

## Context

Tesserix ADK and Google Agent Development Kit are separate runtimes. Consumers need a
standard boundary between them without importing Google objects into the Tesserix core or
calling a second, Tesserix-specific peer protocol by the official A2A name.

The initial serving envelope is 20 task starts per second and 100 concurrent active tasks
per process. Input is at most 64 KiB of UTF-8 text and the final artifact is at most 1 MiB.
The adapter overhead objective is p99 below 10 ms at 100 concurrent tasks, excluding
model, network, gateway, and task-store latency. A task performs one request read and two
or three status/artifact writes. The library stores no durable data at 12 or 36 months;
the deployment-owned A2A `TaskStore` owns retained task volume. Five-times this envelope
requires a capacity test but not a protocol change.

The serving application has a 99.9% monthly availability objective. The bridge must fail
closed and must not create a hidden fallback around its identity resolver, gateway, task
store, or runner.

Prompts, tool authority, tenant data, model spend, and returned artifacts are the assets.
An unauthenticated caller, an authenticated caller from another tenant, or a compromised
dependency could target them. The trust boundary is the A2A server or gateway: it must
authenticate the transport and place a verified principal in server context before an
A2A message reaches the executor. Message metadata is never an identity source.

## Decision

Tesserix ADK provides two optional public helpers:

- `a2a_agent_executor` returns an official A2A 1.x `AgentExecutor` backed by one
  `AgentRunner` and reviewed `AgentDefinition`;
- `google_adk_remote_agent` creates Google ADK 2.8's current, non-legacy
  `RemoteA2aAgent` from an official card or card URL.

The executor accepts user-role text parts only. It resolves a core `Principal` for both
execution and cancellation, binds tenant, subject, scopes, and principal context into the
run, and uses the official A2A task ID as the Tesserix run ID. An active task can only be
cancelled by the same tenant and subject after the resolver has independently authorized
the cancellation request. Rejection and failure messages are generic; internal exception
text is not returned to the peer.

Submitted, working, completed, failed, rejected, and cancelled states map onto official
A2A task events. The authoritative final answer is buffered and emitted as one text or
JSON artifact. Partial answer streaming, input-required/auth-required continuation, and
push-notification delivery are not implemented by this bridge and must not be advertised
unless the surrounding deployment supplies and tests them.

The official request handler and `TaskStore` remain the owners of persistence, duplicate
delivery, task lookup, subscriptions, and resubscription. `InMemoryTaskStore` is for local
development only. A persistent store preserves task state across a process restart, but
does not resume a model call that died with the process; production deployments need a
durable runner or reconciliation policy for tasks left working.

When identity resolution is unavailable or rejects a principal, execution is rejected and
cancellation returns the non-enumerating official task-not-found error. When the runner or
model fails, the task becomes failed with a stable public code. When the task store is
unavailable, the official handler fails; there is no in-memory fallback that could lose
tenant ownership or state. Duplicate sends and resubscription use the selected official
store's semantics. Cancellation racing completion has one terminal-state winner.

The base installation remains unchanged. The `a2a` extra includes the official HTTP
server runtime. The `google-adk` admission adds one direct SDK and up to 20 transitive
packages only for consumers selecting that extra. Per active task, local bridge state is
one cancellation token and identity pair; input and output buffers remain within the
declared limits. Deployment cost is therefore dominated by model, network, and durable
task-store usage.

## Alternatives

Writing a bespoke Google-to-Tesserix adapter was rejected because it would duplicate
official task, cancellation, card, and transport semantics and lock both runtimes
together. Treating Tesserix's typed peer protocol as official A2A was rejected because the
wire contracts differ. Putting an untyped gateway shim in front of `AgentRunner` was
rejected because tenant attribution, bounds, and terminal-state races would remain outside
the tested public surface. The simplest alternative—documenting manual official SDK
wiring—was rejected because every consumer would reimplement the same security boundary.

## Consequences

Google ADK can consume a Tesserix agent while the Tesserix agent keeps any supported model
provider, tool, store, or gateway composition. Other official A2A clients use the same
endpoint, and custom registries or protocol bindings remain possible through the existing
client factory and registry protocols.

The bridge is deliberately not a process host. Applications still mount official routes,
authenticate requests, provide a tenant-scoped task store, apply rate and spend limits,
and operate recovery. A2A SDK 1.1.2 emits a protobuf descriptor deprecation warning under
protobuf 6; the end-to-end test names upstream
[a2aproject/a2a-python#1158](https://github.com/a2aproject/a2a-python/pull/1158) so a future
SDK upgrade removes the narrow expectation rather than hiding all dependency warnings.

Rollback removes the mounted A2A routes and the two optional extras from the consuming
application. No Tesserix run, state, tool, provider, or database schema migration is
required. Existing Tesserix peer integrations and official A2A client/card helpers remain
independent.
