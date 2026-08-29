# Architecture

How `tesserix-adk` is layered, and why the layers sit where they do. Read
[`design-brief.md`](design-brief.md) first — the eight rules there are the reason
this shape exists.

For the larger control-plane, execution-plane and tool-plane design from agent source
through evaluations, registry approval, canary, runtime, monitoring and feedback, see
[Agent lifecycle and platform architecture](agent-lifecycle.md).

---

## 1. The layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Entrypoints                                                          │
│  HTTP · chat · events · webhooks · scheduled · CLI                    │
│  Translate a request into a run. No agent logic lives here.           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  submit(agent, input, context)
┌───────────────────────────────▼──────────────────────────────────────┐
│  Authoring — what an agent IS                                         │
│  Primitives as frozen models · composition · agent + prompt registry  │
│  ReAct · Delegate · Emit · Retrieve                                   │
│  Sequential · Parallel · Loop · Router · Map · Dispatch               │
│  The composed tree is the wire format. Supervisors and managers are   │
│  compositions, not base classes.                                      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  await node.run(input, state)
┌───────────────────────────────▼──────────────────────────────────────┐
│  Runtime — how an agent RUNS                                          │
│  Run loop · caps · cancellation · timeout · checkpoint · streaming    │
│  InProcess (tests, evals, CLI)    Durable (production, resumable)     │
│  Identical bus traversal, so an eval is evidence about production.    │
└──────┬──────────────────────┬──────────────────────┬─────────────────┘
       │                      │                      │
┌──────▼───────┐      ┌───────▼───────┐      ┌───────▼──────────────┐
│  ModelBus    │      │  ToolBus      │      │  Context Engine      │
│              │      │               │      │                      │
│ provider     │      │ registry      │      │ run state            │
│ routing      │      │ allowlist     │      │ prompt assembly      │
│ CPU / GPU    │      │ mutation      │      │ dedupe + eviction    │
│ backends     │      │ class         │      │ retrieval            │
│ cost + usage │      │ sandbox       │      │ episodic memory      │
│              │      │ MCP · A2A     │      │ semantic memory      │
└──────┬───────┘      └───────┬───────┘      └───────┬──────────────┘
       │                      │                      │
       └──────────────────────┴──────────────────────┘
                    every call crosses a bus
                    ───────────────────────
       INLINE, FAIL CLOSED          │        SIDEBAND, FAIL OPEN
  guardrails · policy · budget      │   tracing · metrics · audit
  identity · approvals · schema     │   replay capture
  A trip stops the call.            │   An outage must not stop a run.
```

## 2. Four things this fixes relative to the common layer diagram

The usual rendering of this architecture draws a single **Agent Runtime** box
over three service boxes, with everything else pooled into one bottom
"reliability and control plane". Four things go wrong with that.

**Guardrails are not a plane you consult, they are a boundary you cross.**
Drawn at the bottom next to tracing, they read as something the runtime calls
when it remembers to. They must be interceptors on the model and tool call path,
so that writing a new agent cannot bypass them. There is one code path to review
in an audit, not one per agent.

**Enforcement and observation have opposite failure modes.** Policy, approvals,
identity and budget must fail *closed* — if the check cannot run, the call does
not happen. Tracing, metrics and audit-shipping must fail *open* — a collector
outage must not take production down. Pooling them into one plane invites
exactly the wrong default on one side or the other. They are split above.

**Authoring is missing.** The common diagram shows a runtime but never shows
what a runtime runs. Agent definition, composition, versioning and the prompt
registry are a distinct layer with a distinct lifecycle: definitions are
reviewed, versioned and pinned, while the runtime is deployed. Multi-agent
patterns — supervisor, manager, delegation — live here as compositions.

**The trust boundary is invisible.** Content arriving from the Tool Hub and the
Context Engine is untrusted, and it flows into the model. That is the primary
injection vector in any agent system. It needs to be a marked boundary in the
diagram because it is a marked boundary in the code: retrieved and tool-returned
content is structurally data and cannot become instruction.

## 3. Relationship to the lean-ADK proposal

An external proposal circulated as "Lean ADK Architecture" argues for a small,
model-independent runtime rather than a large framework, split into a
lightweight SDK, a stateless runtime and a durable control plane. That thesis is
right and matches this design. Several of its specifics are worth adopting
verbatim; four need correcting before they are applied here.

**Adopted.** The three-product split maps onto `contracts` / `runtime` /
durable control plane. The four separated forms of state — run state, session
memory, durable memory, retrieved knowledge — are exactly the split in §5. The
insistence that routing, retries and business rules stay as typed configuration
rather than hidden in prompt strings is why control flow here uses a closed
selector union rather than callables. The escalation ladder — one agent with
tools, then a deterministic workflow, then a router with specialists, and
collaborating agents only when measured — is adopted as written. So is the
observation that router, planner, executor, reviewer and approver are roles a
run passes through, not permanently running services.

**The performance objectives measure the wrong thing.** A target of "runtime
overhead under 20 ms" is a rounding error against a CPU model call of five to
thirty seconds. The numbers that decide whether the product is usable are
time-to-first-token, sustained tokens per second on the target CPU, and the
prompt-cache hit ratio. The proposal also asks for both a hard tracing-overhead
ceiling and complete trace coverage of production; at volume those cannot both
hold. Traces are sampled, the audit record is complete, and they are separate
pipelines with opposite failure modes.

**Context engineering is missing.** The proposal's context engine selects
relevant information, applies token budgets and summarises history — all
correct, and all silent on the thing that dominates CPU latency. Prefix
stability, admission dedupe and never evicting the cacheable prefix are what
make CPU inference viable at all, as §6 sets out.

**Routing keyed on cost is the wrong axis for a sovereign deployment.** Cost
limits, data-residency hints and fallback to a "compatible" model assume a
multi-vendor cloud. Where a consumer runs self-hosted with no egress, the
routing axes are tier and capacity, and a fallback that crosses a trust boundary
is a data-handling breach wearing the costume of resilience. Fallback must fail
closed across such a boundary rather than degrade across it.

**`server/` does not belong inside the kit.** The proposed module layout places
an API, streaming and worker under the kit itself, which makes the kit
responsible for its own deployment and drags a web framework into every
consumer. Entrypoints belong in adapters, above the kit.

One further point in the proposal deserves emphasis rather than correction:
guardrails must cover every execution path *including handoffs*, since a
separate delegation pipeline is the easiest control bypass in any agent system.
That is the reason for the single-boundary design in §2.

## 4. Why compositions never touch the runtime

Leaves — `ReAct`, `Delegate`, `Emit`, `Retrieve` — dispatch through the
installed runtime. Compositions — `Sequential`, `Parallel`, `Loop`, `Router`,
`Map`, `Dispatch` — call `await child.run(...)` directly and never see it.

This is what lets the same agent tree run in-process in a unit test and durably
in production. Only the leaves differ between the two, and both walk the same
buses in the same order. If compositions dispatched too, every control-flow
decision would need a durable equivalent and the two paths would diverge.

## 5. Memory is four things

Collapsing these is the most common design error in agent frameworks.

| Kind | Lifetime | Scope | Concern |
|---|---|---|---|
| Run state | one run | one execution | scratch: inputs, outputs, tool calls |
| Working context | one run | one prompt | token budget, dedupe, eviction |
| Episodic | durable | user · tenant | past runs, preferences, personal data |
| Semantic | durable | tenant | facts, entities, contradictions, decay |

The first two are ephemeral and need no privacy machinery. The last two are
personal data: they need tenant scoping at the interface and erasure that
reaches embeddings and derived caches, or a consumer under DPDPA cannot use
them. That is why memory belongs in the kit rather than in each product —
erasure that misses the vector store is the kind of thing every team gets wrong
once.

## 6. CPU first

The default inference target is CPU: llama.cpp with GGUF quantization behind an
OpenAI-compatible endpoint. GPU is a backend swap, not a code change, because
nothing above the ModelBus knows which is in use.

This choice has one large architectural consequence. On CPU, prefill dominates
total latency to a degree it never does on a GPU — a long prompt that costs
about a second on an H100 costs tens of seconds on CPU cores. Prompt-cache reuse
therefore stops being an optimisation and becomes the property the system is
designed around:

- the first three prompt layers are byte-stable, so the cache key holds;
- tool definitions are sorted, so declaration order cannot break the prefix;
- the prefix fingerprint is asserted in tests, so a regression fails a build
  instead of silently doubling latency;
- content is deduped on admission, because re-injecting a chunk the model
  already has is paid again in prefill on every turn;
- eviction takes from the volatile tail and never from the cached prefix.

Cache hit ratio is a first-class metric for the same reason. It is the number
that says whether the above is working.
