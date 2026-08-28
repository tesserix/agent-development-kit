# Core primitives

`tesserix_adk.core` holds the vocabulary every other layer speaks: what a message is,
what a tool call is, what a step cost, what a run is and how it ended, and what an agent
is before anything runs it. Products that invent their own shapes cannot share a tracer,
a budget policy or a fake; these types exist so that they can.

The module is semver-governed. Every name below appears in
[`docs/api-surface.txt`](api-surface.txt), so a change to any of them shows up in the
diff of a pull request and follows [the deprecation policy](versioning.md).

A runnable version of everything here is
[`examples/typed_primitives.py`](https://github.com/tesserix/agent-development-kit/blob/main/examples/typed_primitives.py), which needs no
network and no credentials and is executed in CI.

## The types

| Type | Is |
|---|---|
| `Agent` | A declaration: instructions, model *or* task class, tool allowlist, output type, budget, guardrail chain. It runs nothing and holds no client. |
| `Message` | One turn — a role and one or more content parts. |
| `TextPart` / `BinaryPart` | The content parts, discriminated on `kind`. |
| `ToolCall` | A model's request to run a tool, with the provider's own id. |
| `Usage` | Tokens, and cost where cost is knowable. Adds with `+`. |
| `Run` | One execution, from prompt assembly to a terminal state. |
| `RunState` | Where a run is, and if it is over, why. |
| `TenantContext` / `RunContext` | The identity the runtime threads through every layer. |
| `AdkError` and its subclasses | Every failure the kit raises, each carrying `run_id`, `tenant` and a `details` payload. |

Everything is frozen, validates at construction rather than at first use, and round-trips
through `model_dump_json` without loss.

## Decisions worth knowing

**Unknown is not zero.** A self-hosted model costs something; it does not cost zero. So
`Usage.cost` is `None` when the price is unknown, and a known cost plus an unknown one
totals to unknown — a bill that silently omits a step understates it. A `Usage` recording
nothing at all is the exception: it is the additive identity, because a run starts on an
empty usage and treating that as an unknown price would leave every run unable to report
a cost. Costs in two currencies raise rather than producing a number true in neither.

**Nothing may hold a live collaborator.** A run is checkpointed by one process and
rehydrated by another, so no field may accept a client, a socket or a callable.
`extra="forbid"` makes that a construction-time failure rather than a checkpoint that
fails to serialise in production.

**Binary content is base64 on the wire and withheld from the repr.** JSON has no bytes,
so `BinaryPart.data` serialises as base64 and decodes back to `bytes`; a malformed
payload raises rather than being silently altered. Its `repr` — the form that reaches a
log line or a span attribute — shows the media type and a byte count, never the payload.
Text is *not* hidden: redacting prompt content belongs to the telemetry exporter, and a
type a debugger cannot show helps nobody.

**Deduplication is by id, never by position.** A retried provider response repeats calls
it already sent, and parallel calls to one tool differ only in their arguments.
`deduplicate` keeps the first of each id. Relatedly, `ToolCall.idempotent` defaults to
`False`: a retry that re-sends a payment is worse than a retry that does nothing.

**Every terminal state says why.** `failed` for a budget ceiling and `failed` for a
provider outage is two different bugs in one bucket, so `budget_exhausted`,
`max_iterations_exceeded` and `cancelled` are states of their own.

**Provider fields the kit does not model are kept.** `Usage.extras` holds them. Dropping
them loses evidence; promoting them makes one provider's quirk part of the public API.

## The transition table

`Run.transition_to` returns a new run and refuses anything the table below does not
declare legal, naming the legal set in the message. `legal_transitions(state)` is the
same table as a function.

| From | May go to |
|---|---|
| `pending` | `running`, `cancelled` |
| `running` | `completed`, `failed`, `cancelled`, `budget_exhausted`, `max_iterations_exceeded` |
| any terminal state | nothing |

A run that never started cannot have exhausted a budget, and a terminal run is never
reopened — doing so would make its own audit trail a lie.

## The errors

`AdkError` is the base, so a consumer can catch this kit's failures without also catching
its own bugs. Each carries the run and tenant it happened in, because
`ProviderTimeoutError` in a log with neither is a fact nobody can act on. Both are
optional: configuration fails before any run exists, and a required `run_id` there would
be a value invented to fill a field.

| Error | Raised when |
|---|---|
| `CapabilityError` | A provider is asked for something it does not support. |
| `ProviderError` | A provider call failed. |
| `ProviderTimeoutError` | It failed by timing out. A subclass, so `except ProviderError` does not miss it. |
| `SchemaViolationError` | Output did not validate against the declared type. |
| `ToolExecutionError` | A tool raised. |
| `GuardrailViolationError` | A guardrail refused. |
| `BudgetExceededError` | A ceiling was reached. |
| `CancelledError` | The caller cancelled. |
| `MaxIterationsError` | The loop hit its cap. |

`details` is for debugging — status codes, the offending output, a tool name. Never
credentials, and never message content.

## Known limitations

- **`Usage.extras` holds integers only.** A provider reporting a non-integer usage field
  would need it modelled properly or dropped; nothing in the kit reads `extras`, so
  widening it to `Any` would only make it harder to trust.
- **`Agent.output_type` is excluded from serialisation.** A Python type has no
  serialised form that can be read back safely, so a serialised `Agent` round-trips
  without executable code. `AgentDefinition` records the derived JSON Schema so the
  reviewed contract survives.
- **Unknown cost stays unknown.** `Usage.cost` uses decimal money when a price is known,
  but a self-hosted or unpriced call carries `None`; it is never reported as free.
- **Tool and guardrail fields are names.** `AgentRunner` resolves them against the
  registries supplied by the application. The names remain serializable while concrete
  clients and callables stay outside the declaration.
