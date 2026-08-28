# Idempotency

Idempotency is a cross-cutting guarantee, not one decorator. It prevents a retry,
redelivery, process restart, or duplicate request from repeating an effect whose first
outcome is already known or indeterminate.

## Choose the relevant layer

| Boundary | Mechanism | Reference |
|---|---|---|
| Model-request retry | Retry only normalized transient failures; do not repeat unsafe post-tool turns | [Retry and resilience](resilience.md) |
| Tool call | Declare read-only, idempotent, or effectful policy and use a durable store | [Tool idempotency](tool-idempotency.md) |
| Run checkpoint/resume | Persist the frontier and resolve dispatched calls before continuing | [Checkpointing](checkpointing.md) |
| Event consumer | Deduplicate by event identity before applying the effect | [Idempotent consumption](idempotent-consumption.md) |
| State plus event | Commit state and an outbox record in one transaction | [Transactional outbox](outbox.md) |
| Workflow activity | Keep provider/tool calls outside replayed workflow code and record results | [Replay safety](replay-safety.md) |
| Dead-letter replay | Re-enter through the live idempotent handler under a bounded replay plan | [Dead letters](dead-letters.md) |
| Peer/A2A task | Bind idempotency to authenticated caller, tenant, task/request identity, and operation | [Official A2A](a2a.md) |

## Rules

- A generated retry key must cover tenant, agent/tool operation, normalized arguments,
  and the caller-visible request identity.
- The store must atomically distinguish absent, in-flight, completed, and abandoned work.
- The retention window is part of the guarantee. A request after expiry is not known to
  be a duplicate.
- A timeout after dispatch is indeterminate unless the downstream system exposes a
  status/read-back operation.
- Never turn an unknown outcome into “safe to retry” merely because the local call raised.
- Event acknowledgement happens only after the dedupe decision and effect are durable.
- Replays use the normal authorization, validation, and idempotency path.
- Metrics distinguish suppressed duplicates, in-flight collisions, indeterminate work,
  expired records, and real failures.

For a new side effect, start with [Tool idempotency](tool-idempotency.md) and add
fault-injection tests for a crash before dispatch, after dispatch, after commit, and before
the response reaches the caller.
