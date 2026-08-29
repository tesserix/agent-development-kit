# Complete reference agents

These three programs are production-shaped compositions that run offline by default. They
use the same public interfaces a deployed agent uses, but replace every network, clock,
store, exporter, and model boundary with deterministic fakes. Copy one program, then swap
one boundary at a time.

| Agent | Production property it makes explicit | Run |
| --- | --- | --- |
| [Research](https://github.com/tesserix/agent-development-kit/blob/main/examples/reference_research_agent.py) | Hybrid retrieval, provenance, hostile retrieved instructions kept as data, grounded citations | `uv run python examples/reference_research_agent.py` |
| [Booking](https://github.com/tesserix/agent-development-kit/blob/main/examples/reference_booking_agent.py) | Human approval before an irreversible transaction, payload-bound decision, idempotency | `uv run python examples/reference_booking_agent.py` |
| [Durable](https://github.com/tesserix/agent-development-kit/blob/main/examples/reference_durable_agent.py) | Typed provider failure, partial journal, resumable checkpoint, replay-safe effect, cancellation | `uv run python examples/reference_durable_agent.py` |

Every example carries an explicit tenant, a cost ceiling, ordered guardrails, redacted
telemetry, and a one-case offline eval suite. CI executes the programs with no credentials
or network access.

## Research: retrieved words never become policy

The corpus deliberately contains “ignore previous instructions” style content. Hybrid
keyword and semantic retrieval may rank it, but `AgentRunner(..., memory=...)` places every
passage inside the retrieved untrusted-data envelope. The model can quote evidence; it
cannot use a passage to widen its tools, scopes, budget, or instructions. The output guards
then reject an instruction echo and require a citation marker. Finally, `check_grounding`
binds the claim to the retrieved document version.

Do not concatenate retrieved text into `Agent.instructions`, a trusted prompt variable, or
a tool declaration. That changes data into authority.

## Booking: reasoning proposes; deterministic code transacts

The booking model can propose `confirm_booking`, but the runtime suspends before dispatch.
At that point no booking exists and no worker needs to stay alive. The approver receives a
safe summary and a single-use token bound to the original tenant, run, tool, and arguments.

After approval, the annotated deterministic transaction boundary receives only validated
arguments and the runtime-minted idempotency key. A retry, worker restart, or replay uses
the same key. The model never calls the booking dependency directly and never decides that
its own proposal is approved.

## Durable: failure is state, not plausible prose

The durable worker succeeds through its first model and tool activities, then the provider
fails three times. `ProviderUnavailableError` remains typed and carries the attempt and
step. The completed activity journal and checkpoint stay available for inspection. A new
worker replays the journal, skips the completed effect, and pays only for the remaining
model call. Cancellation follows the same rule: completed state remains inspectable and no
new activity starts.

The in-memory journal demonstrates the contract. Bind the same activities to the durable
engine described in [durable runs](durable-runs.md) for a process-restart guarantee.

## Deployment and rollback

Application deployment belongs in the organisation's cluster chart repository. Follow the
existing cluster chart patterns for workload identity, secret injection, network policy,
probes, resource limits, disruption budgets, autoscaling, and telemetry export; do not copy
Helm or Argo CD manifests into this SDK repository. Durable worker and Temporal namespace
ownership is described in [durable runs](durable-runs.md#deployment-boundary).

Deploy a new agent revision progressively. Keep the prior image and prompt alias available
as the one-action rollback. Durable workflow changes must be replay-compatible before the
new worker receives old histories; database-backed stores use expand, migrate, then
contract. A rollback must never delete journals, approval rows, idempotency records, or
audit evidence.

## Real-provider validation

The `manual-reference-agents` workflow is operator-triggered and environment-protected. It
first reruns the offline suite, then requires an explicit provider choice and a hard cost
ceiling. It must use short-lived repository secrets and records only redacted summaries.
The job is evidence in addition to deterministic CI, never a merge requirement and never a
reason to make ordinary pull requests able to read provider credentials.

Before enabling it, configure the selected provider secret in the protected GitHub
environment, set a cost ceiling the team accepts, and verify the provider/model capability
record. Cancel the job if spend or latency crosses that ceiling; the rollback is to disable
the environment and continue using the offline regression lane.
