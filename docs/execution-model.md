# Choosing an execution model

Once a kit ships a Temporal integration, the failure mode flips. A team reaches for a
workflow to classify one support ticket, inherits a worker fleet, a namespace and replay
constraints they never needed, and concludes the kit is heavy. It is not: they climbed two
rungs they had not earned.

Durability is a ladder with a plain in-process default. This page says which rung a design
belongs on, what earns the next one, and what is not a reason to climb at all.

## The ladder

### Rung 1 — in-process

The **default**. `Runner` drives the loop in the process that received the request, and a
crash loses the run. Nothing is installed beyond the base package.

Ship here unless the table below says otherwise. Most agents are a few model calls and a
handful of reads, and a lost run is a retried request.

### Rung 2 — checkpointed

`CheckpointStore` writes a frontier at each unambiguous boundary, and a later process
resumes from it. Still one process at a time, still no worker fleet, still no engine — a
store and a lease. See [checkpointing.md](checkpointing.md).

**Earned by** a run that will outlive the process that started it: an hours-long human gate,
a queue that will be drained tomorrow, an expensive prefix nobody wants to pay twice.

### Rung 3 — durable workflow

`AgentWorkflow` drives the same loop with every model and tool call as an activity, and the
deployment binds those activities to Temporal. See [durable-runs.md](durable-runs.md).

**Earned by** the combination the lower rungs cannot hold: side effects that are irreversible
or financially meaningful, applied across more than one system, over a run long enough that a
restart is likely rather than possible. That is where a replayable history and a saga
([compensation.md](compensation.md)) stop being paperwork.

## The decision table

Five questions. Take the highest rung any row demands, not the average.

| Question | Answer | Rung |
|---|---|---|
| How long does the run take? | under five minutes | Rung 1 |
| | minutes to hours | Rung 2 |
| | hours to days | Rung 3 |
| Is there a human gate? | none | Rung 1 |
| | one, waiting hours or days, no transactions | Rung 2 |
| | a gate on money or on an irreversible action | Rung 3 |
| Are the side effects irreversible? | reads only | Rung 1 |
| | writes to one system, replayable by hand | Rung 2 |
| | irreversible or financially meaningful, across systems | Rung 3 |
| How wide does it fan out? | a handful of branches | Rung 1 |
| | wide, but cheap and restartable | Rung 1, with concurrency control |
| | wide, and each branch applies something | Rung 3 |
| What does a restart cost? | a retried request | Rung 1 |
| | an expensive prefix paid twice | Rung 2 |
| | work applied twice in someone else's system | Rung 3 |

A cheap run that fans out very wide is a concurrency problem, not a durability one. Cap it
with `fan_out` and a shared `AttemptBudget` ([activity-policies.md](activity-policies.md))
and stay on Rung 1.

## Not a reason to reach for a workflow

- **observability** — spans, a ledger and cost attribution are on every rung.
  See [ledger.md](ledger.md) and [cost-attribution.md](cost-attribution.md).
- **retries** — an activity policy retries on every rung, and a retry budget caps the storm.
  See [activity-policies.md](activity-policies.md).
- **an approval gate** — approval is a grant, not a scheduler. See [tool-approval.md](tool-approval.md).
- **an autonomy grant** — a ceiling is enforced by the run, not by the engine underneath it.
  See [autonomy.md](autonomy.md).

The rule a review cites: durability makes an action **resumable, never authorised**. A
workflow that moves money without an approval gate and a matching autonomy grant is rejected
however durable it is, and a model may not initiate that action in the first place.

## Moving up a rung

No agent, tool or guardrail code changes. The loop is the same loop.

1. Rung 1 → 2: hand the runner a `CheckpointStore` and give the run a stable `run_id`.
2. Rung 2 → 3: construct `AgentWorkflow` instead of `Runner`, hand it a `Journal`, and bind
   the two activity payloads in your worker. Install `tesserix-adk[temporal]` there — and
   only there.

The kit imports `temporalio` nowhere at package import time, so Rung 1 and Rung 2 never
install it and CI runs the whole suite without a Temporal server. That is not a convention:
it is asserted in `tests/test_extras.py`, which fails if any non-optional module reaches an
extra.

A consumer who already runs Temporal can call the kit as a plain library inside their own
activity. That is supported: the runtime is deterministic-friendly and performs no I/O of
its own beyond the activities you hand it.

## Review checklist

Copy this into the design review.

- [ ] The chosen rung is named, and the table row that demanded it is quoted.
- [ ] Nothing on the design climbs a rung for observability, retries, approval or autonomy.
- [ ] Every irreversible or financially meaningful action has an approval gate and an
      autonomy grant, independent of the rung.
- [ ] Every applied step on Rung 3 has a paired reversal, or is declared irreversible.
- [ ] The run has a stable `run_id` and a budget ceiling.
- [ ] Local development and CI run the design with no Temporal server available.
- [ ] Rung 3 only: the worker fleet, namespace and retention are owned by a named team.

This is guidance, not a gate. A team with a legitimate unusual case may ship against the
table — **record the reason** in the product's own ADR, with an owner, so the next reviewer
reads a decision rather than an accident.
