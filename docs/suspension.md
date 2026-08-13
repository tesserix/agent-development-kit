# Suspension

An approval that takes a person a working week is the normal case, not the pathological one.
`TransportGate` handles the minutes-long version by waiting; it cannot handle the three-day
version, because a run that waits three days holds a worker, a connection and a queue slot
across two deploys and a restart, and dies to the first of them.

`DeferringGate` handles it by not waiting. It puts the question on a transport, hands out a
token, writes the frontier down and lets the run go. Nothing is held. Whenever the decision
arrives — same process or not, same week or not — the token resolves back to the original
run and carries it on from where it stopped.

## Stopping

```python
from tesserix_adk.adapters import NatsApprovals
from tesserix_adk.runtime import (
    AgentRunner,
    Checkpointer,
    DeferringGate,
    MemoryCheckpointStore,
    MemorySuspensionStore,
)

gate = DeferringGate(NatsApprovals(nats), MemorySuspensionStore(), hand_to=email_the_approver)
runner = AgentRunner(..., approvals=gate, checkpoints=Checkpointer(MemoryCheckpointStore()))

run = await runner.run(agent, "settle the invoice", tenant="acme", run_id="run_1")
run.state  # RunState.SUSPENDED
```

`RunState.SUSPENDED` is not terminal. It is the one non-terminal state a returned run can be
in, and it means exactly one thing: the run is on somebody's rota and will go again when they
answer. From it a run may only become `RUNNING`, `CANCELLED` or `FAILED` — never `COMPLETED`
without going again.

A transport that answers in its own reply is honoured as the answer and nothing is deferred,
so a console operator typing `y` does not cost a database round trip and a resume.

The gate refuses to defer a run that could never be carried on. A runner with no
checkpointer, or one whose `CheckpointPolicy` does not write at
`CheckpointBoundary.BEFORE_APPROVAL`, raises `ConfigurationError` at the point of deferral
rather than handing out a token that promises a resume nobody can perform.

## The token

`hand_to` is how the token reaches whoever decides. It is deliberately separate from the
transport: the question goes to a queue that a team reads, and the token is a bearer
credential for one decision. They usually go to different places and are worth different
care.

The token is minted over the record the approver was shown, and it is:

- **single-use** — redeeming spends it, and a second presentation raises `ApprovalTokenError`
  and executes nothing;
- **tenant-bound** — presented as another tenant it resolves to nothing at all, rather than
  to somebody else's run;
- **expiring** — three days by default (`DEFAULT_SUSPENSION_SECONDS`), after which it buys a
  denial rather than a grant;
- **bound to the payload** — it carries `arguments_digest`, so an answer cannot be moved onto
  a different set of arguments.

Stores keep `digest`, never `value`. Every presentation is recorded as a `TokenAttempt`
against the identity that made it, accepted or not — a token presented twice is the shape of
an approval being replayed, and that is worth more than a raised exception.

## Carrying it on

```python
run = await runner.resume_with_decision(
    agent,
    "run_1",
    tenant="acme",
    token=token,
    granted=True,
    decided_by="ada",
)
```

The run comes back at the iteration it stopped at, with its own id, tenant, user, scope and
usage ledger — the ledger is restored, so work already paid for is not paid for twice. The
`RUN_RESUMED` event records how long it was stopped for.

**A person saying yes is one of the conditions, not all of them.** The held call goes back
through dispatch rather than round it, so everything that gated it the first time gates it
again against the world as it is now, not as it was three days ago:

| What moved while nobody was looking | What happens |
|---|---|
| The autonomy window closed | The ceiling refuses, as it would for any call |
| The grant was revoked | `GrantRevokedError`, per `revoked_runs` — see [`docs/autonomy.md`](autonomy.md) |
| The tool's schema changed | `SCHEMA_VIOLATION`; the tool does not run |
| The token expired | Closed as a denial decided by `system:timeout`, never by the person |
| The agent now names a different model | `ConfigurationError`, unless `allow_model_drift=True` |

The model check is not fussiness: the approver answered a question about what *that* model
proposed. Carrying the answer onto another model's proposal is a decision nobody made.

## Listing what is waiting

```python
for held in await gate.pending(tenant="acme"):
    show(PendingDecision.of(held))
```

`PendingDecision` is what a person may see: what is being asked, by which agent, why, when it
closes, and the digest of the payload — never the argument values. A rota outlives the run and
is read by people who are not party to it, so the account number is not theirs.

## Answering from a terminal

The process that asked is long gone by the time somebody decides, so the answer has to come
from somewhere else. `tesserix_adk.cli.approvals_main` is that somewhere: three commands over
whatever store and resume the deployment already has.

```python
from tesserix_adk.cli import approvals_main

code = await approvals_main(
    sys.argv[1:],
    waiting=gate,
    answering=lambda token, granted, by, why: runner.resume_with_decision(
        agent, run_id_for(token), tenant="acme", token=token,
        granted=granted, decided_by=by, reason=why,
    ),
)
```

```
adk approvals list --tenant acme
adk approvals approve --token <token> --by ada --reason "invoice checked"
adk approvals deny --token <token> --by ada
```

`waiting` and `answering` are supplied rather than discovered because the kit cannot know
where a deployment keeps its suspensions or which agent to carry on. There is no console
script: a kit that installs a global `adk` binary fights with whatever the consumer ships.

Exit codes are `0` where the command did what it says, `1` where the token bought nothing
(spent, expired, or for a decision already taken — reported, not raised), and `2` for a
command line it could not read. `list` prints the payload digest, never the payload; a
terminal is read over shoulders and scrolls into somebody's session log.

## Storage

`MemorySuspensionStore` is for tests and single-process demos; nothing in it outlives the
process, which is precisely what surviving a three-day approval is not. Implement
`SuspensionStore` against whatever the deployment already runs. The seven methods are
`put`, `get`, `by_token`, `spend`, `pending`, `forget` and `attempted`; `spend` must be atomic
and return `False` for a decision already taken, because that is the whole of the exactly-once
guarantee.

## Stability

`SuspendedRun`, `ApprovalToken`, `PendingDecision`, `TokenAttempt`, `SuspensionStore`,
`TokenRedeemer` and `ApprovalTokenError` are public API. `ApprovalDeferred` is the signal a
gate raises to stop a run and carries both the token and the store the suspension is to wait
in — a gate is free to keep its suspensions wherever it likes without the loop being told.

See also [`docs/tool-approval.md`](tool-approval.md) for the gate and transports, and
[`docs/checkpointing.md`](checkpointing.md) for the frontier a suspension resumes from.
