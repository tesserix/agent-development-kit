# Autonomy

How much an agent may do without asking anybody, declared as a grant the runtime enforces.

Asking a human before every action makes an agent useless. Letting it act freely makes it
dangerous. Most products settle on a number in a config file — no expiry, no audit, and
nobody's name against it. Here that number is a grant: issued by someone, for one class of
action, up to a ceiling, until a moment. Anything an issued grant does not cover escalates
to a human, including everything nobody thought about.

```python
from tesserix_adk.core import (
    ActionClass, ActionRegistry, AutonomyGrant, AutonomyLadder, AutonomyLevel,
    Ceiling, InMemoryGrants,
)
from tesserix_adk.runtime import AgentRunner, AutonomyGate

classes = ActionRegistry({"change_booking": ActionClass(
    name="booking.change", amount_field="amount", currency_field="currency",
)})
grants = InMemoryGrants([AutonomyGrant(
    id="g1", tenant="acme", action_class="booking.change",
    level=AutonomyLevel.ACT_WITHIN_LIMITS, granted_by="ops@acme.example",
    issued_at=now, expires_at=now + 86_400,
    ceiling=Ceiling(amount=Decimal("5000"), currency="INR", window_seconds=86_400),
)])
runner = AgentRunner(
    provider=provider,
    approvals=desk,
    autonomy=AutonomyGate(AutonomyLadder(classes, grants=grants, clock=clock)),
)
```

## Levels

| Level | What it means |
|---|---|
| `ask_always` | Every action of this class goes to a human. The default, and what an unmatched action falls to. |
| `act_within_limits` | Act unattended while the amount fits under the ceiling's remaining headroom. Never valid without a ceiling. |
| `act_and_report` | Act unattended and owe a report. While a report for an earlier action of the class is undelivered, the next one asks. |

These three are stable. A level added later is added beside them; none of them changes
meaning, because a deployment's grants are rows in a database that outlive the release
that wrote them.

## Classes, not tools

A grant is about what an action does to the world, so it names a class — `payment.refund`
— rather than a tool. Three refund tools are one thing to the person deciding how much may
be refunded, and a fourth added next quarter should not silently inherit the grant.

That is also why a tool in no class resolves to `ask_always` rather than to nothing: a tool
registered after the grants were issued is exactly the case where failing open would be a
surprise.

## Fail-closed, everywhere

Each of these escalates to a human rather than acting:

- no grant matches the tenant, the user or the class
- the grant expired, at the clock the ladder holds
- the arguments carry no readable amount for a class that has a ceiling
- the action's currency is not the grant's — a mismatch is a question, never an implicit conversion
- the amount is over the remaining headroom, by any margin. `900.01` against `900.00` of
  headroom escalates; headroom is `Decimal` and is never rounded up to fit

A grant on `acme` does not reach `acme/eu` unless it says `includes_subtenants=True`. A
grant that silently widened as the tenant tree grew would be a grant nobody issued.

## Nobody grants themselves

`AutonomyLadder` holds a `GrantReader`, which can only read. Issuance is a second protocol,
`GrantIssuer`, and the runtime is never given one — so there is no object inside a run that
could widen what the run may do, even if a model asked for it.

On top of that, the reserved class `autonomy.grant` is refused outright: a tool that would
issue autonomy is `AutonomyOutcome.REFUSE`, recorded as `autonomy_refused`, and no human is
offered the chance to wave it through from inside the run. A grant that permits the
reserved class is ignored.

## What the loop does with a decision

Consulted in `_cleared_to_dispatch`, after the `before_tool_dispatch` hooks and before
anything goes out:

| Outcome | Effect | Event |
|---|---|---|
| `ACT` | The call dispatches. An approval the agent or tool declared still applies. | — |
| `ESCALATE` | The call is held for the approval gate, with the ladder's reason. | `autonomy_escalated` |
| `REFUSE` | The run fails with `AutonomyRefusedError`. | `autonomy_refused` |

Autonomy only ever adds a gate. A grant permitting unattended action does not waive an
approval the tool declared, because the two answer different questions: how much this agent
may do, and whether this call is one a human sees. An escalation with no approval gate
configured raises `ConfigurationError` rather than dispatching.

## Reports

`act_and_report` is enforced rather than trusted. `AutonomyGate` records the obligation the
moment it lets an action through, and `ReportLog.outstanding` degrades the next action of
that class to asking a human until it is delivered. Without that, `act_and_report` becomes
`act` the day nobody reads the reports. `InMemoryReports` is the single-process
implementation; a deployment substitutes its own.

## Taking it back

A grant is read from the store on every attempted action, not once at run start, which is
what makes a withdrawal land on the very next action rather than at the next deployment.

```python
await grants.revoke(Revocation(
    grant_id="g1", revoked_by="ops@acme.example", revoked_at=now, reason="card reported stolen",
))
```

A withdrawal names one grant, or a tenant, or a tenant and an action class — enough to stop
a class of work across a fleet without knowing every id that was issued for it. One that
names neither a grant nor a tenant is refused at construction: it would either do nothing or
withdraw the world, and both are the wrong answer to what somebody meant.

Withdrawal is an append, never a delete, so a revoked grant cannot be reactivated: there is
no statement in the kit that removes a revocation, and re-granting mints a new id. What was
withdrawn stays readable as what it permitted while it stood.

### Runs already under way

A run suspended on an approval was asleep while the authority behind it could have been
taken back, and a human approving a call is not the same as the grant that put the call in
front of them still standing. The loop re-checks after the approval returns and before
anything goes out, records `grant_revoked`, and then does what the gate's `revoked_runs`
says:

| `revoked_runs` | What the run does |
|---|---|
| `InFlightPolicy.CANCEL` (default) | Fails with `GrantRevokedError`, naming the grant and who withdrew it. |
| `InFlightPolicy.ASK_ALWAYS` | Proceeds on the approval it has, and every later action of the class asks a human. |

### The bus is an accelerator, never the authority

`RevocationBroadcast` carries withdrawals to every process over NATS or Redis, and
`RevocationWatch.follow` consumes them. A missed message costs latency, not correctness: the
store re-read is what refuses.

The watch fails closed on its own. A view nobody has confirmed within
`stale_after_seconds` (30 by default) refuses unattended action rather than acting on what it
last heard — a process cut off from the bus cannot know a grant is still live. It never
manufactures authority in the other direction: a stale watch turns `act` into `refuse` and
leaves an escalation exactly as it was.

## Grants in PostgreSQL

`PostgresGrantStore` is append-only. An id already in use is refused, never updated: a
decision recorded against an id has to stay readable as what it permitted, so re-granting
mints a new id. The dispatch-path read is one tenant, one class, unexpired at one moment, excluding
anything a row in `adk_grant_revocations` withdrew.

Issuance is not retried — a retried insert that may already have landed is how one id ends
up meaning two things — while reads retry a contended or briefly unreachable database. A
read that ultimately fails raises; it is never read as "no grants", which would be the one
failure mode that widens autonomy.

The DDL is the deployment's, not the kit's. `EXPECTED_GRANT_SCHEMA` is the shape this
adapter was written for, and `verify()` refuses anything else at startup:

```python
store = await PostgresGrantStore.open(pool, clock=clock, settings=settings)
```

## Not here

- **Ceiling arithmetic against splitting and retries.** `CommitmentLedger` is the seam; how
  a deployment counts what is committed, and how it resists an agent splitting one action
  into ten, is its own story.
- **The audit record shape.** The ladder names the grant on every decision, including
  escalations — which grant was not enough is the question an operator asks — but what that
  record looks like on disk belongs elsewhere.
- **Grant administration UI.** Each product's own.

Verify the store against a real PostgreSQL:

```bash
uv run pytest -m integration tests/integration/ -q  # ADK_POSTGRES_DSN=...
```
