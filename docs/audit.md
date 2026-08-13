# Audit

What an agent did unattended, and what it declined to do, kept where sampling cannot reach it.

The question asked after an autonomy incident is never "what does the dashboard show". It is
"what did this agent do without asking anybody, in this tenant, over this period — and what
did it refuse?" Telemetry cannot answer it. Spans are sampled, dropped under load, and
stripped of the context that made the decision. Refusals in particular leave nothing behind,
so nobody can show that a ceiling actually held.

The audit trail is a separate path for exactly that reason. One decision, one record, no
sampling.

```python
from tesserix_adk.adapters import PostgresAuditSink, PostgresAuditSettings
from tesserix_adk.runtime import AgentRunner, AuditTrail, AutonomyGate

sink = await PostgresAuditSink.open(session, settings=PostgresAuditSettings(dsn=dsn), clock=clock)
runner = AgentRunner(
    provider=provider,
    approvals=desk,
    autonomy=AutonomyGate(ladder, audit=AuditTrail(sink, clock=clock)),
)
```

The gate takes the trail because the gate is where the decision is made. Nothing else has to
be wired: the run loop records what it clears, and the gate records what it stops.

## What a record says

An `AuditEvent` is one decision about one attempted call.

| Field | What it is for |
|---|---|
| `run_id`, `sequence` | Which run, and where in it. The sequence is monotonic per run, so the order survives clocks that are not. |
| `tenant`, `user` | Whose isolation boundary, and on whose behalf. |
| `agent_name`, `agent_version` | Which agent, at which revision. A behaviour that changed is a version that changed. |
| `tool`, `action_class`, `level` | What was attempted, what class of thing it is, and the autonomy level applied to it. |
| `decision` | `executed`, `escalated`, `refused` or `revoked`. |
| `reason` | Why, in the ladder's own words, for anything that was not simply executed. |
| `grant_id` | Which grant permitted it — the row with somebody's name on it. |
| `headroom_before`, `headroom_after` | What the ceiling had left either side of the decision. A refusal keeps them equal, which is the ceiling holding. |
| `approver` | Who decided, where a human did. `None` is what makes a record unattended. |
| `arguments_digest` | A digest of the payload. Never the payload. |
| `idempotency_key` | What identifies the call rather than the attempt, so a retried activity writes one record. |
| `recorded_at` | When. |

`event.unattended` is the property the question is usually asked through: executed, with no
approver.

## Refusals weigh the same as executions

Four decisions are recorded, and the three that are not executions are the reason this exists:

- **executed** — recorded by the run loop at the one point a call is cleared past every gate,
  so a call the ladder permitted but a tool-declared approval later stopped is never recorded
  as executed.
- **escalated** — the ladder sent it to a human. Recorded whether or not the human ever
  answers, which is the record that shows a ceiling held: an amount over the headroom stops
  here, with `headroom_before` and `headroom_after` equal.
- **refused** — the ladder would not, and nothing was asked. A tool that would issue autonomy
  is the case that reaches this.
- **revoked** — a grant was withdrawn while the run waited on a person, so authority that
  existed at the start of the wait did not exist at the end of it.

## What is never stored

Arguments are digested, not kept. Before the digest is taken, the payload goes through the
same redaction the guardrails use, so a value that looks like a card number, a key or an
email is replaced first and the digest is of the redacted form. Two calls with the same
payload have the same digest; nobody can read the payload back out of it.

```python
AuditTrail(sink, clock=clock, redact_patterns=[r"CASE-\d{4}-\d{6}"])
```

`redact_patterns` is for shapes a deployment knows about and the built-in ones cannot —
a local case, matter or account reference.

## Nothing acts unaudited

If the sink cannot take the record, the call does not go out. `AuditTrail.record` raises
`AuditUnavailableError` and the run loop fails the run before dispatch, rather than
performing an action nobody can defend afterwards.

That is the deliberate trade: an audit outage stops unattended work. A deployment that would
rather keep acting is a deployment that has decided its audit trail is advisory, and the
kit will not quietly make that decision for it.

## The stores

| Sink | For |
|---|---|
| `MemoryAuditSink` | Tests and single-process demos. Nothing outlives the process. |
| `PostgresAuditSink` | The queryable record. Append-only, one row per decision, `UNIQUE (run_id, idempotency_key, decision)`. |
| `JetStreamAudit` | The same record on a stream, for audit off the transaction path or in a second administrative domain, where the write is a publish that database access cannot undo. |

`EXPECTED_AUDIT_SCHEMA` documents the table. The migration repository applies it; the kit
reads `adk_schema` at startup and refuses a version it was not written for. `REVOKE UPDATE,
DELETE … FROM PUBLIC` is not decoration — the only statement in the adapter that is not an
insert is the erasure, and the schema grants it to a role of its own.

`JetStreamAudit` publishes on `adk.audit.<tenant>`, so a consumer can be authorised for one
tenant and no other. It appends only; pair it with a queryable sink, or with a consumer that
writes one.

## Asking what an agent did

The recipe the store exists for — one tenant, one period, declines included:

```python
records = await sink.records(tenant="acme", since=week_ago, until=now)
unattended = [event for event in records if event.unattended]
declined = [event for event in records if event.decision is not AuditDecision.EXECUTED]
```

Or one kind at a time, which is the cheaper read:

```python
refused = await sink.records(tenant="acme", since=week_ago, decision=AuditDecision.REFUSED)
```

Records come back oldest first, ordered by `(recorded_at, run_id, sequence)`, so a single
run's decisions stay in the order they were taken even when two runs interleave. In SQL, the
same question against the table directly:

```sql
SELECT payload->>'agent_name'  AS agent,
       payload->>'decision'    AS decision,
       count(*)
FROM adk_audit
WHERE tenant = $1 AND recorded_at >= $2 AND recorded_at < $3
GROUP BY 1, 2
ORDER BY 1, 2;
```

`adk_audit_asked (tenant, recorded_at)` is the index that read uses.

## Retention and erasure

Decisions are kept for as long as the deployment is accountable for the actions they
permitted — seven years is the usual floor where money moved. Expiry is a scheduled job in
the migration repository, never a statement the kit can issue.

An erasure request does not delete the record. Deleting it would take the evidence that the
action was permitted along with the person's name, which serves nobody:

```python
changed = await sink.pseudonymise(tenant="acme", subject="ops@acme.example")
```

The person is replaced by a stable stand-in (`anon:` and a digest) wherever they appear as
`user` or `approver`; the decision, the grant, the ceiling and the reason all stay. Pass
`pseudonym_salt` so two deployments cannot join their audit stores on the same person.

## Not the telemetry pipeline

None of this goes through OpenTelemetry, and that is the design. Sampling that is correct for
spans is a missing record here, and the missing one is always the one being asked about.
Spans and metrics answer "is it healthy"; the audit trail answers "was that allowed". Keep
them separate and neither has to compromise for the other.

## See also

- [Autonomy](autonomy.md) — the grants, ceilings and levels the decisions are taken against.
- [Suspension](suspension.md) — the multi-day approvals a `revoked` record comes out of.
- [Erasure](erasure.md) — the wider right-to-erasure flow this participates in.
