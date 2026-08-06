# Repair

An answer that fails validation can be sent back to the model with the reason, a bounded
number of times. Repair is neither of the two things products usually do here:

- It is not **coercion**. Nothing fills a missing field with a default, drops a field to
  make an object fit, or casts a value into the declared type. An object the kit finished
  is an object the caller cannot attribute to anyone.
- It is not a **blind retry**. The same prompt sent again is a second charge with no new
  information in it. The repair prompt names every failing path and what was wrong with
  each, taken from the validation error itself.

## Asking for it

```python
Agent(
    name="planner",
    instructions="Plan trips.",
    model="claude-sonnet-5",
    output_type=TripPlan,
    repair=RepairConfig(max_attempts=2),
)
```

Nothing is repaired by default: `Agent.repair` is `None` and the first violation is
terminal. A high-stakes agent that wants that on the record writes
`repair=RepairConfig(enabled=False)` rather than deleting the block, so the decision reads
as a decision instead of an oversight.

## What goes back

Only what failed, and the schema:

```
That answer did not validate against the schema. What failed:
- nights: Field required

Answer again with one JSON object and nothing else, correcting only what is
listed above and inventing nothing. It must validate against this JSON Schema:
{ ... }
```

The kit never supplies a value for the failing field. A prompt that says what the answer
should be is coercion with extra steps, and the answer would be the kit's.

## What it costs

A repair attempt is an ordinary model call: its tokens land on `run.usage`, it is recorded
against the budget policy, and it is bounded by the run deadline and the iteration cap like
any other. Repair cannot spend past a ceiling — a run whose budget will not pay for the
next attempt ends `budget_exhausted` instead of making it.

## What is recorded

| Event | When |
|---|---|
| `schema_violation` | Every attempt that failed, including the ones that were repaired |
| `repair_requested` | The failure was sent back. Names the type, the failing fields, and which attempt of how many |
| `repair_abandoned` | The same failure came back after being told what it was |
| `output_validated` | A later attempt validated |

Repair rate per agent and prompt version is `repair_requested` over runs; the recovery rate
is how many of those still reach `completed`.

## When repair is futile

A model told exactly which field failed and answering with the identical failure is not
going to converge — the declared constraint cannot be satisfied as instructed. Rather than
spending the rest of the budget proving it, the run stops at that point, records
`repair_abandoned` and fails with a configuration error naming the type. That is a defect
in the declaration, not a bad day for the model.

## Running out

Exhausting the budget fails the run. `run.output` is `None`, the last attempt's violation
is on the run with its failing paths and schema hash, and no partial object is returned.
