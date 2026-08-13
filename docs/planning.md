# Planning — a planner that reasons, an executor that acts

An agent that plans and acts in the same breath executes its own hallucinations. A step is
a sentence, the sentence becomes a call, and nothing between the two ever established that
the tool exists, that this agent may call it, or that the arguments are the shape the tool
declared. The failure is not that the model was wrong; it is that nothing was in a position
to notice.

So the two halves are separated. The planner produces a typed `Plan` and holds no tools at
all. Deterministic code validates that plan in full — registry, allowlist, delegated scope,
argument schemas, dependency graph — and only then runs it.

```python
planner = AgentPlanner(runner, planning_agent, delegation=delegation)
executor = PlanExecutor(tools, contracts, agent=courier, delegation=delegation)

plan = await executor.planned(planner, "get them to New York")
done = await executor.execute(plan)
done.outcomes["s1"]   # what the first step returned
```

## What a plan is

A `PlanStep` names a registered tool and carries arguments; it never carries a sentence
somebody has to interpret. `depends_on` gives the execution order, so the order the planner
happened to write the steps in decides nothing.

```python
Plan(
    goal="get them to New York",
    steps=(
        PlanStep(id="s1", tool="search_flights", arguments={"origin": "LHR", "destination": "JFK"}),
        PlanStep(id="s2", tool="book_flight", arguments={"flight": "BA117", "seats": 1},
                 depends_on=("s1",), intent="hold the seat before the fare moves"),
    ),
)
```

`intent` is for the person reading the plan back. Nothing executes it.

## A planner cannot dispatch

`AgentPlanner` refuses at construction an agent that declares any tool, and an agent whose
`output_type` is not `Plan`. Both are `ConfigurationError`, raised where the planner is
built rather than discovered on the run that mattered:

```python
AgentPlanner(runner, agent_holding_book_flight, delegation=delegation)
# ConfigurationError: ... so it could dispatch what it is planning
```

This is the separation made structural. A convention that says "the planner shouldn't call
tools" is a comment; a constructor that refuses is a rule.

## What the executor checks before anything runs

`validate` refuses the whole plan, and refuses it before the first step touches anything.
`PlanValidationError` carries the step, the tool, a `reason` and the raw planner payload:

| `reason` | What was wrong |
|---|---|
| `empty` | The planning run produced no plan at all. |
| `too_long` | More steps than `max_steps`. Refused, never truncated. |
| `unknown_tool` | The registry has no such tool. |
| `not_allowed` | The agent's allowlist or the delegated scope does not include it. |
| `arguments` | An argument is absent, undeclared, or not the type the tool declared. |
| `dependency` | A step waits on a step nobody planned. |
| `cycle` | Steps wait on each other. Caught here, not as a runtime deadlock. |
| `replan` | The planner produced an invalid plan more times than `max_replans` allowed. |

Nothing is repaired. An executor that dropped an undeclared argument, coerced `"2"` into
`2`, or trimmed a plan to fit would be deciding what the planner meant — which is exactly
the decision this separation exists to keep out of the runtime.

```python
ToolContract(tool="book_flight", accepts=Booking, irreversible=True)
```

A `ToolContract` says what one tool takes, as a Pydantic model validated **strictly**, and
whether the effect can be undone. Every tool the agent may call needs one: a tool without a
contract is a step nothing could have checked, and the executor refuses to be built without
it.

## What touches the world

An irreversible step is cleared by a person or by a matching grant, however confident the
planner was. Where an `AutonomyLadder` classifies the tool, the ladder decides: `ACT`
proceeds under the grant, `ESCALATE` goes to the approval gate, `REFUSE` raises
`AutonomyRefusedError`. Where the tool is in no action class, `contract.irreversible` and
the agent's `approval_required_tools` decide.

Every step is cleared **before the first one runs**, so a denial halfway down the plan
leaves nothing partially executed. A step that needs a person where no gate was configured
is a `ConfigurationError`, not a silent pass.

## Replanning, bounded

`planned(planner, task)` validates what came back and asks again with the refusal as
feedback. The allowance is `max_replans`; past it, the last refusal is raised with
`reason="replan"` and `attempts`. A planner that keeps regenerating an invalid plan is a
loop with a model in it, and an unbounded loop with a model in it is a bill.

Each attempt mints the next `revision`, so two plans for one task are distinguishable in
the record.

## Resuming

Given a `PlanStore`, the plan and its results are written before the first step and after
every step, so a process that dies mid-plan can be picked up:

```python
executor = PlanExecutor(..., plans=InMemoryPlanStore(), idempotency=MemoryIdempotencyStore())
done = await executor.resume()
```

`resume` revalidates against the contracts **as they are now**, because a tool's schema may
have moved since the plan was made; a plan that no longer fits is refused rather than run
against the tool it was not written for. Steps that already ran are replayed from the
record.

The plan store is deliberately not the `CheckpointStore`: a checkpoint is conversation
shaped — messages, frontier, usage — and a plan is a graph of intended effects with a
different lifetime and a different reader.

Where an `IdempotencyStore` is given, each step runs under a key derived from its tenant,
run, tool and arguments (or `key_arguments`, where only some identify the effect). A repeat
returns the recorded outcome with `replayed=True`; a step whose tool failed abandons its key
so the next attempt genuinely runs; a key another caller holds in flight raises
`IndeterminateOutcomeError` rather than being run twice.

Keys and approval records digest the JSON rendering of the validated arguments, while the
tool is called with the types it declared — a `Decimal` amount has no one digest otherwise.

## The record

Everything lands on `executor.events` in order: `PLANNED`, `PLAN_REFUSED`, `REPLANNED`,
`AUTONOMY_ESCALATED`, `AUTONOMY_REFUSED`, `APPROVAL_REQUIRED`, `APPROVAL_GRANTED`,
`APPROVAL_DENIED`, `STEP_EXECUTED`. A plan that was refused is as much a part of the run's
history as one that ran.

## Known limitations

`InMemoryPlanStore` does not outlive the process, which is what a restart-survival
guarantee is not; a deployment that runs more than one replica wants a shared store. Steps
run one at a time in dependency order — independent steps are not yet run concurrently.

See `examples/planner.py` for a runnable end-to-end plan, and
[`docs/autonomy.md`](autonomy.md) for the grant model the ladder consults.
