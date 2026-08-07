"""Declare an agent, walk a run through its states, and account for what it cost.

No network, no credentials, no provider: the primitives are data, so the whole lifecycle
of a run can be modelled — and checkpointed — without one. Run it with
`python examples/typed_primitives.py`.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from tesserix_adk.core import (
    Agent,
    BudgetExceededError,
    Cost,
    Message,
    Run,
    RunState,
    TextPart,
    ToolCall,
    Usage,
    deduplicate,
    legal_transitions,
)


class TripPlan(BaseModel):
    """The shape the agent's answer must take. Anything else is a SchemaViolationError."""

    destination: str
    nights: int


def spent_on(run: object) -> str:
    """What the run cost, or an honest word where nothing priced it."""
    cost = run.usage.cost  # type: ignore[attr-defined]
    return "an unknown amount" if cost is None else f"{cost.total} {cost.currency}"


def main() -> None:
    """Declare, run, account, checkpoint."""
    agent = Agent(
        name="trip-planner",
        instructions="Plan trips. Cite the source of every price.",
        model="claude-sonnet-5",
        tools=("search_flights", "search_hotels"),
        output_type=TripPlan,
        guardrails=("no_pii",),
    )

    run = Run(
        id="run_1",
        tenant="acme",
        user="ada",
        agent_name=agent.name,
        agent_version=agent.version,
        model="claude-sonnet-5",
        messages=[Message(role="user", content=[TextPart(text="Three nights in Kyoto.")])],
    )
    may_go_to = ", ".join(sorted(state.value for state in legal_transitions(run.state)))
    print(f"{run.id} starts {run.state}, may go to: {may_go_to}")  # noqa: T201

    # A retried provider response repeats calls it already sent; running one twice is the bug.
    requested = [
        ToolCall(id="call_1", name="search_flights", arguments={"to": "KIX"}),
        ToolCall(id="call_1", name="search_flights", arguments={"to": "KIX"}),
        ToolCall(id="call_2", name="search_hotels", arguments={"city": "Kyoto"}),
    ]
    calls = deduplicate(requested)
    print(f"provider asked for {len(requested)} calls, {len(calls)} of them distinct")  # noqa: T201

    run = run.transition_to(RunState.RUNNING, at=0.0)
    run = run.record(
        Usage(input_tokens=1_200, output_tokens=300, cost=Cost(input=Decimal("0.004")))
    )
    run = run.record(Usage(input_tokens=800, output_tokens=150, cost=Cost(input=Decimal("0.003"))))
    print(  # noqa: T201
        f"spent {run.usage.input_tokens} in / {run.usage.output_tokens} out, {spent_on(run)}"
    )

    ceiling = 1_500
    if run.input_tokens_spent > ceiling:
        exhausted = run.transition_to(RunState.BUDGET_EXHAUSTED, at=1.0)
        # The state says which ceiling ended the run; "failed" would not.
        raised = BudgetExceededError(
            f"{run.input_tokens_spent} input tokens over a ceiling of {ceiling}",
            run_id=run.id,
            tenant=run.tenant,
        )
        print(f"{exhausted.state}: {raised}")  # noqa: T201
        run = exhausted
    else:
        run = run.transition_to(RunState.COMPLETED, at=1.0)

    # Serialised mid-flight by one process, rehydrated by another. No sockets, no clients.
    rehydrated = Run.model_validate_json(run.model_dump_json())
    print(f"checkpoint round-trips: {rehydrated == run}")  # noqa: T201
    print(f"{run.id} ended {run.state} for tenant {run.context.tenant.tenant}")  # noqa: T201


if __name__ == "__main__":
    main()
