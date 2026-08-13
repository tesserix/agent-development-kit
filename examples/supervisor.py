"""One agent handing work to a roster it cannot let outgrow it.

Six scenarios: routing by declared capability, the allowlist a worker ends up holding, an
allowance sliced off the caller's ledger, a worker stopped by that allowance, two workers
reaching for one key, and the record of who was handed what.
Run it with `python examples/supervisor.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    BudgetLimits,
    BudgetScope,
    DelegationError,
    ScopedLimits,
    Usage,
    most_restrictive,
)
from tesserix_adk.core.budget import RunBudget
from tesserix_adk.runtime import (
    AgentRunner,
    Delegation,
    DelegationScope,
    ModelResponse,
    Roster,
    Specialist,
    Supervisor,
)
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

HELD = frozenset({"search", "refund", "summarise"})


def _agent(name: str, *tools: str) -> Agent:
    """An agent that answers in prose, so the run needs nothing else declared."""
    fields: dict[str, object] = {
        "name": name,
        "instructions": f"You are {name}.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": tools,
    }
    return Agent(**fields)  # type: ignore[arg-type]


def _roster() -> Roster:
    """Three workers: one narrow, one broad, one nobody in these scenarios asks for."""
    return Roster(
        (
            Specialist(
                agent=_agent("researcher", "search", "browse"),
                capabilities=frozenset({"flights", "research"}),
            ),
            Specialist(
                agent=_agent("accountant", "refund"),
                capabilities=frozenset({"refund", "research"}),
                budget=BudgetLimits(max_input_tokens=2_000),
            ),
            Specialist(agent=_agent("writer", "summarise"), capabilities=frozenset({"writing"})),
        )
    )


def _supervising(*answers: str) -> Supervisor:
    """A supervisor over a real ledger, with a script of what its workers come back with."""
    runner = AgentRunner(
        provider=ScriptedProvider(
            *(
                ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))
                for text in answers
            )
        ),
        clock=FakeClock(),
        tools=FakeToolRegistry(dict.fromkeys(("search", "browse", "refund", "summarise"), str)),
    )
    return Supervisor(
        runner,
        _roster(),
        agent=_agent("planner", *sorted(HELD)),
        delegation=Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="planner",
            scope=DelegationScope(tools=HELD),
        ),
        budget=RunBudget(
            most_restrictive(
                ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_input_tokens=10_000))
            ),
            clock=FakeClock(),
        ),
    )


async def routed_by_what_a_worker_declared() -> None:
    """The narrowest worker that covers the need takes it, not the first that could."""
    supervisor = _supervising("Two flights, both refundable.")
    result = await supervisor.delegate("find two flights", needs={"research", "flights"})

    print("=== routed by what a worker declared ===")  # noqa: T201
    print(f"{result.specialist} answered: {result.answered}")  # noqa: T201
    print(result.data)  # noqa: T201


async def what_the_worker_ended_up_holding() -> None:
    """`browse` is the researcher's own and not the planner's, so it is not there."""
    supervisor = _supervising("Two flights.")
    result = await supervisor.delegate("find two flights", needs={"flights"})

    print("\n=== what the worker ended up holding ===")  # noqa: T201
    print(f"declared search+browse, ran with {result.run.grant.tools}")  # type: ignore[union-attr]  # noqa: T201


async def nobody_on_the_roster_can_do_it() -> None:
    """Refused typed, rather than the supervisor doing it with its own wider access."""
    print("\n=== nobody on the roster can do it ===")  # noqa: T201
    try:
        await _supervising().delegate("argue the appeal", needs={"litigation"})
    except DelegationError as refused:
        print(f"{refused.reason}: {refused}")  # noqa: T201


async def an_allowance_off_the_callers_ledger() -> None:
    """What a worker spends is the caller's, under the worker's own name."""
    supervisor = _supervising("Two flights.", "Refunded.")
    await supervisor.delegate("find two flights", needs={"flights"})
    await supervisor.delegate("refund the first", needs={"refund"})

    print("\n=== an allowance off the caller's ledger ===")  # noqa: T201
    for name, usage in supervisor.spent.items():
        print(f"{name}: {usage.input_tokens} in, {usage.output_tokens} out")  # noqa: T201


async def a_worker_that_ran_out() -> None:
    """The worker stops; the supervisor's run does not, unless the call said `fatal`."""
    supervisor = _supervising("Two flights.")
    result = await supervisor.delegate(
        "find two flights", needs={"flights"}, budget=BudgetLimits(max_input_tokens=1)
    )

    print("\n=== a worker that ran out ===")  # noqa: T201
    print(f"{result.run.state}: {result.error.reason}")  # type: ignore[union-attr]  # noqa: T201
    print(f"still attributed: {result.specialist} in {sorted(supervisor.spent)}")  # noqa: T201


async def two_workers_and_one_key() -> None:
    """Concurrent writes to one key are refused rather than silently ordered by luck."""
    supervisor = _supervising("Two flights.", "Refunded.")
    await supervisor.delegate("find two flights", needs={"flights"}, writes="itinerary")

    print("\n=== two workers and one key ===")  # noqa: T201
    try:
        await supervisor.delegate("refund the first", needs={"refund"}, writes="itinerary")
    except DelegationError as refused:
        print(f"{refused.reason}: {refused}")  # noqa: T201
    print(f"handed over: {[(e.kind.value, e.name) for e in supervisor.events]}")  # noqa: T201


def main() -> None:
    """Run every scenario in the order the docs describe them."""
    asyncio.run(routed_by_what_a_worker_declared())
    asyncio.run(what_the_worker_ended_up_holding())
    asyncio.run(nobody_on_the_roster_can_do_it())
    asyncio.run(an_allowance_off_the_callers_ledger())
    asyncio.run(a_worker_that_ran_out())
    asyncio.run(two_workers_and_one_key())


if __name__ == "__main__":
    main()
