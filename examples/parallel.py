"""Several branches at once, capped, aggregated, and never quietly partial.

Six scenarios: every branch under a cap, a quorum that tolerates a loss, a strategy that
fails closed, first-in-declared-order, a caller's own reducer, and a shared ledger that
runs out mid-fan-out.
Run it with `python examples/parallel.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    AggregationError,
    BudgetLimits,
    BudgetScope,
    ScopedLimits,
    Usage,
    most_restrictive,
)
from tesserix_adk.core.budget import RunBudget
from tesserix_adk.runtime import (
    AgentRunner,
    All,
    Branch,
    BranchResult,
    Delegation,
    DelegationScope,
    FirstSuccess,
    ModelResponse,
    Quorum,
    Reduce,
    Roster,
    Specialist,
    Supervisor,
    fan_out,
)
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

HELD = frozenset({"search", "browse", "refund"})


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


def _supervising(*answers: str | Exception, tokens: int = 10, ceiling: int = 10_000) -> Supervisor:
    """A supervisor over one ledger, with a script of what its workers come back with."""
    runner = AgentRunner(
        provider=ScriptedProvider(
            *(
                one
                if isinstance(one, Exception)
                else ModelResponse(content=one, usage=Usage(input_tokens=tokens, output_tokens=5))
                for one in answers
            )
        ),
        clock=FakeClock(),
        tools=FakeToolRegistry(dict.fromkeys(sorted(HELD), str)),
    )
    return Supervisor(
        runner,
        Roster(
            (
                Specialist(
                    agent=_agent("researcher", "search", "browse"),
                    capabilities=frozenset({"research"}),
                ),
                Specialist(agent=_agent("accountant", "refund"), capabilities=frozenset({"sums"})),
            )
        ),
        agent=_agent("planner", *sorted(HELD)),
        delegation=Delegation.root(
            run_id="run_1", tenant="acme", agent="planner", scope=DelegationScope(tools=HELD)
        ),
        budget=RunBudget(
            most_restrictive(
                ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_input_tokens=ceiling))
            ),
            clock=FakeClock(),
        ),
    )


def _branches(count: int) -> tuple[Branch, ...]:
    """Independent lookups, each named so what it contributed can be said afterwards."""
    return tuple(
        Branch(name=f"leg{index}", task=f"price leg {index}", needs={"research"})
        for index in range(count)
    )


def _said(data: str) -> str:
    """What a worker said, out of the untrusted-data envelope it crosses back in."""
    return data.split(">\n", 1)[1].rsplit("\n<", 1)[0]


async def every_branch_under_a_cap() -> None:
    """Four branches, two at a time — a number somebody chose, not the rate limiter's."""
    done = await fan_out(_supervising("LHR", "JFK", "SFO", "NRT"), _branches(4), max_concurrency=2)

    print("=== every branch under a cap ===")  # noqa: T201
    print(f"peaked at {done.peak_in_flight} in flight for {len(done.results)} branches")  # noqa: T201
    print(f"answers: {[_said(one) for one in done.value]}")  # noqa: T201


async def a_quorum_that_tolerates_a_loss() -> None:
    """Enough answered, so the aggregate forms — and says who is not in it."""
    supervisor = _supervising("LHR", RuntimeError("the provider fell over"), "SFO")
    done = await fan_out(supervisor, _branches(3), into=Quorum(2))

    print("\n=== a quorum that tolerates a loss ===")  # noqa: T201
    print(f"contributed: {done.contributed}")  # noqa: T201
    print(f"excluded: {done.excluded}")  # noqa: T201


async def a_strategy_that_fails_closed() -> None:
    """`All` is the default, because a branch missing from an answer is usually a bug."""
    supervisor = _supervising("LHR", RuntimeError("the provider fell over"), "SFO")

    print("\n=== a strategy that fails closed ===")  # noqa: T201
    try:
        await fan_out(supervisor, _branches(3), into=All())
    except AggregationError as refused:
        print(f"{refused.strategy}/{refused.reason}: {refused.contributed} contributed")  # noqa: T201


async def the_first_one_in_declared_order() -> None:
    """Declared order, not finishing order: an answer must not depend on scheduling."""
    supervisor = _supervising(RuntimeError("fell over"), "JFK", "SFO")
    done = await fan_out(supervisor, _branches(3), into=FirstSuccess())

    print("\n=== the first one in declared order ===")  # noqa: T201
    print(f"{done.contributed}: {_said(done.value)}")  # noqa: T201


async def a_reducer_of_the_callers_own() -> None:
    """The reducer sees the branch results, so it can attribute what it used."""
    supervisor = _supervising("7", RuntimeError("fell over"), "11")

    def _total(results: list[BranchResult]) -> int:
        return sum(int(_said(one.data)) for one in results)

    done = await fan_out(supervisor, _branches(3), into=Reduce(_total))

    print("\n=== a reducer of the caller's own ===")  # noqa: T201
    print(f"{done.value} from {done.contributed}")  # noqa: T201


async def a_shared_ledger_that_runs_out() -> None:
    """One ceiling for the whole fan-out, so the branches left are refused, not run."""
    supervisor = _supervising("LHR", "JFK", "SFO", "NRT", tokens=40, ceiling=60)
    done = await fan_out(supervisor, _branches(4), max_concurrency=1, into=Quorum(1))

    print("\n=== a shared ledger that runs out ===")  # noqa: T201
    for one in done.results:
        print(f"{one.branch}: {one.outcome} {one.reason[:52]}")  # noqa: T201
    print(f"spent in total: {done.usage.input_tokens} input tokens")  # noqa: T201


def main() -> None:
    """Run every scenario in the order the docs describe them."""
    asyncio.run(every_branch_under_a_cap())
    asyncio.run(a_quorum_that_tolerates_a_loss())
    asyncio.run(a_strategy_that_fails_closed())
    asyncio.run(the_first_one_in_declared_order())
    asyncio.run(a_reducer_of_the_callers_own())
    asyncio.run(a_shared_ledger_that_runs_out())


if __name__ == "__main__":
    main()
