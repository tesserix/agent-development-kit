"""What a run is allowed to spend, and what happens when it asks for more.

Four scenarios: a ceiling stated in one place and honoured everywhere; two scopes where the
tighter one wins and says so; a sub-agent spending what its parent has left; and a tenant
ceiling shared across runs, including what happens when the ledger holding it is down.

Run it with `python examples/budget.py`. A scripted provider stands in for the vendor, so
nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from tesserix_adk.core import (
    Agent,
    BudgetLimits,
    BudgetScope,
    BudgetUnavailableError,
    ModelCapabilities,
    RunBudget,
    ScopedLimits,
    Usage,
    most_restrictive,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeTenantLedger, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


def runner() -> AgentRunner:
    """A runner given no budget policy at all, which is not a runner without a ceiling."""
    return AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(
                content="Kyoto, four nights.", usage=Usage(input_tokens=900, output_tokens=40)
            ),
            ModelResponse(
                content="Kanazawa next.", usage=Usage(input_tokens=900, output_tokens=40)
            ),
            name="scripted",
            capabilities=CAPABLE,
        ),
        clock=FakeClock(),
    )


def planner(**overrides: object) -> Agent[Any]:
    """The agent every scenario runs."""
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "scripted-1",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


async def nobody_gets_an_unbounded_agent() -> None:
    """No policy, no stated limits, and still a ceiling somebody can read off the run."""
    run = await runner().run(planner(), "Where should I go?", tenant="acme")
    print("=== a runner nobody gave a budget")  # noqa: T201
    print(f"state          {run.state}")  # noqa: T201
    print(f"ceiling        {run.budget.limits.max_model_calls} model calls")  # noqa: T201
    print(f"stated by      {run.budget.sources.get('max_model_calls', 'the defaults')}")  # noqa: T201


def the_tighter_scope_wins() -> None:
    """Nearness does not decide this, and the winner is named."""
    resolved = most_restrictive(
        ScopedLimits(scope=BudgetScope.TENANT, limits=BudgetLimits(max_cost=Decimal("1.00"))),
        ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_cost=Decimal("5.00"))),
    )
    print("\n=== a run that asked for more than its tenant has")  # noqa: T201
    print(f"effective      {resolved.limits.max_cost}")  # noqa: T201
    print(f"attributed to  {resolved.sources['max_cost']}")  # noqa: T201


async def a_child_spends_what_the_parent_has_left() -> None:
    """A sub-agent handed a fresh allowance is a way to spend one ceiling twice."""
    parent = RunBudget(
        resolved=most_restrictive(
            ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_input_tokens=1_000))
        ),
        clock=FakeClock(),
    )
    await parent.record(Usage(input_tokens=700, output_tokens=20))
    print("\n=== a sub-agent's allowance")  # noqa: T201
    print(f"parent had     1000, spent {parent.spent.usage.input_tokens}")  # noqa: T201
    print(f"child starts   {parent.child().limits().max_input_tokens}")  # noqa: T201


async def a_ceiling_two_runs_share() -> None:
    """A tenant ceiling only means anything if every run reads the same total."""
    ledger = FakeTenantLedger()

    def against(store: FakeTenantLedger) -> RunBudget:
        return RunBudget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.TENANT, limits=BudgetLimits(max_input_tokens=1_000))
            ),
            clock=FakeClock(),
            ledger=store,
            tenant="acme",
        )

    await against(ledger).record(Usage(input_tokens=800, output_tokens=10))
    second = against(ledger)
    await second.reserve(10)
    print("\n=== the second run of the hour")  # noqa: T201
    print(f"already spent  {(await ledger.total('acme', 'all')).usage.input_tokens}")  # noqa: T201
    print(f"left for it    {second.limits().max_input_tokens}")  # noqa: T201
    try:
        await against(FakeTenantLedger(reachable=False)).reserve(1)
    except BudgetUnavailableError as unavailable:
        print(f"ledger down    {unavailable}")  # noqa: T201


async def main() -> None:
    """Run the four scenarios in order."""
    await nobody_gets_an_unbounded_agent()
    the_tighter_scope_wins()
    await a_child_spends_what_the_parent_has_left()
    await a_ceiling_two_runs_share()


if __name__ == "__main__":
    asyncio.run(main())
