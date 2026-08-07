"""What a run consumed, what it cost, and how much of that is actually known.

Four scenarios: one workload read the same way whichever vendor answered it; a cache saving
that stays visible instead of vanishing into a total; a negotiated rate laid over the shipped
one; and a model nobody has priced, which is not a model that is free.

Run it with `python examples/cost.py`. Scripted providers stand in for the vendors, so
nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
import warnings
from datetime import date
from decimal import Decimal

from tesserix_adk.core import (
    Agent,
    CountSource,
    ModelCapabilities,
    RateLimitError,
    RetryConfig,
    RunEventKind,
    Usage,
)
from tesserix_adk.models.pricing import (
    PriceCard,
    PriceList,
    Rate,
    UnknownPricing,
    cost_of,
    kit_prices,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)
TODAY = date(2026, 8, 7)
SONNET = "anthropic:claude-sonnet-4-5"


def workload() -> Usage:
    """One recorded call: a long prompt mostly served from cache, with hidden reasoning."""
    return Usage(
        input_tokens=1_000_000,
        cached_tokens=800_000,
        cache_write_tokens=200_000,
        output_tokens=50_000,
        reasoning_tokens=20_000,
    )


def one_workload_one_bill() -> None:
    """The components stay apart, so the cache saving is a number somebody can question."""
    money = cost_of(workload(), SONNET, at=TODAY)
    print("=== what one call came to")  # noqa: T201
    print(f"fresh prompt   {money.input}")  # noqa: T201
    print(f"cache read     {money.cache_read}")  # noqa: T201
    print(f"generated      {money.output}")  # noqa: T201
    print(f"total          {money.quantised(4).total} {money.currency} ({money.confidence})")  # noqa: T201
    naive = workload().input_tokens * Decimal("3.00") / Decimal(1_000_000)
    print(f"billing the whole prompt fresh would have said {naive}")  # noqa: T201


def a_rate_nobody_publishes() -> None:
    """A negotiated list replaces the shipped cards for the models it names, dates included."""
    negotiated = PriceList(
        cards=(
            PriceCard(
                ref=SONNET,
                effective_from=date(2026, 1, 1),
                rate=Rate(
                    input_per_mtok=Decimal("1.80"),
                    output_per_mtok=Decimal("9.00"),
                    cache_read_per_mtok=Decimal("0.18"),
                ),
            ),
        )
    )
    prices = kit_prices().overridden_by(negotiated)
    print("\n=== the same call at an agreed rate")  # noqa: T201
    print(f"list       {cost_of(workload(), SONNET, at=TODAY).quantised(4).total}")  # noqa: T201
    agreed = cost_of(workload(), SONNET, at=TODAY, prices=prices)
    print(f"agreed     {agreed.quantised(4).total}")  # noqa: T201


def a_model_nobody_priced() -> None:
    """Zero components and `UNKNOWN`, never a silent free call."""
    print("\n=== a self-hosted model with no card")  # noqa: T201
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always", UnknownPricing)
        money = cost_of(workload(), "vllm:qwen-3", at=TODAY)
    print(f"warned     {raised[0].message}")  # noqa: T201
    print(f"reported   {money.total} at confidence {money.confidence}")  # noqa: T201


async def a_run_that_had_to_try_twice() -> None:
    """The prompt the first vendor read and then refused is still on the ledger."""
    runner = AgentRunner(
        provider=ScriptedProvider(
            RateLimitError("slow down", provider="anthropic", model="claude-sonnet-4-5"),
            ModelResponse(
                content="Kyoto, four nights.",
                usage=Usage(input_tokens=1200, output_tokens=90),
            ),
            name="anthropic",
            capabilities=CAPABLE,
        ),
        retry=RetryConfig(max_attempts=2),
        clock=FakeClock(),
    )
    agent = Agent(
        name="planner",
        instructions="Plan trips.",
        free_text=True,
        model="claude-sonnet-4-5",
    )
    run = await runner.run(agent, "Where should I go in November?", tenant="acme")
    burned = [event for event in run.events if event.kind is RunEventKind.ATTEMPT_FAILED]
    print("\n=== a run the vendor refused once")  # noqa: T201
    print("the vendor counted   1200 in / 90 out")  # noqa: T201
    print(f"the run recorded     {run.usage.input_tokens} in / {run.usage.output_tokens} out")  # noqa: T201
    print(f"burned on attempt 1  {burned[0].usage.input_tokens if burned[0].usage else 0}")  # noqa: T201
    print(f"so the total is a    {run.usage.source}, not a {CountSource.PROVIDER}")  # noqa: T201


async def main() -> None:
    """Run the four scenarios in order."""
    one_workload_one_bill()
    a_rate_nobody_publishes()
    a_model_nobody_priced()
    await a_run_that_had_to_try_twice()


if __name__ == "__main__":
    asyncio.run(main())
