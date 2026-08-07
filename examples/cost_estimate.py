"""Deciding whether a run is worth starting, before it starts.

Three scenarios: an estimate built from what this agent's runs actually did; a run refused
pre-flight because it does not fit what is left of the budget; and the same estimate put to
a human, who approves it. Nothing here reaches the network — the only thing the provider is
asked for is a token count.

Run it with `python examples/cost_estimate.py`.
"""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from tesserix_adk.core import Agent, ApprovalDecision, ApprovalRecord, BudgetLimits, Cost, Usage
from tesserix_adk.core.errors import BudgetExceededError
from tesserix_adk.models.pricing import pricing_at
from tesserix_adk.runtime import (
    AgentRunner,
    CostEstimate,
    InMemoryHistory,
    ModelResponse,
    approval_for,
    calibrate,
    estimate_run,
    refuse_unaffordable,
)
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core import Message

RESEARCHER = Agent[Any](
    name="researcher",
    instructions="Research the question, then answer it.",
    model="anthropic:claude-opus-4-1",
    free_text=True,
)
TODAY = date(2026, 8, 7)
PROMPT_TOKENS = 1_800


# What five past runs of this agent consumed. They differ, which is the whole reason an
# estimate is a range: a spread collapsed to one number is a promise nobody made.
PAST = (
    (900, 120, "0.02"),
    (1_400, 200, "0.03"),
    (1_800, 240, "0.04"),
    (2_600, 380, "0.06"),
    (4_100, 700, "0.11"),
)


class Scripted(ScriptedProvider):
    """A vendor that answers once, and counts tokens without being paid to."""

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """The prompt this agent sends, as this vendor counts it."""
        return PROMPT_TOKENS * len(messages) // len(messages)


def a_provider(spent: tuple[int, int, str] = PAST[2]) -> Scripted:
    """One scripted answer, consuming what a past run of this agent consumed."""
    read, written, price = spent
    return Scripted(
        ModelResponse(
            content="It was a Greek astronomical calculator, recovered in 1901.",
            usage=Usage(
                input_tokens=read,
                output_tokens=written,
                cost=Cost(input=Decimal(price), currency="USD"),
            ),
        ),
        name="scripted",
        capabilities=CAPABLE,
    )


async def a_history() -> InMemoryHistory:
    """Runs that already happened, which is what makes an estimate a measurement."""
    history = InMemoryHistory()
    for spent in PAST:
        history.record(
            await AgentRunner(provider=a_provider(spent), clock=FakeClock()).run(
                RESEARCHER, "what happened to the Antikythera mechanism?", tenant="acme"
            )
        )
    return history


def an_estimate(history: InMemoryHistory) -> CostEstimate:
    """The same question, priced against those runs."""
    return estimate_run(
        RESEARCHER,
        "what happened to the Antikythera mechanism?",
        provider=a_provider(),
        pricing=pricing_at(TODAY),
        history=history,
    )


async def what_it_will_cost() -> CostEstimate:
    """The number, its range, and what it rests on."""
    estimate = an_estimate(await a_history())
    print("\n=== before starting")  # noqa: T201
    print(  # noqa: T201
        f"{estimate.point.total:.4f} {estimate.point.currency} "
        f"({estimate.low.total:.4f} to {estimate.high.total:.4f}), {estimate.confidence} "
        f"over {estimate.assumptions.runs_observed} runs"
    )
    print(  # noqa: T201
        f"assuming {estimate.assumptions.iterations} turns, "
        f"{estimate.assumptions.tool_calls} tool calls, "
        f"a {estimate.assumptions.prompt_tokens}-token prompt"
    )
    return estimate


def refused_before_the_first_call(estimate: CostEstimate) -> None:
    """A ceiling with nothing left in it stops the run before it spends anything."""
    print("\n=== against what is left of the budget")  # noqa: T201
    try:
        refuse_unaffordable(estimate, BudgetLimits(max_cost=Decimal("0.01"), currency="USD"))
    except BudgetExceededError as refusal:
        print(f"refused: {refusal}")  # noqa: T201


async def put_to_a_human(estimate: CostEstimate) -> None:
    """The same estimate, shown to somebody who decides."""
    record = approval_for(estimate, RESEARCHER, run_id="run_1", tenant="acme")
    decision = await _asked(record)
    print("\n=== put to a human")  # noqa: T201
    print(record.reason)  # noqa: T201
    print(f"{decision.decided_by}: {'approved' if decision.granted else 'declined'}")  # noqa: T201

    finished = await AgentRunner(provider=a_provider(), clock=FakeClock()).run(
        RESEARCHER, "what happened to the Antikythera mechanism?", tenant="acme"
    )
    calibration = calibrate(estimate, finished)
    ratio = calibration.ratio if calibration.ratio is not None else Decimal(0)
    print("\n=== estimate against actual")  # noqa: T201
    print(  # noqa: T201
        f"estimated {calibration.estimated.total:.4f}, actual {calibration.actual.total:.4f}, "
        f"{ratio:.2f}x, within range: {calibration.within_range}"
    )


async def _asked(record: ApprovalRecord) -> ApprovalDecision:
    """A gate that a person is standing at. Here, one that says yes."""
    return ApprovalDecision(record_id=record.id, granted=True, decided_by="ada")


async def main() -> None:
    """Run the three scenarios in order."""
    estimate = await what_it_will_cost()
    refused_before_the_first_call(estimate)
    await put_to_a_human(estimate)


if __name__ == "__main__":
    asyncio.run(main())
