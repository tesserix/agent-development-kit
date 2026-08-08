"""What an agent may call, resolved once, and what a slow tool is allowed to cost.

Four scenarios: two agents sharing one registry with different allowlists; a tool outside
one of them refused without being run; a call held to the ceiling its author declared; and
the spans a registry records about all of it. Run it with `python examples/tool_registry.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import ConcurrencyConfig, ToolNotPermittedError, ToolTimedOutError
from tesserix_adk.tools import ToolCallSpan, ToolRegistry, tool


@tool
async def fare_for(leg: str) -> str:
    """Price one hop of a journey.

    Args:
        leg: The hop to price.
    """
    return f"{leg}: 40 EUR"


@tool
async def rooms_in(city: str) -> str:
    """Find somewhere to stay.

    Args:
        city: Where the traveller is staying.
    """
    return f"{city}: one ryokan, two nights"


@tool
async def refund(booking: str) -> str:
    """Give a fare back, which not every agent may do.

    Args:
        booking: The booking to refund.
    """
    return f"{booking}: refunded"


@tool(timeout=0.05)
async def partner_lookup(reference: str) -> str:
    """Ask a partner that is not answering today.

    Args:
        reference: What to ask about.
    """
    await asyncio.sleep(60)
    return reference


async def main() -> None:
    """Run each scenario and print what the registry decided."""
    spans: list[ToolCallSpan] = []
    registry = ToolRegistry(
        (fare_for, rooms_in, refund, partner_lookup),
        concurrency=ConcurrencyConfig(max_concurrent_tools=4, per_tool={"partner_lookup": 1}),
    )
    registry.observe(spans.append)

    planner = registry.view(allow=("fare_for", "rooms_in"), agent="planner")
    desk = registry.view(allow=("fare_for", "refund"), agent="desk")
    print("planner:    ", planner.names)  # noqa: T201
    print("desk:       ", desk.names)  # noqa: T201

    print("priced:     ", await planner.invoke("fare_for", {"leg": "Osaka to Kyoto"}))  # noqa: T201
    try:
        await planner.invoke("refund", {"booking": "AB-1"})
    except ToolNotPermittedError as refused:
        print("refused:    ", refused)  # noqa: T201
    print("permitted:  ", await desk.invoke("refund", {"booking": "AB-1"}))  # noqa: T201

    try:
        await registry.invoke("partner_lookup", {"reference": "AB-1"})
    except ToolTimedOutError as overran:
        print("overran:    ", overran)  # noqa: T201

    for span in spans:
        print(  # noqa: T201
            f"span:        {span.agent or '-':8} {span.tool:15} "
            f"{span.outcome:10} permitted={span.permitted}"
        )


if __name__ == "__main__":
    asyncio.run(main())
