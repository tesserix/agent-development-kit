"""A turn that asked for four tools, run as one bounded batch.

Four independent lookups cost roughly one lookup rather than four, but not by firing
everything at whatever is downstream: each call stands in the lanes declared for its turn,
its tool and its tenant. A slow tool spends its own ceiling, a failing one is reported
against its own call id, and a tool that cannot be parallelised says so.

Scripted providers stand in for a vendor, so nothing reaches the network and no key is
needed. Run it with `python examples/tool_concurrency.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk.core import Agent, ConcurrencyConfig, ToolCall, Usage
from tesserix_adk.core.provider import ToolDeclaration
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from tesserix_adk.core.run import Run

LOOKUPS = ("timetable", "weather", "hotels", "events")

PLANNER = Agent(
    name="planner",
    instructions="Plan trips.",
    model="claude-sonnet-5",
    free_text=True,
    tools=LOOKUPS,
)


def asked_for(*names: str) -> ModelResponse:
    """One model response requesting each named tool, in the order given."""
    return ModelResponse(
        content="",
        tool_calls=tuple(
            ToolCall(id=f"c{position}", name=name, arguments={"q": "Kyoto"})
            for position, name in enumerate(names, start=1)
        ),
        usage=Usage(input_tokens=812, output_tokens=48),
    )


ANSWER = ModelResponse(
    content="Kyoto, four nights.", usage=Usage(input_tokens=980, output_tokens=9)
)


class Witness:
    """How many tool bodies were in flight at once, and the most there ever were."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    def slow_tool(self, seconds: float = 0.05) -> Callable[..., Any]:
        """A tool that takes real time, so overlap is visible rather than asserted."""

        async def body(**_: object) -> str:
            self.live += 1
            self.peak = max(self.peak, self.live)
            try:
                await asyncio.sleep(seconds)
            finally:
                self.live -= 1
            return "09:12, 11:40"

        return body


def runner(
    response: ModelResponse, tools: dict[str, Callable[..., Any]], **overrides: object
) -> AgentRunner:
    """A runner over a scripted provider and the given tools."""
    fields: dict[str, Any] = {
        "provider": ScriptedProvider(response, ANSWER, capabilities=CAPABLE),
        "clock": FakeClock(),
        "tools": FakeToolRegistry(tools),
    }
    return AgentRunner(**{**fields, **overrides})


def tool_replies(run: Run[Any]) -> list[str]:
    """What went back to the model, in the order the model asked for it.

    The untrusted-data wrapper the model actually sees is stripped for legibility here;
    `docs/trust-boundary.md` is where that wrapper is explained.
    """
    return [
        "".join(part.text for part in message.content if hasattr(part, "text"))
        .partition(">")[2]
        .partition("</")[0]
        .strip()
        for message in run.messages
        if message.role == "tool"
    ]


async def four_lookups_cost_about_one() -> None:
    """The batch is in flight together, so latency is one lookup rather than four."""
    witness = Witness()
    started = asyncio.get_running_loop().time()
    run = await runner(asked_for(*LOOKUPS), {name: witness.slow_tool() for name in LOOKUPS}).run(
        PLANNER, "Four nights near Kyoto.", tenant="acme"
    )
    elapsed = asyncio.get_running_loop().time() - started

    print("\nfour independent lookups, no declared lanes")  # noqa: T201
    print(f"  in flight at once: {witness.peak}")  # noqa: T201
    print(f"  wall clock:        {elapsed:.2f}s against 0.20s serial")  # noqa: T201
    print(f"  state:             {run.state.value}")  # noqa: T201


async def a_lane_two_wide() -> None:
    """A declared width is what keeps one turn from saturating a partner."""
    witness = Witness()
    run = await runner(
        asked_for(*LOOKUPS),
        {name: witness.slow_tool() for name in LOOKUPS},
        concurrency=ConcurrencyConfig(max_concurrent_tools=2),
    ).run(PLANNER, "Four nights near Kyoto.", tenant="acme")

    print("\nthe same turn, two at a time")  # noqa: T201
    print(f"  in flight at once: {witness.peak}")  # noqa: T201
    print(f"  replies:           {len(tool_replies(run))}, still in call order")  # noqa: T201


async def one_failure_is_one_failure() -> None:
    """A tool that raises loses its own call, not its siblings' results."""

    def boom(**_: object) -> str:
        raise RuntimeError("the timetable service is down")

    tools: dict[str, Callable[..., Any]] = {name: lambda **_: "09:12" for name in LOOKUPS}
    tools["timetable"] = boom
    run = await runner(asked_for(*LOOKUPS), tools).run(
        PLANNER, "Four nights near Kyoto.", tenant="acme"
    )

    print("\none of four fails")  # noqa: T201
    for name, reply in zip(LOOKUPS, tool_replies(run), strict=True):
        print(f"  {name:<10} {reply[:48]}")  # noqa: T201


async def a_slow_tool_spends_its_own_ceiling() -> None:
    """Without a per-tool ceiling, the slowest call sets the price of the whole turn."""
    clock = FakeClock(auto_advance=False)

    async def never(**_: object) -> str:
        await asyncio.Event().wait()
        return "unreachable"

    tools: dict[str, Callable[..., Any]] = {name: lambda **_: "09:12" for name in LOOKUPS}
    tools["hotels"] = never
    running = asyncio.ensure_future(
        runner(
            asked_for(*LOOKUPS),
            tools,
            clock=clock,
            concurrency=ConcurrencyConfig(per_tool_seconds={"hotels": 5.0}),
        ).run(PLANNER, "Four nights near Kyoto.", tenant="acme")
    )
    await clock.wait_for_sleep(1)
    clock.advance(5)
    run = await running

    print("\nhotels hangs, with a five-second ceiling of its own")  # noqa: T201
    print(f"  state:  {run.state.value}")  # noqa: T201
    print(f"  hotels: {tool_replies(run)[2][:60]}")  # noqa: T201


async def order_dependent_tools_run_alone() -> None:
    """A booking is not a lookup: it runs with everything before it already resolved."""
    witness = Witness()
    tools = {name: witness.slow_tool(0.02) for name in LOOKUPS}
    await runner(asked_for(*LOOKUPS), tools).run(PLANNER, "Four nights near Kyoto.", tenant="acme")

    print("\nevery tool parallel-safe by default")  # noqa: T201
    print(f"  in flight at once: {witness.peak}")  # noqa: T201

    witness = Witness()
    alone = AgentRunner(
        provider=ScriptedProvider(asked_for(*LOOKUPS), ANSWER, capabilities=CAPABLE),
        clock=FakeClock(),
        tools=FakeToolRegistry(
            {name: witness.slow_tool(0.02) for name in LOOKUPS},
            declarations={
                "hotels": ToolDeclaration(name="hotels", parallel_safe=False),
            },
        ),
    )
    run = await alone.run(PLANNER, "Four nights near Kyoto.", tenant="acme")

    print("  hotels declared order-dependent")  # noqa: T201
    print(f"  in flight at once: {witness.peak}, and {run.state.value}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await four_lookups_cost_about_one()
    await a_lane_two_wide()
    await one_failure_is_one_failure()
    await a_slow_tool_spends_its_own_ceiling()
    await order_dependent_tools_run_alone()


if __name__ == "__main__":
    asyncio.run(main())
