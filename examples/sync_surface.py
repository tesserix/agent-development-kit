"""One async core, reached from a script, and a blocking body that gets caught.

The same agent runs from an async service and from a plain script, and both produce the
same record. A sync helper called from inside a live loop refuses by name instead of
deadlocking. A tool that blocks the loop is attributed to the tool that blocked it, and a
tool that declares itself blocking gets a thread and keeps its tenant.

Scripted providers stand in for a vendor, so nothing reaches the network and no key is
needed. Run it with `python examples/sync_surface.py`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from tesserix_adk.core import Agent, EventLoopStalledError, RunningLoopError, ToolCall, Usage
from tesserix_adk.runtime import (
    AgentRunner,
    Ambient,
    LoopMonitor,
    ModelResponse,
    WorkerPool,
    Workers,
    carrying,
    current_ambient,
)
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

PLANNER = Agent(
    name="planner",
    instructions="Plan trips.",
    model="claude-sonnet-5",
    free_text=True,
    tools=("timetable",),
)

TIMETABLE = ToolCall(id="c1", name="timetable", arguments={"leg": "Osaka to Kyoto"})


def legacy_timetable(leg: str) -> str:
    """A library from before anyone had heard of a coroutine."""
    time.sleep(0.08)
    return f"{leg}: 09:12, 11:40"


def runner(**overrides: object) -> AgentRunner:
    """A run whose first turn calls the timetable and whose second answers."""
    fields: dict[str, Any] = {
        "provider": ScriptedProvider(
            ModelResponse(
                content="",
                tool_calls=(TIMETABLE,),
                usage=Usage(input_tokens=812, output_tokens=18),
            ),
            ModelResponse(
                content="Kyoto, four nights.", usage=Usage(input_tokens=812, output_tokens=24)
            ),
            capabilities=CAPABLE,
        ),
        "clock": FakeClock(),
        "tools": FakeToolRegistry({"timetable": lambda leg: f"{leg}: 09:12"}),
    }
    return AgentRunner(**{**fields, **overrides})


async def the_same_run_from_either_surface() -> None:
    """The sync path is the async path driven differently, so the records match."""
    awaited = await runner().run(PLANNER, "Four nights near Kyoto.", tenant="acme", run_id="r1")
    driven = await asyncio.to_thread(
        lambda: runner().run_sync(PLANNER, "Four nights near Kyoto.", tenant="acme", run_id="r1")
    )

    print("\nthe same agent, two surfaces")  # noqa: T201
    print(f"  awaited: {awaited.state}, {awaited.usage.input_tokens} in")  # noqa: T201
    print(f"  driven:  {driven.state}, {driven.usage.input_tokens} in")  # noqa: T201
    print(f"  identical records: {awaited == driven}")  # noqa: T201


async def a_sync_helper_inside_a_live_loop() -> None:
    """A refusal that names the alternative, rather than a deadlock that names nothing."""
    try:
        runner().run_sync(PLANNER, "Four nights near Kyoto.", tenant="acme")
    except RunningLoopError as refusal:
        print("\ncalled from inside a running loop")  # noqa: T201
        print(f"  refused: {refusal}")  # noqa: T201
        print(f"  await instead: {refusal.async_name}")  # noqa: T201


async def a_tool_that_blocked_the_loop() -> None:
    """The loop's lag is measured while the tool runs, so the tool is named."""
    watchful = LoopMonitor(stall_seconds=0.01, interval=0.005)

    async def undeclared() -> str:
        return legacy_timetable("Osaka to Kyoto")

    try:
        await watchful.watching("tool timetable", undeclared)
    except EventLoopStalledError as stall:
        print("\na blocking body nobody declared")  # noqa: T201
        print(f"  caught: {stall}")  # noqa: T201
        print(f"  blamed: {stall.tool}, {stall.blocked_seconds:.3f}s behind")  # noqa: T201


async def a_tool_that_declared_itself() -> None:
    """Declared blocking bodies get a bounded thread, and keep the run's identity."""
    with WorkerPool(Workers(size=2)) as pool:
        with carrying(Ambient(run_id="r1", tenant="acme", user="dana")):
            leg = await pool.call("timetable", lambda: legacy_timetable("Osaka to Kyoto"))
            who = await pool.call("timetable", current_ambient)

        watchful = LoopMonitor(stall_seconds=0.01, interval=0.005)
        offloaded = await watchful.watching(
            "tool timetable", lambda: pool.call("timetable", lambda: legacy_timetable("Kobe"))
        )

    print("\na blocking body on a worker")  # noqa: T201
    print(f"  returned: {leg}")  # noqa: T201
    print(f"  tenant crossed the hop: {who.tenant if who else None}")  # noqa: T201
    print(f"  the loop kept turning, so nothing was blamed: {offloaded}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await the_same_run_from_either_surface()
    await a_sync_helper_inside_a_live_loop()
    await a_tool_that_blocked_the_loop()
    await a_tool_that_declared_itself()


if __name__ == "__main__":
    asyncio.run(main())
