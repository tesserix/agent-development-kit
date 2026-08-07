"""What a run does when its consumer cannot keep up.

The buffer is bounded, so one slow client costs its own stream rather than the process. Text
deltas merge under pressure and still reassemble to the whole answer; tool calls, usage and
the terminal event are kept whatever the pressure, because a run missing one of those is a
run nobody can account for. A reader that stops reading altogether stops the run.

Scripted providers stand in for a vendor, so nothing reaches the network and no key is
needed. Run it with `python examples/backpressure.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk.core import Agent, ToolCall, Usage
from tesserix_adk.runtime import AgentRunner, Backpressure, ModelResponse
from tesserix_adk.runtime.progress import AnswerDelta
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.runtime.progress import ProgressEvent

PLANNER = Agent(
    name="planner",
    instructions="Plan trips.",
    model="claude-sonnet-5",
    free_text=True,
    tools=("timetable",),
)

LONG = " ".join(f"word{n}" for n in range(120))


def runner(*responses: ModelResponse, clock: FakeClock | None = None) -> AgentRunner:
    """A runner over a scripted provider and one timetable tool."""
    fields: dict[str, Any] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "clock": clock or FakeClock(),
        "tools": FakeToolRegistry({"timetable": lambda leg: f"{leg}: 09:12, 11:40"}),
    }
    return AgentRunner(**fields)


def answer(text: str) -> ModelResponse:
    """A free-text answer with a plausible token count."""
    return ModelResponse(content=text, usage=Usage(input_tokens=812, output_tokens=240))


def texted(events: Sequence[ProgressEvent]) -> str:
    """The answer as a consumer would have rendered it."""
    return "".join(event.text for event in events if isinstance(event, AnswerDelta))


async def deltas_merge_rather_than_disappear() -> None:
    """Coalescing is not dropping: every character still arrives, in order."""
    stream = runner(answer(LONG)).stream(
        PLANNER, "Four nights near Kyoto.", tenant="acme", backpressure=Backpressure(high_water=4)
    )
    events = [event async for event in stream]

    print("\na buffer of four, against a long answer")  # noqa: T201
    print(f"  deltas delivered: {sum(1 for e in events if isinstance(e, AnswerDelta))}")  # noqa: T201
    print(f"  peak occupancy:   {stream.pressure.peak}")  # noqa: T201
    print(f"  merged away:      {stream.pressure.coalesced}")  # noqa: T201
    print(f"  answer intact:    {texted(events) == LONG}")  # noqa: T201


async def what_is_never_merged() -> None:
    """Lose a tool call or the terminal event and the record of the run is a guess."""
    call = ToolCall(id="c1", name="timetable", arguments={"leg": "Osaka to Kyoto"})
    stream = runner(ModelResponse(content="", tool_calls=(call,)), answer(LONG)).stream(
        PLANNER, "Four nights near Kyoto.", tenant="acme", backpressure=Backpressure(high_water=1)
    )
    events = [event async for event in stream]
    kinds = [event.kind for event in events]

    print("\nunder the tightest budget the kit allows")  # noqa: T201
    calls = f"{kinds.count('tool_call_started')} started, {kinds.count('tool_call_finished')} done"
    print(f"  tool calls:     {calls}")  # noqa: T201
    print(f"  usage updates:  {kinds.count('usage_updated')}")  # noqa: T201
    print(f"  terminal event: {kinds[-1]}")  # noqa: T201


async def a_reader_that_walks_away() -> None:
    """A dead client that never disconnected otherwise bills for a run nobody reads."""
    clock = FakeClock()
    call = ToolCall(id="c1", name="timetable", arguments={"leg": "Osaka to Kyoto"})
    slow = FakeToolRegistry({"timetable": lambda leg: _after(clock, 120.0, leg)})
    fields: dict[str, Any] = {
        "provider": ScriptedProvider(
            ModelResponse(content="", tool_calls=(call,)), answer(LONG), capabilities=CAPABLE
        ),
        "clock": clock,
        "tools": slow,
    }
    stream = AgentRunner(**fields).stream(
        PLANNER,
        "Four nights near Kyoto.",
        tenant="acme",
        backpressure=Backpressure(high_water=2, stall_seconds=30.0),
    )
    events = stream.__aiter__()
    await anext(events)
    run = await stream
    await events.aclose()

    print("\na consumer that read once and stopped")  # noqa: T201
    print(f"  state:   {run.state}")  # noqa: T201
    print(f"  stalled: {stream.pressure.stalled}")  # noqa: T201


async def sizing_a_process_rather_than_a_run() -> None:
    """A per-run bound multiplied by however many runs are in flight bounds nothing."""
    shared = Backpressure.shared(total_bytes=64 * 1024 * 1024, streams=200)

    print("\n64 MiB across 200 concurrent runs")  # noqa: T201
    print(f"  per-run byte budget: {shared.byte_budget}")  # noqa: T201
    print(f"  default, for one run: {Backpressure().byte_budget}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await deltas_merge_rather_than_disappear()
    await what_is_never_merged()
    await a_reader_that_walks_away()
    await sizing_a_process_rather_than_a_run()


def _after(clock: FakeClock, seconds: float, value: str) -> str:
    """A tool that takes `seconds` of the clock's time to answer."""
    clock.advance(seconds)
    return value


if __name__ == "__main__":
    asyncio.run(main())
