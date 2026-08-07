"""Consuming a stream: iterate-then-await, await-only, and iterate-and-discard.

The same object serves all three, and only the run's own output is ever validated: what
has arrived mid-stream is a `Provisional`, which the type checker refuses where the
declared output type is required. Scripted providers stand in for a vendor, so nothing here
reaches the network and no key is needed.

Run it with `python examples/stream_consumption.py`.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from tesserix_adk.core import Agent, ToolCall, Usage
from tesserix_adk.runtime import (
    AgentRunner,
    ModelResponse,
    Provisional,
    StructuredDelta,
    ToolCallStarted,
)
from tesserix_adk.testing import CAPABLE, FakeToolRegistry, ScriptedProvider

NATIVE = CAPABLE.declaring(structured_output=True)


class TripPlan(BaseModel):
    """A trip the model proposes.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


PLANNER = Agent(
    name="planner",
    instructions="Plan trips.",
    model="claude-sonnet-5",
    output_type=TripPlan,
    tools=("timetable",),
)


def runner(*responses: ModelResponse) -> AgentRunner:
    """A runner over a scripted provider and one timetable tool."""
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=NATIVE),
        "tools": FakeToolRegistry({"timetable": lambda leg: f"{leg}: 09:12, 11:40"}),
    }
    return AgentRunner(**fields)  # type: ignore[arg-type]


def planned() -> ModelResponse:
    """The structured answer the model settles on."""
    return ModelResponse(
        content='{"destination": "Kyoto", "nights": 4}',
        usage=Usage(input_tokens=812, output_tokens=24),
    )


async def iterate_then_await() -> None:
    """Progress while it happens, then the authoritative record."""
    stream = runner(planned()).stream(PLANNER, "Four nights near Kyoto.", tenant="acme")
    fragments: list[str] = []
    async with stream:
        async for event in stream:
            if isinstance(event, StructuredDelta):
                fragments.append(event.fragment)
    run = await stream

    print("\niterate, then await")  # noqa: T201
    print(f"  the answer arrived in pieces: {fragments}")  # noqa: T201
    print(f"  provisional, a plain mapping: {stream.provisional.snapshot()}")  # noqa: T201
    print(f"  validated result:             {run.output!r}")  # noqa: T201


async def a_half_arrived_object_is_not_a_guess() -> None:
    """What has arrived reads as nothing until it is a whole object."""
    print("\nprovisional content, as it fills in")  # noqa: T201
    for text in ('{"destination"', '{"destination": "Kyoto"', '{"destination": "Kyoto"}'):
        print(f"  {text!r:34} -> {Provisional[TripPlan](text=text).snapshot()}")  # noqa: T201


async def await_only() -> None:
    """A caller that wants the answer and no progress at all."""
    run = await runner(planned()).stream(PLANNER, "Four nights near Kyoto.", tenant="acme")

    print("\nawait, without iterating")  # noqa: T201
    print(f"  state:  {run.state}")  # noqa: T201
    print(f"  output: {run.output!r}")  # noqa: T201


async def iterate_and_discard() -> None:
    """Read until you have seen enough. The run is cancelled, not left running."""
    call = ToolCall(id="c1", name="timetable", arguments={"leg": "Osaka to Kyoto"})
    stream = runner(ModelResponse(content="", tool_calls=(call,)), planned()).stream(
        PLANNER, "Four nights near Kyoto.", tenant="acme"
    )
    async with stream:
        async for event in stream:
            if isinstance(event, ToolCallStarted):
                break

    print("\niterate, then leave early")  # noqa: T201
    print(f"  state:  {stream.run.state}")  # noqa: T201
    print(f"  output: {stream.run.output!r}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await iterate_then_await()
    await a_half_arrived_object_is_not_a_guess()
    await await_only()
    await iterate_and_discard()


if __name__ == "__main__":
    asyncio.run(main())
