"""Stopping a stream, and what both sides know afterwards.

A client that hits stop or closes its tab stops the run, not just its own view of it. The
terminal event carries the reason, what the run had spent and where the stream ended, so the
client's account and the server's are reconcilable. A tool already dispatched is reported
indeterminate rather than claimed undone.

Scripted providers stand in for a vendor, so nothing reaches the network and no key is
needed. Run it with `python examples/stream_cancellation.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters import RunBroker, TransportAuthorizationError
from tesserix_adk.core import Agent, ToolCall, Usage
from tesserix_adk.runtime import (
    AgentRunner,
    CancellationToken,
    ModelResponse,
    RunCancelled,
    ToolCallIndeterminate,
)
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from tesserix_adk.runtime import RunStream

PLANNER = Agent(
    name="planner",
    instructions="Plan trips.",
    model="claude-sonnet-5",
    free_text=True,
    tools=("timetable",),
)

TIMETABLE = ToolCall(id="c1", name="timetable", arguments={"leg": "Osaka to Kyoto"})


async def slow(leg: str) -> str:
    """A tool long enough for a client to change its mind while it runs."""
    await asyncio.sleep(0.05)
    return f"{leg}: 09:12, 11:40"


def dispatching(**overrides: object) -> tuple[RunStream[Any], CancellationToken]:
    """A run that is inside a tool call by the time anyone thinks about stopping it."""
    token = CancellationToken()
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
        "tools": FakeToolRegistry({"timetable": slow}),
    }
    agent = PLANNER.model_copy(update=overrides) if overrides else PLANNER
    stream = AgentRunner(**fields).stream(
        agent, "Four nights near Kyoto.", tenant="acme", cancellation=token
    )
    return stream, token


async def stopping_mid_run() -> None:
    """The stop reaches the run, and the terminal event closes the client's view of it."""
    stream, token = dispatching()
    events = []
    async for event in stream:
        events.append(event)
        if event.kind == "tool_call_started":
            token.cancel("the client pressed stop")

    terminal = events[-1]
    assert isinstance(terminal, RunCancelled)  # noqa: S101 — the example's own claim
    print("\nstopped mid-run")  # noqa: T201
    print(f"  reason:        {terminal.reason}")  # noqa: T201
    print(f"  spent by then: {terminal.usage.input_tokens} in, {terminal.usage.output_tokens} out")  # noqa: T201
    print(f"  last event:    {terminal.last_sequence}, terminal at {terminal.sequence}")  # noqa: T201


async def a_tool_caught_in_flight() -> None:
    """Whether its effect landed cannot be known, so nobody says it was undone."""
    stream, token = dispatching()
    events = []
    async for event in stream:
        events.append(event)
        if event.kind == "tool_call_started":
            token.cancel("the client pressed stop")

    caught = [event for event in events if isinstance(event, ToolCallIndeterminate)]
    print("\na tool that was already running")  # noqa: T201
    print(f"  reported as: {caught[0].kind}")  # noqa: T201
    print(f"  detail:      {caught[0].detail}")  # noqa: T201


async def a_stop_that_lost_the_race() -> None:
    """The run's own record decides, so a client never sees two endings."""
    stream, token = dispatching()
    kinds = [event.kind async for event in stream]
    token.cancel("too late")

    print("\na stop that arrived after the run finished")  # noqa: T201
    print(f"  terminal event: {kinds[-1]}")  # noqa: T201
    print(f"  recorded state: {stream.run.state}")  # noqa: T201


async def stopping_over_a_transport() -> None:
    """A run id from a client is a claim; the broker checks it before acting on it."""
    broker: RunBroker[Any] = RunBroker()
    run_id = broker.register(dispatching()[0], tenant="acme")

    try:
        await broker.cancel(run_id, tenant="rival")
    except TransportAuthorizationError as refusal:
        print("\nstopping someone else's run")  # noqa: T201
        print(f"  refused: {refusal}")  # noqa: T201

    await broker.cancel(run_id, tenant="acme")
    await broker.cancel(run_id, tenant="acme")
    print(f"  the owner's stop, sent twice: {broker.run(run_id, tenant='acme').state}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await stopping_mid_run()
    await a_tool_caught_in_flight()
    await a_stop_that_lost_the_race()
    await stopping_over_a_transport()


if __name__ == "__main__":
    asyncio.run(main())
