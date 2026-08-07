"""Run an agent end to end — tool call, structured answer, full record — with no network.

The provider is scripted and the tool is a plain function, so the whole loop is exercised
without credentials. Swapping in a real provider changes nothing else here.

Run it with `python examples/run_loop.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from pydantic import BaseModel

from tesserix_adk.core import Agent, Cost, RunEventKind, ToolCall, Usage
from tesserix_adk.runtime import AgentRunner, ModelResponse, ToolDeclaration
from tesserix_adk.testing import FakeToolRegistry, ScriptedProvider


class TripPlan(BaseModel):
    """The shape the answer must take. Anything else fails the run."""

    destination: str
    nights: int


def timetable(origin: str, destination: str) -> dict[str, object]:
    """A tool. Ordinary function, ordinary return value."""
    return {"origin": origin, "destination": destination, "trains": 4}


def spent_on(run: object) -> str:
    """What the run cost, or an honest word where nothing priced it."""
    cost = run.usage.cost  # type: ignore[attr-defined]
    return "an unknown amount" if cost is None else f"{cost.total} {cost.currency}"


def main() -> None:
    """Plan a trip: one tool call, one structured answer, one complete record."""
    agent = Agent(
        name="planner",
        version="1.2.0",
        instructions="Plan trips. Cite the timetable before recommending a leg.",
        model="claude-sonnet-5",
        tools=("timetable",),
        output_type=TripPlan,
    )

    # The model asks for the tool, then answers with it. Real providers do the same.
    provider = ScriptedProvider(
        ModelResponse(
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="timetable",
                    arguments={"origin": "Osaka", "destination": "Kyoto"},
                ),
            ),
            usage=Usage(input_tokens=420, output_tokens=18, cost=Cost(input=Decimal("0.004"))),
        ),
        ModelResponse(
            content='{"destination": "Kyoto", "nights": 4}',
            usage=Usage(input_tokens=610, output_tokens=24, cost=Cost(input=Decimal("0.006"))),
        ),
    )
    tools = FakeToolRegistry(
        {"timetable": timetable},
        {
            "timetable": ToolDeclaration(
                name="timetable",
                description="Trains between two stations.",
                parameters={
                    "type": "object",
                    "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
                },
            )
        },
    )

    runner = AgentRunner(provider=provider, tools=tools)
    run = asyncio.run(
        runner.run(agent, "Four nights near Kyoto, arriving from Osaka.", tenant="acme", user="ada")
    )

    print(f"state:   {run.state}")  # noqa: T201
    print(f"answer:  {run.output}")  # noqa: T201
    print(f"prompt:  {run.agent_name} {run.agent_version} @ {run.prompt_version}")  # noqa: T201
    spent = run.usage.input_tokens + run.usage.output_tokens
    print(f"spent:   {spent} tokens, {spent_on(run)}")  # noqa: T201

    print("\nwhat happened:")  # noqa: T201
    for event in run.events:
        detail = f" — {event.name}" if event.name else ""
        print(f"  {event.kind}{detail}")  # noqa: T201

    # The tool result came back as data, not as prose the model could take orders from.
    result = next(message for message in run.messages if message.role == "tool")
    print(f"\ntool result reached the model wrapped: {RunEventKind.TOOL_RESULT} ->")  # noqa: T201
    print("  " + result.content[0].text.replace("\n", "\n  "))  # type: ignore[union-attr]  # noqa: T201


if __name__ == "__main__":
    main()
