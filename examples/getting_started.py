"""Run a typed, tool-using agent with a readable trace and no network."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from tesserix_adk import Agent, AgentRunner, ToolRegistry, tool
from tesserix_adk.core import BudgetLimits
from tesserix_adk.testing import FakeModelProvider, ScriptedTurn


class PackingTip(BaseModel):
    """A packing suggestion validated before it reaches the application."""

    suggestion: str


@tool(idempotency="read_only")
def current_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"{city} is 21°C and clear"


async def main() -> None:
    """Run the same declaration and registry used with a real provider."""
    agent: Agent[PackingTip] = Agent(
        name="weather-agent",
        instructions="Use current_weather, then return one packing suggestion.",
        model="demo-model",
        output_type=PackingTip,
        tools=("current_weather",),
        idempotent_tools=("current_weather",),
        budget=BudgetLimits(max_model_calls=2, max_tool_calls=1),
    )
    provider = FakeModelProvider(
        ScriptedTurn.calling("current_weather", {"city": "Melbourne"}),
        ScriptedTurn.returning({"suggestion": "Pack a light jacket."}),
    )
    stream = AgentRunner(provider=provider, tools=ToolRegistry((current_weather,))).stream(
        agent, "What should I pack for Melbourne?", tenant="demo", user="local-user"
    )
    async for event in stream:
        print(f"trace: {event.sequence} {event.kind}")  # noqa: T201
    print(await stream)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
