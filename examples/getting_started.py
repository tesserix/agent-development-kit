"""Create and run one custom agent without network access or credentials.

The scripted provider keeps the first run deterministic. Replace it with any provider
from ``tesserix_adk.models.providers`` when connecting a real model.
"""

from __future__ import annotations

import asyncio

from tesserix_adk import Agent, AgentRunner, ToolRegistry, __version__, tool
from tesserix_adk.testing import FakeModelProvider, ScriptedTurn


@tool(idempotency="read_only")
def current_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: City whose weather should be returned.
    """
    result = f"{city} is 21°C and clear"
    print(f"tool: {result}")  # noqa: T201 — visible proof that the agent called its tool
    return result


async def main() -> None:
    """Run the same declaration and registry used with a real model provider."""
    agent = Agent(
        name="weather-agent",
        instructions="Use current_weather, then give one concise packing suggestion.",
        model="demo-model",
        free_text=True,
        tools=("current_weather",),
        idempotent_tools=("current_weather",),
    )
    provider = FakeModelProvider(
        ScriptedTurn.calling("current_weather", {"city": "Melbourne"}),
        ScriptedTurn.saying("Pack a light jacket."),
    )
    runner = AgentRunner(
        provider=provider,
        tools=ToolRegistry((current_weather,)),
    )

    run = await runner.run(agent, "What should I pack for Melbourne?", tenant="demo")

    print(f"tesserix-adk {__version__}")  # noqa: T201
    print(f"agent: {run.agent_name}")  # noqa: T201
    print(f"answer: {run.text}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
