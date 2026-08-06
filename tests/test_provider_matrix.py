"""One agent, one answer type, three vendors, from committed recordings.

Each adapter passing its own suite says each translation is right. It does not say the
three are interchangeable, which is the whole claim a provider abstraction makes. This
runs the same agent over the same tool against all three and asserts the run came out the
same: same tool, same arguments, same typed answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from tesserix_adk.core import Agent
from tesserix_adk.models.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock, FakeSecrets, FakeToolRegistry, HttpCassette, HttpReplay

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import ModelProvider

CASSETTES = Path(__file__).parent / "cassettes"

MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
}
PROVIDERS: dict[str, type[AnthropicProvider] | type[OpenAIProvider] | type[GeminiProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class Weather(BaseModel):
    """What the agent is asked to answer with.

    Args:
        city: Where the reading is from.
        summary: The reading, in a few words.
    """

    city: str
    summary: str


def forecaster(vendor: str) -> Agent[Weather]:
    """The same agent every time but the model id, which every vendor spells its own way."""
    return Agent(
        name="forecaster",
        instructions="Answer with the weather.",
        model=MODELS[vendor],
        output_type=Weather,
        tools=("lookup",),
    )


def recorded(vendor: str) -> tuple[ModelProvider, HttpReplay]:
    replay = HttpReplay(
        HttpCassette.load(CASSETTES / f"{vendor}-weather.json"), expect_provider=vendor
    )
    provider = PROVIDERS[vendor](
        MODELS[vendor],
        secrets=FakeSecrets({KEYS[vendor]: "test-key"}),
        transport=replay.transport,
    )
    return provider, replay


def registry() -> FakeToolRegistry:
    return FakeToolRegistry({"lookup": lambda city: f"{city}: clear"})


@pytest.mark.parametrize("vendor", sorted(MODELS))
class TestTheSameRunOnEveryVendor:
    async def test_the_agent_answers_in_its_declared_type(self, vendor: str) -> None:
        provider, _ = recorded(vendor)
        tools = registry()
        finished = await AgentRunner(provider=provider, tools=tools, clock=FakeClock()).run(
            forecaster(vendor), "what is the weather in Delhi", tenant="acme"
        )
        assert finished.output == Weather(city="Delhi", summary="clear")

    async def test_the_tool_ran_with_the_arguments_the_model_chose(self, vendor: str) -> None:
        """Three wire formats, and one of them mints the id the other two send."""
        provider, _ = recorded(vendor)
        tools = registry()
        await AgentRunner(provider=provider, tools=tools, clock=FakeClock()).run(
            forecaster(vendor), "what is the weather in Delhi", tenant="acme"
        )
        assert tools.calls == [("lookup", {"city": "Delhi"})]

    async def test_the_whole_recording_was_used(self, vendor: str) -> None:
        """A run that stops early leaves an exchange behind, and still asserts green."""
        provider, replay = recorded(vendor)
        await AgentRunner(provider=provider, tools=registry(), clock=FakeClock()).run(
            forecaster(vendor), "what is the weather in Delhi", tenant="acme"
        )
        assert replay.remaining == 0

    async def test_the_tool_result_was_sent_back(self, vendor: str) -> None:
        """Each vendor spells a result differently, and none of them may drop it."""
        provider, replay = recorded(vendor)
        await AgentRunner(provider=provider, tools=registry(), clock=FakeClock()).run(
            forecaster(vendor), "what is the weather in Delhi", tenant="acme"
        )
        assert "Delhi: clear" in str(replay.sent[1].body)

    async def test_spend_is_reported_in_the_same_units(self, vendor: str) -> None:
        """A usage total that only some vendors fill in is a budget nothing can enforce."""
        provider, _ = recorded(vendor)
        finished = await AgentRunner(provider=provider, tools=registry(), clock=FakeClock()).run(
            forecaster(vendor), "what is the weather in Delhi", tenant="acme"
        )
        assert finished.usage.input_tokens > 0
        assert finished.usage.output_tokens > 0
