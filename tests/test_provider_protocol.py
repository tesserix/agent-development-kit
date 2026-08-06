"""One provider protocol, typed, with the capability record as part of it.

A provider that is a duck is a provider whose limits are found out by exceeding them. The
protocol therefore carries `capabilities`, and the runner reads that record before it
sends anything: an agent asking for tool calling from a model that does not do tool
calling fails where it was wired, not on the call that needed it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    BinaryPart,
    Capability,
    CapabilityError,
    ContextWindowExceededError,
    Message,
    ModelCapabilities,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    ProtocolConformanceError,
    TextPart,
    members_of,
)
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import (
    CAPABLE,
    Cassette,
    FakeToolRegistry,
    ModelProviderConformance,
    ReplayingProvider,
    ScriptedProvider,
    StallingProvider,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

TALKS = ModelCapabilities(context_window_tokens=1_000)
CALLS_TOOLS = ModelCapabilities(tool_calling=True, context_window_tokens=1_000)


def agent(**overrides: Any) -> Agent[Any]:
    declared: dict[str, Any] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "model": "claude-sonnet-5",
        "free_text": True,
    }
    return Agent(**(declared | overrides))


def runner(provider: ScriptedProvider, **overrides: Any) -> AgentRunner:
    return AgentRunner(provider=provider, **overrides)


class TestTheProtocolStatesWhatAProviderMustOffer:
    def test_it_requires_a_capability_record(self) -> None:
        assert "capabilities" in members_of(ModelProvider)

    def test_it_requires_the_three_calls_a_provider_serves(self) -> None:
        assert {"complete", "stream", "count_tokens"} <= set(members_of(ModelProvider))

    def test_a_provider_without_a_capability_record_is_refused_at_wiring(self) -> None:
        """Absent is not "supports nothing": it is a provider nobody can check."""

        class Undeclared:
            name = "undeclared"

            async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002
                return ModelResponse()

            async def stream(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002
                return ModelResponse()

            def count_tokens(self, messages: Sequence[Message]) -> int:  # noqa: ARG002
                return 0

        with pytest.raises(ProtocolConformanceError, match="capabilities"):
            AgentRunner(provider=Undeclared())  # type: ignore[arg-type]


class TestACapabilityIsCheckedBeforeTheCall:
    def test_a_registry_wired_to_a_model_that_cannot_call_tools_fails_at_construction(
        self,
    ) -> None:
        """The wiring is wrong, and the wiring is what the caller can still change."""
        with pytest.raises(CapabilityError) as raised:
            runner(ScriptedProvider(capabilities=TALKS), tools=FakeToolRegistry())
        assert raised.value.capability == Capability.TOOL_CALLING
        assert raised.value.provider == "scripted"

    async def test_an_agent_naming_tools_fails_before_the_first_request(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="hi"), capabilities=TALKS)
        with pytest.raises(CapabilityError, match="tool_calling"):
            await runner(provider, tools=FakeToolRegistry({"lookup": lambda: "x"})).run(
                agent(tools=("lookup",)), "go", tenant="acme"
            )
        assert provider.requests == []

    async def test_an_image_sent_to_a_text_only_model_fails_before_the_first_request(
        self,
    ) -> None:
        """A model that cannot see does not say so; it answers about the text and stops."""
        provider = ScriptedProvider(ModelResponse(content="hi"), capabilities=TALKS)
        seen = Message(
            role="user",
            content=[
                BinaryPart(media_type="image/png", data=b"\x89PNG"),
                TextPart(text="what is it"),
            ],
        )
        with pytest.raises(CapabilityError, match="vision"):
            await runner(provider).run(agent(), "look", tenant="acme", history=(seen,))
        assert provider.requests == []

    async def test_a_prompt_past_the_declared_window_fails_before_the_first_request(
        self,
    ) -> None:
        """The vendor's answer to an over-long prompt is to truncate it and not mention it."""
        provider = ScriptedProvider(
            ModelResponse(content="hi"), capabilities=ModelCapabilities(context_window_tokens=8)
        )
        with pytest.raises(ContextWindowExceededError) as raised:
            await runner(provider).run(agent(), "a much longer question " * 20, tenant="acme")
        assert raised.value.limit == 8
        assert raised.value.counted > 8
        assert provider.requests == []

    async def test_a_window_nobody_declared_is_not_a_limit_to_check(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="hi"), capabilities=ModelCapabilities())
        run = await runner(provider).run(agent(), "go", tenant="acme")
        assert run.state.is_terminal

    async def test_a_model_that_can_call_tools_runs_normally(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="done"), capabilities=CALLS_TOOLS)
        run = await runner(provider, tools=FakeToolRegistry({"lookup": lambda: "x"})).run(
            agent(tools=("lookup",)), "go", tenant="acme"
        )
        assert run.state.is_terminal


class TestAPayloadThatIsNotAResponseIsRefused:
    async def test_it_raises_rather_than_carrying_the_shape_into_the_run(self) -> None:
        """A duck-typed provider returning a string is a wiring mistake, not an answer."""
        provider = ScriptedProvider("not a response", capabilities=TALKS)  # type: ignore[arg-type]
        with pytest.raises(ModelResponseError) as raised:
            await runner(provider).run(agent(), "go", tenant="acme")
        assert raised.value.payload == "not a response"
        assert raised.value.provider == "scripted"


class TestTheStructuredOutputPathReadsTheSameRecord:
    async def test_a_declared_capability_puts_the_schema_on_the_request(self) -> None:
        provider = ScriptedProvider(
            ModelResponse(content='{"nights": 4}'),
            capabilities=ModelCapabilities(structured_output=True, context_window_tokens=1_000),
        )
        await runner(provider).run(_typed_agent(), "go", tenant="acme")
        assert provider.requests[0].output_schema is not None

    async def test_an_undeclared_capability_falls_back_to_the_prompt(self) -> None:
        """The fallback is the point: a model without the feature still answers in shape."""
        provider = ScriptedProvider(ModelResponse(content='{"nights": 4}'), capabilities=TALKS)
        await runner(provider).run(_typed_agent(), "go", tenant="acme")
        assert provider.requests[0].output_schema is None


def _typed_agent() -> Agent[Any]:
    from pydantic import BaseModel

    class TripPlan(BaseModel):
        nights: int

    return agent(free_text=False, output_type=TripPlan)


class TestAProviderIsCheckedAgainstTheSuite(ModelProviderConformance):
    """The suite a third-party provider inherits, applied to the kit's own fake."""

    def make_provider(self) -> ScriptedProvider:
        return ScriptedProvider(*[ModelResponse(content="hi")] * 8, capabilities=CAPABLE)


class TestAProviderThatDoesNotStreamIsCheckedToo(ModelProviderConformance):
    """The same suite where streaming is undeclared, which is the case that must refuse."""

    def make_provider(self) -> ScriptedProvider:
        return ScriptedProvider(
            *[ModelResponse(content="hi")] * 8, capabilities=ModelCapabilities()
        )


class TestAStallingFakeIsCheckedToo(ModelProviderConformance):
    """The cancellation fake serves the same protocol, so it answers to the same suite."""

    def make_provider(self) -> StallingProvider:
        return StallingProvider(*[ModelResponse(content="hi")] * 8, capabilities=CAPABLE)


class TestAReplayedProviderCountsTokensToo:
    def test_it_estimates_since_the_tokeniser_is_not_on_the_cassette(self) -> None:
        provider = ReplayingProvider(Cassette(provider="scripted"))
        counted = provider.count_tokens([Message(role="user", content=[TextPart(text="a" * 40)])])
        assert counted > 0
