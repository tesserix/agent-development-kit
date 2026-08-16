"""A model provider a test can script, with no vendor SDK and no network."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import Cost, Message, StopReason, StreamEnd, TextDelta, TextPart
from tesserix_adk.models import ModelCapabilities, ModelRequest
from tesserix_adk.testing import (
    FakeModelProvider,
    Fault,
    ScriptedTurn,
    ScriptExhaustedError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.anyio


def _request(text: str = "refund the charge", model: str = "fake-1") -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=(Message(role="user", content=[TextPart(text=text)]),),
    )


def _asked() -> ScriptedTurn:
    return ScriptedTurn.calling("lookup_charge", {"id": "ch_1"}, input_tokens=40, output_tokens=8)


def _answered() -> ScriptedTurn:
    return ScriptedTurn.saying("refunded", input_tokens=60, output_tokens=12)


class TestReplayingAScript:
    async def test_turns_come_back_in_the_order_they_were_scripted(self) -> None:
        provider = FakeModelProvider(_asked(), _answered())
        assert (await provider.complete(_request())).tool_calls[0].name == "lookup_charge"
        assert (await provider.complete(_request())).content == "refunded"

    async def test_a_tool_call_carries_the_arguments_it_was_scripted_with(self) -> None:
        provider = FakeModelProvider(_asked())
        call = (await provider.complete(_request())).tool_calls[0]
        assert call.arguments == {"id": "ch_1"}

    async def test_a_tool_call_turn_stops_for_the_tool(self) -> None:
        provider = FakeModelProvider(_asked())
        assert (await provider.complete(_request())).stop_reason is StopReason.TOOL_CALLS

    async def test_a_structured_payload_comes_back_as_the_content(self) -> None:
        turn = ScriptedTurn.returning({"status": "refunded", "amount": 12})
        provider = FakeModelProvider(turn)
        assert '"status"' in (await provider.complete(_request())).content

    async def test_a_finish_reason_can_be_scripted(self) -> None:
        turn = ScriptedTurn.saying("as far as I got", stop_reason=StopReason.MAX_TOKENS)
        provider = FakeModelProvider(turn)
        assert (await provider.complete(_request())).stop_reason is StopReason.MAX_TOKENS


class TestExactAccounting:
    async def test_the_scripted_token_counts_are_what_comes_back(self) -> None:
        """A budget assertion against an approximation is an assertion that flakes."""
        provider = FakeModelProvider(_answered())
        usage = (await provider.complete(_request())).usage
        assert (usage.input_tokens, usage.output_tokens) == (60, 12)

    async def test_a_scripted_cost_is_returned_exactly(self) -> None:
        turn = ScriptedTurn.saying("done", cost=Cost(input=Decimal("0.25")))
        provider = FakeModelProvider(turn)
        usage = (await provider.complete(_request())).usage
        assert usage.cost is not None
        assert usage.cost.total == Decimal("0.25")

    async def test_counting_tokens_is_deterministic(self) -> None:
        provider = FakeModelProvider(_answered())
        messages = (Message(role="user", content=[TextPart(text="a" * 40)]),)
        assert provider.count_tokens(messages) == provider.count_tokens(messages)


class TestCapabilities:
    async def test_a_fake_declares_what_a_test_needs_it_to_declare(self) -> None:
        provider = FakeModelProvider(capabilities=ModelCapabilities(structured_output=True))
        assert provider.capabilities.structured_output

    async def test_a_fake_can_declare_a_capability_it_lacks_so_the_error_path_is_testable(
        self,
    ) -> None:
        provider = FakeModelProvider(capabilities=ModelCapabilities(tool_calling=False))
        assert not provider.capabilities.tool_calling

    async def test_the_provider_name_is_recorded_on_the_run(self) -> None:
        assert FakeModelProvider(name="pretend").name == "pretend"


class TestInjectedFaults:
    async def test_a_timeout_can_be_scripted(self) -> None:
        provider = FakeModelProvider(ScriptedTurn.failing(Fault.TIMEOUT))
        with pytest.raises(Exception, match="timed out"):
            await provider.complete(_request())

    async def test_a_rate_limit_can_be_scripted(self) -> None:
        provider = FakeModelProvider(ScriptedTurn.failing(Fault.RATE_LIMIT))
        with pytest.raises(Exception, match="rate"):
            await provider.complete(_request())

    async def test_a_transport_error_can_be_scripted(self) -> None:
        provider = FakeModelProvider(ScriptedTurn.failing(Fault.TRANSPORT))
        with pytest.raises(Exception, match=r"transport|connection"):
            await provider.complete(_request())

    async def test_a_malformed_body_can_be_scripted(self) -> None:
        provider = FakeModelProvider(ScriptedTurn.failing(Fault.MALFORMED))
        with pytest.raises(Exception, match=r"malformed|decode"):
            await provider.complete(_request())

    async def test_a_raised_fault_carries_the_raw_payload_for_the_report(self) -> None:
        provider = FakeModelProvider(ScriptedTurn.failing(Fault.MALFORMED, payload="<html>502"))
        with pytest.raises(Exception, match="502") as raised:
            await provider.complete(_request())
        assert "502" in str(raised.value)

    async def test_a_schema_violating_payload_is_returned_rather_than_raised(self) -> None:
        """The runtime, not the fake, decides what an invalid payload means."""
        provider = FakeModelProvider(ScriptedTurn.returning({"wrong": True}))
        assert "wrong" in (await provider.complete(_request())).content

    async def test_a_fault_then_a_good_turn_exercises_a_retry(self) -> None:
        provider = FakeModelProvider(ScriptedTurn.failing(Fault.TIMEOUT), _answered())
        with pytest.raises(Exception, match="timed out"):
            await provider.complete(_request())
        assert (await provider.complete(_request())).content == "refunded"


class TestARunawayLoop:
    async def test_an_unscripted_call_raises_rather_than_inventing_a_reply(self) -> None:
        provider = FakeModelProvider(_answered())
        await provider.complete(_request())
        with pytest.raises(ScriptExhaustedError):
            await provider.complete(_request())

    async def test_the_error_says_how_many_calls_were_made(self) -> None:
        provider = FakeModelProvider()
        with pytest.raises(ScriptExhaustedError, match="1"):
            await provider.complete(_request())

    async def test_a_lenient_fake_answers_instead_of_raising(self) -> None:
        provider = FakeModelProvider(strict=False)
        assert (await provider.complete(_request())).stop_reason is StopReason.END_TURN

    async def test_turns_the_run_never_reached_can_be_asserted_on(self) -> None:
        """More script than the run consumed is a run that stopped early."""
        provider = FakeModelProvider(_asked(), _answered())
        await provider.complete(_request())
        assert provider.remaining == 1


class TestTheCallLog:
    async def test_the_prompts_the_runtime_assembled_are_recorded(self) -> None:
        provider = FakeModelProvider(_answered())
        await provider.complete(_request("refund it"))
        assert "refund it" in str(provider.requests[0].messages)

    async def test_the_log_counts_the_calls_made(self) -> None:
        provider = FakeModelProvider(_asked(), _answered())
        await provider.complete(_request())
        await provider.complete(_request())
        assert provider.calls == 2

    async def test_the_model_asked_for_is_recorded(self) -> None:
        provider = FakeModelProvider(_answered())
        await provider.complete(_request(model="fake-9"))
        assert provider.requests[0].model == "fake-9"

    async def test_a_test_can_assert_no_further_call_was_made(self) -> None:
        provider = FakeModelProvider(ScriptedTurn.failing(Fault.TIMEOUT))
        with pytest.raises(Exception, match="timed out"):
            await provider.complete(_request())
        assert provider.calls == 1


class TestStreaming:
    async def test_a_scripted_turn_streams_and_ends(self) -> None:
        provider = FakeModelProvider(_answered())
        events = [event async for event in await provider.stream(_request())]
        assert isinstance(events[-1], StreamEnd)

    async def test_a_consumer_that_stops_part_way_does_not_break_the_fake(self) -> None:
        provider = FakeModelProvider(_answered(), _answered())
        stream = await provider.stream(_request())
        assert isinstance(await anext(stream), TextDelta)
        await stream.aclose()  # type: ignore[attr-defined]
        assert (await provider.complete(_request())).content == "refunded"


class TestTheFixtures:
    async def test_the_fixture_hands_over_a_fake_a_test_can_script(
        self, fake_model: FakeModelProvider
    ) -> None:
        fake_model.script(_answered())
        assert (await fake_model.complete(_request())).content == "refunded"

    async def test_the_fixture_is_strict_by_default(self, fake_model: FakeModelProvider) -> None:
        with pytest.raises(ScriptExhaustedError):
            await fake_model.complete(_request())

    async def test_the_factory_builds_a_fake_per_run(
        self, fake_model_factory: Callable[..., FakeModelProvider]
    ) -> None:
        first, second = fake_model_factory(_answered()), fake_model_factory(_answered())
        await first.complete(_request())
        assert (first.calls, second.calls) == (1, 0)


class TestConcurrentRuns:
    async def test_two_runs_can_hold_their_own_scripts(self) -> None:
        """One shared fake whose scripts interleave makes both tests lie."""
        factory = FakeModelProvider.factory(_asked(), _answered())
        first, second = factory(), factory()
        assert (await first.complete(_request())).tool_calls
        assert (await second.complete(_request())).tool_calls

    async def test_a_factory_gives_each_run_its_own_call_log(self) -> None:
        factory = FakeModelProvider.factory(_answered(), _answered())
        first, second = factory(), factory()
        await first.complete(_request())
        assert (first.calls, second.calls) == (1, 0)

    async def test_one_fake_serialises_concurrent_callers_rather_than_interleaving(self) -> None:
        provider = FakeModelProvider(_asked(), _answered())
        results = await asyncio.gather(provider.complete(_request()), provider.complete(_request()))
        assert {result.stop_reason for result in results} == {
            StopReason.TOOL_CALLS,
            StopReason.END_TURN,
        }
