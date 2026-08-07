"""Consuming a stream: what is provisional, what is final, and what happens on the way out.

A partially-streamed itinerary looks exactly like an itinerary, so a consumer holding one
mid-stream cannot tell whether acting on it is safe. Here it can: partial structured
content is a `Provisional`, which the type checker refuses everywhere the declared output
type is required, and the run's own result is only ever the validated one.

`assert_type` states what the checker must infer and is a no-op at runtime; a
`type: ignore[code]` on a line of deliberate misuse states that the checker must reject it,
and `warn_unused_ignores` fails the build if it stops doing so.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, assert_type

import pytest
from pydantic import BaseModel

from tesserix_adk.core import Agent, NoOutput, Run, RunState, StreamInterruptedError, Usage
from tesserix_adk.runtime import (
    AgentRunner,
    AnswerDelta,
    CancellationToken,
    ModelResponse,
    Provisional,
    RunStream,
)
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider, StallingProvider

if TYPE_CHECKING:
    from tesserix_adk.runtime import ProgressEvent

NATIVE = CAPABLE.declaring(structured_output=True)


class TripPlan(BaseModel):
    """A trip the model proposes.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


def planner() -> Agent[TripPlan]:
    return Agent(
        name="planner", instructions="Plan trips.", model="claude-sonnet-5", output_type=TripPlan
    )


def chatter() -> Agent[NoOutput]:
    return Agent(name="chatter", instructions="Chat.", model="claude-sonnet-5", free_text=True)


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def runner(*responses: ModelResponse | BaseException, **overrides: object) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=NATIVE),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


def stream_of(
    *responses: ModelResponse | BaseException, **overrides: object
) -> RunStream[NoOutput]:
    return runner(*responses, **overrides).stream(
        chatter(), "plan a trip", tenant="acme", run_id="run_1"
    )


class TestAwaitingTheResult:
    async def test_awaiting_without_iterating_still_drives_the_run(self) -> None:
        """The await-only pattern: a caller that wants the answer and no progress."""
        run = await stream_of(answer())
        assert run.state is RunState.COMPLETED

    async def test_iterate_then_await_gives_the_same_run_the_iteration_produced(self) -> None:
        stream = stream_of(answer())
        events = [event async for event in stream]
        assert await stream is stream.run
        assert events[-1].kind == "run_completed"

    async def test_two_consumers_awaiting_one_stream_do_not_run_it_twice(self) -> None:
        provider = ScriptedProvider(answer(), capabilities=NATIVE)
        stream = runner(provider=provider).stream(
            chatter(), "plan a trip", tenant="acme", run_id="run_1"
        )
        first, second = await asyncio.gather(_awaited(stream), _awaited(stream))
        assert first is second
        assert len(provider.requests) == 1

    async def test_the_awaited_run_is_typed_by_the_agent(self) -> None:
        stream = runner(ModelResponse(content='{"destination": "Kyoto", "nights": 4}')).stream(
            planner(), "plan a trip", tenant="acme"
        )
        run = await stream
        assert_type(run, Run[TripPlan])
        assert run.output == TripPlan(destination="Kyoto", nights=4)


class TestProvisionalIsNotFinal:
    def test_a_provisional_cannot_be_used_where_the_output_type_is_required(self) -> None:
        """The distinction is enforced by the checker, not by a naming convention."""
        provisional: Provisional[TripPlan] = Provisional(text='{"destination": "Kyoto"}')
        booked: TripPlan = provisional  # type: ignore[assignment]
        assert isinstance(booked, Provisional)

    def test_a_provisional_that_would_validate_is_still_not_the_result(self) -> None:
        """Syntactically complete JSON arrives before the run is over. It is still partial."""
        provisional: Provisional[TripPlan] = Provisional(
            text='{"destination": "Kyoto", "nights": 4}'
        )
        assert provisional.snapshot() == {"destination": "Kyoto", "nights": 4}
        assert not isinstance(provisional.snapshot(), TripPlan)

    def test_a_half_arrived_object_reads_as_nothing_rather_than_as_a_guess(self) -> None:
        assert Provisional(text='{"destination": "Kyo').snapshot() is None

    def test_a_bare_json_value_is_not_an_object_and_reads_as_nothing(self) -> None:
        """`4` parses. It is not the declared output type, so it is not a snapshot of one."""
        assert Provisional(text="4").snapshot() is None

    def test_the_stream_exposes_what_has_arrived_as_provisional(self) -> None:
        assert stream_of(answer()).provisional.text == ""

    async def test_partial_structured_output_reaches_the_consumer_as_provisional(self) -> None:
        stream = runner(ModelResponse(content='{"destination": "Kyoto", "nights": 4}')).stream(
            planner(), "plan a trip", tenant="acme"
        )
        seen: list[str] = []
        async for event in stream:
            seen.append(stream.provisional.text)
            assert_type(stream.provisional, Provisional[TripPlan])
            del event
        assert seen[-1] == '{"destination": "Kyoto", "nights": 4}'
        assert stream.run.output == TripPlan(destination="Kyoto", nights=4)


class TestLeavingEarly:
    async def test_a_consumer_that_stops_reading_cancels_the_run(self) -> None:
        """An abandoned run that keeps calling a provider is a bill nobody is waiting for."""
        provider = StallingProvider(capabilities=NATIVE)
        stream = runner(provider=provider).stream(chatter(), "plan a trip", tenant="acme")
        async with stream:
            async for _ in stream:
                break
        assert stream.run.state is RunState.CANCELLED

    async def test_an_exception_in_the_loop_body_still_ends_the_run(self) -> None:
        provider = StallingProvider(capabilities=NATIVE)
        stream = runner(provider=provider).stream(chatter(), "plan a trip", tenant="acme")

        async def read_and_fail() -> None:
            async with stream:
                async for _ in stream:
                    _ = 1 / 0

        with pytest.raises(ZeroDivisionError):
            await read_and_fail()
        assert stream.run.state is RunState.CANCELLED

    async def test_leaving_a_finished_stream_changes_nothing(self) -> None:
        stream = stream_of(answer())
        async with stream:
            events = [event async for event in stream]
        assert stream.run.state is RunState.COMPLETED
        assert events[-1].kind == "run_completed"

    async def test_leaving_a_stream_that_was_never_read_starts_nothing(self) -> None:
        """Nothing runs until the stream is read, so there is nothing to cancel."""
        provider = ScriptedProvider(answer(), capabilities=NATIVE)
        stream = runner(provider=provider).stream(chatter(), "plan a trip", tenant="acme")
        async with stream:
            pass
        assert not provider.requests

    async def test_awaiting_an_abandoned_stream_refuses_to_promote_what_arrived(self) -> None:
        """Accumulated partial content returned as a result is a wrong answer that looks right."""
        provider = StallingProvider(capabilities=NATIVE)
        stream = runner(provider=provider).stream(chatter(), "plan a trip", tenant="acme")
        async with stream:
            async for _ in stream:
                break
        with pytest.raises(StreamInterruptedError) as raised:
            await stream
        assert "cancelled" in str(raised.value)

    async def test_the_caller_s_own_cancellation_still_reaches_the_run(self) -> None:
        provider = StallingProvider(capabilities=NATIVE)
        token = CancellationToken()
        stream = runner(provider=provider).stream(
            chatter(), "plan a trip", tenant="acme", cancellation=token
        )
        async with stream:
            events: list[ProgressEvent] = []
            async for event in stream:
                events.append(event)
                token.cancel("caller went away")
        assert stream.run.state is RunState.CANCELLED
        assert stream.run.events[-1].detail is not None


class TestWhatIsStillTrueOfTheEvents:
    async def test_deltas_reassemble_to_the_answer_under_the_context_manager(self) -> None:
        async with stream_of(answer()) as stream:
            events = [event async for event in stream]
        streamed = "".join(e.text for e in events if isinstance(e, AnswerDelta))
        assert streamed == "Kyoto, four nights."


async def _awaited(stream: RunStream[NoOutput]) -> Run[NoOutput]:
    return await stream
