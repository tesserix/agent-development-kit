"""What happens to a run whose consumer cannot keep up.

An unbounded buffer turns one slow client into an outage for every run in the process, and
blocking the run on the client turns it into a run that costs more and answers later. So
the buffer is bounded, text deltas merge under pressure rather than vanish, everything that
makes a run attributable is kept whatever the pressure, and a consumer that has stopped
reading altogether stops the run rather than paying for it in silence.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tesserix_adk.core import Agent, RunState, ToolCall, Usage
from tesserix_adk.runtime import AgentRunner, Backpressure, ModelResponse
from tesserix_adk.runtime.progress import (
    AnswerDelta,
    ProgressEvent,
    RunCompleted,
    StructuredDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence


class TripPlan(BaseModel):
    destination: str
    nights: int


LONG = " ".join(f"word{n}" for n in range(200))


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str = "Kyoto, four nights.", **overrides: object) -> ModelResponse:
    fields: dict[str, object] = {
        "content": text,
        "usage": Usage(input_tokens=10, output_tokens=5),
    }
    return ModelResponse(**{**fields, **overrides})  # type: ignore[arg-type]


def runner(*responses: ModelResponse, **overrides: object) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


def kinds(events: Sequence[ProgressEvent]) -> list[str]:
    return [event.kind for event in events]


def texted(events: Sequence[ProgressEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, AnswerDelta))


class TestTheBufferIsBounded:
    async def test_a_slow_consumer_does_not_grow_the_buffer_without_limit(self) -> None:
        """The failure this exists to prevent is one client taking the process down."""
        stream = runner(answer(LONG)).stream(
            agent(), "plan", tenant="acme", backpressure=Backpressure(high_water=4)
        )
        events = [event async for event in stream]
        assert stream.pressure.peak <= 4
        assert kinds(events)[-1] == "run_completed"

    async def test_merged_deltas_still_reassemble_to_the_whole_answer(self) -> None:
        """Coalescing is not dropping: the answer a consumer renders is still the answer."""
        stream = runner(answer(LONG)).stream(
            agent(), "plan", tenant="acme", backpressure=Backpressure(high_water=4)
        )
        events = [event async for event in stream]
        assert texted(events) == LONG

    async def test_a_merged_delta_says_it_was_merged(self) -> None:
        """A consumer measuring the shape of a stream is told, rather than left to guess."""
        stream = runner(answer(LONG)).stream(
            agent(), "plan", tenant="acme", backpressure=Backpressure(high_water=4)
        )
        events = [event async for event in stream]
        merged = [event for event in events if isinstance(event, AnswerDelta) and event.coalesced]
        assert merged
        assert stream.pressure.coalesced == sum(event.coalesced for event in merged)

    async def test_nothing_that_makes_the_run_attributable_is_merged_away(self) -> None:
        """Lose a tool call or a usage update and the record of what happened is a guess."""
        stream = runner(
            answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"}),)),
            answer(LONG),
            tools=FakeToolRegistry({"lookup": lambda q: f"{q}: sunny"}),
        ).stream(
            agent(tools=("lookup",)),
            "plan",
            tenant="acme",
            backpressure=Backpressure(high_water=1),
        )
        events = [event async for event in stream]
        assert kinds(events).count("tool_call_started") == 1
        assert kinds(events).count("tool_call_finished") == 1
        assert kinds(events).count("usage_updated") >= 1
        assert kinds(events).count("run_completed") == 1

    async def test_the_terminal_event_arrives_however_full_the_buffer_was(self) -> None:
        """A run whose terminal event was dropped is a run nobody can account for."""
        stream = runner(answer(LONG)).stream(
            agent(), "plan", tenant="acme", backpressure=Backpressure(high_water=1)
        )
        events = [event async for event in stream]
        assert isinstance(events[-1], RunCompleted)
        assert stream.run.state is RunState.COMPLETED

    async def test_an_event_larger_than_the_whole_budget_is_admitted_and_counted(self) -> None:
        """Dropping it would lose a tool call; growing for it is the unbounded case."""
        stream = runner(
            answer(
                "",
                tool_calls=(ToolCall(id="c1", name="lookup", arguments={"q": "kyoto" * 5000}),),
            ),
            answer(LONG),
            tools=FakeToolRegistry({"lookup": lambda q: q[:20]}),
        ).stream(
            agent(tools=("lookup",)),
            "plan",
            tenant="acme",
            backpressure=Backpressure(high_water=64, byte_budget=2048),
        )
        events = [event async for event in stream]
        assert any(isinstance(event, ToolCallStarted) for event in events)
        assert any(isinstance(event, ToolCallFinished) for event in events)
        assert stream.pressure.oversize == 1


class TestStructuredAnswersUnderPressure:
    async def test_structured_fragments_merge_and_still_parse(self) -> None:
        """Half a JSON object is not a smaller one, so merging must keep every character."""
        payload = '{"destination": "' + "Kyoto " * 200 + '", "nights": 4}'
        stream = runner(answer(payload)).stream(
            agent(free_text=False, output_type=TripPlan),
            "plan",
            tenant="acme",
            backpressure=Backpressure(high_water=4),
        )
        events = [event async for event in stream]
        fragments = [event for event in events if isinstance(event, StructuredDelta)]
        assert "".join(event.fragment for event in fragments) == payload
        assert any(event.coalesced for event in fragments)
        assert stream.run.output is not None

    async def test_merging_everything_is_a_configuration_not_a_special_case(self) -> None:
        """A high-water mark of zero says merge from the first delta, and it holds."""
        stream = runner(answer(LONG)).stream(
            agent(), "plan", tenant="acme", backpressure=Backpressure(high_water=0)
        )
        events = [event async for event in stream]
        assert texted(events) == LONG
        assert len([event for event in events if isinstance(event, AnswerDelta)]) == 1


class TestNobodyReading:
    async def test_awaiting_without_reading_buffers_nothing(self) -> None:
        """Await-only is a supported pattern, not a slow consumer: there is no reader."""
        stream = runner(answer(LONG)).stream(agent(), "plan", tenant="acme")
        run = await stream
        assert run.state is RunState.COMPLETED
        assert stream.pressure.peak == 0

    async def test_a_reader_that_stops_reading_stops_the_run(self) -> None:
        """A dead client that never disconnects otherwise bills for a run nobody reads."""
        clock = FakeClock()
        stream = runner(
            answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"}),)),
            answer(LONG),
            clock=clock,
            tools=FakeToolRegistry({"lookup": lambda q: _after(clock, 120.0, q)}),
        ).stream(
            agent(tools=("lookup",)),
            "plan",
            tenant="acme",
            backpressure=Backpressure(high_water=2, stall_seconds=30.0),
        )
        async with aclosing(stream.__aiter__()) as events:
            await anext(events)
            run = await stream
        assert run.state is RunState.CANCELLED
        assert stream.pressure.stalled

    async def test_a_stalled_run_says_why_it_was_cancelled(self) -> None:
        clock = FakeClock()
        stream = runner(
            answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"}),)),
            answer(LONG),
            clock=clock,
            tools=FakeToolRegistry({"lookup": lambda q: _after(clock, 120.0, q)}),
        ).stream(
            agent(tools=("lookup",)),
            "plan",
            tenant="acme",
            backpressure=Backpressure(high_water=2, stall_seconds=30.0),
        )
        async with aclosing(stream.__aiter__()) as events:
            await anext(events)
            run = await stream
        detail = " ".join(event.detail or "" for event in run.events)
        assert "read" in detail

    async def test_a_consumer_that_comes_back_in_time_keeps_its_run(self) -> None:
        """Reading slowly is not the same as having gone away."""
        clock = FakeClock()
        stream = runner(
            answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"}),)),
            answer(LONG),
            clock=clock,
            tools=FakeToolRegistry({"lookup": lambda q: _after(clock, 20.0, q)}),
        ).stream(
            agent(tools=("lookup",)),
            "plan",
            tenant="acme",
            backpressure=Backpressure(high_water=2, stall_seconds=30.0),
        )
        events = [event async for event in stream]
        assert isinstance(events[-1], RunCompleted)
        assert not stream.pressure.stalled


class TestNothingBlocks:
    async def test_a_tool_whose_result_feeds_the_stalled_consumer_still_finishes(self) -> None:
        """A buffer that blocks the run deadlocks exactly the run it was protecting."""
        stream = runner(
            answer("", tool_calls=(ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"}),)),
            answer(LONG),
            clock=FakeClock(),
            tools=FakeToolRegistry({"lookup": lambda q: q * 200}),
        ).stream(
            agent(tools=("lookup",)),
            "plan",
            tenant="acme",
            backpressure=Backpressure(high_water=1, byte_budget=256),
        )
        events = [event async for event in stream]
        assert any(isinstance(event, ToolCallStarted) for event in events)
        assert isinstance(events[-1], RunCompleted)


class TestPressureAcrossManyStreams:
    def test_an_aggregate_allowance_divides_into_a_per_run_budget(self) -> None:
        """A per-run bound multiplied by hundreds of runs is not a bound on the process."""
        shared = Backpressure.shared(total_bytes=64 * 1024 * 1024, streams=200)
        assert shared.byte_budget * 200 <= 64 * 1024 * 1024

    async def test_the_default_is_documented_and_bounded(self) -> None:
        default = Backpressure()
        assert default.high_water > 0
        assert default.byte_budget > 0
        assert default.stall_seconds > 0


class TestWhatPressureReports:
    async def test_an_unpressured_run_reports_no_coalescing(self) -> None:
        stream = runner(answer()).stream(agent(), "plan", tenant="acme")
        events = [event async for event in stream]
        assert texted(events) == "Kyoto, four nights."
        assert stream.pressure.coalesced == 0
        assert stream.pressure.oversize == 0
        assert not stream.pressure.stalled

    async def test_occupancy_is_readable_while_the_run_is_still_going(self) -> None:
        """Pressure a consumer only learns about after the run is pressure it cannot act on."""
        stream = runner(answer(LONG)).stream(
            agent(), "plan", tenant="acme", backpressure=Backpressure(high_water=4)
        )
        seen: list[int] = []
        async for _ in stream:
            seen.append(stream.pressure.buffered)
        assert max(seen) <= 4
        assert stream.pressure.buffered == 0


def _after(clock: FakeClock, seconds: float, value: Any) -> Any:
    """A tool that takes `seconds` of the clock's time to answer."""
    clock.advance(seconds)
    return value
