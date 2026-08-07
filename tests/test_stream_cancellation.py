"""Stopping a stream stops the work, and both sides agree on what happened.

A tab closed mid-answer used to stop nothing: the connection went away, the provider kept
being paid and dispatched tools ran to completion with nobody reading the result. So a stop
propagates in both directions — client into the run, run termination back out — and the
terminal event carries enough for a client to reconcile its view with the server's.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters import RunBroker, TransportAuthorizationError
from tesserix_adk.core import Agent, RunEventKind, RunState, ToolCall, Usage
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse
from tesserix_adk.runtime.progress import (
    ProgressEvent,
    RunCancelled,
    RunCompleted,
    ToolCallFailed,
    ToolCallIndeterminate,
)
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

LOOKUP = ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"})


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


def stopping(token: CancellationToken, reason: str) -> Any:
    """A tool that stops the run from underneath itself, as a client hitting stop does."""

    def tool(q: str) -> str:
        token.cancel(reason)
        return q

    return tool


class TestStoppingAStreamStopsTheWork:
    async def test_a_stop_mid_stream_ends_the_run_as_cancelled(self) -> None:
        token = CancellationToken()
        stream = runner(
            answer("", tool_calls=(LOOKUP,)),
            answer(),
            tools=FakeToolRegistry({"lookup": stopping(token, "the client pressed stop")}),
        ).stream(agent(tools=("lookup",)), "plan", tenant="acme", cancellation=token)
        events = [event async for event in stream]

        assert isinstance(events[-1], RunCancelled)
        assert stream.run.state is RunState.CANCELLED

    async def test_the_terminal_event_says_why_it_stopped(self) -> None:
        """A client told only that it stopped cannot tell a stop from a failure."""
        token = CancellationToken()
        stream = runner(
            answer("", tool_calls=(LOOKUP,)),
            answer(),
            tools=FakeToolRegistry({"lookup": stopping(token, "the client pressed stop")}),
        ).stream(agent(tools=("lookup",)), "plan", tenant="acme", cancellation=token)
        events = [event async for event in stream]

        terminal = events[-1]
        assert isinstance(terminal, RunCancelled)
        assert "stop" in terminal.reason

    async def test_the_terminal_event_carries_what_the_run_had_already_spent(self) -> None:
        """A run whose spend is knowable only on completion is unattributable when it stops."""
        token = CancellationToken()
        stream = runner(
            answer("", tool_calls=(LOOKUP,)),
            answer(),
            tools=FakeToolRegistry({"lookup": stopping(token, "stop")}),
        ).stream(agent(tools=("lookup",)), "plan", tenant="acme", cancellation=token)
        events = [event async for event in stream]

        terminal = events[-1]
        assert isinstance(terminal, RunCancelled)
        assert terminal.usage.input_tokens > 0
        assert terminal.usage == stream.run.usage

    async def test_the_terminal_event_names_the_last_event_before_it(self) -> None:
        """The client reconciles what it received against what the server sent."""
        token = CancellationToken()
        stream = runner(
            answer("", tool_calls=(LOOKUP,)),
            answer(),
            tools=FakeToolRegistry({"lookup": stopping(token, "stop")}),
        ).stream(agent(tools=("lookup",)), "plan", tenant="acme", cancellation=token)
        events = [event async for event in stream]

        terminal = events[-1]
        assert isinstance(terminal, RunCancelled)
        assert terminal.last_sequence == events[-2].sequence
        assert terminal.sequence == terminal.last_sequence + 1

    async def test_a_stop_before_anything_was_emitted_still_gives_a_whole_stream(self) -> None:
        """A stream that ends without a terminal event is one the client cannot close out."""
        token = CancellationToken()
        token.cancel("stopped before it started")
        stream = runner(answer()).stream(agent(), "plan", tenant="acme", cancellation=token)
        events = [event async for event in stream]

        assert isinstance(events[-1], RunCancelled)
        assert kinds(events).count("run_cancelled") == 1
        assert events[-1].last_sequence == len(events) - 2


class TestExactlyOneTerminalEvent:
    async def test_nothing_is_emitted_after_the_terminal_event(self) -> None:
        token = CancellationToken()
        stream = runner(
            answer("", tool_calls=(LOOKUP,)),
            answer(),
            tools=FakeToolRegistry({"lookup": stopping(token, "stop")}),
        ).stream(agent(tools=("lookup",)), "plan", tenant="acme", cancellation=token)
        events = [event async for event in stream]

        terminals = [event for event in events if event.kind.startswith("run_")]
        assert [event.kind for event in terminals] == ["run_started", "run_cancelled"]
        assert events[-1] is terminals[-1]

    async def test_a_late_event_from_the_runtime_never_reaches_the_consumer(self) -> None:
        """Teardown is what makes 'terminal is last' true, not the runtime's good manners."""
        stream = runner(answer()).stream(agent(), "plan", tenant="acme")
        events = [event async for event in stream]
        sink = stream._sink
        late = RunCompleted(state=RunState.COMPLETED, usage=Usage(input_tokens=0, output_tokens=0))
        sink.emit(late)

        assert await sink.next_event() is None
        assert kinds(events).count("run_completed") == 1

    async def test_a_stop_that_races_a_natural_completion_gives_one_outcome(self) -> None:
        """The client must never see both completed and cancelled for one run."""
        token = CancellationToken()
        stream = runner(answer()).stream(agent(), "plan", tenant="acme", cancellation=token)
        events: list[ProgressEvent] = []
        async for event in stream:
            events.append(event)
            token.cancel("stopped as it finished")

        ends = {"run_completed", "run_cancelled"}
        terminals = [event.kind for event in events if event.kind in ends]
        assert len(terminals) == 1
        assert stream.run.state.value == terminals[0].removeprefix("run_")

    async def test_the_record_the_run_reached_first_is_the_one_that_stands(self) -> None:
        """The rule: a stop after the loop recorded a state does not rewrite it."""
        token = CancellationToken()
        stream = runner(answer()).stream(agent(), "plan", tenant="acme", cancellation=token)
        run = await stream
        token.cancel("too late")

        assert run.state is RunState.COMPLETED
        assert stream.run.state is RunState.COMPLETED

    async def test_duplicate_stops_from_a_retrying_client_tear_down_once(self) -> None:
        token = CancellationToken()
        stream = runner(
            answer("", tool_calls=(LOOKUP,)),
            answer(),
            tools=FakeToolRegistry({"lookup": stopping(token, "the first reason")}),
        ).stream(agent(tools=("lookup",)), "plan", tenant="acme", cancellation=token)
        events = [event async for event in stream]
        token.cancel("a second, different reason")
        await stream.aclose()

        terminal = events[-1]
        assert isinstance(terminal, RunCancelled)
        assert "first" in terminal.reason
        assert kinds(events).count("run_cancelled") == 1


class TestToolsCaughtInFlight:
    async def test_a_tool_stopped_after_dispatch_is_reported_indeterminate(self) -> None:
        """Claiming a side effect was rolled back when nobody rolled it back is the worse answer."""
        events = await _stopped_mid_tool(agent(tools=("lookup",)))

        stopped = [event for event in events if isinstance(event, ToolCallIndeterminate)]
        assert [event.tool for event in stopped] == ["lookup"]
        assert stopped[0].call_id == "c1"
        assert "cannot be known" in stopped[0].detail

    async def test_the_stream_says_what_the_run_record_says(self) -> None:
        """Two accounts of one tool call is one account too many."""
        stream, token = _dispatching(agent(tools=("lookup",)))
        events = await _drained(stream, token)

        recorded = [
            event for event in stream.run.events if event.kind is RunEventKind.TOOL_INDETERMINATE
        ]
        assert len(recorded) == 1
        assert sum(isinstance(event, ToolCallIndeterminate) for event in events) == 1

    async def test_a_tool_the_agent_declared_idempotent_is_reported_retryable(self) -> None:
        """Indeterminacy is the default; a tool escapes it by declaring itself safe to retry."""
        events = await _stopped_mid_tool(agent(tools=("lookup",), idempotent_tools=("lookup",)))

        assert not any(isinstance(event, ToolCallIndeterminate) for event in events)
        failed = [event for event in events if isinstance(event, ToolCallFailed)]
        assert "safe to retry" in failed[0].detail


class TestStoppingOverATransport:
    async def test_a_client_stop_reaches_the_run(self) -> None:
        broker: RunBroker[Any] = RunBroker()
        run_id = broker.register(_dispatching(agent(tools=("lookup",)))[0], tenant="acme")
        events = [event async for event in _stopped(broker, run_id)]

        assert isinstance(events[-1], RunCancelled)
        assert broker.run(run_id, tenant="acme").state is RunState.CANCELLED

    async def test_a_stop_for_another_tenants_run_is_refused_and_not_applied(self) -> None:
        broker: RunBroker[Any] = RunBroker()
        run_id = broker.register(
            runner(answer()).stream(agent(), "plan", tenant="acme"), tenant="acme"
        )

        try:
            await broker.cancel(run_id, tenant="rival")
        except TransportAuthorizationError:
            pass
        else:  # pragma: no cover - the refusal is the assertion
            raise AssertionError("another tenant stopped a run it does not own")

        assert [event async for event in broker.subscribe(run_id, tenant="acme")]
        assert broker.run(run_id, tenant="acme").state is RunState.COMPLETED

    async def test_a_run_whose_client_is_gone_is_still_recorded_as_cancelled(self) -> None:
        """Attribution does not depend on the client being there to be told."""
        broker: RunBroker[Any] = RunBroker()
        run_id = broker.register(_dispatching(agent(tools=("lookup",)))[0], tenant="acme")
        listener = broker.subscribe(run_id, tenant="acme")
        await anext(listener)
        await listener.aclose()
        await broker.cancel(run_id, tenant="acme")

        assert broker.run(run_id, tenant="acme").state is RunState.CANCELLED

    async def test_duplicate_stops_over_the_transport_are_idempotent(self) -> None:
        broker: RunBroker[Any] = RunBroker()
        run_id = broker.register(_dispatching(agent(tools=("lookup",)))[0], tenant="acme")
        await broker.cancel(run_id, tenant="acme")
        await broker.cancel(run_id, tenant="acme")

        assert broker.run(run_id, tenant="acme").state is RunState.CANCELLED


async def _stopped(broker: RunBroker[Any], run_id: str) -> Any:
    """Read a subscription and stop the run once the first event has arrived."""
    async for event in broker.subscribe(run_id, tenant="acme"):
        yield event
        if event.kind == "run_started":
            await broker.cancel(run_id, tenant="acme")


def _dispatching(watched: Agent) -> tuple[Any, CancellationToken]:
    """A stream whose run is inside a tool call by the time the client changes its mind."""
    token = CancellationToken()
    stream = runner(
        answer("", tool_calls=(LOOKUP,)),
        answer(),
        tools=FakeToolRegistry({"lookup": _slow}),
    ).stream(watched, "plan", tenant="acme", cancellation=token)
    return stream, token


async def _drained(stream: Any, token: CancellationToken) -> list[ProgressEvent]:
    """Read a stream, stopping it the moment its tool has been dispatched."""
    events: list[ProgressEvent] = []
    async for event in stream:
        events.append(event)
        if event.kind == "tool_call_started":
            token.cancel("the client pressed stop")
    return events


async def _stopped_mid_tool(watched: Agent) -> list[ProgressEvent]:
    """Every event of a run stopped while one tool was in flight."""
    return await _drained(*_dispatching(watched))


async def _slow(q: str) -> str:
    """A tool long enough for a client to change its mind while it runs."""
    await asyncio.sleep(0.05)
    return q
