"""A turn that asked for four tools should not cost four round trips.

Firing them all at once is the obvious fix and the obvious way to turn one agent turn into
a rate-limit breach at a partner. So the batch runs concurrently but inside declared lanes,
resolves in call order however it finishes, and reports each failure against the call that
caused it rather than folding the whole batch into one.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import Agent, RunState, ToolCall, Usage
from tesserix_adk.core.agent import ToolFailurePolicy
from tesserix_adk.core.config import ConcurrencyConfig
from tesserix_adk.core.errors import ToolTimedOutError
from tesserix_adk.core.provider import ToolDeclaration
from tesserix_adk.core.run import RunEventKind
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse
from tesserix_adk.runtime.fanout import Lanes, phased
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tesserix_adk.core.primitives import Message
    from tesserix_adk.core.provider import ModelRequest
    from tesserix_adk.core.run import Run

GUARD = 10.0
"""Seconds a rendezvous waits before failing, so a serial regression fails rather than hangs."""


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": ("alpha", "beta", "gamma", "delta"),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def batch(*names: str) -> ModelResponse:
    """One model response asking for each named tool, in the order given."""
    return ModelResponse(
        content="",
        tool_calls=tuple(
            ToolCall(id=f"c{position}", name=name, arguments={"q": name})
            for position, name in enumerate(names, start=1)
        ),
        usage=Usage(input_tokens=8, output_tokens=2),
    )


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


class EveryRunAlike(ScriptedProvider):
    """A provider that answers each run the same way, so two runs may share one runner."""

    def __init__(self, first: ModelResponse, then: ModelResponse) -> None:
        super().__init__(capabilities=CAPABLE)
        self._first = first
        self._then = then

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Ask for the tools on the first turn of any run, and answer once they are back."""
        self.requests.append(request)
        answered = any(message.role == "tool" for message in request.messages)
        return self._then if answered else self._first


def fanning(
    response: ModelResponse, tools: dict[str, Callable[..., Any]], **overrides: object
) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(response, answer(), capabilities=CAPABLE),
        "clock": FakeClock(),
        "tools": FakeToolRegistry(tools),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


class Overlap:
    """How many tool bodies were in flight at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def entered(self) -> None:
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)

    def left(self) -> None:
        with self._lock:
            self.live -= 1


def meeting(overlap: Overlap, rendezvous: asyncio.Barrier, name: str) -> Callable[..., Any]:
    """A tool that can only return if `rendezvous.parties` of them are running together."""

    async def body(**_: object) -> str:
        overlap.entered()
        try:
            await asyncio.wait_for(rendezvous.wait(), GUARD)
        finally:
            overlap.left()
        return f"{name} done"

    return body


def tool_messages(run: Run) -> list[Message]:
    return [message for message in run.messages if message.role == "tool"]


def said(message: Message) -> str:
    return "".join(part.text for part in message.content if hasattr(part, "text"))


def details(run: Run, kind: RunEventKind) -> list[str]:
    return [event.detail or "" for event in run.events if event.kind is kind]


class TestNestedRuns:
    """A tool that runs a sub-agent spends the lane it is already standing in."""

    async def test_a_sub_agent_does_not_queue_behind_its_own_parent(self) -> None:
        started: list[AgentRunner] = []
        depth = 0

        async def nested(**_: object) -> str:
            nonlocal depth
            if depth:
                return "the leaf"
            depth += 1
            run = await started[0].run(agent(), "look it up", tenant="acme")
            return f"nested: {run.state.value}"

        runner = AgentRunner(
            provider=EveryRunAlike(batch("alpha"), answer()),
            clock=FakeClock(),
            tools=FakeToolRegistry({"alpha": nested}),
            concurrency=ConcurrencyConfig(per_tenant=1, per_tool={"alpha": 1}),
        )
        started.append(runner)
        run = await asyncio.wait_for(runner.run(agent(), "plan a trip", tenant="acme"), GUARD)

        assert run.state is RunState.COMPLETED
        assert "nested: completed" in said(tool_messages(run)[-1])


class TestSyncTools:
    """A blocking tool body is the registry's to offload, and siblings keep moving."""

    async def test_an_offloaded_sync_tool_does_not_hold_up_its_siblings(self) -> None:
        overlap = Overlap()
        rendezvous = asyncio.Barrier(2)

        def blocking(**_: object) -> str:
            overlap.entered()
            try:
                asyncio.run_coroutine_threadsafe(rendezvous.wait(), loop).result(GUARD)
            finally:
                overlap.left()
            return "alpha done"

        async def offloaded(**arguments: object) -> str:
            return await asyncio.to_thread(blocking, **arguments)

        loop = asyncio.get_running_loop()
        runner = fanning(
            batch("alpha", "beta"),
            {"alpha": offloaded, "beta": meeting(overlap, rendezvous, "beta")},
        )
        run = await asyncio.wait_for(runner.run(agent(), "plan a trip", tenant="acme"), GUARD)

        assert run.state is RunState.COMPLETED
        assert overlap.peak == 2


class TestOneTurnManyTools:
    """Four independent lookups are one bounded batch, not four round trips."""

    async def test_the_batch_is_in_flight_together(self) -> None:
        overlap = Overlap()
        rendezvous = asyncio.Barrier(4)
        names = ("alpha", "beta", "gamma", "delta")
        runner = fanning(
            batch(*names), {name: meeting(overlap, rendezvous, name) for name in names}
        )

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert overlap.peak == 4

    async def test_results_are_ordered_by_the_call_not_by_the_finish(self) -> None:
        names = ("alpha", "beta", "gamma", "delta")
        started = [asyncio.Event() for _ in names]
        finished = [asyncio.Event() for _ in names]
        release = asyncio.Event()

        def body(position: int) -> Callable[..., Any]:
            async def call(**_: object) -> str:
                started[position].set()
                waiting = release if position == len(names) - 1 else finished[position + 1]
                await asyncio.wait_for(waiting.wait(), GUARD)
                finished[position].set()
                return f"result {position}"

            return call

        async def let_the_last_one_go() -> None:
            for event in started:
                await asyncio.wait_for(event.wait(), GUARD)
            release.set()

        runner = fanning(
            batch(*names), {name: body(position) for position, name in enumerate(names)}
        )
        opener = asyncio.ensure_future(let_the_last_one_go())
        run = await runner.run(agent(), "plan a trip", tenant="acme")
        await opener

        assert [message.tool_call_id for message in tool_messages(run)] == ["c1", "c2", "c3", "c4"]
        assert [
            f"result {position}" in said(message)
            for position, message in enumerate(tool_messages(run))
        ] == [True, True, True, True]

    async def test_a_single_call_turn_still_works(self) -> None:
        runner = fanning(batch("alpha"), {"alpha": lambda **_: "one"})

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert "one" in said(tool_messages(run)[0])


class TestLanes:
    """Concurrency is declared per run, per tool and per tenant, and the tightest one wins."""

    async def test_the_per_run_cap_bounds_the_batch(self) -> None:
        overlap = Overlap()
        rendezvous = asyncio.Barrier(2)
        names = ("alpha", "beta", "gamma", "delta")
        runner = fanning(
            batch(*names),
            {name: meeting(overlap, rendezvous, name) for name in names},
            concurrency=ConcurrencyConfig(max_concurrent_tools=2),
        )

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert overlap.peak == 2

    async def test_a_per_tool_cap_bounds_that_tool_alone(self) -> None:
        alpha, others = Overlap(), Overlap()
        alphas = asyncio.Barrier(1)
        rendezvous = asyncio.Barrier(2)
        calls = ModelResponse(
            content="",
            tool_calls=(
                ToolCall(id="c1", name="alpha", arguments={"q": "one"}),
                ToolCall(id="c2", name="alpha", arguments={"q": "two"}),
                ToolCall(id="c3", name="beta", arguments={}),
                ToolCall(id="c4", name="gamma", arguments={}),
            ),
            usage=Usage(input_tokens=8, output_tokens=2),
        )
        runner = fanning(
            calls,
            {
                "alpha": meeting(alpha, alphas, "alpha"),
                "beta": meeting(others, rendezvous, "beta"),
                "gamma": meeting(others, rendezvous, "gamma"),
            },
            concurrency=ConcurrencyConfig(per_tool={"alpha": 1}),
        )

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert alpha.peak == 1
        assert others.peak == 2

    async def test_a_tenant_cap_is_shared_across_that_tenant_s_runs(self) -> None:
        overlap = Overlap()
        turns = 0

        async def body(**_: object) -> str:
            nonlocal turns
            overlap.entered()
            for _turn in range(4):
                turns += 1
                await asyncio.sleep(0)
            overlap.left()
            return "done"

        runner = AgentRunner(
            provider=EveryRunAlike(batch("alpha"), answer()),
            clock=FakeClock(),
            tools=FakeToolRegistry({"alpha": body}),
            concurrency=ConcurrencyConfig(per_tenant=1),
        )

        first = asyncio.ensure_future(runner.run(agent(), "one", tenant="acme"))
        second = asyncio.ensure_future(runner.run(agent(), "two", tenant="acme"))
        await asyncio.gather(first, second)

        assert overlap.peak == 1
        assert turns == 8

    async def test_two_tenants_do_not_queue_behind_each_other(self) -> None:
        overlap = Overlap()
        rendezvous = asyncio.Barrier(2)
        runner = AgentRunner(
            provider=EveryRunAlike(batch("alpha"), answer()),
            clock=FakeClock(),
            tools=FakeToolRegistry({"alpha": meeting(overlap, rendezvous, "alpha")}),
            concurrency=ConcurrencyConfig(per_tenant=1),
        )

        await asyncio.gather(
            runner.run(agent(), "one", tenant="acme"),
            runner.run(agent(), "two", tenant="globex"),
        )

        assert overlap.peak == 2

    def test_an_agent_narrows_the_runner_s_lanes_and_cannot_widen_them(self) -> None:
        runner_wide = ConcurrencyConfig(max_concurrent_tools=4, per_tool={"alpha": 3})
        agent_narrow = ConcurrencyConfig(max_concurrent_tools=8, per_tool={"alpha": 1, "beta": 2})

        composed = runner_wide.narrowed_to(agent_narrow)

        assert composed.max_concurrent_tools == 4
        assert composed.per_tool == {"alpha": 1, "beta": 2}

    def test_narrowing_to_nothing_leaves_the_lanes_alone(self) -> None:
        lanes = ConcurrencyConfig(max_concurrent_tools=3)

        assert lanes.narrowed_to(None) == lanes

    def test_a_lane_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ConcurrencyConfig(max_concurrent_tools=0)

        with pytest.raises(ValueError, match="at least one"):
            ConcurrencyConfig(per_tool={"alpha": 0})

        with pytest.raises(ValueError, match="at least one"):
            ConcurrencyConfig(per_tenant=0)

    def test_a_per_tool_ceiling_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ConcurrencyConfig(per_tool_seconds={"alpha": 0.0})

    async def test_lanes_are_acquired_in_one_order_whichever_tool_asks(self) -> None:
        """Deadlock between two lanes is avoided by ordering, not by luck."""
        lanes = Lanes(ConcurrencyConfig(max_concurrent_tools=1, per_tenant=1))

        async with lanes.held("alpha", tenant="acme"):
            pass
        async with lanes.held("beta", tenant="acme"):
            pass


class TestPerToolTimeouts:
    """A slow tool spends its own ceiling, not the batch's."""

    async def test_the_slow_tool_times_out_and_its_siblings_are_kept(self) -> None:
        clock = FakeClock(auto_advance=False)

        async def stall(**_: object) -> str:
            await asyncio.Event().wait()
            return "never"

        runner = fanning(
            batch("alpha", "beta"),
            {"alpha": stall, "beta": lambda **_: "beta done"},
            clock=clock,
            concurrency=ConcurrencyConfig(per_tool_seconds={"alpha": 5.0}),
        )
        running = asyncio.ensure_future(runner.run(agent(), "plan a trip", tenant="acme"))
        await clock.wait_for_sleep(1)
        clock.advance(5)
        run = await running

        assert run.state is RunState.COMPLETED
        assert "5" in said(tool_messages(run)[0])
        assert "beta done" in said(tool_messages(run)[1])

    async def test_a_tool_that_answers_inside_its_ceiling_is_left_alone(self) -> None:
        runner = fanning(
            batch("alpha"),
            {"alpha": lambda **_: "alpha done"},
            concurrency=ConcurrencyConfig(per_tool_seconds={"alpha": 5.0}),
        )
        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert "alpha done" in said(tool_messages(run)[0])

    async def test_the_timeout_is_the_tool_s_own_error(self) -> None:
        failure = ToolTimedOutError("alpha", 5.0)

        assert failure.tool == "alpha"
        assert failure.seconds == pytest.approx(5.0)
        assert "5" in str(failure)


class TestPartialFailure:
    """Two succeed, one times out, one raises — and the model sees exactly which."""

    async def test_each_failure_is_reported_against_its_own_call(self) -> None:
        clock = FakeClock(auto_advance=False)

        async def stall(**_: object) -> str:
            await asyncio.Event().wait()
            return "never"

        def boom(**_: object) -> str:
            raise ValueError("upstream said no")

        runner = fanning(
            batch("alpha", "beta", "gamma", "delta"),
            {
                "alpha": lambda **_: "alpha done",
                "beta": stall,
                "gamma": boom,
                "delta": lambda **_: "delta done",
            },
            clock=clock,
            concurrency=ConcurrencyConfig(per_tool_seconds={"beta": 5.0}),
        )
        running = asyncio.ensure_future(runner.run(agent(), "plan a trip", tenant="acme"))
        await clock.wait_for_sleep(1)
        clock.advance(5)
        run = await running

        spoken = {message.tool_call_id: said(message) for message in tool_messages(run)}
        assert list(spoken) == ["c1", "c2", "c3", "c4"]
        assert "alpha done" in spoken["c1"]
        assert "delta done" in spoken["c4"]
        assert "beta" in spoken["c2"]
        assert "upstream said no" in spoken["c3"]

    async def test_no_placeholder_stands_in_for_a_failed_call(self) -> None:
        def boom(**_: object) -> str:
            raise ValueError("upstream said no")

        runner = fanning(batch("alpha", "beta"), {"alpha": lambda **_: "alpha done", "beta": boom})

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        spoken = {message.tool_call_id: said(message) for message in tool_messages(run)}
        assert "alpha done" not in spoken["c2"]
        assert "None" not in spoken["c2"]
        assert spoken["c2"].count("error") >= 1

    async def test_a_batch_aborts_where_the_agent_declared_it_should(self) -> None:
        entered = asyncio.Event()

        def boom(**_: object) -> str:
            raise ValueError("upstream said no")

        async def stall(**_: object) -> str:
            entered.set()
            await asyncio.Event().wait()
            return "never"

        runner = fanning(batch("alpha", "beta"), {"alpha": boom, "beta": stall})

        run = await runner.run(
            agent(on_tool_error=ToolFailurePolicy.FAIL_RUN), "plan a trip", tenant="acme"
        )

        assert run.state is RunState.FAILED
        assert any("upstream said no" in detail for detail in details(run, RunEventKind.TOOL_ERROR))


class TestCancellingTheBatch:
    """Queued and in-flight are terminated together and recorded apart."""

    async def test_what_was_running_and_what_was_queued_are_recorded_distinctly(self) -> None:
        token = CancellationToken()
        running = asyncio.Event()

        async def stall(**_: object) -> str:
            running.set()
            await asyncio.Event().wait()
            return "never"

        async def queued(**_: object) -> str:
            return "should not run"

        runner = fanning(
            batch("alpha", "beta"),
            {"alpha": stall, "beta": queued},
            concurrency=ConcurrencyConfig(max_concurrent_tools=1),
        )
        task = asyncio.ensure_future(
            runner.run(agent(), "plan a trip", tenant="acme", cancellation=token)
        )
        await asyncio.wait_for(running.wait(), GUARD)
        token.cancel("caller went away")
        run = await task

        assert run.state is RunState.CANCELLED
        assert any(
            "cannot be known" in detail for detail in details(run, RunEventKind.TOOL_ERROR)
        ) or [event for event in run.events if event.kind is RunEventKind.TOOL_INDETERMINATE]
        assert any("never dispatched" in detail for detail in details(run, RunEventKind.TOOL_ERROR))


class TestToolsThatCannotBeParallelised:
    """A tool that declares itself order-dependent is run alone."""

    async def test_a_serial_tool_never_overlaps_a_sibling(self) -> None:
        overlap = Overlap()
        witnessed: dict[str, int] = {}

        def body(name: str) -> Callable[..., Any]:
            async def call(**_: object) -> str:
                overlap.entered()
                for _turn in range(3):
                    witnessed[name] = max(witnessed.get(name, 0), overlap.live)
                    await asyncio.sleep(0)
                overlap.left()
                return "done"

            return call

        names = ("alpha", "beta", "gamma", "delta")
        runner = AgentRunner(
            provider=ScriptedProvider(batch(*names), answer(), capabilities=CAPABLE),
            clock=FakeClock(),
            tools=FakeToolRegistry(
                {name: body(name) for name in names},
                declarations={"beta": ToolDeclaration(name="beta", parallel_safe=False)},
            ),
        )

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert witnessed["beta"] == 1
        assert witnessed["delta"] == 2

    def test_a_declaration_is_parallel_safe_unless_it_says_otherwise(self) -> None:
        assert ToolDeclaration(name="alpha").parallel_safe is True

    def test_phases_keep_call_order_and_isolate_the_serial_ones(self) -> None:
        calls: Sequence[ToolCall] = [
            ToolCall(id=f"c{position}", name=name, arguments={})
            for position, name in enumerate(("alpha", "beta", "gamma", "delta"), start=1)
        ]

        grouped = phased(calls, serial={"beta"})

        assert [[call.name for call in phase] for phase in grouped] == [
            ["alpha"],
            ["beta"],
            ["gamma", "delta"],
        ]

    def test_an_empty_turn_has_no_phases(self) -> None:
        assert phased([], serial=frozenset()) == ()

    def test_serial_calls_back_to_back_each_get_their_own_phase(self) -> None:
        calls: Sequence[ToolCall] = [
            ToolCall(id=f"c{position}", name=name, arguments={})
            for position, name in enumerate(("alpha", "beta", "gamma"), start=1)
        ]

        grouped = phased(calls, serial={"alpha", "beta"})

        assert [[call.name for call in phase] for phase in grouped] == [
            ["alpha"],
            ["beta"],
            ["gamma"],
        ]

    async def test_a_phase_behind_a_failed_one_is_never_dispatched(self) -> None:
        def boom(**_: object) -> str:
            raise RuntimeError("the index is down")

        runner = AgentRunner(
            provider=ScriptedProvider(batch("alpha", "beta"), capabilities=CAPABLE),
            clock=FakeClock(),
            tools=FakeToolRegistry(
                {"alpha": boom, "beta": lambda **_: "beta done"},
                declarations={"beta": ToolDeclaration(name="beta", parallel_safe=False)},
            ),
        )
        run = await runner.run(
            agent(on_tool_error=ToolFailurePolicy.FAIL_RUN), "plan a trip", tenant="acme"
        )

        assert run.state is RunState.FAILED
        assert "never dispatched" in details(run, RunEventKind.TOOL_ERROR)[-1]
