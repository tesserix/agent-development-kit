"""One async core, a small sync surface over it, and no event-loop surprises.

Mixed paradigms fail in three recurring ways: a sync helper called inside a running loop
that deadlocks or nests a second loop, a tool that blocks the loop and slows every other
run sharing it, and identity that is dropped the moment work hops to a thread. Each has a
named, tested answer here rather than a convention nobody reads.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import signal
import threading
import time
import warnings
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from tesserix_adk.core import Agent, NoOutput, RunState, ToolCall, TypedAgent, Usage
from tesserix_adk.core.errors import EventLoopStalledError, RunningLoopError, WorkersBusyError
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse
from tesserix_adk.runtime.blocking import (
    Ambient,
    LoopMonitor,
    WorkerPool,
    Workers,
    carrying,
    current_ambient,
)
from tesserix_adk.testing import CAPABLE, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Callable

LOOKUP = ToolCall(id="c1", name="lookup", arguments={"q": "kyoto"})


class TripRequest(BaseModel):
    destination: str


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def calling(tool: Callable[..., Any], **overrides: object) -> AgentRunner:
    """A runner whose first turn calls `lookup` and whose second answers."""
    dispatch = ModelResponse(
        content="", tool_calls=(LOOKUP,), usage=Usage(input_tokens=8, output_tokens=2)
    )
    fields: dict[str, object] = {
        "provider": ScriptedProvider(dispatch, answer(), capabilities=CAPABLE),
        "clock": FakeClock(),
        "tools": FakeToolRegistry({"lookup": tool}),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


def plain() -> AgentRunner:
    return AgentRunner(provider=ScriptedProvider(answer(), capabilities=CAPABLE), clock=FakeClock())


class TestOneAgentFromEitherSurface:
    """The sync path is the async path driven differently, not a second implementation."""

    async def test_the_no_loop_script_and_the_async_service_agree(self) -> None:
        expected = await plain().run(agent(), "Where to?", tenant="acme", run_id="r1")
        produced = await asyncio.to_thread(
            lambda: plain().run_sync(agent(), "Where to?", tenant="acme", run_id="r1")
        )

        assert produced == expected

    async def test_a_tool_sees_the_tenant_from_either_surface(self) -> None:
        seen: list[Ambient | None] = []

        def note(q: str) -> str:
            seen.append(current_ambient())
            return q

        await calling(note).run(agent(tools=("lookup",)), "Where to?", tenant="acme", user="dana")
        await asyncio.to_thread(
            lambda: calling(note).run_sync(
                agent(tools=("lookup",)), "Where to?", tenant="acme", user="dana"
            )
        )

        assert [(a.tenant, a.user) for a in seen if a is not None] == [("acme", "dana")] * 2

    async def test_the_ambient_is_gone_once_the_run_is(self) -> None:
        await plain().run(agent(), "Where to?", tenant="acme")

        assert current_ambient() is None

    async def test_streaming_to_a_list_is_a_sync_helper_too(self) -> None:
        events = await asyncio.to_thread(
            lambda: plain().stream_sync(agent(), "Where to?", tenant="acme")
        )

        assert [event.kind for event in events][-1] == "run_completed"

    async def test_the_sync_wrapper_carries_the_run_budget(self) -> None:
        run = await asyncio.to_thread(
            lambda: plain().run_sync(agent(), "Where to?", tenant="acme", budget=None)
        )

        assert run.state is RunState.COMPLETED

    async def test_structured_input_uses_the_same_sync_and_stream_surfaces(self) -> None:
        typed: TypedAgent[TripRequest, NoOutput] = TypedAgent(
            name="typed-planner",
            instructions="Plan trips.",
            model="claude-sonnet-5",
            input_type=TripRequest,
            free_text=True,
        )
        request = TripRequest(destination="Kyoto")

        run = await asyncio.to_thread(lambda: plain().run_typed_sync(typed, request, tenant="acme"))
        events = await asyncio.to_thread(
            lambda: plain().stream_typed_sync(typed, request, tenant="acme")
        )

        assert run.state is RunState.COMPLETED
        assert events[-1].kind == "run_completed"


class TestCalledFromInsideARunningLoop:
    """A wrapper that deadlocks teaches nothing; one that names the alternative does."""

    async def test_run_sync_refuses_and_names_run(self) -> None:
        with pytest.raises(RunningLoopError) as raised:
            plain().run_sync(agent(), "Where to?", tenant="acme")

        assert "run_sync" in str(raised.value)
        assert "await AgentRunner.run" in str(raised.value)

    async def test_stream_sync_refuses_and_names_stream(self) -> None:
        with pytest.raises(RunningLoopError) as raised:
            plain().stream_sync(agent(), "Where to?", tenant="acme")

        assert "AgentRunner.stream" in str(raised.value)

    async def test_the_refusal_is_still_a_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            plain().run_sync(agent(), "Where to?", tenant="acme")

    async def test_the_documented_pattern_out_of_a_live_loop_is_a_thread(self) -> None:
        run = await asyncio.to_thread(lambda: plain().run_sync(agent(), "Where to?", tenant="acme"))

        assert run.state is RunState.COMPLETED

    async def test_a_refusal_leaves_no_coroutine_to_abandon(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(RunningLoopError):
                plain().run_sync(agent(), "Where to?", tenant="acme")
            gc.collect()

        assert [w for w in caught if "never awaited" in str(w.message)] == []


class TestAToolThatBlocksTheLoop:
    """A blocking body does not slow one run; it slows every run sharing the loop."""

    async def test_the_offending_tool_is_named(self) -> None:
        monitor = LoopMonitor(stall_seconds=0.01, interval=0.005)

        async def blocking() -> str:
            time.sleep(0.08)  # noqa: ASYNC251 — the stall being detected
            return "ok"

        with pytest.raises(EventLoopStalledError) as raised:
            await monitor.watching("tool lookup", blocking)

        assert "tool lookup" in str(raised.value)
        assert raised.value.tool == "tool lookup"

    async def test_a_tool_that_awaits_is_left_alone(self) -> None:
        monitor = LoopMonitor(stall_seconds=1.0, interval=0.005)

        async def polite() -> str:
            await asyncio.sleep(0.05)
            return "ok"

        assert await monitor.watching("tool lookup", polite) == "ok"

    async def test_the_tool_s_own_failure_is_not_relabelled(self) -> None:
        monitor = LoopMonitor(stall_seconds=0.01, interval=0.005)

        async def cross() -> str:
            time.sleep(0.05)  # noqa: ASYNC251 — stalls, then fails on its own terms
            raise ValueError("the tool's own complaint")

        with pytest.raises(ValueError, match="own complaint"):
            await monitor.watching("tool lookup", cross)

    async def test_a_run_reports_the_stall_against_the_tool(self) -> None:
        def blocking(q: str) -> str:
            time.sleep(0.08)
            return q

        runner = calling(blocking, monitor=LoopMonitor(stall_seconds=0.01, interval=0.005))
        run = await runner.run(agent(tools=("lookup",)), "Where to?", tenant="acme")

        stalls = [event for event in run.events if "stalled" in (event.detail or "")]
        assert [event.name for event in stalls] == ["lookup"]

    async def test_a_deployment_may_take_the_latency_instead(self) -> None:
        def blocking(q: str) -> str:
            time.sleep(0.05)
            return q

        run = await calling(blocking, monitor=None).run(
            agent(tools=("lookup",)), "Where to?", tenant="acme"
        )

        assert run.state is RunState.COMPLETED
        assert [event for event in run.events if "stalled" in (event.detail or "")] == []


class TestSyncBodiesOnABoundedPool:
    """Declaring a body sync buys a thread; it does not buy an unbounded number of them."""

    async def test_the_body_runs_off_the_loop(self) -> None:
        with WorkerPool(Workers(size=2)) as pool:
            elsewhere = await pool.call("lookup", threading.get_ident)

        assert elsewhere != threading.get_ident()

    async def test_the_loop_keeps_running_while_a_body_blocks(self) -> None:
        release = threading.Event()
        ticks = 0

        async def counting() -> None:
            nonlocal ticks
            while not release.is_set():
                ticks += 1
                await asyncio.sleep(0.001)

        with WorkerPool(Workers(size=1)) as pool:
            spinner = asyncio.create_task(counting())
            await pool.call("lookup", lambda: time.sleep(0.05))
            release.set()
            await spinner

        assert ticks > 1

    async def test_threads_are_capped_however_many_ask(self) -> None:
        with WorkerPool(Workers(size=2)) as pool:
            idents = await asyncio.gather(
                *(pool.call("lookup", threading.get_ident) for _ in range(8))
            )

        assert len(set(idents)) <= 2

    async def test_a_saturated_pool_refuses_rather_than_grows(self) -> None:
        release = threading.Event()
        with WorkerPool(Workers(size=1, queue_seconds=0.01)) as pool:
            held = asyncio.create_task(pool.call("slow", release.wait))
            await asyncio.sleep(0.02)
            with pytest.raises(WorkersBusyError, match="queued"):
                await pool.call("lookup", lambda: "never reached")
            release.set()
            await held

    async def test_the_ambient_crosses_the_hop(self) -> None:
        with carrying(Ambient(run_id="r1", tenant="acme", user="dana")), WorkerPool() as pool:
            seen = await pool.call("lookup", current_ambient)

        assert seen is not None
        assert seen.tenant == "acme"

    async def test_one_run_s_context_does_not_leak_into_the_next(self) -> None:
        stray: contextvars.ContextVar[str] = contextvars.ContextVar("stray")

        def first() -> None:
            stray.set("acme")

        with WorkerPool(Workers(size=1)) as pool:
            await pool.call("first", first)
            second = await pool.call("second", lambda: stray.get("unset"))
            tenants = []
            for tenant in ("acme", "globex"):
                with carrying(Ambient(run_id="r", tenant=tenant)):
                    tenants.append(await pool.call("third", lambda: current_ambient()))

        assert second == "unset"
        assert [a.tenant for a in tenants if a is not None] == ["acme", "globex"]

    async def test_a_body_may_run_its_own_event_loop(self) -> None:
        async def nested() -> str:
            await asyncio.sleep(0)
            return "nested"

        with WorkerPool() as pool:
            assert await pool.call("lookup", lambda: asyncio.run(nested())) == "nested"

    async def test_a_body_raising_is_the_caller_s_exception(self) -> None:
        def cross() -> str:
            raise ValueError("the tool's own complaint")

        with WorkerPool() as pool, pytest.raises(ValueError, match="own complaint"):
            await pool.call("lookup", cross)

    async def test_a_closed_pool_refuses_further_work(self) -> None:
        pool = WorkerPool()
        pool.close()

        with pytest.raises(WorkersBusyError, match="closed"):
            await pool.call("lookup", lambda: "never reached")

    def test_a_pool_sized_at_nothing_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="at least one worker"):
            Workers(size=0)

    def test_a_wait_that_cannot_be_waited_is_refused_too(self) -> None:
        with pytest.raises(ValueError, match="queue_seconds"):
            Workers(queue_seconds=-1.0)


class TestCancellationAcrossTheBoundary:
    """A thread cannot be interrupted, so a sync body is told rather than killed."""

    async def test_a_body_can_cooperate_with_the_token(self) -> None:
        token = CancellationToken()
        token.cancel("the caller went away")

        def obedient() -> str:
            ambient = current_ambient()
            assert ambient is not None
            ambient.raise_if_cancelled()
            return "never reached"

        with (
            carrying(Ambient(run_id="r1", tenant="acme", cancellation=token)),
            WorkerPool() as pool,
            pytest.raises(Exception, match="went away"),
        ):
            await pool.call("lookup", obedient)

    async def test_an_ambient_without_a_token_never_claims_cancellation(self) -> None:
        Ambient(run_id="r1", tenant="acme").raise_if_cancelled()

    def test_a_signal_handler_stops_a_sync_run(self) -> None:
        token = CancellationToken()

        async def slow(q: str) -> str:
            await asyncio.sleep(2)
            return q

        def stop(*_: object) -> None:
            token.cancel("interrupted at the terminal")

        previous = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, 0.05)
        try:
            run = calling(slow).run_sync(
                agent(tools=("lookup",)), "Where to?", tenant="acme", cancellation=token
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

        assert run.state is RunState.CANCELLED
