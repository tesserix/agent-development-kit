"""What an agent may call, and how much it may spend calling it, is declared in one place.

The failures this file exists to prevent are a tool added for one agent becoming reachable
by every agent sharing the process, and a hanging call inside a tool stalling a whole run
until some outer request timeout notices. An allowlist resolved once at construction cannot
widen mid-run, and a tool that will not stop is abandoned rather than waited for.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    ConcurrencyConfig,
    ConfigurationError,
    ModelCapabilities,
    RetryConfig,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    ToolDefinitionError,
    ToolFailurePolicy,
    ToolNotFoundError,
    ToolNotPermittedError,
    ToolTimedOutError,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolCallSpan, ToolRegistry, tool

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)

if TYPE_CHECKING:
    from collections.abc import Callable

entered: list[str] = []


@tool
def look_up_fare(leg: str) -> str:
    """Price one leg.

    Args:
        leg: The hop to price.
    """
    entered.append(f"fare:{leg}")
    return f"{leg}: 40 EUR"


@tool
def look_up_hotel(city: str) -> str:
    """Find a room.

    Args:
        city: Where to stay.
    """
    entered.append(f"hotel:{city}")
    return f"{city}: Ryokan"


@tool
def refund_fare(booking: str) -> str:
    """Give a fare back.

    Args:
        booking: What to refund.
    """
    entered.append(f"refund:{booking}")
    return f"{booking}: refunded"


@pytest.fixture(autouse=True)
def _clear() -> None:
    entered.clear()


def registry_of(*, clock: FakeClock | None = None, **kwargs: Any) -> ToolRegistry:
    """A registry holding the three tools above, in that order."""
    return ToolRegistry(
        (look_up_fare, look_up_hotel, refund_fare), clock=clock or FakeClock(), **kwargs
    )


class TestWhatTheRegistryHoldsAndWhatItRefusesToHold:
    def test_declarations_come_back_in_registration_order_every_time(self) -> None:
        registry = registry_of()

        first = registry.declarations()
        second = registry.declarations()

        assert [declaration.name for declaration in first] == [
            "look_up_fare",
            "look_up_hotel",
            "refund_fare",
        ]
        assert first == second

    def test_a_declaration_carries_the_schema_the_model_is_shown(self) -> None:
        declaration = registry_of().declarations()[0]

        assert declaration.description == "Price one leg."
        assert declaration.parameters["required"] == ["leg"]

    def test_two_tools_answering_to_one_name_are_refused_with_both_origins(self) -> None:
        registry = registry_of()

        @tool(name="look_up_ferry")
        def crossings(leg: str) -> str:  # noqa: ARG001 — only its name matters here
            """Price a crossing."""
            return ""

        crossings.release()

        @tool(name="look_up_ferry")
        def also_crossings(leg: str) -> str:  # noqa: ARG001 — the same name from elsewhere
            """Price a crossing, differently."""
            return ""

        registry.register(crossings, origin="itinerary.tools")
        with pytest.raises(ToolDefinitionError) as conflict:
            registry.register(also_crossings, origin="baggage.tools")

        assert "look_up_ferry" in str(conflict.value)
        assert "itinerary.tools" in str(conflict.value)
        assert "baggage.tools" in str(conflict.value)
        also_crossings.release()

    def test_registering_the_same_tool_twice_is_not_a_conflict(self) -> None:
        registry = registry_of()

        registry.register(look_up_fare)

        assert len(registry.declarations()) == 3

    def test_a_name_nothing_is_registered_under_is_a_typed_refusal(self) -> None:
        with pytest.raises(ToolNotFoundError) as missing:
            registry_of().resolve("look_up_ferry")

        assert missing.value.tool == "look_up_ferry"

    async def test_calling_a_name_nothing_is_registered_under_runs_nothing(self) -> None:
        with pytest.raises(ToolNotFoundError):
            await registry_of().invoke("look_up_ferry", {"leg": "Osaka"})

        assert entered == []


class TestAnAgentsAllowlistIsResolvedOnceAndCannotWiden:
    def test_a_view_declares_only_what_its_agent_may_call(self) -> None:
        view = registry_of().view(allow=("look_up_fare", "look_up_hotel"), agent="planner")

        assert [declaration.name for declaration in view.declarations()] == [
            "look_up_fare",
            "look_up_hotel",
        ]
        assert view.names == ("look_up_fare", "look_up_hotel")

    async def test_a_tool_outside_the_allowlist_is_refused_without_running(self) -> None:
        view = registry_of().view(allow=("look_up_fare",), agent="planner")

        with pytest.raises(ToolNotPermittedError) as refused:
            await view.invoke("refund_fare", {"booking": "AB-1"})

        assert refused.value.tool == "refund_fare"
        assert refused.value.agent == "planner"
        assert entered == []

    async def test_a_tool_inside_the_allowlist_runs_and_returns_its_result(self) -> None:
        view = registry_of().view(allow=("look_up_fare",), agent="planner")

        assert await view.invoke("look_up_fare", {"leg": "Osaka"}) == "Osaka: 40 EUR"
        assert entered == ["fare:Osaka"]

    async def test_one_agents_refusal_says_nothing_about_another_agents_view(self) -> None:
        registry = registry_of()
        planner = registry.view(allow=("look_up_fare",), agent="planner")
        desk = registry.view(allow=("refund_fare",), agent="desk")

        with pytest.raises(ToolNotPermittedError):
            await planner.invoke("refund_fare", {"booking": "AB-1"})

        assert await desk.invoke("refund_fare", {"booking": "AB-1"}) == "AB-1: refunded"

    def test_a_tool_registered_after_a_view_was_built_is_not_in_it(self) -> None:
        registry = registry_of()
        view = registry.view(allow=("look_up_fare",), agent="planner")

        @tool
        def look_up_ferry(leg: str) -> str:  # noqa: ARG001 — registered after the fact
            """Price a crossing."""
            return ""

        registry.register(look_up_ferry)

        assert view.names == ("look_up_fare",)
        look_up_ferry.release()

    def test_an_allowlist_naming_a_tool_nobody_registered_fails_at_construction(self) -> None:
        with pytest.raises(ToolNotFoundError) as missing:
            registry_of().view(allow=("look_up_fare", "look_up_ferry"), agent="planner")

        assert missing.value.tool == "look_up_ferry"

    def test_an_empty_allowlist_fails_at_construction_rather_than_at_the_first_call(self) -> None:
        with pytest.raises(ConfigurationError) as empty:
            registry_of().view(allow=(), agent="planner")

        assert "planner" in str(empty.value)

    def test_a_view_names_each_tool_once_however_often_the_allowlist_does(self) -> None:
        view = registry_of().view(allow=("look_up_fare", "look_up_fare"), agent="planner")

        assert view.names == ("look_up_fare",)


class TestATimeoutIsTheCallsOwnCeilingAndItIsRealCancellation:
    async def test_a_tool_that_outruns_its_declared_ceiling_is_a_typed_failure(self) -> None:
        clock = FakeClock(auto_advance=False)
        stopped = asyncio.Event()

        @tool(timeout=5.0)
        async def stall() -> str:
            """Block until something stops it."""
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopped.set()
                raise
            return ""

        registry = ToolRegistry((stall,), clock=clock)
        call = asyncio.ensure_future(registry.invoke("stall", {}))
        await _settled(lambda: clock.slept == [5.0])
        clock.advance(5.0)

        with pytest.raises(ToolTimedOutError) as timed_out:
            await call

        assert timed_out.value.tool == "stall"
        assert timed_out.value.seconds == 5.0
        assert stopped.is_set()
        stall.release()

    async def test_the_registry_may_tighten_the_ceiling_a_tool_declared(self) -> None:
        clock = FakeClock(auto_advance=False)

        @tool(timeout=30.0)
        async def dawdle() -> str:
            """Block forever."""
            await asyncio.Event().wait()
            return ""

        registry = ToolRegistry((dawdle,), clock=clock, timeouts={"dawdle": 2.0})
        call = asyncio.ensure_future(registry.invoke("dawdle", {}))
        await _settled(lambda: clock.slept == [2.0])
        clock.advance(2.0)

        with pytest.raises(ToolTimedOutError) as timed_out:
            await call

        assert timed_out.value.seconds == 2.0
        dawdle.release()

    async def test_a_tool_that_finishes_inside_its_ceiling_returns_normally(self) -> None:
        clock = FakeClock(auto_advance=False)

        @tool(timeout=5.0)
        async def prompt_enough() -> str:
            """Answer without needing the ceiling."""
            return "in time"

        registry = ToolRegistry((prompt_enough,), clock=clock)
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)

        assert await registry.invoke("prompt_enough", {}) == "in time"
        assert spans[-1].outcome == "ok"
        assert spans[-1].abandoned is False
        prompt_enough.release()

    async def test_a_tool_that_stops_when_asked_is_not_recorded_as_abandoned(self) -> None:
        clock = FakeClock(auto_advance=False)

        @tool(timeout=1.0)
        async def obedient() -> str:
            """Stop when the caller says so."""
            await asyncio.Event().wait()
            return ""

        registry = ToolRegistry((obedient,), clock=clock)
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)
        call = asyncio.ensure_future(registry.invoke("obedient", {}))
        await _settled(lambda: clock.slept == [1.0])
        clock.advance(1.0)

        with pytest.raises(ToolTimedOutError):
            await call

        assert spans[-1].outcome == "timed_out"
        assert spans[-1].abandoned is False
        obedient.release()

    async def test_a_ceiling_of_no_time_at_all_is_refused_where_it_is_declared(self) -> None:
        with pytest.raises(ToolDefinitionError, match="no time at all"):

            @tool(timeout=0.0)
            def instant() -> str:
                """Never get the chance to run."""
                return ""

    async def test_a_tool_with_no_declared_ceiling_is_left_alone(self) -> None:
        clock = FakeClock(auto_advance=False)

        assert await registry_of(clock=clock).invoke("look_up_fare", {"leg": "Kyoto"})
        assert clock.slept == []

    async def test_a_tool_that_ignores_cancellation_is_abandoned_not_waited_for(self) -> None:
        clock = FakeClock(auto_advance=False)
        released = asyncio.Event()
        finished = asyncio.Event()

        @tool(timeout=1.0)
        async def deaf() -> str:
            """Keep going whatever the caller decided."""
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            await released.wait()
            finished.set()
            return "too late"

        registry = ToolRegistry((deaf,), clock=clock, abandon_after=4.0)
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)
        call = asyncio.ensure_future(registry.invoke("deaf", {}))
        await _settled(lambda: clock.slept == [1.0])
        clock.advance(1.0)
        await _settled(lambda: clock.slept == [1.0, 4.0])
        clock.advance(4.0)

        with pytest.raises(ToolTimedOutError):
            await call

        assert spans[-1].abandoned is True
        assert spans[-1].outcome == "timed_out"
        released.set()
        await _settled(finished.is_set)
        deaf.release()


class TestHowWideTheCallsMayRun:
    async def test_no_more_calls_are_in_flight_than_the_declared_width(self) -> None:
        registry, peak = _counting_registry(ConcurrencyConfig(max_concurrent_tools=2))

        await asyncio.gather(*(registry.invoke("counted", {}) for _ in range(6)))

        assert peak() == 2

    async def test_a_tools_own_width_is_tighter_than_the_registrys(self) -> None:
        registry, peak = _counting_registry(
            ConcurrencyConfig(max_concurrent_tools=8, per_tool={"counted": 1})
        )

        await asyncio.gather(*(registry.invoke("counted", {}) for _ in range(4)))

        assert peak() == 1

    async def test_a_tool_declaring_itself_order_dependent_never_overlaps_itself(self) -> None:
        live = [0]
        peak = [0]

        @tool(parallel_safe=False)
        async def one_at_a_time() -> str:
            """Refuse to overlap with itself."""
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            await asyncio.sleep(0)
            live[0] -= 1
            return "done"

        registry = ToolRegistry(
            (one_at_a_time,),
            clock=FakeClock(),
            concurrency=ConcurrencyConfig(max_concurrent_tools=8),
        )
        await asyncio.gather(*(registry.invoke("one_at_a_time", {}) for _ in range(4)))

        assert peak[0] == 1
        one_at_a_time.release()

    def test_a_registry_outliving_the_loop_it_was_first_used_on_still_bounds_it(self) -> None:
        registry, peak = _counting_registry(ConcurrencyConfig(max_concurrent_tools=1))

        async def twice() -> None:
            await asyncio.gather(*(registry.invoke("counted", {}) for _ in range(2)))

        asyncio.run(twice())
        asyncio.run(twice())

        assert peak() == 1


class TestWhatIsRecordedAboutACall:
    async def test_a_span_records_the_outcome_and_who_was_permitted_what(self) -> None:
        registry = registry_of()
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)
        view = registry.view(allow=("look_up_fare",), agent="planner")

        await view.invoke("look_up_fare", {"leg": "Osaka"})

        assert spans[-1].tool == "look_up_fare"
        assert spans[-1].agent == "planner"
        assert spans[-1].permitted is True
        assert spans[-1].outcome == "ok"

    async def test_a_refusal_is_recorded_as_one_rather_than_going_unattributed(self) -> None:
        registry = registry_of()
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)
        view = registry.view(allow=("look_up_fare",), agent="planner")

        with pytest.raises(ToolNotPermittedError):
            await view.invoke("refund_fare", {"booking": "AB-1"})

        assert spans[-1].outcome == "refused"
        assert spans[-1].permitted is False

    async def test_a_tool_that_raised_is_recorded_by_the_class_of_its_failure(self) -> None:
        @tool
        def unlucky() -> str:
            """Fail."""
            raise RuntimeError("the partner said no")

        registry = ToolRegistry((unlucky,), clock=FakeClock())
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)

        with pytest.raises(RuntimeError):
            await registry.invoke("unlucky", {})

        assert spans[-1].outcome == "error"
        assert spans[-1].failure == "RuntimeError"
        unlucky.release()

    async def test_a_span_carries_no_argument_and_no_result(self) -> None:
        registry = registry_of()
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)

        await registry.invoke("look_up_fare", {"leg": "Osaka"})

        assert "Osaka" not in repr(spans[-1])
        assert "40 EUR" not in repr(spans[-1])

    async def test_an_observer_that_raises_does_not_take_the_call_down_with_it(self) -> None:
        registry = registry_of()

        def unhelpful(span: ToolCallSpan) -> None:
            raise RuntimeError(span.tool)

        registry.observe(unhelpful)

        assert await registry.invoke("look_up_fare", {"leg": "Osaka"}) == "Osaka: 40 EUR"

    async def test_the_duration_recorded_is_the_clocks_and_not_the_wall(self) -> None:
        clock = FakeClock()

        @tool
        async def waits(seconds: float) -> str:
            """Take a measurable amount of time.

            Args:
                seconds: How long to take.
            """
            await clock.sleep(seconds)
            return "done"

        registry = ToolRegistry((waits,), clock=clock)
        spans: list[ToolCallSpan] = []
        registry.observe(spans.append)

        await registry.invoke("waits", {"seconds": 3.0})

        assert spans[-1].duration_seconds == 3.0
        waits.release()


class TestTheRegistryIsWhatTheRunLoopWasPromised:
    def test_a_registry_and_a_view_both_satisfy_the_protocol_the_runner_asks_for(self) -> None:
        from tesserix_adk.core import ToolRegistry as ToolRegistryProtocol

        view = registry_of().view(allow=("look_up_fare",), agent="planner")

        assert isinstance(view, ToolRegistryProtocol)
        assert isinstance(registry_of(), ToolRegistryProtocol)

    def test_a_view_is_frozen_so_nothing_widens_it_mid_run(self) -> None:
        view = registry_of().view(allow=("look_up_fare",), agent="planner")

        with pytest.raises((AttributeError, TypeError)):
            view.names = ("look_up_fare", "refund_fare")  # type: ignore[misc]

    def test_a_tool_declaring_itself_order_dependent_says_so_in_its_declaration(self) -> None:
        @tool(parallel_safe=False)
        def bookings() -> str:
            """Take a booking, which is not safe to do twice at once."""
            return ""

        registry = ToolRegistry((bookings,), clock=FakeClock())

        assert registry.declarations()[0].parallel_safe is False
        bookings.release()


class TestARunCallingThroughAView:
    async def test_a_permitted_call_reaches_the_tool_and_the_run_completes(self) -> None:
        run = await _run(_calling("look_up_fare", leg="Osaka"), _answer())

        assert run.state is RunState.COMPLETED
        assert entered == ["fare:Osaka"]

    async def test_a_call_outside_the_allowlist_is_a_tool_error_the_model_is_shown(self) -> None:
        run = await _run(
            _calling("refund_fare", booking="AB-1"),
            _answer(),
            agent_tools=("look_up_fare", "refund_fare"),
        )

        assert entered == []
        assert "allowlist" in "".join(
            record.detail or "" for record in run.events if record.kind is RunEventKind.TOOL_ERROR
        )

    async def test_a_refusal_fails_the_run_where_that_is_the_declared_policy(self) -> None:
        run = await _run(
            _calling("refund_fare", booking="AB-1"),
            _answer(),
            agent_tools=("look_up_fare", "refund_fare"),
            on_tool_error=ToolFailurePolicy.FAIL_RUN,
        )

        assert run.state is RunState.FAILED
        assert entered == []

    async def test_a_refusal_is_not_retried_even_where_the_tool_is_declared_idempotent(
        self,
    ) -> None:
        spans: list[ToolCallSpan] = []
        await _run(
            _calling("refund_fare", booking="AB-1"),
            _answer(),
            spans=spans,
            retry=RetryConfig(max_attempts=3, base_delay_seconds=0.0),
            agent_tools=("look_up_fare", "refund_fare"),
            idempotent_tools=("refund_fare",),
        )

        assert [span.outcome for span in spans] == ["refused"]

    async def test_the_model_is_told_about_the_permitted_tools_only(self) -> None:
        provider = ScriptedProvider(
            _calling("look_up_fare", leg="Osaka"), _answer(), capabilities=CAPABLE
        )
        await _run(provider=provider)

        assert [declaration.name for declaration in provider.requests[0].tools] == ["look_up_fare"]


async def _run(
    *responses: ModelResponse,
    provider: ScriptedProvider | None = None,
    spans: list[ToolCallSpan] | None = None,
    retry: RetryConfig | None = None,
    agent_tools: tuple[str, ...] = ("look_up_fare",),
    **overrides: object,
) -> Run[Any]:
    """One run whose view permits `look_up_fare` and nothing else."""
    registry = registry_of()
    if spans is not None:
        registry.observe(spans.append)
    runner = AgentRunner(
        provider=provider or ScriptedProvider(*responses, capabilities=CAPABLE),
        clock=FakeClock(),
        tools=registry.view(allow=("look_up_fare",), agent="planner"),
        retry=retry,
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Plan trips.",
        free_text=True,
        model="scripted-1",
        tools=agent_tools,
        **overrides,  # type: ignore[arg-type]
    )
    return await runner.run(agent, "price it", tenant="acme", run_id="run_1")


def _calling(name: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id="call_1", name=name, arguments=arguments),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _answer() -> ModelResponse:
    return ModelResponse(content="40 EUR.", usage=Usage(input_tokens=10, output_tokens=5))


async def _settled(until: Callable[[], bool], *, turns: int = 200) -> None:
    """Let the loop run until `until` holds, without sleeping on the wall clock."""
    for _ in range(turns):
        if until():
            return
        await asyncio.sleep(0)
    raise AssertionError("the loop never reached the state the test was waiting for")


def _counting_registry(config: ConcurrencyConfig) -> tuple[ToolRegistry, Callable[[], int]]:
    """A registry with one tool that records how many of its bodies overlap."""
    live = [0]
    peak = [0]

    @tool(name="counted")
    async def counted() -> str:
        """Record its own overlap with its siblings."""
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0)
        live[0] -= 1
        return "counted"

    registry = ToolRegistry((counted,), clock=FakeClock(), concurrency=config)
    counted.release()
    return registry, lambda: peak[0]
