"""Cancellation and deadlines: work that nobody is waiting for has to actually stop.

The failure this file exists to prevent is the expensive one: a caller abandons a
request, the HTTP handler returns, and the loop keeps calling a provider against a socket
nobody is reading. A cancelled run must stop issuing model calls, come back inside its
grace window, and still say what it spent and how far it got.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    Agent,
    CancelledError,
    DeadlineConfig,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, CancellationToken, Deadline, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, StallingProvider

if TYPE_CHECKING:
    from collections.abc import Callable


def tool_that(effect: Callable[[], object], returns: object) -> Callable[[], object]:
    """A tool that changes the world before it answers, which is where indeterminacy lives."""

    def run() -> object:
        effect()
        return returns

    return run


class StallingGuardrail:
    """A guardrail that never reaches a verdict, so a test can cancel one mid-check."""

    def __init__(self, name: str = "toxicity") -> None:
        self._name = name
        self.entered = asyncio.Event()

    @property
    def name(self) -> str:
        return self._name

    async def check(self, subject: object) -> bool:  # noqa: ARG002 — it never gets that far
        self.entered.set()
        await asyncio.Event().wait()
        return True


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def calling(tool: str = "timetable") -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name=tool, arguments={}),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def kinds(run: Run) -> list[RunEventKind]:
    return [event.kind for event in run.events]


class TestTheToken:
    def test_a_fresh_token_is_not_cancelled(self) -> None:
        assert CancellationToken().cancelled is False

    def test_cancelling_records_the_reason(self) -> None:
        token = CancellationToken()
        token.cancel("caller went away")
        assert token.cancelled is True
        assert token.reason == "caller went away"

    def test_the_first_reason_is_the_one_that_stands(self) -> None:
        """Two cancellations are one cancellation; the second must not rewrite the why."""
        token = CancellationToken()
        token.cancel("caller went away")
        token.cancel("deadline elapsed")
        assert token.reason == "caller went away"

    def test_raising_is_a_no_op_until_cancelled(self) -> None:
        CancellationToken().raise_if_cancelled()

    def test_raising_after_cancellation_carries_the_reason(self) -> None:
        token = CancellationToken()
        token.cancel("caller went away")
        with pytest.raises(CancelledError, match="caller went away"):
            token.raise_if_cancelled()

    async def test_waiting_returns_the_reason(self) -> None:
        token = CancellationToken()
        token.cancel("caller went away")
        assert await token.wait() == "caller went away"

    async def test_waiting_wakes_when_cancellation_arrives(self) -> None:
        token = CancellationToken()
        waiter = asyncio.ensure_future(token.wait())
        await asyncio.sleep(0)
        token.cancel()
        assert await waiter


class TestTheDeadline:
    def test_it_is_an_instant_not_a_duration(self) -> None:
        assert Deadline.in_seconds(30, now=100.0).at == 130.0

    def test_remaining_counts_down(self) -> None:
        assert Deadline.in_seconds(30, now=100.0).remaining(110.0) == 20.0

    def test_remaining_never_goes_negative(self) -> None:
        """A negative remaining reads as 'plenty of time' to anything taking a minimum."""
        assert Deadline.in_seconds(30, now=100.0).remaining(200.0) == 0.0

    def test_it_expires_at_the_instant(self) -> None:
        deadline = Deadline.in_seconds(30, now=100.0)
        assert deadline.expired(129.9) is False
        assert deadline.expired(130.0) is True

    def test_a_deadline_narrows_to_the_earlier_one(self) -> None:
        assert Deadline(at=130.0).narrowed_to(Deadline(at=120.0)).at == 120.0

    def test_a_deadline_is_never_extended(self) -> None:
        """A sub-agent inherits its parent's ceiling and cannot declare a longer one."""
        assert Deadline(at=120.0).narrowed_to(Deadline(at=999.0)).at == 120.0

    def test_narrowing_to_nothing_leaves_it_alone(self) -> None:
        assert Deadline(at=120.0).narrowed_to(None).at == 120.0

    def test_a_deadline_in_no_time_at_all_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Deadline.in_seconds(0, now=100.0)


class TestDeclaringDeadlines:
    def test_nothing_is_bounded_by_default(self) -> None:
        """The kit will not guess how long a model takes; an invented ceiling kills good runs."""
        assert DeadlineConfig().run_seconds is None
        assert DeadlineConfig().model_call_seconds is None

    def test_a_zero_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="model_call_seconds"):
            DeadlineConfig(model_call_seconds=0)

    def test_a_negative_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="run_seconds"):
            DeadlineConfig(run_seconds=-1)

    def test_an_agent_may_declare_its_own(self) -> None:
        declared = agent(deadlines=DeadlineConfig(run_seconds=30))
        assert declared.deadlines is not None
        assert declared.deadlines.run_seconds == 30

    def test_a_tool_can_be_declared_safe_to_retry(self) -> None:
        declared = agent(tools=("search",), idempotent_tools=("search",))
        assert declared.idempotent_tools == ("search",)

    def test_a_tool_cannot_be_declared_idempotent_without_being_declared(self) -> None:
        with pytest.raises(ValueError, match="not on the allowlist"):
            agent(tools=("search",), idempotent_tools=("charge_card",))


def runner(provider: StallingProvider, clock: FakeClock, **overrides: object) -> AgentRunner:
    fields: dict[str, object] = {"provider": provider, "clock": clock}
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


async def started(runner_: AgentRunner, agent_: Agent, **overrides: object) -> asyncio.Task[Run]:
    return asyncio.ensure_future(
        runner_.run(agent_, "plan a trip", tenant="acme", run_id="run_1", **overrides)  # type: ignore[arg-type]
    )


class TestCancellingARun:
    async def test_a_run_cancelled_before_it_starts_calls_no_provider(self) -> None:
        provider = StallingProvider()
        token = CancellationToken()
        token.cancel("caller went away")
        run = await runner(provider, FakeClock()).run(
            agent(), "plan a trip", tenant="acme", cancellation=token
        )
        assert run.state is RunState.CANCELLED
        assert provider.calls == 0

    async def test_an_in_flight_model_call_is_aborted(self) -> None:
        provider = StallingProvider()
        token = CancellationToken()
        task = await started(runner(provider, FakeClock()), agent(), cancellation=token)
        await provider.entered.wait()
        token.cancel("caller went away")

        run = await task
        assert run.state is RunState.CANCELLED
        assert provider.calls == 1

    async def test_the_reason_is_on_the_run(self) -> None:
        provider = StallingProvider()
        token = CancellationToken()
        task = await started(runner(provider, FakeClock()), agent(), cancellation=token)
        await provider.entered.wait()
        token.cancel("caller went away")

        run = await task
        terminated = [event for event in run.events if event.kind is RunEventKind.TERMINATED]
        assert "caller went away" in str(terminated[-1].detail)

    async def test_cancellation_is_recorded_as_its_own_event(self) -> None:
        provider = StallingProvider()
        token = CancellationToken()
        task = await started(runner(provider, FakeClock()), agent(), cancellation=token)
        await provider.entered.wait()
        token.cancel()

        assert RunEventKind.CANCELLATION_REQUESTED in kinds(await task)

    async def test_a_cancelled_run_still_says_what_it_spent(self) -> None:
        """A cancelled run that reports zero usage is a bill nobody can reconcile."""
        provider = StallingProvider(calling())
        tools = FakeToolRegistry({"timetable": lambda: {"trains": 4}})
        token = CancellationToken()
        task = await started(
            runner(provider, FakeClock(), tools=tools),
            agent(tools=("timetable",)),
            cancellation=token,
        )
        await provider.entered.wait()
        token.cancel()

        run = await task
        assert run.state is RunState.CANCELLED
        assert run.usage.input_tokens == 10
        assert RunEventKind.TOOL_RESULT in kinds(run)

    async def test_no_further_model_call_is_issued_after_cancellation(self) -> None:
        provider = StallingProvider(calling())
        tools = FakeToolRegistry({"timetable": lambda: {"trains": 4}})
        token = CancellationToken()
        task = await started(
            runner(provider, FakeClock(), tools=tools),
            agent(tools=("timetable",)),
            cancellation=token,
        )
        await provider.entered.wait()
        token.cancel()
        await task

        assert provider.calls == 2

    async def test_two_cancellations_produce_exactly_one_cancelled_run(self) -> None:
        provider = StallingProvider()
        token = CancellationToken()
        task = await started(runner(provider, FakeClock()), agent(), cancellation=token)
        await provider.entered.wait()
        token.cancel("caller went away")
        token.cancel("and again")

        run = await task
        assert run.state is RunState.CANCELLED
        assert kinds(run).count(RunEventKind.TERMINATED) == 1

    async def test_a_run_without_a_token_still_completes(self) -> None:
        provider = StallingProvider(ModelResponse(content="Kyoto."))
        run = await runner(provider, FakeClock()).run(agent(), "plan a trip", tenant="acme")
        assert run.state is RunState.COMPLETED


class TestDeadlines:
    async def test_a_model_call_that_overruns_its_ceiling_is_cut_off(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider()
        task = await started(
            runner(provider, clock, deadlines=DeadlineConfig(model_call_seconds=30)), agent()
        )
        await clock.wait_for_sleep(1)
        clock.advance(30)

        run = await task
        assert run.state is RunState.CANCELLED
        assert RunEventKind.DEADLINE_EXCEEDED in kinds(run)

    async def test_the_overall_run_deadline_bounds_a_single_call(self) -> None:
        """A run ceiling with no per-call ceiling still has to stop the call it is inside."""
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider()
        task = await started(
            runner(provider, clock, deadlines=DeadlineConfig(run_seconds=45)), agent()
        )
        await clock.wait_for_sleep(1)
        assert clock.slept[0] == 45

        clock.advance(45)
        assert (await task).state is RunState.CANCELLED

    async def test_the_tighter_of_the_two_ceilings_wins(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider()
        task = await started(
            runner(
                provider,
                clock,
                deadlines=DeadlineConfig(run_seconds=45, model_call_seconds=90),
            ),
            agent(),
        )
        await clock.wait_for_sleep(1)
        assert clock.slept[0] == 45

        clock.advance(45)
        await task

    async def test_an_agents_own_deadlines_beat_the_runners(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider()
        task = await started(
            runner(provider, clock, deadlines=DeadlineConfig(model_call_seconds=90)),
            agent(deadlines=DeadlineConfig(model_call_seconds=15)),
        )
        await clock.wait_for_sleep(1)
        assert clock.slept[0] == 15

        clock.advance(15)
        await task

    async def test_a_caller_deadline_narrows_the_agents(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider()
        task = await started(
            runner(provider, clock),
            agent(deadlines=DeadlineConfig(run_seconds=600)),
            deadline=Deadline.in_seconds(20, now=clock.now()),
        )
        await clock.wait_for_sleep(1)
        assert clock.slept[0] == 20

        clock.advance(20)
        await task

    async def test_a_caller_deadline_cannot_extend_the_agents(self) -> None:
        """A sub-agent handed a longer deadline than its parent would outlive the parent."""
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider()
        task = await started(
            runner(provider, clock),
            agent(deadlines=DeadlineConfig(run_seconds=20)),
            deadline=Deadline.in_seconds(600, now=clock.now()),
        )
        await clock.wait_for_sleep(1)
        assert clock.slept[0] == 20

        clock.advance(20)
        await task

    async def test_an_elapsed_deadline_stops_the_run_before_the_next_model_call(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider(calling(), ModelResponse(content="late"))
        tools = FakeToolRegistry(
            {"timetable": tool_that(lambda: clock.advance(100), {"trains": 4})}
        )
        task = await started(
            runner(provider, clock, tools=tools, deadlines=DeadlineConfig(run_seconds=50)),
            agent(tools=("timetable",)),
        )

        run = await task
        assert run.state is RunState.CANCELLED
        assert provider.calls == 1
        assert RunEventKind.DEADLINE_EXCEEDED in kinds(run)

    async def test_a_deadline_already_elapsed_calls_no_provider_at_all(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider()
        run = await runner(provider, clock).run(
            agent(),
            "plan a trip",
            tenant="acme",
            deadline=Deadline(at=clock.now() - 1),
        )
        assert run.state is RunState.CANCELLED
        assert provider.calls == 0


class TestToolsCaughtMidFlight:
    async def test_a_tool_cancelled_after_dispatch_is_indeterminate(self) -> None:
        """The kit never claims a side effect did not happen when it cannot know."""
        clock = FakeClock(auto_advance=False)
        started_tool = asyncio.Event()

        async def charge_card() -> str:
            started_tool.set()
            await asyncio.Event().wait()
            return "charged"

        provider = StallingProvider(calling("charge_card"))
        tools = FakeToolRegistry({"charge_card": charge_card})
        token = CancellationToken()
        task = await started(
            runner(provider, clock, tools=tools),
            agent(tools=("charge_card",)),
            cancellation=token,
        )
        await started_tool.wait()
        token.cancel("caller went away")

        run = await task
        assert run.state is RunState.CANCELLED
        assert RunEventKind.TOOL_INDETERMINATE in kinds(run)

    async def test_a_tool_declared_idempotent_is_reported_as_safe_to_retry(self) -> None:
        clock = FakeClock(auto_advance=False)
        started_tool = asyncio.Event()

        async def search() -> str:
            started_tool.set()
            await asyncio.Event().wait()
            return "found"

        provider = StallingProvider(calling("search"))
        tools = FakeToolRegistry({"search": search})
        token = CancellationToken()
        task = await started(
            runner(provider, clock, tools=tools),
            agent(tools=("search",), idempotent_tools=("search",)),
            cancellation=token,
        )
        await started_tool.wait()
        token.cancel()

        run = await task
        assert RunEventKind.TOOL_INDETERMINATE not in kinds(run)
        assert RunEventKind.TOOL_ERROR in kinds(run)

    async def test_cancellation_between_two_tool_calls_stops_the_second(self) -> None:
        """The gap between validating a call's arguments and running it is a gap to check in."""
        token = CancellationToken()
        provider = StallingProvider(
            ModelResponse(
                tool_calls=(
                    ToolCall(id="call_1", name="first", arguments={}),
                    ToolCall(id="call_2", name="second", arguments={}),
                ),
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        )
        tools = FakeToolRegistry(
            {
                "first": tool_that(lambda: token.cancel("caller went away"), "done"),
                "second": lambda: "done",
            }
        )
        run = await runner(provider, FakeClock(), tools=tools).run(
            agent(tools=("first", "second")),
            "plan a trip",
            tenant="acme",
            cancellation=token,
        )
        assert run.state is RunState.CANCELLED
        assert [name for name, _ in tools.calls] == ["first"]

    async def test_a_tool_is_not_dispatched_with_no_time_left(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider(
            ModelResponse(
                tool_calls=(
                    ToolCall(id="call_1", name="slow", arguments={}),
                    ToolCall(id="call_2", name="next", arguments={}),
                ),
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        )
        tools = FakeToolRegistry(
            {"slow": tool_that(lambda: clock.advance(100), "done"), "next": lambda: "done"}
        )
        run = await runner(
            provider, clock, tools=tools, deadlines=DeadlineConfig(run_seconds=50)
        ).run(agent(tools=("slow", "next")), "plan a trip", tenant="acme")

        assert run.state is RunState.CANCELLED
        assert [name for name, _ in tools.calls] == ["slow"]
        assert RunEventKind.DEADLINE_EXCEEDED in kinds(run)

    async def test_a_tool_that_overruns_its_own_ceiling_is_cut_off(self) -> None:
        clock = FakeClock(auto_advance=False)
        started_tool = asyncio.Event()

        async def slow() -> str:
            started_tool.set()
            await asyncio.Event().wait()
            return "eventually"

        provider = StallingProvider(calling("slow"))
        tools = FakeToolRegistry({"slow": slow})
        task = await started(
            runner(provider, clock, tools=tools, deadlines=DeadlineConfig(tool_call_seconds=10)),
            agent(tools=("slow",)),
        )
        await started_tool.wait()
        await clock.wait_for_sleep(1)
        clock.advance(10)

        run = await task
        assert run.state is RunState.CANCELLED
        assert RunEventKind.DEADLINE_EXCEEDED in kinds(run)


class TestWorkThatIgnoresTheAbort:
    async def test_the_run_still_resolves_inside_the_grace_window(self) -> None:
        """A provider that keeps streaming is dropped, not waited for."""
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider(ignores_cancellation=1)
        task = await started(
            runner(
                provider,
                clock,
                deadlines=DeadlineConfig(model_call_seconds=30, grace_seconds=5),
            ),
            agent(),
        )
        await clock.wait_for_sleep(1)
        clock.advance(30)
        await clock.wait_for_sleep(2)
        clock.advance(5)

        run = await task
        assert run.state is RunState.CANCELLED
        assert RunEventKind.WORK_ORPHANED in kinds(run)
        provider.release()

    async def test_work_that_lands_inside_the_grace_window_is_not_orphaned(self) -> None:
        """Orphaned is a claim about knowledge, so it is only made once the grace is spent."""
        clock = FakeClock(auto_advance=False)
        provider = StallingProvider(ignores_cancellation=1)
        task = await started(
            runner(
                provider,
                clock,
                deadlines=DeadlineConfig(model_call_seconds=30, grace_seconds=5),
            ),
            agent(),
        )
        await clock.wait_for_sleep(1)
        clock.advance(30)
        await clock.wait_for_sleep(2)
        provider.release()

        run = await task
        assert run.state is RunState.CANCELLED
        assert RunEventKind.WORK_ORPHANED not in kinds(run)

    async def test_a_guardrail_caught_mid_check_stops_the_run(self) -> None:
        guardrail = StallingGuardrail()
        provider = StallingProvider(ModelResponse(content="here is a plan"))
        token = CancellationToken()
        task = await started(
            runner(provider, FakeClock(), guardrails={"toxicity": guardrail}),
            agent(guardrails=("toxicity",)),
            cancellation=token,
        )
        await guardrail.entered.wait()
        token.cancel("caller went away")

        run = await task
        assert run.state is RunState.CANCELLED
        assert RunEventKind.CANCELLATION_REQUESTED in kinds(run)


class TestTheTestClock:
    """The manual clock is what makes every deadline test above deterministic."""

    async def test_advancing_short_of_a_sleeper_leaves_it_sleeping(self) -> None:
        clock = FakeClock(auto_advance=False)
        sleeper = asyncio.ensure_future(clock.sleep(30))
        await clock.wait_for_sleep(1)
        clock.advance(29)
        await asyncio.sleep(0)
        assert sleeper.done() is False

        clock.advance(1)
        await sleeper

    async def test_waiting_for_a_sleep_that_never_comes_fails_the_test(self) -> None:
        with pytest.raises(AssertionError, match="expected 1 sleeps"):
            await FakeClock(auto_advance=False).wait_for_sleep()
