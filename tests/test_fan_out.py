"""Caps on the shape of a run, held in the same policy as the money.

A cost ceiling reacts after the money is gone. A fan-out of two hundred tool calls, or two
agents delegating to each other, spends a whole budget in seconds and hammers whatever is
downstream on the way. These caps stop that by structure, before it converts into spend.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    BudgetLimits,
    BudgetScope,
    Consumed,
    ModelCapabilities,
    Run,
    RunBudget,
    RunContext,
    RunEventKind,
    RunState,
    ScopedLimits,
    TenantContext,
    ToolCall,
    Usage,
    most_restrictive,
)
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Callable

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


def agent(**overrides: object) -> Agent[Any]:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "scripted-1",
        "tools": ("lookup",),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def fanning_out(count: int, tool: str = "lookup") -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=tuple(
            ToolCall(id=f"call_{n}", name=tool, arguments={"page": n}) for n in range(count)
        ),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def budget(clock: FakeClock, **limits: object) -> RunBudget:
    return RunBudget(
        resolved=most_restrictive(
            ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(**limits))  # type: ignore[arg-type]
        ),
        clock=clock,
    )


def runner(clock: FakeClock, *responses: ModelResponse, **limits: object) -> AgentRunner:
    return AgentRunner(
        provider=ScriptedProvider(*responses, name="scripted", capabilities=CAPABLE),
        clock=clock,
        budget=budget(clock, **limits),
        tools=FakeToolRegistry({"lookup": lambda page=0: {"page": page}}),
    )


def detail_of(run: Run[Any], kind: RunEventKind) -> str:
    return next((event.detail or "" for event in run.events if event.kind is kind), "")


class TestTheCapsLiveWithTheMoney:
    def test_fan_out_and_depth_are_dimensions_of_the_one_ceiling(self) -> None:
        """Two policies is two places to raise a cap, and one of them gets forgotten."""
        limits = BudgetLimits.conservative()
        assert limits.max_parallel_tool_calls is not None
        assert limits.max_delegation_depth is not None
        assert limits.max_peer_invocations is not None

    def test_a_cap_of_zero_is_refused_like_any_other_ceiling(self) -> None:
        with pytest.raises(ValueError, match="max_delegation_depth"):
            BudgetLimits(max_delegation_depth=0)

    def test_what_a_run_has_delegated_is_counted(self) -> None:
        assert (Consumed(peer_invocations=1) + Consumed(peer_invocations=2)).peer_invocations == 3


class TestATurnWiderThanTheCap:
    async def test_no_tool_is_dispatched_beyond_the_cap(self) -> None:
        """Ten of two hundred executed, presented as the answer, is the worst outcome."""
        clock = FakeClock()
        registry = FakeToolRegistry({"lookup": lambda page=0: {"page": page}})
        run = await AgentRunner(
            provider=ScriptedProvider(fanning_out(200), name="scripted", capabilities=CAPABLE),
            clock=clock,
            budget=budget(clock, max_parallel_tool_calls=10),
            tools=registry,
        ).run(agent(), "look it all up", tenant="acme")
        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert registry.calls == []
        assert run.output is None

    async def test_the_refusal_names_the_cap_and_what_was_asked_for(self) -> None:
        clock = FakeClock()
        run = await runner(clock, fanning_out(200), max_parallel_tool_calls=10).run(
            agent(), "look it all up", tenant="acme"
        )
        detail = detail_of(run, RunEventKind.FAN_OUT_REFUSED)
        assert "max_parallel_tool_calls" in detail
        assert "10" in detail
        assert "200" in detail

    async def test_a_fan_out_inside_the_cap_runs_every_call(self) -> None:
        clock = FakeClock()
        registry = FakeToolRegistry({"lookup": lambda page=0: {"page": page}})
        run = await AgentRunner(
            provider=ScriptedProvider(
                fanning_out(3), answer(), name="scripted", capabilities=CAPABLE
            ),
            clock=clock,
            budget=budget(clock, max_parallel_tool_calls=10),
            tools=registry,
        ).run(agent(), "look it up", tenant="acme")
        assert run.state is RunState.COMPLETED
        assert len(registry.calls) == 3


class TestARunThatDelegatesTooDeep:
    async def test_the_run_stops_at_the_cap_and_prints_the_path_it_took(self) -> None:
        """A→B→A is the shape of the bug, and naming it is how somebody finds it."""
        clock = FakeClock()
        run = await runner(clock, answer(), max_delegation_depth=3).run(
            agent(name="alpha"),
            "delegate",
            tenant="acme",
            parent=RunContext(
                run_id="run_0",
                tenant=TenantContext(tenant="acme"),
                depth=3,
                path=("alpha", "beta"),
            ),
        )
        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert "alpha→beta→alpha" in detail_of(run, RunEventKind.DEPTH_EXCEEDED)

    async def test_what_it_spent_getting_there_is_still_recorded(self) -> None:
        clock = FakeClock()
        limit = budget(clock, max_delegation_depth=1)
        await AgentRunner(
            provider=ScriptedProvider(answer(), name="scripted", capabilities=CAPABLE),
            clock=clock,
            budget=limit,
        ).run(agent(tools=()), "go", tenant="acme")
        assert limit.spent.model_calls == 1

    async def test_a_child_is_counted_against_the_whole_tree(self) -> None:
        """Per-hop counting is how a tree of runs each stays under a cap they broke together."""
        clock = FakeClock()
        limit = budget(clock, max_peer_invocations=2)
        called = RunContext(run_id="run_0", tenant=TenantContext(tenant="acme"))
        for _ in range(2):
            child = AgentRunner(
                provider=ScriptedProvider(answer(), name="scripted", capabilities=CAPABLE),
                clock=clock,
                budget=limit.child(),
            )
            await child.run(agent(tools=()), "look it up", tenant="acme", parent=called)
        assert limit.spent.peer_invocations == 2
        run = await AgentRunner(
            provider=ScriptedProvider(answer(), name="scripted", capabilities=CAPABLE),
            clock=clock,
            budget=limit.child(),
        ).run(agent(tools=()), "again", tenant="acme", parent=called)
        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert "max_peer_invocations" in detail_of(run, RunEventKind.DEPTH_EXCEEDED)


class TestAChildCannotVoteItselfMoreRope:
    async def test_a_delegated_agent_asking_for_a_wider_cap_gets_the_parent_s(self) -> None:
        clock = FakeClock()
        limit = budget(clock, max_parallel_tool_calls=2)
        run = await AgentRunner(
            provider=ScriptedProvider(fanning_out(5), name="scripted", capabilities=CAPABLE),
            clock=clock,
            budget=limit.child(),
            tools=FakeToolRegistry({"lookup": lambda page=0: {"page": page}}),
        ).run(
            agent(budget=BudgetLimits(max_parallel_tool_calls=50)),
            "look it up",
            tenant="acme",
            parent=RunContext(run_id="run_0", tenant=TenantContext(tenant="acme"), depth=1),
        )
        assert run.state is RunState.LOOP_LIMIT_EXCEEDED


class TestConcurrencyThatDoesNotOverwhelmWhatIsDownstream:
    async def test_a_cleared_fan_out_never_has_more_downstream_than_the_cap(self) -> None:
        """One call at a time is inside any cap, and it is what keeps the next one stoppable."""
        clock = FakeClock()
        watcher = _Watching()
        run = await AgentRunner(
            provider=ScriptedProvider(
                fanning_out(4), answer(), name="scripted", capabilities=CAPABLE
            ),
            clock=clock,
            budget=budget(clock, max_parallel_tool_calls=4),
            tools=FakeToolRegistry({"lookup": watcher}),
        ).run(agent(), "look it up", tenant="acme")
        assert run.state is RunState.COMPLETED
        assert watcher.peak <= 4

    async def test_a_fan_out_cancelled_part_way_records_what_ran_and_stops(self) -> None:
        """The calls that landed are on the record; the ones that had not gone out do not."""
        clock = FakeClock()
        token = CancellationToken()
        registry = FakeToolRegistry({"lookup": _cancelling(token)})
        limit = budget(clock, max_parallel_tool_calls=4)
        run = await AgentRunner(
            provider=ScriptedProvider(fanning_out(4), name="scripted", capabilities=CAPABLE),
            clock=clock,
            budget=limit,
            tools=registry,
        ).run(agent(), "look it up", tenant="acme", cancellation=token)
        assert run.state is RunState.CANCELLED
        assert len(registry.calls) == 1
        assert [event.kind for event in run.events].count(RunEventKind.TOOL_RESULT) == 1
        assert limit.spent.model_calls == 1


class _Watching:
    """A tool that says how many copies of itself ran at once."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    async def __call__(self, page: int = 0) -> dict[str, int]:
        """Hold the slot long enough for a wider fan-out to overlap if it were allowed to."""
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0)
        self.live -= 1
        return {"page": page}


def _cancelling(token: CancellationToken) -> Callable[..., dict[str, int]]:
    def tool(page: int = 0) -> dict[str, int]:
        token.cancel("caller went away")
        return {"page": page}

    return tool
