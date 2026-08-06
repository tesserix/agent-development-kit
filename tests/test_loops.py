"""Bounding the shape of a run: how deep, how wide, and how often the same thing.

The failure this file exists to prevent is a run nobody can interrupt — two agents calling
each other, or one model asking for the same tool forever, until a provider quota or a
support ticket ends it.
"""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    Agent,
    FanOutLimitError,
    LoopConfig,
    RecursionLimitError,
    RepeatedCallError,
    Run,
    RunContext,
    RunEventKind,
    RunState,
    TenantContext,
    ToolCall,
    ToolExecutionError,
    ToolFailurePolicy,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeBudgetPolicy, FakeToolRegistry, ScriptedProvider


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def kinds(run: Run) -> list[RunEventKind]:
    return [event.kind for event in run.events]


def details(run: Run, kind: RunEventKind) -> list[str]:
    return [event.detail or "" for event in run.events if event.kind is kind]


def calling(tool: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name=tool, arguments=arguments),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def fanning_out(tool: str, count: int) -> ModelResponse:
    return ModelResponse(
        tool_calls=tuple(
            ToolCall(id=f"call_{n}", name=tool, arguments={"page": n}) for n in range(count)
        ),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def registry() -> FakeToolRegistry:
    return FakeToolRegistry({"search": lambda **_: "a result"})


class TestDeclaringLoopBounds:
    def test_a_run_is_bounded_by_default(self) -> None:
        """Unlike a deadline, a loop cap costs nothing when it does not bind, and the
        run it stops is one nobody can otherwise interrupt."""
        bounds = LoopConfig()
        assert bounds.max_depth >= 1
        assert bounds.max_tool_calls_per_turn >= 1
        assert bounds.max_tool_calls_per_run >= 1
        assert bounds.max_repeated_calls >= 1

    @pytest.mark.parametrize(
        "field",
        [
            "max_depth",
            "max_tool_calls_per_turn",
            "max_tool_calls_per_run",
            "max_repeated_calls",
        ],
    )
    def test_a_cap_of_zero_is_refused(self, field: str) -> None:
        """Zero reads as 'never do this at all', which is not a cap, it is a broken run."""
        with pytest.raises(ValueError, match=field):
            LoopConfig(**{field: 0})

    def test_an_agent_narrows_the_runner_and_never_widens_it(self) -> None:
        tight = LoopConfig(max_depth=2, max_tool_calls_per_run=4)
        loose = LoopConfig(max_depth=9, max_tool_calls_per_run=99)
        narrowed = loose.narrowed_to(tight)
        assert narrowed.max_depth == 2
        assert narrowed.max_tool_calls_per_run == 4

    def test_narrowing_against_nothing_leaves_the_bounds_alone(self) -> None:
        bounds = LoopConfig(max_depth=3)
        assert bounds.narrowed_to(None) == bounds


class TestRecursionDepth:
    async def test_a_run_nobody_called_is_at_the_top(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="done"))
        run = await AgentRunner(provider=provider).run(agent(), "plan a trip", tenant="acme")
        assert run.depth == 0

    async def test_a_child_run_is_one_deeper_than_its_parent(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="done"))
        parent = RunContext(run_id="run_0", tenant=TenantContext(tenant="acme"))

        run = await AgentRunner(provider=provider).run(
            agent(), "plan a trip", tenant="acme", parent=parent
        )

        assert run.depth == 1
        assert run.context.depth == 1

    async def test_the_deepest_run_fails_closed_without_calling_a_model(self) -> None:
        provider = ScriptedProvider()
        runner = AgentRunner(provider=provider, loop=LoopConfig(max_depth=1))
        parent = RunContext(run_id="run_1", tenant=TenantContext(tenant="acme"), depth=1)

        run = await runner.run(agent(), "again", tenant="acme", parent=parent)

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert RunEventKind.DEPTH_EXCEEDED in kinds(run)
        assert provider.requests == []
        assert "RecursionLimitError" in details(run, RunEventKind.TERMINATED)[0]

    async def test_a_cycle_between_two_agents_ends_and_every_level_records_its_depth(self) -> None:
        """A calls B calls A: the chain stops, and no level invents a result to keep going."""
        depths: list[int] = []

        async def peer() -> str:
            depth = len(depths)
            parent = RunContext(
                run_id=f"run_{depth}", tenant=TenantContext(tenant="acme"), depth=depth
            )
            depths.append(depth + 1)
            child = await runner.run(
                _peer_agent(), "again", tenant="acme", parent=parent, run_id=f"run_{depth + 1}"
            )
            if child.state is not RunState.COMPLETED:
                raise ToolExecutionError(
                    f"peer run {child.id} ended {child.state} at depth {child.depth}"
                )
            return "the peer answered"

        runner = AgentRunner(
            provider=_always_calls_peer(),
            tools=FakeToolRegistry({"ask_peer": peer}),
            loop=LoopConfig(max_depth=2),
        )

        run = await runner.run(_peer_agent(), "start", tenant="acme", run_id="run_0")

        assert run.state is RunState.FAILED
        assert depths == [1, 2, 3]

    async def test_an_agent_cannot_raise_the_ceiling_it_was_given(self) -> None:
        provider = ScriptedProvider()
        runner = AgentRunner(provider=provider, loop=LoopConfig(max_depth=1))
        parent = RunContext(run_id="run_1", tenant=TenantContext(tenant="acme"), depth=1)

        run = await runner.run(
            agent(loop=LoopConfig(max_depth=99)), "again", tenant="acme", parent=parent
        )

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED

    async def test_an_agent_may_tighten_it(self) -> None:
        provider = ScriptedProvider()
        runner = AgentRunner(provider=provider, loop=LoopConfig(max_depth=9))
        parent = RunContext(run_id="run_1", tenant=TenantContext(tenant="acme"), depth=1)

        run = await runner.run(
            agent(loop=LoopConfig(max_depth=1)), "again", tenant="acme", parent=parent
        )

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED


def _peer_agent() -> Agent:
    return agent(tools=("ask_peer",), on_tool_error=ToolFailurePolicy.FAIL_RUN)


def _always_calls_peer() -> ScriptedProvider:
    return ScriptedProvider(*[calling("ask_peer") for _ in range(8)])


class TestFanOut:
    async def test_a_turn_wider_than_the_cap_dispatches_nothing(self) -> None:
        """Refusing before dispatch is the point: half a fan-out is a side effect nobody chose."""
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(fanning_out("search", 5)),
            tools=tools,
            loop=LoopConfig(max_tool_calls_per_turn=2),
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert tools.calls == []
        assert "FanOutLimitError" in details(run, RunEventKind.TERMINATED)[0]
        assert RunEventKind.FAN_OUT_REFUSED in kinds(run)

    async def test_the_run_total_binds_across_turns(self) -> None:
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(*[fanning_out("search", 2) for _ in range(4)]),
            tools=tools,
            loop=LoopConfig(max_tool_calls_per_turn=2, max_tool_calls_per_run=3),
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert len(tools.calls) == 2

    async def test_a_run_inside_the_caps_is_untouched(self) -> None:
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(fanning_out("search", 2), ModelResponse(content="done")),
            tools=tools,
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert len(tools.calls) == 2


class TestRepeatedCalls:
    async def test_the_same_call_over_and_over_ends_the_run(self) -> None:
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(*[calling("search", q="kyoto") for _ in range(6)]),
            tools=tools,
            loop=LoopConfig(max_repeated_calls=2),
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert RunEventKind.REPEAT_DETECTED in kinds(run)
        assert "RepeatedCallError" in details(run, RunEventKind.TERMINATED)[0]
        assert len(tools.calls) == 2

    async def test_a_different_argument_is_a_different_call(self) -> None:
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(
                calling("search", q="kyoto"),
                calling("search", q="osaka"),
                calling("search", q="nara"),
                ModelResponse(content="done"),
            ),
            tools=tools,
            loop=LoopConfig(max_repeated_calls=2),
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert len(tools.calls) == 3

    async def test_argument_order_does_not_make_a_call_look_new(self) -> None:
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(
                calling("search", q="kyoto", page=1),
                calling("search", page=1, q="kyoto"),
                calling("search", q="kyoto", page=1),
            ),
            tools=tools,
            loop=LoopConfig(max_repeated_calls=2),
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED

    async def test_a_polling_tool_declared_idempotent_may_repeat_itself(self) -> None:
        """Polling the same status endpoint with the same arguments is the design, not a cycle."""
        tools = FakeToolRegistry({"poll": lambda **_: "pending"})
        runner = AgentRunner(
            provider=ScriptedProvider(
                *[calling("poll", job="j1") for _ in range(3)], ModelResponse(content="done")
            ),
            tools=tools,
            loop=LoopConfig(max_repeated_calls=2),
        )

        run = await runner.run(
            agent(tools=("poll",), idempotent_tools=("poll",)), "watch it", tenant="acme"
        )

        assert run.state is RunState.COMPLETED
        assert len(tools.calls) == 3


class TestWhicheverCapBindsFirst:
    async def test_a_model_that_never_answers_stops_at_the_iteration_cap(self) -> None:
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(*[calling("search", page=n) for n in range(6)]),
            tools=tools,
            max_iterations=6,
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.MAX_ITERATIONS_EXCEEDED
        assert kinds(run).count(RunEventKind.MODEL_CALL) == 6
        assert run.output is None
        assert run.usage.input_tokens == 60
        assert "MaxIterationsError" in details(run, RunEventKind.TERMINATED)[0]

    async def test_exactly_one_terminal_state_is_recorded_when_two_ceilings_bind(self) -> None:
        """A cap and a budget reached on the same turn is still one ending and one reason."""
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(*[calling("search", page=n) for n in range(4)]),
            tools=tools,
            budget=FakeBudgetPolicy(limit=40),
            loop=LoopConfig(max_tool_calls_per_run=1),
        )

        run = await runner.run(agent(tools=("search",)), "plan a trip", tenant="acme")

        assert run.state is RunState.LOOP_LIMIT_EXCEEDED
        assert kinds(run).count(RunEventKind.TERMINATED) == 1


class TestTheTypedErrors:
    @pytest.mark.parametrize(
        "error",
        [
            RecursionLimitError("too deep"),
            FanOutLimitError("too wide"),
            RepeatedCallError("again and again"),
        ],
    )
    def test_a_cap_is_never_retryable(self, error: Exception) -> None:
        """Asking again gets the same cap: the run is over, not unlucky."""
        assert getattr(error, "retryable") is False  # noqa: B009
