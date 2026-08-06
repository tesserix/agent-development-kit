"""The run loop, from prompt assembly to exactly one terminal state.

Every product that hand-rolled this loop disagreed about what "finished" means. Here it
is one thing: `run` returns a `Run` whose state is terminal, always, and the events on it
say how it got there. A wedged run is the failure mode this file exists to prevent.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from tesserix_adk.core import (
    Agent,
    BudgetConfig,
    CancelledError,
    ConfigurationError,
    ProviderError,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    ToolFailurePolicy,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import (
    FakeBudgetPolicy,
    FakeClock,
    FakeGuardrail,
    FakeToolRegistry,
    ScriptedProvider,
)


class TripPlan(BaseModel):
    destination: str
    nights: int


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def runner(*responses: ModelResponse | BaseException, **overrides: object) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


async def start(runner_: AgentRunner, agent_: Agent, text: str = "plan a trip") -> Run:
    return await runner_.run(agent_, text, tenant="acme", run_id="run_1")


class TestTheHappyPath:
    async def test_a_model_that_stops_without_tool_calls_completes_the_run(self) -> None:
        run = await start(runner(answer()), agent())
        assert run.state is RunState.COMPLETED

    async def test_the_answer_is_the_last_message(self) -> None:
        run = await start(runner(answer("Kyoto, four nights.")), agent())
        assert run.messages[-1].role == "assistant"
        assert run.messages[-1].content[0].text == "Kyoto, four nights."  # type: ignore[union-attr]

    async def test_the_run_carries_who_and_what_it_was(self) -> None:
        """Cost attribution reads these off the run rather than being wired per product."""
        run = await start(runner(answer()), agent())
        assert (run.tenant, run.agent_name, run.agent_version) == ("acme", "planner", "1.0.0")
        assert run.model == "claude-sonnet-5"
        assert run.prompt_version is not None

    async def test_usage_is_totalled_onto_the_run(self) -> None:
        run = await start(runner(answer()), agent())
        assert run.usage == Usage(input_tokens=10, output_tokens=5)

    async def test_the_run_is_timed(self) -> None:
        run = await start(runner(answer()), agent())
        assert run.started_at is not None
        assert run.ended_at is not None

    async def test_the_steps_are_recorded_in_order(self) -> None:
        run = await start(runner(answer()), agent())
        assert [event.kind for event in run.events] == [
            RunEventKind.PROMPT_ASSEMBLED,
            RunEventKind.MODEL_CALL,
            RunEventKind.MODEL_RESPONSE,
            RunEventKind.TERMINATED,
        ]

    async def test_the_model_call_event_carries_what_it_cost(self) -> None:
        run = await start(runner(answer()), agent())
        response = next(e for e in run.events if e.kind is RunEventKind.MODEL_RESPONSE)
        assert response.usage == Usage(input_tokens=10, output_tokens=5)

    async def test_an_empty_registry_and_no_output_type_still_finishes(self) -> None:
        """The floor case: the first response already satisfies everything asked of it."""
        run = await start(runner(answer()), agent())
        assert run.state is RunState.COMPLETED
        assert run.output is None

    def test_the_sync_wrapper_runs_the_same_loop(self) -> None:
        """Not every consumer is async; the wrapper owns the loop, so it cannot run inside one."""
        run = runner(answer()).run_sync(agent(), "plan a trip", tenant="acme")
        assert run.state is RunState.COMPLETED


class TestTools:
    def registry(self) -> FakeToolRegistry:
        return FakeToolRegistry({"search": lambda q: f"3 results for {q}"})

    def calling(self) -> ModelResponse:
        return ModelResponse(
            tool_calls=(ToolCall(id="call_1", name="search", arguments={"q": "Kyoto"}),),
            usage=Usage(input_tokens=8, output_tokens=2),
        )

    async def test_a_tool_call_is_dispatched_and_its_result_fed_back(self) -> None:
        tools = self.registry()
        run = await start(runner(self.calling(), answer(), tools=tools), agent(tools=("search",)))
        assert tools.calls == [("search", {"q": "Kyoto"})]
        assert run.state is RunState.COMPLETED

    async def test_the_tool_result_re_enters_the_conversation_as_data(self) -> None:
        """A result pasted in as prose is an instruction channel for whatever produced it."""
        run = await start(
            runner(self.calling(), answer(), tools=self.registry()), agent(tools=("search",))
        )
        result = next(message for message in run.messages if message.role == "tool")
        assert result.tool_call_id == "call_1"
        assert "untrusted-data" in result.content[0].text  # type: ignore[union-attr]

    async def test_the_dispatch_is_recorded(self) -> None:
        run = await start(
            runner(self.calling(), answer(), tools=self.registry()), agent(tools=("search",))
        )
        kinds = [event.kind for event in run.events]
        assert RunEventKind.TOOL_CALL in kinds
        assert RunEventKind.TOOL_RESULT in kinds

    async def test_a_tool_the_agent_was_not_given_is_never_dispatched(self) -> None:
        """The allowlist is the boundary; a call outside it means something upstream is wrong."""
        tools = FakeToolRegistry({"wire_money": lambda **_: "sent"})
        response = ModelResponse(
            tool_calls=(ToolCall(id="call_1", name="wire_money"),),
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        run = await start(runner(response, tools=tools), agent(tools=("search",)))
        assert tools.calls == []
        assert run.state is RunState.FAILED
        assert any(event.kind is RunEventKind.TOOL_REFUSED for event in run.events)

    async def test_an_oversized_tool_result_is_truncated_and_says_so(self) -> None:
        """Silently dropping half a result is a wrong answer nobody can account for."""
        tools = FakeToolRegistry({"search": lambda **_: "x" * 5_000})
        run = await start(
            runner(self.calling(), answer(), tools=tools, max_tool_result_chars=100),
            agent(tools=("search",)),
        )
        assert any(event.kind is RunEventKind.TOOL_RESULT_TRUNCATED for event in run.events)
        result = next(message for message in run.messages if message.role == "tool")
        assert len(result.content[0].text) < 1_000  # type: ignore[union-attr]

    async def test_a_repeated_tool_call_is_dispatched_once(self) -> None:
        """A retried provider response repeats calls it already sent."""
        call = ToolCall(id="call_1", name="search", arguments={"q": "Kyoto"})
        tools = self.registry()
        await start(
            runner(
                ModelResponse(
                    tool_calls=(call, call), usage=Usage(input_tokens=1, output_tokens=1)
                ),
                answer(),
                tools=tools,
            ),
            agent(tools=("search",)),
        )
        assert len(tools.calls) == 1

    async def test_an_async_tool_is_awaited(self) -> None:
        """Most real tools do I/O; a coroutine handed back as a result is not a result."""

        async def search(q: str) -> str:
            return f"3 results for {q}"

        tools = FakeToolRegistry({"search": search})
        run = await start(runner(self.calling(), answer(), tools=tools), agent(tools=("search",)))
        result = next(message for message in run.messages if message.role == "tool")
        assert "3 results for Kyoto" in result.content[0].text  # type: ignore[union-attr]

    async def test_a_tool_result_that_is_not_text_is_rendered_as_data(self) -> None:
        tools = FakeToolRegistry({"search": lambda **_: {"hits": 3}})
        run = await start(runner(self.calling(), answer(), tools=tools), agent(tools=("search",)))
        result = next(message for message in run.messages if message.role == "tool")
        assert '"hits": 3' in result.content[0].text  # type: ignore[union-attr]

    async def test_a_self_referential_tool_result_still_renders(self) -> None:
        """A result that cannot be serialised must not take the run down with it."""
        circular: list[object] = []
        circular.append(circular)
        tools = FakeToolRegistry({"search": lambda **_: circular})
        run = await start(runner(self.calling(), answer(), tools=tools), agent(tools=("search",)))
        assert run.state is RunState.COMPLETED

    async def test_an_allowlisted_tool_the_registry_does_not_have_is_a_tool_error(self) -> None:
        """The allowlist and the registry disagreeing is a wiring fault, not a fake result."""
        tools = FakeToolRegistry({})
        run = await start(runner(self.calling(), answer(), tools=tools), agent(tools=("search",)))
        failure = next(e for e in run.events if e.kind is RunEventKind.TOOL_ERROR)
        assert failure.detail is not None
        assert "search" in failure.detail

    async def test_an_agent_that_declares_tools_without_a_registry_is_refused(self) -> None:
        """Failing at the first tool call, mid-run, is a worse time to find out."""
        with pytest.raises(ConfigurationError, match="search"):
            await start(runner(answer()), agent(tools=("search",)))


class TestToolFailure:
    def exploding(self) -> FakeToolRegistry:
        def boom(**_: object) -> str:
            raise RuntimeError("upstream 503")

        return FakeToolRegistry({"search": boom})

    def calling(self) -> ModelResponse:
        return ModelResponse(
            tool_calls=(ToolCall(id="call_1", name="search"),),
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def test_an_unexpected_exception_does_not_escape_untyped(self) -> None:
        run = await start(
            runner(self.calling(), answer(), tools=self.exploding()), agent(tools=("search",))
        )
        assert run.state is RunState.COMPLETED

    async def test_the_failure_is_recorded_with_the_tool_that_caused_it(self) -> None:
        run = await start(
            runner(self.calling(), answer(), tools=self.exploding()), agent(tools=("search",))
        )
        failure = next(e for e in run.events if e.kind is RunEventKind.TOOL_ERROR)
        assert failure.name == "search"
        assert failure.detail is not None
        assert "503" in failure.detail

    async def test_the_model_is_told_its_tool_failed_rather_than_handed_a_fake_result(
        self,
    ) -> None:
        """An invented result is a wrong answer the model has no way to notice."""
        run = await start(
            runner(self.calling(), answer(), tools=self.exploding()), agent(tools=("search",))
        )
        result = next(message for message in run.messages if message.role == "tool")
        assert "error" in result.content[0].text.lower()  # type: ignore[union-attr]

    async def test_an_agent_may_declare_that_a_tool_failure_ends_the_run(self) -> None:
        run = await start(
            runner(self.calling(), tools=self.exploding()),
            agent(tools=("search",), on_tool_error=ToolFailurePolicy.FAIL_RUN),
        )
        assert run.state is RunState.FAILED

    async def test_the_partially_built_run_comes_back_with_everything_so_far(self) -> None:
        """A failure that discards the record leaves nobody able to say what happened."""
        run = await start(
            runner(self.calling(), tools=self.exploding()),
            agent(tools=("search",), on_tool_error=ToolFailurePolicy.FAIL_RUN),
        )
        assert [event.kind for event in run.events][:3] == [
            RunEventKind.PROMPT_ASSEMBLED,
            RunEventKind.MODEL_CALL,
            RunEventKind.MODEL_RESPONSE,
        ]
        assert run.usage.input_tokens == 1


class TestStructuredOutput:
    async def test_a_valid_answer_is_parsed_onto_the_run(self) -> None:
        run = await start(
            runner(answer('{"destination": "Kyoto", "nights": 4}')),
            agent(output_type=TripPlan),
        )
        assert run.output == {"destination": "Kyoto", "nights": 4}
        assert run.state is RunState.COMPLETED

    async def test_the_validation_is_recorded(self) -> None:
        run = await start(
            runner(answer('{"destination": "Kyoto", "nights": 4}')),
            agent(output_type=TripPlan),
        )
        assert any(event.kind is RunEventKind.OUTPUT_VALIDATED for event in run.events)

    async def test_an_answer_of_the_wrong_shape_fails_the_run(self) -> None:
        """A half-parsed plan handed on as if it were whole is the bug this prevents."""
        run = await start(runner(answer('{"destination": "Kyoto"}')), agent(output_type=TripPlan))
        assert run.state is RunState.FAILED
        assert any(event.kind is RunEventKind.SCHEMA_VIOLATION for event in run.events)

    async def test_an_answer_that_is_not_json_fails_the_run(self) -> None:
        run = await start(runner(answer("Kyoto, four nights.")), agent(output_type=TripPlan))
        assert run.state is RunState.FAILED


class TestTerminalStates:
    async def test_a_provider_failure_ends_the_run_rather_than_escaping(self) -> None:
        run = await start(runner(ProviderError("upstream 500")), agent())
        assert run.state is RunState.FAILED
        assert run.ended_at is not None

    async def test_zero_content_and_zero_tool_calls_is_terminal_not_a_retry(self) -> None:
        """Asking again for the same nothing is how a loop wedges."""
        empty = ModelResponse(usage=Usage(input_tokens=1, output_tokens=0))
        run = await start(runner(empty), agent())
        assert run.state is RunState.FAILED

    async def test_a_loop_that_will_not_settle_hits_the_cap(self) -> None:
        tools = FakeToolRegistry({"search": lambda **_: "again"})
        calling = ModelResponse(
            tool_calls=(ToolCall(id="call_1", name="search"),),
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        run = await start(
            runner(calling, calling, calling, tools=tools, max_iterations=3),
            agent(tools=("search",)),
        )
        assert run.state is RunState.MAX_ITERATIONS_EXCEEDED

    async def test_an_exhausted_budget_ends_the_run_in_its_own_state(self) -> None:
        """A budget ceiling and a provider outage are different failures."""
        run = await start(
            runner(answer(), budget=FakeBudgetPolicy(limit=1)),
            agent(budget=BudgetConfig(max_tokens_per_run=1)),
        )
        assert run.state is RunState.BUDGET_EXHAUSTED

    async def test_a_budget_that_is_not_breached_records_what_was_spent(self) -> None:
        """A ceiling nobody reports against is a ceiling nobody can raise or lower."""
        policy = FakeBudgetPolicy(limit=1_000)
        run = await start(
            runner(answer(), budget=policy), agent(budget=BudgetConfig(max_tokens_per_run=1_000))
        )
        assert run.state is RunState.COMPLETED
        assert policy.spent == 15

    async def test_every_declared_guardrail_is_consulted(self) -> None:
        """Stopping at the first pass would leave the rest of the chain never running."""
        second = FakeGuardrail("no_prompt_leak")
        run = await start(
            runner(
                answer(),
                guardrails={"no_pii": FakeGuardrail("no_pii"), "no_prompt_leak": second},
            ),
            agent(guardrails=("no_pii", "no_prompt_leak")),
        )
        assert run.state is RunState.COMPLETED
        assert second.checked == ["Kyoto, four nights."]

    async def test_a_run_that_outlives_its_script_fails_rather_than_hanging(self) -> None:
        """The fake refuses to invent a response, so a runaway loop is visible immediately."""
        tools = FakeToolRegistry({"search": lambda **_: "again"})
        calling = ModelResponse(
            tool_calls=(ToolCall(id="call_1", name="search"),),
            usage=Usage(input_tokens=1, output_tokens=1),
        )
        run = await start(runner(calling, tools=tools), agent(tools=("search",)))
        assert run.state is RunState.FAILED
        terminated = run.events[-1]
        assert terminated.detail is not None
        assert "scripted" in terminated.detail

    async def test_a_guardrail_refusal_ends_the_run_and_says_which_one(self) -> None:
        run = await start(
            runner(answer(), guardrails={"no_pii": FakeGuardrail("no_pii", allow=False)}),
            agent(guardrails=("no_pii",)),
        )
        assert run.state is RunState.FAILED
        refusal = next(e for e in run.events if e.kind is RunEventKind.GUARDRAIL_REFUSAL)
        assert refusal.name == "no_pii"

    async def test_a_guardrail_that_cannot_decide_is_a_refusal(self) -> None:
        """Fail closed: a check that did not run is not a check that passed."""
        broken = FakeGuardrail("no_pii", raises=RuntimeError("classifier down"))
        run = await start(
            runner(answer(), guardrails={"no_pii": broken}), agent(guardrails=("no_pii",))
        )
        assert run.state is RunState.FAILED

    async def test_a_cancelled_run_says_so_rather_than_reporting_a_failure(self) -> None:
        """Cancelled and failed are different outcomes and belong in different buckets."""
        run = await start(runner(CancelledError("operator stopped it")), agent())
        assert run.state is RunState.CANCELLED

    async def test_every_path_lands_on_exactly_one_terminal_state(self) -> None:
        """The metric this issue is measured by: zero runs left in an undefined state."""
        empty = ModelResponse(usage=Usage(input_tokens=1, output_tokens=0))
        for scripted in (answer(), ProviderError("500"), empty):
            run = await start(runner(scripted), agent())
            assert run.state.is_terminal


class TestFailingClosed:
    async def test_a_guardrail_the_runner_was_not_given_stops_the_run_before_it_starts(
        self,
    ) -> None:
        """Starting anyway would run the agent with a check it declared and never got."""
        with pytest.raises(ConfigurationError, match="no_pii"):
            await start(runner(answer()), agent(guardrails=("no_pii",)))

    def test_a_guardrail_filed_under_the_wrong_name_is_refused(self) -> None:
        """The agent declares a name; a guardrail answering to another one is not that check."""
        with pytest.raises(ConfigurationError, match="no_pii"):
            runner(answer(), guardrails={"no_pii": FakeGuardrail("no_prompt_leak")})

    async def test_a_declared_budget_without_a_policy_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="budget"):
            await start(runner(answer()), agent(budget=BudgetConfig(max_tokens_per_run=10)))

    async def test_an_agent_routed_by_task_class_is_refused_until_there_is_a_router(self) -> None:
        """Guessing a model would attribute the run to one that never ran it."""
        with pytest.raises(ConfigurationError, match="task_class"):
            await start(runner(answer()), agent(model=None, task_class="reasoning"))

    async def test_the_sync_wrapper_refuses_to_run_inside_a_loop(self) -> None:
        """Nesting one loop in another deadlocks; saying so beats hanging."""
        with pytest.raises(RuntimeError, match="run_sync"):
            runner(answer()).run_sync(agent(), "plan a trip", tenant="acme")

    async def test_cancellation_is_not_swallowed(self) -> None:
        """A cancelled task that returns normally leaves its canceller waiting forever."""
        run = runner(asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await start(run, agent())
