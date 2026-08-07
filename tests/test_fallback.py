"""When the vendor a run was routed to will not answer, and what may be done about it.

A fallback that is silent is worse than a failure: nobody can tell afterwards which model
answered, whether a tool ran twice, or why the bill has two calls on it for one question.
So every attempt is on the record, a fallback only happens after that vendor's own retries
are spent, and it never happens where it could repeat a side effect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    CHEAP,
    Agent,
    AuthenticationError,
    BudgetExceededError,
    Capability,
    CapabilityError,
    ContentFilteredError,
    ContextWindowExceededError,
    FallbackChain,
    InvalidRequestError,
    ModelCapabilities,
    ModelRef,
    ModelRequest,
    ModelRequirements,
    ModelResponseError,
    ModelSpec,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    RetryConfig,
    RunEventKind,
    RunState,
    StreamInterruptedError,
    ToolCall,
    Usage,
    fallback_eligible,
)
from tesserix_adk.models.routing import RoutingRule, RoutingTable, TableRouter
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse
from tesserix_adk.testing import FakeBudgetPolicy, FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

CAPABLE = ModelCapabilities(
    tool_calling=True, streaming=True, vision=True, context_window_tokens=200_000
)
NARROW = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=1)


def spec(ref: str, capabilities: ModelCapabilities = CAPABLE) -> ModelSpec:
    provider, _, model = ref.partition(":")
    return ModelSpec(provider=provider, model=model, capabilities=capabilities)


def table(*candidates: ModelSpec) -> RoutingTable:
    return RoutingTable(rules=(RoutingRule(task_class=CHEAP, candidates=candidates),))


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def limited() -> RateLimitError:
    return RateLimitError("slow down", provider="openai", model="gpt-4o-mini")


def runner(
    routing: RoutingTable,
    fleet: dict[str, ScriptedProvider],
    **overrides: Any,
) -> AgentRunner:
    fields: dict[str, Any] = {
        "provider": next(iter(fleet.values())),
        "providers": fleet,
        "router": TableRouter(routing),
        "retry": RetryConfig(max_attempts=1),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})


def agent(**overrides: Any) -> Agent:
    fields: dict[str, Any] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "task_class": CHEAP,
    }
    return Agent(**{**fields, **overrides})


class Cancelling(ScriptedProvider):
    """A provider that flips the caller's switch on its way to failing."""

    def __init__(
        self,
        token: CancellationToken,
        *responses: ModelResponse | BaseException,
        name: str = "scripted",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        super().__init__(*responses, name=name, capabilities=capabilities)
        self._token = token

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Cancel, then fail, so the chain is asked to continue under a cancelled token."""
        self._token.cancel("the caller changed their mind")
        return await super().complete(request)


def fleet_of(**scripts: Sequence[ModelResponse | BaseException]) -> dict[str, ScriptedProvider]:
    return {
        name: ScriptedProvider(*script, name=name, capabilities=CAPABLE)
        for name, script in scripts.items()
    }


def terminated(run: Any) -> str:
    return next(
        event.detail or ""
        for event in reversed(run.events)
        if event.kind is RunEventKind.TERMINATED
    )


class TestTheChainIsWhatTheTableAlreadySaid:
    """A fallback order invented separately from the routing order is a second opinion on
    the same question, and the two drift."""

    def test_every_eligible_candidate_is_in_the_chain_in_written_order(self) -> None:
        decided = TableRouter(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5"))
        ).resolve(CHEAP)
        assert decided.chain == ("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5")

    def test_the_chosen_model_is_the_head_of_the_chain(self) -> None:
        decided = TableRouter(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5"))
        ).resolve(CHEAP)
        assert decided.chain[0] == str(decided.chosen)

    def test_a_candidate_that_cannot_do_the_work_is_not_a_fallback_for_it(self) -> None:
        blind = ModelCapabilities(tool_calling=True, context_window_tokens=1_000)
        decided = TableRouter(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5", blind))
        ).resolve(CHEAP, requirements=None)
        assert decided.chain == ("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5")

    def test_the_capability_floor_holds_for_the_whole_chain(self) -> None:
        """A fallback that quietly loses vision answers a different question."""
        blind = ModelCapabilities(tool_calling=True, context_window_tokens=1_000)
        decided = TableRouter(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5", blind))
        ).resolve(
            CHEAP, requirements=ModelRequirements(capabilities=frozenset({Capability.VISION}))
        )
        assert decided.chain == ("openai:gpt-4o-mini",)

    def test_a_pin_has_no_fallback(self) -> None:
        """A pin names the model on purpose; falling off it answers a different question."""
        decided = TableRouter(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5"))
        ).resolve(CHEAP, pinned=ModelRef(provider="openai", model="gpt-4o-mini"))
        assert decided.chain == ("openai:gpt-4o-mini",)


class TestWhichFailuresAreWorthAnotherVendor:
    @pytest.mark.parametrize(
        "failure",
        [
            RateLimitError("slow down"),
            RateLimitError("out of credit", quota=True),
            ProviderUnavailableError("overloaded"),
            ProviderTimeoutError("no answer"),
        ],
    )
    def test_a_failure_of_this_vendor_may_be_asked_of_another(self, failure: ProviderError) -> None:
        assert fallback_eligible(failure)

    def test_a_spent_quota_is_not_retryable_but_is_another_vendors_to_answer(self) -> None:
        """The two questions are different: waiting will not help, another allowance will."""
        spent = RateLimitError("out of credit", quota=True)
        assert not spent.retryable
        assert fallback_eligible(spent)

    @pytest.mark.parametrize(
        "failure",
        [
            AuthenticationError("bad key"),
            InvalidRequestError("no such model"),
            ContentFilteredError("refused"),
            ContextWindowExceededError("too long"),
            CapabilityError("cannot do it"),
            BudgetExceededError("no money left"),
            ModelResponseError("that is not a response"),
        ],
    )
    def test_a_failure_another_vendor_would_repeat_is_terminal(self, failure: Exception) -> None:
        assert not fallback_eligible(failure)

    def test_a_stream_that_already_emitted_is_the_callers_to_restart(self) -> None:
        """Restarting under a consumer that has seen half an answer shows it two answers."""
        assert not fallback_eligible(StreamInterruptedError("cut off", partial="Kyo"))

    def test_an_unmapped_failure_does_not_open_a_second_bill(self) -> None:
        assert not fallback_eligible(RuntimeError("who knows"))


class TestFallingBackToTheNextVendor:
    async def test_the_secondary_answers_when_the_primary_will_not(self) -> None:
        fleet = fleet_of(openai=[limited()], anthropic=[answer()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")), fleet
        ).run(agent(), "plan a trip", tenant="acme")
        assert run.state is RunState.COMPLETED
        assert run.model == "claude-haiku-4-5"

    async def test_the_run_names_the_model_that_actually_answered(self) -> None:
        fleet = fleet_of(openai=[limited()], anthropic=[answer()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")), fleet
        ).run(agent(), "plan a trip", tenant="acme")
        fell_back = [e for e in run.events if e.kind is RunEventKind.MODEL_FELL_BACK]
        assert [event.name for event in fell_back] == ["anthropic:claude-haiku-4-5"]
        assert "RateLimitError" in (fell_back[0].detail or "")

    async def test_both_attempts_are_on_the_record_with_their_error_classes(self) -> None:
        fleet = fleet_of(openai=[limited()], anthropic=[answer()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")), fleet
        ).run(agent(), "plan a trip", tenant="acme")
        failed = [e for e in run.events if e.kind is RunEventKind.ATTEMPT_FAILED]
        assert [event.name for event in failed] == ["gpt-4o-mini"]
        assert "RateLimitError" in (failed[0].detail or "")

    async def test_the_chain_waits_for_this_vendors_retries_to_be_spent(self) -> None:
        """Leaving a vendor on the first 429 gives up an allowance that was about to clear."""
        fleet = fleet_of(openai=[limited(), limited(), answer()], anthropic=[answer()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")),
            fleet,
            retry=RetryConfig(max_attempts=3),
        ).run(agent(), "plan a trip", tenant="acme")
        assert run.model == "gpt-4o-mini"
        assert not fleet["anthropic"].requests

    async def test_a_terminal_failure_never_starts_a_second_bill(self) -> None:
        fleet = fleet_of(openai=[AuthenticationError("bad key")], anthropic=[answer()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")), fleet
        ).run(agent(), "plan a trip", tenant="acme")
        assert run.state is RunState.FAILED
        assert not fleet["anthropic"].requests

    async def test_a_run_with_no_fallback_configured_fails_as_it_did_before(self) -> None:
        fleet = fleet_of(openai=[limited()])
        run = await runner(table(spec("openai:gpt-4o-mini")), fleet).run(
            agent(), "plan a trip", tenant="acme"
        )
        assert run.state is RunState.FAILED
        assert "RateLimitError" in terminated(run)


class TestWhenEveryVendorRefuses:
    async def test_the_failure_names_every_attempt_rather_than_the_last(self) -> None:
        fleet = fleet_of(
            openai=[limited()], anthropic=[ProviderUnavailableError("overloaded", status=529)]
        )
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")), fleet
        ).run(agent(), "plan a trip", tenant="acme")
        assert run.state is RunState.FAILED
        detail = terminated(run)
        assert "FallbackExhaustedError" in detail
        assert "openai:gpt-4o-mini" in detail
        assert "anthropic:claude-haiku-4-5" in detail

    async def test_a_vendor_that_already_failed_is_not_asked_twice(self) -> None:
        """A table may list one model in two rules; a chain that loops is a loop."""
        fleet = fleet_of(openai=[limited()], anthropic=[limited()])
        chain = FallbackChain(links=("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"))
        assert chain.after("openai:gpt-4o-mini", failed={"anthropic:claude-haiku-4-5"}) is None
        assert len(fleet) == 2

    async def test_a_vendor_the_runner_was_not_given_is_not_a_fallback(self) -> None:
        fleet = fleet_of(openai=[limited()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")), fleet
        ).run(agent(), "plan a trip", tenant="acme")
        assert run.state is RunState.FAILED
        assert "anthropic" in terminated(run)


class TestTheChainDoesNotEvadeTheRunsLimits:
    async def test_a_failed_attempt_still_spends_the_budget_it_reserved(self) -> None:
        """Otherwise a chain is a way to buy two calls against a ceiling that permits one."""
        budget = FakeBudgetPolicy()
        fleet = fleet_of(openai=[limited()], anthropic=[answer()])
        await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")),
            fleet,
            budget=budget,
        ).run(agent(), "plan a trip", tenant="acme")
        assert len(budget.reservations) == 2

    async def test_a_candidate_that_cannot_hold_the_prompt_is_skipped(self) -> None:
        fleet = {
            "openai": ScriptedProvider(limited(), name="openai", capabilities=CAPABLE),
            "anthropic": ScriptedProvider(answer(), name="anthropic", capabilities=NARROW),
            "vllm": ScriptedProvider(answer(), name="vllm", capabilities=CAPABLE),
        }
        run = await runner(
            table(
                spec("openai:gpt-4o-mini"),
                spec("anthropic:claude-haiku-4-5"),
                spec("vllm:qwen"),
            ),
            fleet,
        ).run(agent(), "plan a trip", tenant="acme")
        assert run.model == "qwen"
        assert not fleet["anthropic"].requests

    async def test_cancellation_between_candidates_stops_the_chain(self) -> None:
        token = CancellationToken()
        fleet: dict[str, ScriptedProvider] = {
            "openai": Cancelling(token, limited(), name="openai", capabilities=CAPABLE),
            "anthropic": ScriptedProvider(answer(), name="anthropic", capabilities=CAPABLE),
        }
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")), fleet
        ).run(agent(), "plan a trip", tenant="acme", cancellation=token)
        assert run.state is RunState.CANCELLED
        assert not fleet["anthropic"].requests


class TestASideEffectThatMustNotHappenTwice:
    def tools(self) -> FakeToolRegistry:
        return FakeToolRegistry({"charge": lambda **_: "charged", "weather": lambda **_: "clear"})

    def calling(self, name: str) -> ModelResponse:
        return ModelResponse(
            content="",
            tool_calls=(ToolCall(id="call_1", name=name, arguments={"city": "Delhi"}),),
            usage=Usage(input_tokens=5, output_tokens=1),
        )

    async def test_a_run_that_already_ran_a_tool_does_not_fall_back(self) -> None:
        registry = self.tools()
        fleet = fleet_of(openai=[self.calling("charge"), limited()], anthropic=[answer()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")),
            fleet,
            tools=registry,
        ).run(agent(tools=("charge",)), "charge the card", tenant="acme")
        assert run.state is RunState.FAILED
        assert "FallbackUnsafeError" in terminated(run)
        assert [name for name, _ in registry.calls] == ["charge"]
        assert not fleet["anthropic"].requests

    async def test_an_idempotent_tool_does_not_block_the_fallback(self) -> None:
        registry = self.tools()
        fleet = fleet_of(openai=[self.calling("weather"), limited()], anthropic=[answer()])
        run = await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")),
            fleet,
            tools=registry,
        ).run(
            agent(tools=("weather",), idempotent_tools=("weather",)),
            "what is the weather",
            tenant="acme",
        )
        assert run.state is RunState.COMPLETED
        assert run.model == "claude-haiku-4-5"

    async def test_the_recorded_result_is_replayed_rather_than_the_tool_re_invoked(self) -> None:
        registry = self.tools()
        fleet = fleet_of(openai=[self.calling("weather"), limited()], anthropic=[answer()])
        await runner(
            table(spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")),
            fleet,
            tools=registry,
        ).run(
            agent(tools=("weather",), idempotent_tools=("weather",)),
            "what is the weather",
            tenant="acme",
        )
        assert [name for name, _ in registry.calls] == ["weather"]
        assert any("weather" in str(m.content) for m in fleet["anthropic"].requests[0].messages)
