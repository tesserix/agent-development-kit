"""Resilience that never weakens a data-handling guarantee, and routing that says why.

A fallback chain that crosses from a self-hosted model to a hosted vendor, or from a sealed
tier to a standard one, is a data-handling breach dressed up as availability. It has to
fail closed. And a model choice nobody can explain after the fact is a bill nobody can
explain after the fact, so the inputs that produced the choice are recorded with it.
"""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    CHEAP,
    Agent,
    Capability,
    FallbackChain,
    ModelCapabilities,
    ModelRequirements,
    ModelSpec,
    ProviderUnavailableError,
    RetryConfig,
    RunEventKind,
    RunState,
    TrustBoundary,
    TrustBoundaryError,
)
from tesserix_adk.models.routing import RoutingRule, RoutingTable, TableRouter
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

SEALED = TrustBoundary(tier="sealed", hosting="self-hosted", residency="in-central")
STANDARD = TrustBoundary(tier="standard", hosting="self-hosted", residency="in-central")
VENDOR = TrustBoundary(tier="standard", hosting="vendor-api", residency="us")

SMALL = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=8_000)
BIG = ModelCapabilities(
    tool_calling=True,
    structured_output=True,
    streaming=True,
    context_window_tokens=200_000,
)


def spec(
    ref: str, trust: TrustBoundary | None = None, caps: ModelCapabilities = SMALL
) -> ModelSpec:
    provider, _, model = ref.partition(":")
    return ModelSpec(
        provider=provider, model=model, capabilities=caps, trust=trust or TrustBoundary()
    )


def _strings(value: object) -> set[str]:
    """Every string anywhere in a serialised decision, however deeply nested."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {s for item in value.values() for s in _strings(item)}
    if isinstance(value, list | tuple | set | frozenset):
        return {s for item in value for s in _strings(item)}
    return set()


def router(*candidates: ModelSpec) -> TableRouter:
    return TableRouter(RoutingTable(rules=(RoutingRule(task_class=CHEAP, candidates=candidates),)))


class TestWhatABoundaryAdmits:
    def test_the_same_boundary_admits_itself(self) -> None:
        assert SEALED.admits(SEALED)

    def test_a_different_tier_is_not_the_same_boundary(self) -> None:
        """Degrading a sealed matter to the standard tier is a breach, not a fallback."""
        assert not SEALED.admits(STANDARD)

    def test_different_hosting_is_not_the_same_boundary(self) -> None:
        assert not STANDARD.admits(VENDOR)

    def test_the_axes_that_differ_are_named(self) -> None:
        """'Not equivalent' leaves the operator to work out which of three axes to change."""
        assert STANDARD.differs_from(VENDOR) == ("hosting", "residency")

    def test_a_target_that_states_nothing_is_refused_by_a_source_that_states_something(
        self,
    ) -> None:
        """An undeclared boundary is an unknown one, and unknown is not equal."""
        assert not SEALED.admits(TrustBoundary())

    def test_a_source_that_states_nothing_constrains_nothing(self) -> None:
        """The kit cannot enforce a boundary nobody declared; it says so rather than guesses."""
        assert TrustBoundary().admits(VENDOR)
        assert not TrustBoundary().stated


class TestTheChainCannotLeaveTheBoundary:
    def test_a_candidate_outside_the_boundary_is_not_a_link(self) -> None:
        decision = router(spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR)).resolve(
            CHEAP
        )
        assert decision.chain == ("vllm:qwen",)

    def test_what_the_boundary_excluded_is_recorded_rather_than_dropped_quietly(self) -> None:
        decision = router(spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR)).resolve(
            CHEAP
        )
        assert decision.excluded_by_boundary == ("openai:gpt-4o-mini",)
        (rejected,) = [c for c in decision.rejected if c.ref == "openai:gpt-4o-mini"]
        assert "tier" in rejected.reason
        assert "hosting" in rejected.reason

    def test_an_equivalent_candidate_stays_in_the_chain(self) -> None:
        decision = router(spec("vllm:qwen", SEALED), spec("vllm:qwen-2", SEALED)).resolve(CHEAP)
        assert decision.chain == ("vllm:qwen", "vllm:qwen-2")

    def test_an_undeclared_deployment_routes_as_it_did_before(self) -> None:
        """Nothing declared means nothing enforced, not everything refused."""
        decision = router(spec("vllm:qwen"), spec("openai:gpt-4o-mini")).resolve(CHEAP)
        assert decision.chain == ("vllm:qwen", "openai:gpt-4o-mini")
        assert decision.excluded_by_boundary == ()

    def test_a_fallback_that_would_lose_structured_output_is_not_a_link(self) -> None:
        """Trust is not the only floor: a legal target that cannot do the work is not one."""
        decision = router(
            spec("vllm:qwen", STANDARD, BIG), spec("vllm:qwen-2", STANDARD, SMALL)
        ).resolve(
            CHEAP,
            requirements=ModelRequirements(capabilities=frozenset({Capability.STRUCTURED_OUTPUT})),
        )
        assert decision.chain == ("vllm:qwen",)


class TestFailingClosed:
    def test_the_chain_refuses_the_link_the_boundary_excluded(self) -> None:
        chain = FallbackChain(links=("vllm:qwen",), excluded=("openai:gpt-4o-mini",))
        assert chain.after("vllm:qwen") is None
        assert chain.excluded == ("openai:gpt-4o-mini",)

    def test_asking_for_the_refused_link_by_name_raises_rather_than_returning_it(self) -> None:
        chain = FallbackChain(links=("vllm:qwen",), excluded=("openai:gpt-4o-mini",))
        with pytest.raises(TrustBoundaryError, match="openai:gpt-4o-mini"):
            chain.refuse_the_excluded()

    def test_the_refusal_names_what_it_would_not_send_to(self) -> None:
        chain = FallbackChain(
            links=("vllm:qwen",), excluded=("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5")
        )
        with pytest.raises(TrustBoundaryError) as refused:
            chain.refuse_the_excluded()
        assert refused.value.excluded == ("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5")

    def test_a_chain_with_nothing_excluded_refuses_nothing(self) -> None:
        FallbackChain(links=("vllm:qwen",)).refuse_the_excluded()

    def test_a_run_that_was_never_routed_has_no_chain_and_nothing_excluded(self) -> None:
        assert FallbackChain.of(None) == FallbackChain()

    def test_the_chain_carries_the_boundary_forward_from_the_decision(self) -> None:
        decision = router(spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR)).resolve(
            CHEAP
        )
        chain = FallbackChain.of(decision)
        assert chain.links == ("vllm:qwen",)
        assert chain.excluded == ("openai:gpt-4o-mini",)


class TestTheRecordedRationale:
    def test_the_inputs_that_chose_the_model_are_recorded_with_it(self) -> None:
        decision = router(spec("vllm:qwen", STANDARD, BIG)).resolve(
            CHEAP,
            requirements=ModelRequirements(
                capabilities=frozenset({Capability.STRUCTURED_OUTPUT}),
                min_context_window_tokens=32_000,
            ),
        )
        assert decision.required == ("structured_output",)
        assert decision.min_context_window_tokens == 32_000
        assert decision.boundary == STANDARD

    def test_the_rationale_explains_the_choice_in_one_line(self) -> None:
        decision = router(spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR)).resolve(
            CHEAP
        )
        explained = decision.explain()
        assert "vllm:qwen" in explained
        assert "1 excluded by trust boundary" in explained

    def test_the_rationale_is_drawn_from_a_closed_vocabulary(self) -> None:
        """A trace that could quote the prompt is a trace nobody may keep for a sealed matter."""
        decision = router(spec("vllm:qwen", SEALED), spec("openai:gpt-4o-mini", VENDOR)).resolve(
            CHEAP, requirements=ModelRequirements(capabilities=frozenset({Capability.TOOL_CALLING}))
        )
        allowed = {
            "vllm:qwen",
            "vllm",
            "qwen",
            "openai:gpt-4o-mini",
            "cheap",
            "cheap@*/*",
            "tool_calling",
            "sealed",
            "self-hosted",
            "in-central",
            "",
        }
        assert _strings(decision.model_dump()) - allowed == {
            "outside the trust boundary: tier, hosting, residency"
        }


class TestARunThatCannotStayInsideTheBoundary:
    async def test_the_run_fails_closed_rather_than_reaching_the_vendor(self) -> None:
        """The self-hosted endpoint is down and the only alternative is out of boundary."""
        fleet = {
            "vllm": ScriptedProvider(
                ProviderUnavailableError("the endpoint is down", status=503),
                name="vllm",
                capabilities=BIG,
            ),
            "openai": ScriptedProvider(
                ModelResponse(content="Kyoto."), name="openai", capabilities=BIG
            ),
        }
        run = await AgentRunner(
            provider=fleet["vllm"],
            providers=fleet,
            router=router(spec("vllm:qwen", SEALED, BIG), spec("openai:gpt-4o-mini", VENDOR, BIG)),
            retry=RetryConfig(max_attempts=1),
            clock=FakeClock(),
        ).run(
            Agent(name="clerk", instructions="Cite the page.", free_text=True, task_class=CHEAP),
            "what does page 12 say?",
            tenant="acme",
        )
        assert run.state is RunState.FAILED
        assert not fleet["openai"].requests

    async def test_the_refusal_says_it_was_the_boundary_and_names_what_it_would_not_use(
        self,
    ) -> None:
        fleet = {
            "vllm": ScriptedProvider(
                ProviderUnavailableError("the endpoint is down", status=503),
                name="vllm",
                capabilities=BIG,
            ),
            "openai": ScriptedProvider(
                ModelResponse(content="Kyoto."), name="openai", capabilities=BIG
            ),
        }
        run = await AgentRunner(
            provider=fleet["vllm"],
            providers=fleet,
            router=router(spec("vllm:qwen", SEALED, BIG), spec("openai:gpt-4o-mini", VENDOR, BIG)),
            retry=RetryConfig(max_attempts=1),
            clock=FakeClock(),
        ).run(
            Agent(name="clerk", instructions="Cite the page.", free_text=True, task_class=CHEAP),
            "what does page 12 say?",
            tenant="acme",
        )
        detail = next(
            event.detail or ""
            for event in reversed(run.events)
            if event.kind is RunEventKind.TERMINATED
        )
        assert "TrustBoundaryError" in detail
        assert "openai:gpt-4o-mini" in detail

    async def test_an_equivalent_alternative_still_answers(self) -> None:
        """Failing closed is about the boundary, not about refusing to fall back at all."""
        fleet = {
            "vllm": ScriptedProvider(
                ProviderUnavailableError("the endpoint is down", status=503),
                ModelResponse(content="Kyoto."),
                name="vllm",
                capabilities=BIG,
            )
        }
        run = await AgentRunner(
            provider=fleet["vllm"],
            providers=fleet,
            router=router(spec("vllm:qwen", SEALED, BIG), spec("vllm:qwen-2", SEALED, BIG)),
            retry=RetryConfig(max_attempts=1),
            clock=FakeClock(),
        ).run(
            Agent(name="clerk", instructions="Cite the page.", free_text=True, task_class=CHEAP),
            "what does page 12 say?",
            tenant="acme",
        )
        assert run.state is RunState.COMPLETED
        assert run.model == "qwen-2"
