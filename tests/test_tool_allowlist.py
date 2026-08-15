"""Resolving what a run may call, once, and refusing everything else at dispatch."""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    Agent,
    ConfigurationError,
    DenyReason,
    RunEventKind,
    RunState,
    TenantLimits,
    TenantPolicy,
    ToolAllowlist,
    ToolCall,
    ToolNotPermittedError,
    Usage,
    canonical,
    tenant_policy,
)
from tesserix_adk.guardrails import ToolAllowlistGuard
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

pytestmark = pytest.mark.anyio

DECLARED = ("search", "book", "refund")


class TestNormalisingAName:
    def test_case_and_width_do_not_make_a_second_tool(self) -> None:
        wide = "".join(chr(0xFF41 + ord(letter) - ord("a")) for letter in "search")

        assert canonical("Search") == canonical(wide) == "search"

    def test_surrounding_space_is_not_part_of_the_name(self) -> None:
        assert canonical("  book\n") == "book"

    def test_a_declaration_that_names_one_tool_twice_under_two_spellings_is_refused(self) -> None:
        with pytest.raises(ConfigurationError) as refused:
            ToolAllowlist.resolve(("search", "Search"))

        assert "search" in str(refused.value)

    def test_a_namespaced_tool_is_not_the_native_one_of_the_same_stem(self) -> None:
        allowlist = ToolAllowlist.resolve(("search", "mcp:search"))

        assert allowlist.permits("mcp:search")
        assert allowlist.permits("search")


class TestResolvingTheThreeSources:
    def test_a_declaration_alone_is_the_allowlist(self) -> None:
        assert ToolAllowlist.resolve(DECLARED).names == DECLARED

    def test_a_source_that_states_nothing_narrows_nothing(self) -> None:
        assert ToolAllowlist.resolve(DECLARED, tenant=None, caller=None).names == DECLARED

    def test_a_tenant_narrows_and_never_widens(self) -> None:
        allowlist = ToolAllowlist.resolve(DECLARED, tenant={"search", "book", "settle"})

        assert allowlist.names == ("search", "book")
        assert not allowlist.permits("settle")

    def test_caller_scopes_narrow_what_the_tenant_left(self) -> None:
        allowlist = ToolAllowlist.resolve(DECLARED, tenant={"search", "book"}, caller={"search"})

        assert allowlist.names == ("search",)

    def test_a_source_stating_nothing_is_not_a_source_permitting_everything(self) -> None:
        assert ToolAllowlist.resolve(DECLARED, tenant=frozenset()).names == ()

    def test_an_empty_allowlist_is_valid_and_refuses_the_lot(self) -> None:
        allowlist = ToolAllowlist.resolve((), agent="reader")

        assert allowlist.names == ()
        assert not allowlist.permits("search")

    def test_the_sources_are_compared_after_normalisation(self) -> None:
        assert ToolAllowlist.resolve(DECLARED, tenant={"SEARCH"}).names == ("search",)

    def test_declaration_order_survives_so_the_prompt_prefix_is_stable(self) -> None:
        assert ToolAllowlist.resolve(DECLARED, tenant={"refund", "search"}).names == (
            "search",
            "refund",
        )

    def test_it_cannot_be_appended_to_once_resolved(self) -> None:
        allowlist = ToolAllowlist.resolve(DECLARED)

        with pytest.raises(AttributeError):
            allowlist.names = (*DECLARED, "settle")  # type: ignore[misc]


class TestWhySomethingWasRefused:
    def test_a_tool_nobody_declared_says_so(self) -> None:
        decision = ToolAllowlist.resolve(DECLARED).decide("settle")

        assert not decision.permitted
        assert decision.reason is DenyReason.NOT_DECLARED

    def test_a_tool_the_tenant_cut_says_it_was_the_tenant(self) -> None:
        allowlist = ToolAllowlist.resolve(DECLARED, tenant={"search"})

        assert allowlist.decide("refund").reason is DenyReason.TENANT

    def test_a_tool_the_caller_lacks_the_scope_for_says_it_was_the_caller(self) -> None:
        allowlist = ToolAllowlist.resolve(DECLARED, caller={"search"})

        assert allowlist.decide("book").reason is DenyReason.CALLER

    def test_the_narrowest_source_is_the_one_named(self) -> None:
        allowlist = ToolAllowlist.resolve(DECLARED, tenant={"search"}, caller={"search", "book"})

        assert allowlist.decide("book").reason is DenyReason.TENANT

    def test_a_permitted_tool_carries_no_reason(self) -> None:
        decision = ToolAllowlist.resolve(DECLARED).decide("search")

        assert decision.permitted
        assert decision.reason is None

    def test_the_refusal_names_the_tool_asked_for_and_the_reason(self) -> None:
        allowlist = ToolAllowlist.resolve(DECLARED, tenant={"search"}, agent="concierge")

        with pytest.raises(ToolNotPermittedError) as refused:
            allowlist.check("Refund")

        assert refused.value.tool == "Refund"
        assert refused.value.details["reason"] == "tenant"
        assert refused.value.agent == "concierge"

    def test_a_permitted_tool_passes_the_check_under_any_spelling(self) -> None:
        ToolAllowlist.resolve(DECLARED).check("SEARCH")


class TestDelegatingToAPeer:
    def test_a_peer_is_held_to_the_intersection_and_not_its_own_declaration(self) -> None:
        parent = ToolAllowlist.resolve(DECLARED, tenant={"search"})

        peer = parent.narrowed(("search", "book", "settle"), agent="pricing")

        assert peer.names == ("search",)
        assert peer.agent == "pricing"

    def test_a_peer_that_declares_more_cannot_proxy_what_the_parent_lost(self) -> None:
        parent = ToolAllowlist.resolve(("search",))

        assert parent.narrowed(("refund",)).decide("refund").reason is DenyReason.CALLER

    def test_a_chain_of_delegations_only_ever_shrinks(self) -> None:
        parent = ToolAllowlist.resolve(DECLARED)

        chain = parent.narrowed(("search", "book")).narrowed(("book", "refund"))

        assert chain.names == ("book",)


class TestTheGuardAtDispatch:
    def test_it_resolves_the_same_three_sources(self) -> None:
        guard = ToolAllowlistGuard.resolving(DECLARED, tenant={"search"}, agent="concierge")

        assert guard.allowlist.names == ("search",)
        assert guard.name == "tool_allowlist"

    def test_a_refused_call_counts_as_an_attempt_against_the_run(self) -> None:
        guard = ToolAllowlistGuard.resolving(DECLARED, tenant={"search"})

        guard.check("search")
        with pytest.raises(ToolNotPermittedError):
            guard.check("refund")

        assert guard.attempts == 2
        assert guard.refusals == 1

    def test_it_tells_the_model_only_about_what_survived(self) -> None:
        guard = ToolAllowlistGuard.resolving(DECLARED, tenant={"search", "book"})

        assert guard.permitted(DECLARED) == ("search", "book")
        assert guard.permitted(("refund", "search")) == ("search",)

    def test_a_peer_guard_starts_from_the_parent_allowlist(self) -> None:
        guard = ToolAllowlistGuard.resolving(DECLARED, caller={"search", "book"})

        peer = guard.delegating(("book", "refund"), agent="pricing")

        assert peer.allowlist.names == ("book",)
        assert peer.attempts == 0

    def test_an_approval_gated_tool_is_refused_before_anyone_is_asked(self) -> None:
        agent = Agent(
            name="concierge",
            instructions="book what was asked for",
            model="gpt-5",
            free_text=True,
            tools=DECLARED,
            approval_required_tools=("refund",),
        )
        guard = ToolAllowlistGuard.resolving(agent.tools, tenant={"search"}, agent=agent.name)

        with pytest.raises(ToolNotPermittedError):
            guard.check("refund")

    def test_a_tenant_policy_supplies_the_middle_source(self) -> None:
        policy = TenantPolicy(tenant="acme", limits=TenantLimits(tools=frozenset({"search"})))
        guard = ToolAllowlistGuard.resolving(DECLARED, tenant=policy.limits.tools)

        assert guard.allowlist.names == ("search",)


def _agent() -> Agent:
    return Agent(
        name="concierge",
        instructions="book what was asked for",
        model="claude-sonnet-5",
        free_text=True,
        tools=("search", "refund"),
    )


def _runner(*responses: ModelResponse) -> AgentRunner:
    return AgentRunner(
        provider=ScriptedProvider(*responses),
        tools=FakeToolRegistry({"search": lambda **_: "one seat", "refund": lambda **_: "done"}),
        clock=FakeClock(),
    )


def _said(text: str) -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=1, output_tokens=1))


def _asked_for(tool: str) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id="call_1", name=tool, arguments={}),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


class TestTheRunLoopEnforcesIt:
    async def test_a_tool_the_tenant_excludes_is_refused_though_the_agent_declares_it(
        self,
    ) -> None:
        policy = TenantPolicy(tenant="acme", limits=TenantLimits(tools=frozenset({"search"})))
        runner = _runner(_asked_for("refund"))

        with tenant_policy(policy):
            run = await runner.run(_agent(), "refund my fare", tenant="acme", run_id="run_1")

        assert run.state is RunState.FAILED
        assert any(event.kind is RunEventKind.TOOL_REFUSED for event in run.events)

    async def test_the_model_is_only_told_about_what_survived_the_tenant(self) -> None:
        policy = TenantPolicy(tenant="acme", limits=TenantLimits(tools=frozenset({"search"})))
        provider = ScriptedProvider(_said("one seat"))
        runner = AgentRunner(
            provider=provider,
            tools=FakeToolRegistry({"search": lambda **_: "one seat", "refund": lambda **_: "x"}),
            clock=FakeClock(),
        )

        with tenant_policy(policy):
            await runner.run(_agent(), "find me a seat", tenant="acme", run_id="run_1")

        assert [tool.name for tool in provider.requests[0].tools] == ["search"]

    async def test_without_a_tenant_policy_the_declaration_is_the_allowlist(self) -> None:
        runner = _runner(_asked_for("refund"), _said("refunded"))

        run = await runner.run(_agent(), "refund my fare", tenant="acme", run_id="run_1")

        assert run.state is RunState.COMPLETED
