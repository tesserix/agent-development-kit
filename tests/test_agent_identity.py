"""An agent acts for its caller, so what it holds is never more than what the caller does."""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    Agent,
    AgentIdentity,
    AuthMethod,
    AuthorisationError,
    Principal,
    RunEventKind,
    RunState,
    ScopeSet,
    ToolCall,
    Usage,
    current_principal,
    principal_here,
    principal_scope,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

pytestmark = pytest.mark.anyio

READ = "bookings:read"
WRITE = "bookings:write"


class TestAScopeSet:
    def test_spelling_drift_between_products_is_one_scope(self) -> None:
        assert ScopeSet.of("Bookings:Read") == ScopeSet.of("  bookings:read ")

    def test_it_can_be_narrowed(self) -> None:
        held = ScopeSet.of(READ, WRITE).narrowed_to(ScopeSet.of(READ))

        assert tuple(held) == (READ,)

    def test_narrowing_never_finds_what_neither_side_holds(self) -> None:
        assert not ScopeSet.of(READ).narrowed_to(ScopeSet.of(WRITE))

    def test_nothing_on_it_widens_a_grant(self) -> None:
        widening = [name for name in dir(ScopeSet) if name in {"union", "widened", "__or__"}]

        assert widening == []

    def test_it_reads_the_same_twice(self) -> None:
        assert repr(ScopeSet.of(WRITE, READ)) == "ScopeSet.of('bookings:read', 'bookings:write')"

    def test_it_answers_the_questions_a_set_is_asked(self) -> None:
        held = ScopeSet.of(READ)

        assert READ in held
        assert 7 not in held
        assert len(held) == 1
        assert not ScopeSet()

    def test_it_names_what_is_missing_in_the_order_asked_for(self) -> None:
        assert ScopeSet.of(READ).missing((READ, WRITE)) == (WRITE,)

    def test_a_blank_name_is_not_a_scope(self) -> None:
        assert not ScopeSet.of("", "   ")


class TestAPrincipal:
    def test_it_names_who_is_acting(self) -> None:
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}))

        assert caller.granted == ScopeSet.of(READ)
        assert caller.method is AuthMethod.USER_SESSION

    def test_a_blank_subject_is_not_a_principal(self) -> None:
        with pytest.raises(ValueError, match="names a subject"):
            Principal(subject=" ", tenant="acme")

    def test_a_scheduled_job_is_a_principal_rather_than_an_absence(self) -> None:
        job = Principal(
            subject="nightly-reconcile",
            tenant="acme",
            scopes=frozenset({READ}),
            method=AuthMethod.SCHEDULED_JOB,
        )

        assert job.granted.permits(READ)

    def test_authority_lapses_at_its_expiry(self) -> None:
        caller = Principal(subject="ada", tenant="acme", expires_at=100.0)

        assert not caller.expired(99.0)
        assert caller.expired(100.0)

    def test_authority_with_no_expiry_never_lapses(self) -> None:
        assert not Principal(subject="ada", tenant="acme").expired(1e9)

    def test_a_continuation_runs_on_a_recorded_delegation_with_its_own_expiry(self) -> None:
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ, WRITE}))

        later = caller.delegating(until=500.0)

        assert later.method is AuthMethod.DELEGATED
        assert later.expires_at == 500.0
        assert later.granted == caller.granted

    def test_a_delegation_can_ask_for_less_and_never_for_more(self) -> None:
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}))

        later = caller.delegating(until=500.0, scopes=(READ, WRITE))

        assert later.granted == ScopeSet.of(READ)


class TestResolvingWhatARunHolds:
    def test_the_effective_set_is_the_intersection(self) -> None:
        identity = _identity(declared=(READ, WRITE), held=(READ,))

        assert identity.effective == ScopeSet.of(READ)
        assert identity.declared == ScopeSet.of(READ, WRITE)

    def test_a_caller_holding_more_than_the_agent_declares_does_not_widen_it(self) -> None:
        identity = _identity(declared=(READ,), held=(READ, WRITE))

        assert identity.effective == ScopeSet.of(READ)

    def test_a_lapsed_authority_is_refused_before_the_run_starts(self) -> None:
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}), expires_at=10.0)

        with pytest.raises(AuthorisationError, match="expired"):
            AgentIdentity.resolve(agent="desk", declared=(READ,), principal=caller, now=11.0)

    def test_expiry_is_not_judged_where_no_clock_reading_is_given(self) -> None:
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}), expires_at=10.0)

        assert AgentIdentity.resolve(agent="desk", declared=(READ,), principal=caller).permits(READ)

    def test_a_refusal_names_the_missing_scope_rather_than_failing_generically(self) -> None:
        identity = _identity(declared=(READ, WRITE), held=(READ,))

        with pytest.raises(AuthorisationError) as refused:
            identity.check((WRITE,), where="cancel_booking")

        assert refused.value.scope == WRITE
        assert refused.value.subject == "ada"
        assert "cancel_booking" in str(refused.value)

    def test_holding_what_was_asked_for_is_not_an_event(self) -> None:
        _identity(declared=(READ,), held=(READ,)).check((READ,))

    def test_a_run_holding_nothing_says_so_rather_than_listing_nothing(self) -> None:
        identity = _identity(declared=(WRITE,), held=())

        with pytest.raises(AuthorisationError, match="it holds nothing"):
            identity.check((WRITE,))

    def test_a_run_does_not_outlive_the_session_that_authorised_it(self) -> None:
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}), expires_at=10.0)
        identity = AgentIdentity.resolve(agent="desk", declared=(READ,), principal=caller, now=1.0)

        identity.check_live(9.0)
        with pytest.raises(AuthorisationError, match="lapsed"):
            identity.check_live(10.0, where="search")

    def test_declared_scopes_nothing_reached_for_are_visible_to_trim(self) -> None:
        identity = _identity(declared=(READ, WRITE), held=(READ, WRITE))

        assert identity.unused((READ,)) == (WRITE,)


class TestNarrowingForAPeer:
    def test_a_peer_holds_no_more_than_the_agent_that_called_it(self) -> None:
        desk = _identity(declared=(READ,), held=(READ,))

        peer = desk.narrowed(agent="billing", declared=(READ, WRITE))

        assert peer.effective == ScopeSet.of(READ)
        assert peer.chain == ("desk",)

    def test_a_peer_cannot_re_widen_by_calling_on_again(self) -> None:
        desk = _identity(declared=(READ, WRITE), held=(READ,))

        third = desk.narrowed(agent="billing", declared=(READ,)).narrowed(
            agent="ledger", declared=(READ, WRITE)
        )

        assert third.effective == ScopeSet.of(READ)
        assert third.chain == ("desk", "billing")

    def test_the_peer_acts_for_the_same_caller(self) -> None:
        desk = _identity(declared=(READ,), held=(READ,))

        assert desk.narrowed(agent="billing", declared=(READ,)).principal.subject == "ada"


class TestBindingWhoTheRunActsFor:
    def test_nothing_is_bound_by_default(self) -> None:
        assert principal_here() is None

    def test_the_bound_principal_is_what_is_read_below(self) -> None:
        caller = Principal(subject="ada", tenant="acme")

        with principal_scope(caller):
            assert current_principal() is caller
        assert principal_here() is None

    def test_asking_with_nothing_bound_refuses_rather_than_falling_back(self) -> None:
        with pytest.raises(AuthorisationError, match="none is bound"):
            current_principal(where="a tool call")


class TestAnAgentDeclaringScopes:
    def test_a_tool_may_only_need_scopes_the_agent_declares(self) -> None:
        with pytest.raises(ValueError, match="does not declare"):
            _agent(scopes=(READ,), tool_scopes={"refund": (WRITE,)})

    def test_scopes_may_only_be_required_of_declared_tools(self) -> None:
        with pytest.raises(ValueError, match="not on the allowlist"):
            _agent(scopes=(READ,), tool_scopes={"audit": (READ,)})


class TestTheRunLoopEnforcesIt:
    async def test_a_write_is_refused_where_the_caller_only_holds_read(self) -> None:
        runner = _runner(_asked_for("refund"))
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}))

        with principal_scope(caller):
            run = await runner.run(_privileged(), "refund my fare", tenant="acme", run_id="run_1")

        assert run.state is RunState.FAILED
        assert any(event.kind is RunEventKind.TOOL_REFUSED for event in run.events)

    async def test_the_refused_tool_is_not_declared_to_the_model_either(self) -> None:
        provider = ScriptedProvider(_said("one seat"))
        runner = AgentRunner(provider=provider, tools=_registry(), clock=FakeClock())
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ}))

        with principal_scope(caller):
            await runner.run(_privileged(), "find me a seat", tenant="acme", run_id="run_1")

        assert [tool.name for tool in provider.requests[0].tools] == ["search"]

    async def test_a_caller_holding_the_scope_is_dispatched(self) -> None:
        runner = _runner(_asked_for("refund"), _said("refunded"))
        caller = Principal(subject="ada", tenant="acme", scopes=frozenset({READ, WRITE}))

        with principal_scope(caller):
            run = await runner.run(_privileged(), "refund my fare", tenant="acme", run_id="run_1")

        assert run.state is RunState.COMPLETED

    async def test_a_run_declaring_scopes_with_no_caller_fails_closed(self) -> None:
        runner = _runner(_said("one seat"))

        with pytest.raises(AuthorisationError, match="none is bound"):
            await runner.run(_privileged(), "find me a seat", tenant="acme", run_id="run_1")

    async def test_a_run_whose_caller_has_already_lapsed_never_starts(self) -> None:
        clock = FakeClock()
        runner = AgentRunner(
            provider=ScriptedProvider(_asked_for("search"), _said("one seat")),
            tools=_registry(),
            clock=clock,
        )
        caller = Principal(
            subject="ada",
            tenant="acme",
            scopes=frozenset({READ, WRITE}),
            expires_at=clock.now() + 1.0,
        )
        clock.advance(2.0)

        with principal_scope(caller), pytest.raises(AuthorisationError, match="expired"):
            await runner.run(_privileged(), "find me a seat", tenant="acme", run_id="run_1")

    async def test_an_agent_declaring_no_scopes_runs_as_it_always_did(self) -> None:
        runner = _runner(_asked_for("refund"), _said("refunded"))

        run = await runner.run(_agent(), "refund my fare", tenant="acme", run_id="run_1")

        assert run.state is RunState.COMPLETED


def _identity(*, declared: tuple[str, ...], held: tuple[str, ...]) -> AgentIdentity:
    return AgentIdentity.resolve(
        agent="desk",
        declared=declared,
        principal=Principal(subject="ada", tenant="acme", scopes=frozenset(held)),
    )


def _agent(
    *, scopes: tuple[str, ...] = (), tool_scopes: dict[str, tuple[str, ...]] | None = None
) -> Agent:
    return Agent(
        name="desk",
        instructions="book what was asked for",
        model="claude-sonnet-5",
        free_text=True,
        tools=("search", "refund"),
        scopes=scopes,
        tool_scopes=tool_scopes or {},
    )


def _privileged() -> Agent:
    return _agent(scopes=(READ, WRITE), tool_scopes={"search": (READ,), "refund": (WRITE,)})


def _registry() -> FakeToolRegistry:
    return FakeToolRegistry({"search": lambda **_: "one seat", "refund": lambda **_: "done"})


def _runner(*responses: ModelResponse) -> AgentRunner:
    return AgentRunner(provider=ScriptedProvider(*responses), tools=_registry(), clock=FakeClock())


def _said(text: str) -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=1, output_tokens=1))


def _asked_for(tool: str) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id="call_1", name=tool, arguments={}),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )
