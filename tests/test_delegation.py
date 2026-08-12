"""How far an agent may delegate, and what a child may hold that its parent did not."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tesserix_adk.core import ConfigurationError, DelegationLimitError, ScopeEscalationError
from tesserix_adk.runtime import Delegation, DelegationLimits, DelegationScope
from tesserix_adk.testing import FakeClock

TOOLS = frozenset({"search", "summarise", "file_bug"})


def _root(**overrides: object) -> Delegation:
    """A root delegation with room to move, so a test narrows only what it is about."""
    limits = DelegationLimits(**overrides) if overrides else DelegationLimits()  # type: ignore[arg-type]
    return Delegation.root(
        run_id="run_1",
        tenant="acme",
        agent="supervisor",
        scope=DelegationScope(tools=TOOLS),
        limits=limits,
    )


class TestHowDeep:
    def test_a_child_is_one_deeper_and_says_who_it_came_through(self) -> None:
        child = _root().to("researcher")

        assert child.depth == 1
        assert child.path == ("supervisor", "researcher")

    def test_the_depth_ceiling_refuses_rather_than_creating_the_child(self) -> None:
        deep = _root(max_depth=2).to("a").to("b")

        with pytest.raises(DelegationLimitError) as refused:
            deep.to("c")

        assert refused.value.reason == "depth"
        assert refused.value.path == ("supervisor", "a", "b", "c")

    def test_a_refusal_does_not_spend_the_run_s_allowance(self) -> None:
        parent = _root(max_depth=1)
        parent.to("a")
        before = parent.delegations

        with pytest.raises(DelegationLimitError):
            parent.to("a").to("b")

        assert parent.delegations == before + 1

    def test_the_same_refusal_happens_again_rather_than_letting_a_retry_through(self) -> None:
        parent = _root(max_depth=1).to("a")

        for _ in range(3):
            with pytest.raises(DelegationLimitError):
                parent.to("b")


class TestHowWide:
    def test_one_parent_may_not_exceed_its_fan_out(self) -> None:
        parent = _root(max_fan_out=2)
        parent.to("a")
        parent.to("b")

        with pytest.raises(DelegationLimitError) as refused:
            parent.to("c")

        assert refused.value.reason == "fan_out"

    def test_the_run_s_own_ceiling_bounds_breadth_times_depth(self) -> None:
        parent = _root(max_delegations=3, max_fan_out=8, max_depth=8)
        parent.to("a").to("b")
        parent.to("c")

        with pytest.raises(DelegationLimitError) as refused:
            parent.to("d")

        assert refused.value.reason == "run"

    def test_every_delegation_in_the_tree_counts_against_the_one_run(self) -> None:
        parent = _root()
        parent.to("a").to("b")

        assert parent.delegations == 2


class TestGoingRoundInCircles:
    def test_an_agent_may_not_be_asked_by_something_it_is_already_working_for(self) -> None:
        chain = _root().to("a")

        with pytest.raises(DelegationLimitError) as refused:
            chain.to("supervisor")

        assert refused.value.reason == "cycle"

    def test_alternation_shallower_than_the_depth_ceiling_is_still_a_cycle(self) -> None:
        chain = _root(max_depth=8).to("a").to("b")

        with pytest.raises(DelegationLimitError) as refused:
            chain.to("a")

        assert refused.value.reason == "cycle"

    def test_the_same_agent_on_two_separate_branches_is_not_a_cycle(self) -> None:
        parent = _root()
        parent.to("a").to("worker")

        assert parent.to("b").to("worker").depth == 2


class TestWhatAChildHolds:
    def test_a_child_that_asks_for_nothing_holds_exactly_what_its_parent_held(self) -> None:
        child = _root().to("researcher")

        assert child.scope.tools == TOOLS

    def test_a_child_holds_the_intersection_of_the_parent_and_what_it_needs(self) -> None:
        child = _root().to("researcher", tools={"search"})

        assert child.scope.tools == frozenset({"search"})

    def test_a_tool_the_parent_never_held_is_an_escalation_rather_than_a_grant(self) -> None:
        with pytest.raises(ScopeEscalationError) as refused:
            _root().to("researcher", tools={"search", "wire_transfer"})

        assert refused.value.requested == ("wire_transfer",)
        assert refused.value.path == ("supervisor", "researcher")

    def test_a_mutation_class_the_parent_never_held_is_refused_the_same_way(self) -> None:
        parent = Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="supervisor",
            scope=DelegationScope(tools=TOOLS, mutations=frozenset({"read"})),
        )

        with pytest.raises(ScopeEscalationError, match="write"):
            parent.to("researcher", mutations={"write"})

    def test_a_scope_answers_whether_it_holds_a_tool(self) -> None:
        scope = _root().to("researcher", tools={"search"}).scope

        assert scope.holds("search")
        assert not scope.holds("file_bug")

    def test_a_child_asking_for_no_tool_at_all_is_refused_rather_than_created_inert(self) -> None:
        with pytest.raises(ConfigurationError, match="no tool"):
            _root().to("researcher", tools=set())

    def test_the_child_belongs_to_the_tenant_its_parent_belonged_to(self) -> None:
        child = _root().to("researcher")

        assert child.context.tenant.tenant == "acme"
        assert child.context.run_id == "run_1"

    @given(
        held=st.frozensets(st.sampled_from(sorted(TOOLS)), min_size=1),
        wanted=st.frozensets(st.sampled_from(sorted(TOOLS)), min_size=1),
    )
    def test_a_child_scope_is_never_wider_than_its_parent(
        self, held: frozenset[str], wanted: frozenset[str]
    ) -> None:
        parent = Delegation.root(
            run_id="run_1", tenant="acme", agent="supervisor", scope=DelegationScope(tools=held)
        )

        if wanted <= held:
            assert parent.to("child", tools=wanted).scope.tools <= held
        else:
            with pytest.raises(ScopeEscalationError):
                parent.to("child", tools=wanted)


class TestLimitsAndTime:
    def test_a_child_may_tighten_its_own_ceilings(self) -> None:
        child = _root(max_depth=4).to("a", limits=DelegationLimits(max_depth=2))

        assert child.limits.max_depth == 2

    def test_a_child_asking_for_more_room_keeps_its_parent_s(self) -> None:
        child = _root(max_depth=2).to("a", limits=DelegationLimits(max_depth=9))

        assert child.limits.max_depth == 2

    def test_a_scope_that_has_expired_refuses_rather_than_delegating_on_it(self) -> None:
        clock = FakeClock(start=100.0)
        parent = Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="supervisor",
            scope=DelegationScope(tools=TOOLS, expires_at=99.0),
            clock=clock,
        )

        with pytest.raises(DelegationLimitError) as refused:
            parent.to("researcher")

        assert refused.value.reason == "expired"

    def test_a_scope_still_in_date_delegates(self) -> None:
        parent = Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="supervisor",
            scope=DelegationScope(tools=TOOLS, expires_at=101.0),
            clock=FakeClock(start=100.0),
        )

        assert parent.to("researcher").depth == 1

    def test_ceilings_below_one_could_never_permit_anything(self) -> None:
        with pytest.raises(ConfigurationError, match="max_depth"):
            DelegationLimits(max_depth=0)

    def test_a_scope_that_expires_with_nothing_to_read_it_against_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="clock"):
            Delegation.root(
                run_id="run_1",
                tenant="acme",
                agent="supervisor",
                scope=DelegationScope(tools=TOOLS, expires_at=99.0),
            )


class TestBuildingOneByHand:
    def test_a_delegation_built_directly_is_refused_rather_than_running_unscoped(self) -> None:
        with pytest.raises(ConfigurationError, match="root"):
            Delegation()

    def test_a_scope_holding_no_tool_is_refused_where_it_is_written(self) -> None:
        with pytest.raises(ConfigurationError, match="no tool"):
            DelegationScope(tools=frozenset())
