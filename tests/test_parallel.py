"""Several branches at once, bounded, attributed, ordered, and never silently partial.

Hand-rolled `asyncio.gather` gets the same three things wrong every time: nothing caps how
many branches are in flight, nothing stops one branch draining the run's ledger, and a
result built from three branches out of five reads exactly like one built from five. This
file is the counter-argument.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    AggregationError,
    BudgetLimits,
    BudgetScope,
    ConfigurationError,
    ScopedLimits,
    Usage,
    most_restrictive,
)
from tesserix_adk.core.budget import RunBudget
from tesserix_adk.core.guards import GuardrailPipeline, GuardResult
from tesserix_adk.runtime import (
    All,
    Branch,
    BranchOutcome,
    BranchResult,
    FirstSuccess,
    ModelResponse,
    Quorum,
    Reduce,
    Roster,
    Specialist,
    Supervisor,
    fan_out,
)
from tesserix_adk.runtime.delegation import Delegation, DelegationLimits, DelegationScope
from tesserix_adk.runtime.loop import AgentRunner
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from tesserix_adk.core.provider import ModelRequest

HELD = frozenset({"search", "browse", "refund", "summarise"})


class Counting(ScriptedProvider):
    """A provider that records how many calls were in flight at once."""

    def __init__(self, *responses: ModelResponse | BaseException) -> None:
        super().__init__(*responses)
        self.in_flight = 0
        self.peak = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer, holding the call open long enough for its siblings to arrive."""
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return await super().complete(request)
        finally:
            self.in_flight -= 1


class Staggered(ScriptedProvider):
    """A provider that answers in the reverse of the order it was asked.

    It answers by arrival rather than by completion, so what a branch is told stays the
    same and only when it is told changes. Otherwise the fake, not the fan-out, is what
    reorders the results.
    """

    def __init__(self, *responses: ModelResponse) -> None:
        super().__init__(*responses)
        self._scripted = list(responses)
        self.arrived = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002 — scripted
        """Answer the nth caller with the nth response, the later ones sooner."""
        mine = self.arrived
        self.arrived += 1
        for _ in range(8 - self.arrived):
            await asyncio.sleep(0)
        return self._scripted[mine]


class Held(ScriptedProvider):
    """A provider that does not answer until it is let go, for the cancellation path."""

    def __init__(self, *responses: ModelResponse | BaseException) -> None:
        super().__init__(*responses)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Announce the call, wait to be released, then answer from the script."""
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return await super().complete(request)


class Sieve:
    """A guard that blocks one particular thing, so one branch of several is excluded."""

    def __init__(self, blocked: str) -> None:
        self._blocked = blocked

    @property
    def name(self) -> str:
        """What the pipeline records this verdict against."""
        return "sieve"

    async def check_input(self, content: str) -> GuardResult:
        """Stop the one string this guard was built to stop."""
        if self._blocked in content:
            return GuardResult.blocked(code="sieve_refusal")
        return GuardResult.allow()

    async def check_output(self, content: str) -> GuardResult:  # noqa: ARG002 — one stage only
        """Nothing on the way out is this guard's business."""
        return GuardResult.allow()


def agent(name: str, *tools: str) -> Agent[Any]:
    fields: dict[str, object] = {
        "name": name,
        "instructions": f"You are {name}.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": tools,
    }
    return Agent(**fields)  # type: ignore[arg-type]


def answer(text: str = "found one", tokens: int = 10) -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=tokens, output_tokens=5))


def spoken(data: str) -> str:
    """What a branch said, out of the untrusted-data envelope it crosses back in."""
    return data.split(">\n", 1)[1].rsplit("\n<", 1)[0]


def roster() -> Roster:
    return Roster(
        (
            Specialist(
                agent=agent("researcher", "search", "browse"),
                capabilities=frozenset({"research"}),
            ),
            Specialist(agent=agent("accountant", "refund"), capabilities=frozenset({"sums"})),
        )
    )


def ledger(max_input_tokens: int = 10_000) -> RunBudget:
    return RunBudget(
        most_restrictive(
            ScopedLimits(
                scope=BudgetScope.RUN, limits=BudgetLimits(max_input_tokens=max_input_tokens)
            )
        ),
        clock=FakeClock(),
    )


def supervising(
    *responses: ModelResponse | BaseException,
    provider: ScriptedProvider | None = None,
    budget: RunBudget | None = None,
    guardrails: GuardrailPipeline | None = None,
    limits: DelegationLimits | None = None,
) -> Supervisor:
    runner = AgentRunner(
        provider=provider or ScriptedProvider(*responses),
        clock=FakeClock(),
        tools=FakeToolRegistry(dict.fromkeys(sorted(HELD), str)),
    )
    return Supervisor(
        runner,
        roster(),
        agent=agent("supervisor", *sorted(HELD)),
        delegation=Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="supervisor",
            scope=DelegationScope(tools=HELD),
            limits=limits,
        ),
        budget=budget if budget is not None else ledger(),
        guardrails=guardrails,
    )


def branches(count: int = 3, *, needs: str = "research") -> tuple[Branch, ...]:
    return tuple(
        Branch(name=f"b{index}", task=f"look at {index}", needs={needs}) for index in range(count)
    )


class TestWhatABranchIs:
    """A branch is named, so what it contributed can be said afterwards."""

    def test_a_branch_carries_its_own_name_and_task(self) -> None:
        one = Branch(name="fares", task="check fares", needs={"research"})
        assert one.name == "fares"
        assert one.needs == frozenset({"research"})

    def test_a_branch_that_needs_nothing_could_not_be_routed(self) -> None:
        with pytest.raises(ConfigurationError, match="needs nothing"):
            Branch(name="fares", task="check fares", needs=())

    def test_a_branch_without_a_name_could_not_be_attributed(self) -> None:
        with pytest.raises(ConfigurationError, match="unnamed"):
            Branch(name="", task="check fares", needs={"research"})

    async def test_two_branches_answering_to_one_name_are_refused(self) -> None:
        same = (branches(1)[0], branches(1)[0])
        with pytest.raises(ConfigurationError, match="one name"):
            await fan_out(supervising(answer(), answer()), same)

    async def test_fanning_out_over_nothing_is_a_mistake_rather_than_an_empty_aggregate(
        self,
    ) -> None:
        with pytest.raises(ConfigurationError, match="no branches"):
            await fan_out(supervising(), ())


class TestBoundedConcurrency:
    """Concurrency is capped, because a provider rate limit is not a cap anybody chose."""

    async def test_no_more_branches_are_in_flight_than_the_cap_allows(self) -> None:
        provider = Counting(*[answer() for _ in range(5)])
        done = await fan_out(supervising(provider=provider), branches(5), max_concurrency=2)
        assert provider.peak <= 2
        assert done.peak_in_flight <= 2
        assert len(done.results) == 5

    async def test_every_branch_still_runs_under_a_cap_of_one(self) -> None:
        provider = Counting(*[answer() for _ in range(3)])
        done = await fan_out(supervising(provider=provider), branches(3), max_concurrency=1)
        assert provider.peak == 1
        assert [one.outcome for one in done.results] == [BranchOutcome.OK] * 3

    async def test_a_cap_below_one_is_a_configuration_mistake(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one"):
            await fan_out(supervising(answer()), branches(1), max_concurrency=0)


class TestDeterministicOrder:
    """The same inputs aggregate identically however the branches happened to finish."""

    async def test_results_are_in_declared_order_not_completion_order(self) -> None:
        provider = Staggered(answer("first"), answer("second"), answer("third"))
        done = await fan_out(supervising(provider=provider), branches(3))
        assert [one.branch for one in done.results] == ["b0", "b1", "b2"]

    async def test_the_aggregate_does_not_depend_on_who_finished_first(self) -> None:
        ordered = await fan_out(
            supervising(provider=ScriptedProvider(answer("a"), answer("b"))), branches(2)
        )
        staggered = await fan_out(
            supervising(provider=Staggered(answer("a"), answer("b"))), branches(2)
        )
        assert ordered.value == staggered.value
        assert ordered.contributed == staggered.contributed


class TestAggregation:
    """What counts as an answer is declared, and failing closed is the default."""

    async def test_all_is_the_default_and_gives_every_branch_in_order(self) -> None:
        done = await fan_out(supervising(answer("a"), answer("b")), branches(2))
        assert tuple(map(spoken, done.value)) == ("a", "b")
        assert done.contributed == ("b0", "b1")
        assert done.strategy == "all"

    async def test_all_fails_closed_with_the_failing_branch_named(self) -> None:
        provider = ScriptedProvider(answer("a"), RuntimeError("the provider fell over"), answer())
        with pytest.raises(AggregationError) as refused:
            await fan_out(supervising(provider=provider), branches(3), into=All())
        assert refused.value.reason == "failed"
        assert refused.value.excluded["b1"]
        assert refused.value.contributed == ("b0", "b2")

    async def test_quorum_answers_from_the_branches_that_did_answer(self) -> None:
        provider = ScriptedProvider(
            answer("a"), RuntimeError("fell over"), answer("c"), answer("d"), answer("e")
        )
        done = await fan_out(supervising(provider=provider), branches(5), into=Quorum(3))
        assert tuple(map(spoken, done.value)) == ("a", "c", "d", "e")
        assert done.contributed == ("b0", "b2", "b3", "b4")
        assert "b1" in done.excluded

    async def test_a_quorum_nobody_reached_is_refused_rather_than_rounded_down(self) -> None:
        provider = ScriptedProvider(answer("a"), RuntimeError("fell over"), RuntimeError("again"))
        with pytest.raises(AggregationError) as refused:
            await fan_out(supervising(provider=provider), branches(3), into=Quorum(3))
        assert refused.value.reason == "quorum"
        assert refused.value.contributed == ("b0",)

    def test_a_quorum_of_none_is_a_configuration_mistake(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one"):
            Quorum(0)

    async def test_first_success_takes_the_first_in_declared_order(self) -> None:
        provider = ScriptedProvider(RuntimeError("fell over"), answer("b"), answer("c"))
        done = await fan_out(supervising(provider=provider), branches(3), into=FirstSuccess())
        assert spoken(done.value) == "b"
        assert done.contributed == ("b1",)

    async def test_nothing_answered_is_a_refusal_rather_than_an_empty_answer(self) -> None:
        provider = ScriptedProvider(RuntimeError("fell over"), RuntimeError("again"))
        with pytest.raises(AggregationError) as refused:
            await fan_out(supervising(provider=provider), branches(2), into=FirstSuccess())
        assert refused.value.reason == "none"
        assert refused.value.strategy == "first_success"

    async def test_a_reducer_sees_the_branches_that_answered_in_order(self) -> None:
        provider = ScriptedProvider(answer("7"), RuntimeError("fell over"), answer("11"))
        done = await fan_out(
            supervising(provider=provider),
            branches(3),
            into=Reduce(lambda results: sum(int(spoken(one.data)) for one in results)),
        )
        assert done.value == 18
        assert done.excluded["b1"]

    async def test_a_reducer_with_nothing_to_reduce_is_refused(self) -> None:
        provider = ScriptedProvider(RuntimeError("fell over"))
        with pytest.raises(AggregationError) as refused:
            await fan_out(
                supervising(provider=provider), branches(1), into=Reduce(lambda one: len(one))
            )
        assert refused.value.reason == "none"


class TestProvenance:
    """An aggregate says which branches are in it and why the others are not."""

    async def test_the_aggregate_names_what_was_left_out_and_why(self) -> None:
        provider = ScriptedProvider(answer("a"), RuntimeError("the provider fell over"))
        done = await fan_out(supervising(provider=provider), branches(2), into=Quorum(1))
        assert done.contributed == ("b0",)
        assert "failed" in done.excluded["b1"]

    async def test_every_branch_is_on_the_result_answered_or_not(self) -> None:
        provider = ScriptedProvider(answer("a"), RuntimeError("fell over"))
        done = await fan_out(supervising(provider=provider), branches(2), into=Quorum(1))
        assert [one.outcome for one in done.results] == [BranchOutcome.OK, BranchOutcome.FAILED]

    async def test_spend_is_attributed_to_the_branch_that_spent_it(self) -> None:
        done = await fan_out(supervising(answer("a", 10), answer("b", 30)), branches(2))
        assert done.spent["b0"].input_tokens == 10
        assert done.spent["b1"].input_tokens == 30
        assert done.usage.input_tokens == 40

    async def test_what_a_branch_handed_back_passed_the_guardrails(self) -> None:
        done = await fan_out(
            supervising(
                answer("secret"), answer("fine"), guardrails=GuardrailPipeline((Sieve("secret"),))
            ),
            branches(2),
            into=Quorum(1),
        )
        assert done.contributed == ("b1",)
        assert "blocked" in done.excluded["b0"]

    async def test_the_supervisor_recorded_every_branch_it_ran(self) -> None:
        supervisor = supervising(answer("a"), answer("b"))
        await fan_out(supervisor, branches(2))
        assert len(supervisor.events) == 2


class TestScopeAndAttribution:
    """A branch holds the intersection of what it declares and what its caller holds."""

    async def test_a_branch_runs_narrowed_to_its_callers_scope(self) -> None:
        done = await fan_out(supervising(answer()), branches(1))
        assert done.results[0].specialist == "researcher"
        assert done.results[0].outcome is BranchOutcome.OK

    async def test_a_branch_nobody_on_the_roster_can_do_is_excluded_not_raised(self) -> None:
        done = await fan_out(
            supervising(answer()),
            (
                Branch(name="b0", task="argue it", needs={"litigation"}),
                Branch(name="b1", task="look it up", needs={"research"}),
            ),
            into=Quorum(1),
        )
        assert done.results[0].outcome is BranchOutcome.FAILED
        assert "no_worker" in done.excluded["b0"]
        assert done.contributed == ("b1",)

    async def test_a_wiring_mistake_is_a_failed_branch_which_all_then_refuses(self) -> None:
        with pytest.raises(AggregationError) as refused:
            await fan_out(
                supervising(answer()), (Branch(name="b0", task="argue it", needs={"litigation"}),)
            )
        assert refused.value.reason == "failed"

    async def test_two_branches_writing_one_key_is_refused_rather_than_last_writer_wins(
        self,
    ) -> None:
        pair = (
            Branch(name="b0", task="one", needs={"research"}, writes="itinerary"),
            Branch(name="b1", task="two", needs={"sums"}, writes="itinerary"),
        )
        done = await fan_out(
            supervising(answer("a"), answer("b")),
            pair,
            max_concurrency=1,
            into=Quorum(1),
        )
        assert done.contributed == ("b0",)
        assert "conflict" in done.excluded["b1"]

    async def test_the_run_wide_delegation_ceiling_bounds_breadth_times_depth(self) -> None:
        supervisor = supervising(
            *[answer() for _ in range(4)], limits=DelegationLimits(max_delegations=2)
        )
        done = await fan_out(supervisor, branches(4), max_concurrency=1, into=Quorum(2))
        assert done.contributed == ("b0", "b1")
        assert sorted(done.excluded) == ["b2", "b3"]


class TestTheLedger:
    """The aggregate ceiling is one ceiling, not one ceiling per branch."""

    async def test_a_ledger_that_runs_out_stops_the_branches_not_yet_started(self) -> None:
        provider = Counting(*[answer("a", 40) for _ in range(4)])
        done = await fan_out(
            supervising(provider=provider, budget=ledger(max_input_tokens=60)),
            branches(4),
            max_concurrency=1,
            into=Quorum(1),
        )
        assert done.contributed == ("b0",)
        assert [one.outcome for one in done.results[1:]] == [BranchOutcome.BUDGET_EXHAUSTED] * 3
        assert "ledger" in done.excluded["b3"]

    async def test_a_branch_that_outruns_its_own_slice_is_only_that_branch(self) -> None:
        done = await fan_out(
            supervising(answer("a"), answer("b")),
            (
                Branch(
                    name="b0",
                    task="one",
                    needs={"research"},
                    budget=BudgetLimits(max_input_tokens=1),
                ),
                Branch(name="b1", task="two", needs={"research"}),
            ),
            max_concurrency=1,
            into=Quorum(1),
        )
        assert done.results[0].outcome is BranchOutcome.BUDGET_EXHAUSTED
        assert done.contributed == ("b1",)


class TestCancellation:
    """A fan-out stopped mid-flight produces a typed refusal, never a fabricated aggregate."""

    async def test_cancelling_mid_flight_refuses_rather_than_aggregating_what_arrived(self) -> None:
        provider = Held(*[answer() for _ in range(3)])
        supervisor = supervising(provider=provider)
        running = asyncio.ensure_future(fan_out(supervisor, branches(3), max_concurrency=3))
        await provider.started.wait()
        supervisor.cancel("the caller changed their mind")
        provider.release.set()
        with pytest.raises(AggregationError) as stopped:
            await running
        assert stopped.value.reason == "cancelled"

    async def test_a_cancelled_branch_is_still_on_the_record(self) -> None:
        provider = Held(*[answer() for _ in range(2)])
        supervisor = supervising(provider=provider)
        running = asyncio.ensure_future(
            fan_out(supervisor, branches(2), max_concurrency=2, into=Quorum(1))
        )
        await provider.started.wait()
        supervisor.cancel("stop")
        provider.release.set()
        with pytest.raises(AggregationError) as stopped:
            await running
        assert sorted(stopped.value.excluded) == ["b0", "b1"]

    async def test_a_branch_still_queued_when_the_stop_lands_never_starts(self) -> None:
        provider = Held(*[answer() for _ in range(2)])
        supervisor = supervising(provider=provider)
        running = asyncio.ensure_future(fan_out(supervisor, branches(2), max_concurrency=1))
        await provider.started.wait()
        supervisor.cancel("stop")
        provider.release.set()
        with pytest.raises(AggregationError):
            await running
        assert provider.calls == 1


class TestBranchResults:
    """A branch result reads the same whether it answered or not."""

    def test_an_outcome_that_is_not_ok_has_no_data_to_read(self) -> None:
        result = BranchResult(
            branch="b0",
            specialist="researcher",
            outcome=BranchOutcome.FAILED,
            data="",
            usage=Usage(input_tokens=0, output_tokens=0),
            reason="failed: the provider fell over",
        )
        assert not result.ok
        assert result.reason
