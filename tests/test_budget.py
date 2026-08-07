"""What a run is allowed to spend, said once and honoured everywhere.

A ceiling nobody can state in one vocabulary is a ceiling each product invents for itself:
one counts iterations, one counts nothing, and the looping agent produces a bill somebody
finds at the end of the month. So limits are a value object, the effective ceiling is the
most restrictive applicable one with its source on the record, and unlimited is a thing you
have to name rather than a thing you get by forgetting.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from tesserix_adk.core import (
    Agent,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    BudgetUnavailableError,
    ConfigurationError,
    Consumed,
    Cost,
    LedgerFailure,
    ModelCapabilities,
    NoOutput,
    ResolvedBudget,
    RunBudget,
    RunState,
    ScopedLimits,
    ToolCall,
    UnlimitedBudget,
    Usage,
    most_restrictive,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import (
    FakeBudgetPolicy,
    FakeClock,
    FakeTenantLedger,
    FakeToolRegistry,
    ScriptedProvider,
)


def limits(**overrides: object) -> BudgetLimits:
    return BudgetLimits(**overrides)  # type: ignore[arg-type]


def scoped(scope: BudgetScope, **overrides: object) -> ScopedLimits:
    return ScopedLimits(scope=scope, limits=limits(**overrides))


class TestALimitNobodySetIsNotNoLimit:
    """A field left None is a field nobody thought about, and an unbounded agent is what
    that costs. Forgetting is not a way to opt out."""

    def test_limits_with_nothing_set_resolve_to_the_conservative_defaults(self) -> None:
        assert limits().filled() == BudgetLimits.conservative()

    def test_the_conservative_defaults_bound_every_dimension(self) -> None:
        default = BudgetLimits.conservative()
        assert default.max_cost is not None
        assert default.max_input_tokens is not None
        assert default.max_output_tokens is not None
        assert default.max_model_calls is not None
        assert default.max_tool_calls is not None
        assert default.max_iterations is not None
        assert default.max_seconds is not None

    def test_a_field_that_was_set_survives_filling(self) -> None:
        filled = limits(max_model_calls=2).filled()
        assert filled.max_model_calls == 2
        assert filled.max_cost == BudgetLimits.conservative().max_cost

    def test_unlimited_is_a_thing_you_say_rather_than_a_thing_you_forget(self) -> None:
        assert not limits().unlimited
        assert BudgetLimits.unbounded().unlimited

    def test_filling_unbounded_limits_leaves_them_unbounded(self) -> None:
        """Otherwise the one explicit way to say 'no ceiling' would quietly grow one."""
        assert BudgetLimits.unbounded().filled().unlimited

    def test_money_is_decimal_and_not_a_float(self) -> None:
        assert isinstance(BudgetLimits.conservative().max_cost, Decimal)

    def test_a_ceiling_of_zero_is_refused_as_a_field_somebody_meant_to_disable(self) -> None:
        with pytest.raises(ValueError, match="max_model_calls"):
            limits(max_model_calls=0)


class TestTheMostRestrictiveLimitWins:
    """Two scopes both apply; the answer is not the nearer one, it is the tighter one."""

    def test_the_tighter_ceiling_is_the_effective_one(self) -> None:
        resolved = most_restrictive(
            scoped(BudgetScope.TENANT, max_cost=Decimal("5.00")),
            scoped(BudgetScope.RUN, max_cost=Decimal("1.00")),
        )
        assert resolved.limits.max_cost == Decimal("1.00")

    def test_a_looser_run_limit_does_not_widen_the_tenant_ceiling(self) -> None:
        resolved = most_restrictive(
            scoped(BudgetScope.TENANT, max_cost=Decimal("1.00")),
            scoped(BudgetScope.RUN, max_cost=Decimal("5.00")),
        )
        assert resolved.limits.max_cost == Decimal("1.00")

    def test_each_dimension_is_resolved_on_its_own(self) -> None:
        resolved = most_restrictive(
            scoped(BudgetScope.TENANT, max_cost=Decimal("1.00"), max_model_calls=100),
            scoped(BudgetScope.RUN, max_cost=Decimal("5.00"), max_model_calls=4),
        )
        assert (resolved.limits.max_cost, resolved.limits.max_model_calls) == (Decimal("1.00"), 4)

    def test_the_winning_scope_is_recorded_for_every_dimension(self) -> None:
        """A ceiling nobody can attribute is one nobody can raise."""
        resolved = most_restrictive(
            scoped(BudgetScope.TENANT, max_cost=Decimal("1.00"), max_model_calls=100),
            scoped(BudgetScope.RUN, max_cost=Decimal("5.00"), max_model_calls=4),
        )
        assert resolved.sources["max_cost"] is BudgetScope.TENANT
        assert resolved.sources["max_model_calls"] is BudgetScope.RUN

    def test_a_dimension_no_scope_set_falls_to_the_conservative_default(self) -> None:
        resolved = most_restrictive(scoped(BudgetScope.RUN, max_cost=Decimal("1.00")))
        assert resolved.limits.max_iterations == BudgetLimits.conservative().max_iterations

    def test_resolving_nothing_at_all_is_the_conservative_default_and_not_unlimited(self) -> None:
        assert most_restrictive().limits == BudgetLimits.conservative()

    def test_an_explicitly_unbounded_scope_still_loses_to_a_scope_with_a_ceiling(self) -> None:
        resolved = most_restrictive(
            ScopedLimits(scope=BudgetScope.TENANT, limits=BudgetLimits.unbounded()),
            scoped(BudgetScope.RUN, max_cost=Decimal("1.00")),
        )
        assert resolved.limits.max_cost == Decimal("1.00")

    def test_two_currencies_are_refused_rather_than_converted(self) -> None:
        """A conversion the kit invented would be a rate nobody agreed to."""
        with pytest.raises(ConfigurationError, match="EUR and USD"):
            most_restrictive(
                scoped(BudgetScope.TENANT, max_cost=Decimal("5.00")),
                scoped(BudgetScope.RUN, max_cost=Decimal("1.00"), currency="EUR"),
            )

    def test_currencies_may_differ_where_neither_scope_limits_money(self) -> None:
        resolved = most_restrictive(
            scoped(BudgetScope.TENANT, max_model_calls=4, currency="EUR"),
            scoped(BudgetScope.RUN, max_model_calls=2),
        )
        assert resolved.limits.max_model_calls == 2

    def test_two_scopes_of_the_same_kind_are_refused_as_one_question_answered_twice(
        self,
    ) -> None:
        with pytest.raises(ConfigurationError, match="two sets of limits at run scope"):
            most_restrictive(
                scoped(BudgetScope.RUN, max_model_calls=4),
                scoped(BudgetScope.RUN, max_model_calls=2),
            )


def usage(cost: str = "0", **counts: int) -> Usage:
    fields: dict[str, object] = {"input_tokens": 0, "output_tokens": 0, **counts}
    return Usage(cost=Cost(input=Decimal(cost)), **fields)  # type: ignore[arg-type]


def budget(clock: FakeClock | None = None, **overrides: object) -> RunBudget:
    fields: dict[str, object] = {
        "resolved": most_restrictive(ScopedLimits(scope=BudgetScope.RUN, limits=limits())),
        "clock": clock or FakeClock(),
    }
    return RunBudget(**{**fields, **overrides})  # type: ignore[arg-type]


def capped(**overrides: object) -> ResolvedBudget:
    return most_restrictive(ScopedLimits(scope=BudgetScope.RUN, limits=limits(**overrides)))


class TestSpendingIsCheckedBeforeItHappens:
    """A ceiling enforced after the call is a report, not a ceiling."""

    @pytest.mark.anyio
    async def test_a_reservation_within_the_ceiling_is_permitted(self) -> None:
        under = budget(resolved=capped(max_input_tokens=100))
        await under.reserve(50)
        assert under.check().permitted

    @pytest.mark.anyio
    async def test_a_reservation_past_the_ceiling_raises_before_the_call(self) -> None:
        under = budget(resolved=capped(max_input_tokens=100))
        with pytest.raises(BudgetExceededError):
            await under.reserve(101)

    @pytest.mark.anyio
    async def test_the_error_says_which_limit_broke_and_by_how_much(self) -> None:
        under = budget(resolved=capped(max_input_tokens=100))
        await under.reserve(80)
        with pytest.raises(BudgetExceededError) as refused:
            await under.reserve(80)
        assert refused.value.breached == "max_input_tokens"
        assert refused.value.limit == Decimal(100)
        assert refused.value.consumed == Decimal(80)
        assert refused.value.remaining == Decimal(20)
        assert refused.value.scope is BudgetScope.RUN

    @pytest.mark.anyio
    async def test_recording_actual_usage_releases_the_over_reservation(self) -> None:
        under = budget(resolved=capped(max_input_tokens=100))
        await under.reserve(90)
        await under.record(usage(input_tokens=10))
        await under.reserve(80)  # must not raise: only 10 was actually spent

    @pytest.mark.anyio
    async def test_money_is_a_ceiling_of_its_own(self) -> None:
        under = budget(resolved=capped(max_cost=Decimal("0.10")))
        with pytest.raises(BudgetExceededError, match="max_cost"):
            await under.record(usage("0.11"))

    @pytest.mark.anyio
    async def test_an_unpriced_call_does_not_silently_pass_a_money_ceiling(self) -> None:
        """A cost nobody could compute is not a cost of zero, so the decision says so."""
        under = budget(resolved=capped(max_cost=Decimal("0.10")))
        await under.record(Usage(input_tokens=1, output_tokens=1))
        assert under.check().priced is False

    @pytest.mark.anyio
    async def test_model_calls_are_counted_including_the_ones_that_failed(self) -> None:
        under = budget(resolved=capped(max_model_calls=2))
        await under.record(usage(), model_calls=1)
        with pytest.raises(BudgetExceededError, match="max_model_calls"):
            await under.record(usage(), model_calls=2)

    @pytest.mark.anyio
    async def test_tool_calls_and_iterations_are_counted_too(self) -> None:
        under = budget(resolved=capped(max_tool_calls=1, max_iterations=1))
        await under.record(usage(), tool_calls=1, iterations=1)
        with pytest.raises(BudgetExceededError, match="max_tool_calls"):
            await under.record(usage(), tool_calls=1)

    @pytest.mark.anyio
    async def test_wall_clock_time_is_a_ceiling_the_clock_decides(self) -> None:
        clock = FakeClock()
        under = budget(clock, resolved=capped(max_seconds=10.0))
        await clock.sleep(11)
        with pytest.raises(BudgetExceededError, match="max_seconds"):
            await under.reserve(1)

    @pytest.mark.anyio
    async def test_unbounded_limits_permit_what_the_defaults_would_have_refused(self) -> None:
        under = budget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits.unbounded())
            )
        )
        await under.reserve(10_000_000)
        assert under.check().permitted


class TestAChildRunSpendsWhatTheParentHasLeft:
    """A sub-agent handed a fresh allowance is a way to spend the ceiling twice."""

    @pytest.mark.anyio
    async def test_a_child_starts_from_what_the_parent_has_left(self) -> None:
        parent = budget(resolved=capped(max_input_tokens=100))
        await parent.reserve(60)
        await parent.record(usage(input_tokens=60))
        assert parent.child().limits().max_input_tokens == 40

    @pytest.mark.anyio
    async def test_a_child_spending_counts_against_the_parent(self) -> None:
        parent = budget(resolved=capped(max_input_tokens=100))
        child = parent.child()
        await child.record(usage(input_tokens=90))
        with pytest.raises(BudgetExceededError):
            await parent.reserve(20)

    @pytest.mark.anyio
    async def test_a_child_cannot_widen_what_the_parent_was_given(self) -> None:
        parent = budget(resolved=capped(max_model_calls=2))
        child = parent.child()
        await child.record(usage(), model_calls=2)
        with pytest.raises(BudgetExceededError, match="max_model_calls"):
            await child.record(usage(), model_calls=1)


class TestATenantCeilingIsSharedBetweenRuns:
    """A ceiling each run gets in full is not a tenant ceiling, it is a run ceiling with a
    misleading name."""

    def shared(self, ledger: FakeTenantLedger, **overrides: object) -> RunBudget:
        return RunBudget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.TENANT, limits=limits(max_input_tokens=100))
            ),
            clock=FakeClock(),
            ledger=ledger,
            tenant="acme",
            **overrides,  # type: ignore[arg-type]
        )

    @pytest.mark.anyio
    async def test_a_second_run_sees_what_the_first_spent(self) -> None:
        ledger = FakeTenantLedger()
        await self.shared(ledger).record(usage(input_tokens=80))
        with pytest.raises(BudgetExceededError, match="max_input_tokens"):
            await self.shared(ledger).reserve(30)

    @pytest.mark.anyio
    async def test_concurrent_runs_do_not_each_get_the_whole_ceiling(self) -> None:
        ledger = FakeTenantLedger()
        runs = [self.shared(ledger) for _ in range(3)]
        spent = await asyncio.gather(
            *(one.record(usage(input_tokens=60)) for one in runs),
            return_exceptions=True,
        )
        refused = [outcome for outcome in spent if isinstance(outcome, BudgetExceededError)]
        assert len(refused) == 2

    @pytest.mark.anyio
    async def test_a_ledger_that_cannot_be_reached_fails_closed(self) -> None:
        """Spending against a ceiling nobody can read is spending without a ceiling."""
        with pytest.raises(BudgetUnavailableError, match="acme"):
            await self.shared(FakeTenantLedger(reachable=False)).reserve(1)

    @pytest.mark.anyio
    async def test_proceeding_without_the_ledger_takes_an_explicit_choice(self) -> None:
        under = self.shared(FakeTenantLedger(reachable=False), on_ledger_failure="proceed")
        await under.reserve(1)
        assert under.resolved.on_ledger_failure is LedgerFailure.PROCEED

    @pytest.mark.anyio
    async def test_a_run_straddling_a_window_boundary_gets_no_second_allowance(self) -> None:
        """Otherwise a run started at 10:59 is a way to spend two hours of one hour's budget."""
        clock = FakeClock()
        ledger = FakeTenantLedger()
        under = RunBudget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.TENANT_WINDOW, limits=limits(max_input_tokens=100))
            ),
            clock=clock,
            ledger=ledger,
            tenant="acme",
            window_seconds=3600.0,
        )
        await under.record(usage(input_tokens=90))
        await clock.sleep(7200)
        with pytest.raises(BudgetExceededError):
            await under.reserve(20)

    @pytest.mark.anyio
    async def test_a_run_scoped_ceiling_is_not_checked_against_the_tenant_ledger(self) -> None:
        ledger = FakeTenantLedger()
        first = RunBudget(
            resolved=capped(max_input_tokens=100),
            clock=FakeClock(),
            ledger=ledger,
            tenant="acme",
        )
        await first.record(usage(input_tokens=90))
        second = RunBudget(
            resolved=capped(max_input_tokens=100),
            clock=FakeClock(),
            ledger=ledger,
            tenant="acme",
        )
        await second.reserve(90)


class TestNoRuntimeCanBeBuiltWithoutACeiling:
    """A runtime with no policy is the unbounded agent, arrived at by omission."""

    def runner(self, **overrides: object) -> AgentRunner:
        fields: dict[str, object] = {
            "provider": ScriptedProvider(
                ModelResponse(content="done", usage=Usage(input_tokens=5_000, output_tokens=10)),
                name="scripted",
                capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
            ),
            "clock": FakeClock(),
        }
        return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]

    def agent(self, **overrides: object) -> Agent[NoOutput]:
        fields: dict[str, object] = {
            "name": "planner",
            "instructions": "Plan trips.",
            "free_text": True,
            "model": "scripted-1",
        }
        return Agent(**{**fields, **overrides})  # type: ignore[arg-type]

    @pytest.mark.anyio
    async def test_a_runner_given_no_policy_still_enforces_a_ceiling(self) -> None:
        run = await self.runner().run(
            self.agent(budget=limits(max_input_tokens=10)), "Where to?", tenant="acme"
        )
        assert run.state is RunState.BUDGET_EXHAUSTED

    @pytest.mark.anyio
    async def test_the_resolved_ceiling_and_its_source_are_recorded_on_the_run(self) -> None:
        run = await self.runner().run(
            self.agent(budget=limits(max_model_calls=3)), "Where to?", tenant="acme"
        )
        assert run.budget is not None
        assert run.budget.limits.max_model_calls == 3
        assert run.budget.sources["max_model_calls"] is BudgetScope.AGENT

    @pytest.mark.anyio
    async def test_a_dimension_nobody_stated_falls_to_the_documented_default(self) -> None:
        run = await self.runner().run(self.agent(), "Where to?", tenant="acme")
        assert run.budget is not None
        assert run.budget.limits.max_iterations == BudgetLimits.conservative().max_iterations

    @pytest.mark.anyio
    async def test_unlimited_is_named_in_configuration_and_recorded_on_the_run(self) -> None:
        run = await self.runner(budget=UnlimitedBudget(reason="batch backfill, RFC-114")).run(
            self.agent(), "Where to?", tenant="acme"
        )
        assert run.budget is not None
        assert run.budget.limits.unlimited
        assert run.budget.unlimited_reason == "batch backfill, RFC-114"

    def test_unlimited_without_a_stated_reason_is_refused(self) -> None:
        """A ceiling removed for a reason nobody wrote down is one nobody can review."""
        with pytest.raises(ValueError, match="reason"):
            UnlimitedBudget(reason="")


class _WriteOnlyBrokenLedger(FakeTenantLedger):
    """Reads fine, refuses to write — the outage that starts mid-run."""

    async def consume(self, tenant: str, window: str, spent: Consumed) -> Consumed:
        _ = (window, spent)
        raise BudgetUnavailableError(f"the ledger for {tenant} stopped answering")


class TestTheEdgesOfTheCeiling:
    """The refusals and the corners, each of which is a way a ceiling stops meaning
    anything if it is wrong."""

    def test_unlimited_alongside_a_ceiling_is_a_contradiction(self) -> None:
        with pytest.raises(ValueError, match="unlimited"):
            BudgetLimits(unlimited=True, max_model_calls=2)

    def test_a_ledger_with_no_tenant_is_a_ceiling_shared_with_nobody(self) -> None:
        with pytest.raises(ConfigurationError, match="no tenant"):
            RunBudget(resolved=capped(), clock=FakeClock(), ledger=FakeTenantLedger())

    def test_what_is_left_of_no_ceiling_is_no_ceiling(self) -> None:
        under = budget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits.unbounded())
            )
        )
        assert under.limits().unlimited

    def test_check_reports_a_breach_rather_than_raising_it(self) -> None:
        """`check` answers a question; only spending refuses."""
        under = RunBudget(
            resolved=capped(max_model_calls=1),
            clock=FakeClock(),
            spent=Consumed(model_calls=4),
        )
        decision = under.check()
        assert not decision.permitted
        assert decision.breached == "max_model_calls"
        assert decision.remaining == Decimal(0)

    @pytest.mark.anyio
    async def test_a_ledger_that_fails_on_the_write_fails_closed_too(self) -> None:
        under = RunBudget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.TENANT, limits=limits(max_input_tokens=100))
            ),
            clock=FakeClock(),
            ledger=_WriteOnlyBrokenLedger(),
            tenant="acme",
        )
        with pytest.raises(BudgetUnavailableError):
            await under.record(usage(input_tokens=1))

    @pytest.mark.anyio
    async def test_an_unlimited_budget_permits_and_records_nothing(self) -> None:
        under = UnlimitedBudget(reason="batch backfill, RFC-114")
        await under.reserve(10_000_000)
        await under.record(usage(input_tokens=10_000_000), model_calls=99)
        assert under.limits().unlimited
        assert under.child() is under
        assert under.check().permitted
        assert under.reservations == [10_000_000]

    def test_the_fake_policy_reports_a_breach_it_was_pushed_past(self) -> None:
        policy = FakeBudgetPolicy(limit=10)
        assert policy.check().permitted
        policy.spent = 11
        assert not policy.check().permitted
        assert policy.check().breached == "max_input_tokens"


class TestARunThatCannotReadItsCeilingStops:
    """Carrying on without the ledger is how one outage becomes an unbounded bill."""

    def runner(self, **overrides: object) -> AgentRunner:
        fields: dict[str, object] = {
            "provider": ScriptedProvider(
                ModelResponse(content="done", usage=Usage(input_tokens=5, output_tokens=2)),
                name="scripted",
                capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
            ),
            "clock": FakeClock(),
        }
        return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]

    def shared(self, ledger: FakeTenantLedger) -> RunBudget:
        return RunBudget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.TENANT, limits=limits(max_input_tokens=1_000))
            ),
            clock=FakeClock(),
            ledger=ledger,
            tenant="acme",
        )

    def agent(self) -> Agent[NoOutput]:
        return Agent(name="planner", instructions="Plan trips.", free_text=True, model="s-1")

    @pytest.mark.anyio
    async def test_a_ledger_down_before_the_call_stops_the_run(self) -> None:
        run = await self.runner().run(
            self.agent(),
            "Where to?",
            tenant="acme",
            budget=self.shared(FakeTenantLedger(reachable=False)),
        )
        assert run.state is RunState.FAILED

    @pytest.mark.anyio
    async def test_a_ledger_that_goes_down_after_the_call_stops_the_run(self) -> None:
        run = await self.runner().run(
            self.agent(), "Where to?", tenant="acme", budget=self.shared(_WriteOnlyBrokenLedger())
        )
        assert run.state is RunState.FAILED


class _LedgerThatGoesDownMidRun(FakeTenantLedger):
    """Answers once, then stops — the outage that arrives between two steps."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def total(self, tenant: str, window: str) -> Consumed:
        self.reads += 1
        if self.reads > 1:
            raise BudgetUnavailableError(f"the ledger for {tenant} stopped answering")
        return await super().total(tenant, window)


class TestAToolCallIsSpendToo:
    """A tool call reserves against the same ceiling the model call does."""

    @pytest.mark.anyio
    async def test_a_ledger_lost_before_a_tool_call_stops_the_run(self) -> None:
        runner = AgentRunner(
            provider=ScriptedProvider(
                ModelResponse(
                    content="",
                    tool_calls=(ToolCall(id="call_0", name="search", arguments={"q": "kyoto"}),),
                    usage=Usage(input_tokens=5, output_tokens=2),
                ),
                name="scripted",
                capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
            ),
            tools=FakeToolRegistry({"search": lambda **_: "a result"}),
            clock=FakeClock(),
        )
        agent = Agent(
            name="planner",
            instructions="Plan trips.",
            free_text=True,
            model="s-1",
            tools=("search",),
        )
        run = await runner.run(
            agent,
            "Where to?",
            tenant="acme",
            budget=RunBudget(
                resolved=most_restrictive(
                    ScopedLimits(scope=BudgetScope.TENANT, limits=limits(max_input_tokens=1_000))
                ),
                clock=FakeClock(),
                ledger=_LedgerThatGoesDownMidRun(),
                tenant="acme",
            ),
        )
        assert run.state is RunState.FAILED
