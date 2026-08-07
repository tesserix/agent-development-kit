"""What the run loop does when the ceiling is reached with work still in flight.

A ceiling checked only before the run starts does not stop the loop that discovered the
spend on its fortieth iteration. Worse is the product that handled overspend by truncating
context and answering anyway: money spent, and a degraded answer presented as a real one.
Here the ceiling can be hit at any point, and hitting it ends the run.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from tesserix_adk.core import (
    Agent,
    BudgetDecision,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    BudgetUnavailableError,
    Consumed,
    Cost,
    ModelCapabilities,
    ProviderTimeoutError,
    RepairConfig,
    RetryConfig,
    Run,
    RunBudget,
    RunEventKind,
    RunState,
    ScopedLimits,
    StreamEvent,
    TextDelta,
    ToolCall,
    Usage,
    UsageDelta,
    most_restrictive,
)
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse, budgeted_stream
from tesserix_adk.testing import (
    FakeClock,
    FakeTenantLedger,
    FakeToolRegistry,
    ScriptedProvider,
    StallingProvider,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)
NATIVE = CAPABLE.declaring(structured_output=True)
BAD = '{"destination": "Kyoto"}'


class TripPlan(BaseModel):
    destination: str
    nights: int


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "scripted-1",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str = "Kyoto, four nights.", **usage: object) -> ModelResponse:
    fields: dict[str, object] = {"input_tokens": 10, "output_tokens": 5}
    return ModelResponse(content=text, usage=Usage(**{**fields, **usage}))  # type: ignore[arg-type]


def calling(tool: str = "lookup", cost: str | None = None) -> ModelResponse:
    """A response that keeps the loop going, so a run can reach a ceiling mid-flight."""
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id="call_1", name=tool, arguments={}),),
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            cost=None if cost is None else Cost(input=Decimal(cost), currency="USD"),
        ),
    )


def unvalidatable() -> Agent:
    """An agent whose answers never validate, so the loop keeps asking and keeps paying."""
    return agent(output_type=TripPlan, free_text=False, repair=RepairConfig(max_attempts=5))


def tools() -> FakeToolRegistry:
    return FakeToolRegistry({"lookup": lambda: {"trains": 4}})


def budget(clock: FakeClock, **limits: object) -> RunBudget:
    return RunBudget(
        resolved=most_restrictive(
            ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(**limits))  # type: ignore[arg-type]
        ),
        clock=clock,
    )


def runner(*responses: ModelResponse | BaseException, **overrides: object) -> AgentRunner:
    clock = FakeClock()
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "clock": clock,
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


async def start(runner_: AgentRunner, agent_: Agent, text: str = "plan a trip") -> Run:
    return await runner_.run(agent_, text, tenant="acme", run_id="run_1")


def detail_of(run: Run, kind: RunEventKind) -> str:
    return next(event.detail or "" for event in run.events if event.kind is kind)


def spending(
    clock: FakeClock, *costs: str, **limits: object
) -> tuple[AgentRunner, ScriptedProvider]:
    """A runner whose model keeps calling a tool, so a ceiling can be reached mid-flight."""
    provider = ScriptedProvider(*(calling(cost=cost) for cost in costs), capabilities=CAPABLE)
    return (
        AgentRunner(provider=provider, clock=clock, budget=budget(clock, **limits), tools=tools()),
        provider,
    )


def repairing(
    clock: FakeClock, *costs: str, **limits: object
) -> tuple[AgentRunner, ScriptedProvider]:
    """A runner whose answers never validate, so every turn is a priced model call.

    Money is the dimension a tool call takes out of reach: a tool nobody priced makes the
    run's total unknown by design, so the ceiling here is exercised on model calls alone.
    """
    provider = ScriptedProvider(
        *(ModelResponse(content=BAD, usage=_priced(cost)) for cost in costs), capabilities=NATIVE
    )
    return (
        AgentRunner(provider=provider, clock=clock, budget=budget(clock, **limits)),
        provider,
    )


def _priced(cost: str) -> Usage:
    return Usage(input_tokens=10, output_tokens=5, cost=Cost(input=Decimal(cost), currency="USD"))


class TestTheCallThatWouldBreakTheCeilingIsNotMade:
    async def test_no_further_call_is_made_once_the_money_ceiling_is_reached(self) -> None:
        clock = FakeClock()
        runner_, provider = repairing(
            clock, "0.20", "0.20", "0.20", max_cost=Decimal("0.30"), currency="USD"
        )
        run = await start(runner_, unvalidatable())
        assert run.state is RunState.BUDGET_EXHAUSTED
        assert len(provider.requests) == 2

    async def test_a_call_that_would_not_fit_the_token_ceiling_is_never_dispatched(self) -> None:
        """Tokens are estimated before dispatch, so this one is refused rather than billed."""
        clock = FakeClock()
        runner_, provider = spending(clock, "0.10", max_input_tokens=4)
        run = await start(runner_, agent(tools=("lookup",)))
        assert run.state is RunState.BUDGET_EXHAUSTED
        assert provider.requests == []

    async def test_the_run_says_which_limit_stopped_it_and_what_it_had_spent(self) -> None:
        clock = FakeClock()
        runner_, _ = repairing(
            clock, "0.20", "0.20", "0.20", max_cost=Decimal("0.30"), currency="USD"
        )
        run = await start(runner_, unvalidatable())
        detail = detail_of(run, RunEventKind.TERMINATED)
        assert "max_cost" in detail
        assert "0.30" in detail
        assert "0.40" in detail

    async def test_an_exhausted_run_carries_no_answer(self) -> None:
        """A partial run record, never a result shaped like a real one."""
        clock = FakeClock()
        runner_, _ = repairing(
            clock, "0.20", "0.20", "0.20", max_cost=Decimal("0.30"), currency="USD"
        )
        run = await start(runner_, unvalidatable())
        assert run.output is None
        assert run.state is RunState.BUDGET_EXHAUSTED

    async def test_the_work_that_did_happen_is_still_on_the_run(self) -> None:
        clock = FakeClock()
        runner_, _ = repairing(
            clock, "0.20", "0.20", "0.20", max_cost=Decimal("0.30"), currency="USD"
        )
        run = await start(runner_, unvalidatable())
        assert run.usage.input_tokens > 0
        assert [event.kind for event in run.events].count(RunEventKind.MODEL_RESPONSE) == 2

    async def test_a_run_nobody_could_price_says_so_rather_than_reporting_nothing_spent(
        self,
    ) -> None:
        """A money ceiling checked against calls nobody priced is not enforcement."""
        clock = FakeClock()
        runner_, _ = spending(clock, "0.10", "0.10", max_cost=Decimal("1.00"), currency="USD")
        limit = runner_._budget
        await start(runner_, agent(tools=("lookup",)))
        assert limit is not None
        assert limit.check().priced is False


class TestTheCeilingIsCheckedBetweenIterationsToo:
    async def test_an_iteration_ceiling_stops_the_loop_before_the_next_call(self) -> None:
        clock = FakeClock()
        runner_, provider = spending(clock, "0.01", "0.01", "0.01", max_iterations=2)
        run = await start(runner_, agent(tools=("lookup",)))
        assert run.state is RunState.BUDGET_EXHAUSTED
        assert len(provider.requests) == 2

    async def test_a_wall_clock_ceiling_reached_while_idle_stops_the_next_iteration(self) -> None:
        """Time passes between iterations too, and a ceiling on it has to notice."""
        clock = FakeClock()
        runner_, provider = spending(clock, "0.01", max_seconds=1.0)
        clock.advance(5.0)
        run = await start(runner_, agent(tools=("lookup",)))
        assert run.state is RunState.BUDGET_EXHAUSTED
        assert provider.requests == []


class TestNothingIsSqueezedUnderTheCeiling:
    async def test_the_prompt_is_not_truncated_to_fit_the_remaining_budget(self) -> None:
        clock = FakeClock()
        runner_, provider = spending(clock, "0.01", "0.01", "0.01", max_iterations=2)
        run = await start(runner_, agent(tools=("lookup",)))
        assert run.state is RunState.BUDGET_EXHAUSTED
        lengths = [len(request.messages) for request in provider.requests]
        assert lengths == sorted(lengths)

    async def test_the_tools_are_not_dropped_to_make_the_call_cheaper(self) -> None:
        clock = FakeClock()
        runner_, provider = spending(clock, "0.01", "0.01", "0.01", max_iterations=2)
        await start(runner_, agent(tools=("lookup",)))
        assert {len(request.tools) for request in provider.requests} == {1}

    async def test_the_model_is_not_downgraded_to_make_the_call_cheaper(self) -> None:
        clock = FakeClock()
        runner_, provider = spending(clock, "0.01", "0.01", "0.01", max_iterations=2)
        await start(runner_, agent(tools=("lookup",)))
        assert {request.model for request in provider.requests} == {"scripted-1"}


class TestARetriedAttemptIsSpendToo:
    async def test_attempts_that_failed_are_charged_against_the_call_ceiling(self) -> None:
        clock = FakeClock()
        provider = ScriptedProvider(
            ProviderTimeoutError("no answer in 30s"),
            ProviderTimeoutError("no answer in 30s"),
            answer(),
            capabilities=CAPABLE,
        )
        run = await start(
            AgentRunner(provider=provider, clock=clock, budget=budget(clock, max_model_calls=2)),
            agent(retry=RetryConfig(max_attempts=3)),
        )
        assert run.state is RunState.BUDGET_EXHAUSTED

    async def test_a_failed_attempt_that_fits_leaves_the_run_free_to_finish(self) -> None:
        clock = FakeClock()
        provider = ScriptedProvider(
            ProviderTimeoutError("no answer in 30s"),
            answer(),
            capabilities=CAPABLE,
        )
        run = await start(
            AgentRunner(provider=provider, clock=clock, budget=budget(clock, max_model_calls=5)),
            agent(retry=RetryConfig(max_attempts=3)),
        )
        assert run.state is RunState.COMPLETED

    async def test_what_a_failed_attempt_burned_is_on_the_run(self) -> None:
        clock = FakeClock()
        provider = ScriptedProvider(
            ProviderTimeoutError("no answer in 30s"),
            answer(),
            capabilities=CAPABLE,
        )
        run = await start(
            AgentRunner(provider=provider, clock=clock, budget=budget(clock, max_model_calls=5)),
            agent(retry=RetryConfig(max_attempts=3)),
        )
        failed = next(event for event in run.events if event.kind is RunEventKind.ATTEMPT_FAILED)
        assert failed.usage is not None
        assert failed.usage.input_tokens > 0


class TestACancelledRunSettlesWhatItSpent:
    async def test_a_cancelled_call_still_charges_what_it_had_reserved(self) -> None:
        """A cancelled run that reports nothing spent is a bill nobody can reconcile."""
        clock = FakeClock()
        limit = budget(clock, max_input_tokens=100_000)
        run = await self._cancelled_mid_call(clock, limit)
        assert run.state is RunState.CANCELLED
        assert limit.spent.usage.input_tokens > 0

    async def test_cancellation_wins_over_a_ceiling_reached_in_the_same_breath(self) -> None:
        """Two terminal states cannot both be true, and the caller's switch decides."""
        clock = FakeClock()
        limit = budget(clock, max_input_tokens=8)
        run = await self._cancelled_mid_call(clock, limit)
        assert run.state is RunState.CANCELLED

    async def _cancelled_mid_call(self, clock: FakeClock, limit: RunBudget) -> Run:
        provider = StallingProvider()
        token = CancellationToken()
        task = asyncio.ensure_future(
            AgentRunner(provider=provider, clock=clock, budget=limit).run(
                agent(), "plan a trip", tenant="acme", cancellation=token
            )
        )
        await provider.entered.wait()
        token.cancel("the caller changed their mind")
        return await task


class TestASideEffectThatOutlivedTheRun:
    async def test_a_tool_that_already_ran_is_marked_as_needing_compensation(self) -> None:
        clock = FakeClock()
        registry = tools()
        provider = ScriptedProvider(calling("lookup"), calling("lookup"), capabilities=CAPABLE)
        run = await start(
            AgentRunner(
                provider=provider,
                clock=clock,
                budget=budget(clock, max_tool_calls=1),
                tools=registry,
            ),
            agent(tools=("lookup",)),
        )
        assert run.state is RunState.BUDGET_EXHAUSTED
        compensate = [
            event.name for event in run.events if event.kind is RunEventKind.COMPENSATION_REQUIRED
        ]
        assert compensate == ["lookup"]

    async def test_the_tool_is_not_dispatched_again_while_the_run_unwinds(self) -> None:
        clock = FakeClock()
        registry = tools()
        provider = ScriptedProvider(calling("lookup"), calling("lookup"), capabilities=CAPABLE)
        await start(
            AgentRunner(
                provider=provider,
                clock=clock,
                budget=budget(clock, max_tool_calls=1),
                tools=registry,
            ),
            agent(tools=("lookup",)),
        )
        assert [name for name, _ in registry.calls] == ["lookup"]

    async def test_a_tool_declared_idempotent_needs_no_compensation(self) -> None:
        clock = FakeClock()
        provider = ScriptedProvider(calling("lookup"), calling("lookup"), capabilities=CAPABLE)
        run = await start(
            AgentRunner(
                provider=provider,
                clock=clock,
                budget=budget(clock, max_tool_calls=1),
                tools=tools(),
            ),
            agent(tools=("lookup",), idempotent_tools=("lookup",)),
        )
        assert run.state is RunState.BUDGET_EXHAUSTED
        assert RunEventKind.COMPENSATION_REQUIRED not in {event.kind for event in run.events}

    async def test_a_run_that_completed_leaves_nothing_to_compensate(self) -> None:
        run = await start(
            runner(calling("lookup"), answer(), tools=tools()),
            agent(tools=("lookup",)),
        )
        assert run.state is RunState.COMPLETED
        assert RunEventKind.COMPENSATION_REQUIRED not in {event.kind for event in run.events}


class TestTheLastIterationIsNotAFreeOne:
    async def test_a_ceiling_reached_where_the_answer_was_due_still_stops_the_run(self) -> None:
        """The turn that would have produced the result is a turn, and it is charged."""
        clock = FakeClock()
        runner_, provider = spending(clock, "0.01", "0.01", max_iterations=1)
        run = await start(runner_, agent(tools=("lookup",)))
        assert run.state is RunState.BUDGET_EXHAUSTED
        assert run.output is None
        assert len(provider.requests) == 1


class TestBookkeepingThatDoesNotDominateTheRun:
    async def test_a_run_scoped_ceiling_never_touches_the_shared_ledger(self) -> None:
        """Cheap high-volume calls pay no IO for a ceiling nobody else shares."""
        clock = FakeClock()
        ledger = _CountingLedger()
        run = await start(
            AgentRunner(
                provider=ScriptedProvider(
                    calling(), calling(tool="lookup"), answer(), capabilities=CAPABLE
                ),
                clock=clock,
                budget=RunBudget(
                    resolved=most_restrictive(
                        ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(max_model_calls=10))
                    ),
                    clock=clock,
                    ledger=ledger,
                    tenant="acme",
                ),
                tools=tools(),
            ),
            agent(tools=("lookup",)),
        )
        assert run.state is RunState.COMPLETED
        assert ledger.calls == 0

    async def test_a_shared_ceiling_costs_a_bounded_number_of_round_trips_per_call(self) -> None:
        """A round trip per chargeable operation — never one per dimension checked."""
        clock = FakeClock()
        ledger = _CountingLedger()
        run = await start(
            AgentRunner(
                provider=ScriptedProvider(calling(), answer(), capabilities=CAPABLE),
                clock=clock,
                budget=RunBudget(
                    resolved=most_restrictive(
                        ScopedLimits(
                            scope=BudgetScope.TENANT, limits=BudgetLimits(max_model_calls=10)
                        )
                    ),
                    clock=clock,
                    ledger=ledger,
                    tenant="acme",
                ),
                tools=tools(),
            ),
            agent(tools=("lookup",)),
        )
        assert run.state is RunState.COMPLETED
        kinds = [event.kind for event in run.events]
        chargeable = kinds.count(RunEventKind.MODEL_CALL) + kinds.count(RunEventKind.TOOL_CALL)
        assert ledger.reads == chargeable
        assert ledger.writes <= 2 * chargeable


class _CountingLedger(FakeTenantLedger):
    """A ledger that says how often it was asked, so a hot loop cannot quietly hammer it."""

    reads = 0
    writes = 0

    async def total(self, tenant: str, window: str) -> Consumed:
        """Count the read, then answer as usual."""
        self.reads += 1
        return await super().total(tenant, window)

    async def consume(self, tenant: str, window: str, spent: Consumed) -> Consumed:
        """Count the write, then apply it as usual."""
        self.writes += 1
        return await super().consume(tenant, window, spent)

    @property
    def calls(self) -> int:
        """How many round trips this ledger has served."""
        return self.reads + self.writes


class TestAnOvershootIsRecordedNotHidden:
    async def test_a_response_dearer_than_its_reservation_says_by_how_much(self) -> None:
        """A call dearer than the estimate lands the run over; the excess is the record."""
        clock = FakeClock()
        runner_, _ = repairing(clock, "2.00", "2.00", max_cost=Decimal("0.50"), currency="USD")
        run = await start(runner_, unvalidatable())
        assert run.state is RunState.BUDGET_EXHAUSTED
        assert "over by 1.50" in detail_of(run, RunEventKind.BUDGET_EXCEEDED)

    async def test_a_call_refused_before_dispatch_overshot_nothing(self) -> None:
        clock = FakeClock()
        runner_, provider = spending(clock, "0.10", max_input_tokens=4)
        run = await start(runner_, agent(tools=("lookup",)))
        assert provider.requests == []
        assert "over by 0" in detail_of(run, RunEventKind.BUDGET_EXCEEDED)


class TestASubAgentSpendsTheSameAllowance:
    async def test_a_tool_that_runs_an_agent_draws_on_the_calling_run(self) -> None:
        clock = FakeClock()
        limit = budget(clock, max_model_calls=10)
        inner = AgentRunner(
            provider=ScriptedProvider(answer(), capabilities=CAPABLE),
            clock=clock,
            budget=limit.child(),
        )
        await inner.run(agent(name="researcher"), "look it up", tenant="acme")
        assert limit.spent.model_calls == 1


class _UnreadableLedger(FakeTenantLedger):
    """Takes writes, cannot answer what the tenant has spent — the read-side outage."""

    async def total(self, tenant: str, window: str) -> Consumed:
        _ = window
        raise BudgetUnavailableError(f"the ledger for {tenant} is not answering reads")


class TestALedgerThatStoppedAnsweringMidRun:
    async def test_a_reservation_against_an_unreadable_ceiling_ends_the_run(self) -> None:
        """A ceiling nobody can read is not one to spend against on the optimistic reading."""
        clock = FakeClock()
        run = await start(
            AgentRunner(
                provider=ScriptedProvider(answer(), capabilities=CAPABLE),
                clock=clock,
                budget=self.tenant_wide(clock, _UnreadableLedger()),
            ),
            agent(),
        )
        assert run.state is RunState.FAILED
        assert "could not be read" in detail_of(run, RunEventKind.TERMINATED)

    def tenant_wide(self, clock: FakeClock, ledger: FakeTenantLedger) -> RunBudget:
        return RunBudget(
            resolved=most_restrictive(
                ScopedLimits(scope=BudgetScope.TENANT, limits=BudgetLimits(max_input_tokens=1_000))
            ),
            clock=clock,
            ledger=ledger,
            tenant="acme",
        )

    async def test_the_run_fails_rather_than_carrying_on_uncounted(self) -> None:
        """A ceiling nobody can read is not a ceiling, so the run stops rather than spend."""
        clock = FakeClock()
        run = await start(
            AgentRunner(
                provider=ScriptedProvider(answer(), capabilities=CAPABLE),
                clock=clock,
                budget=self.tenant_wide(clock, FakeTenantLedger(reachable=False)),
            ),
            agent(),
        )
        assert run.state is RunState.FAILED
        assert "acme" in detail_of(run, RunEventKind.TERMINATED)


class TestAnOvershootAgainstNoCeilingIsNothing:
    def test_a_dimension_with_no_limit_cannot_have_been_overshot(self) -> None:
        assert BudgetDecision(permitted=True).overshoot == Decimal(0)

    def test_what_went_past_the_ceiling_is_the_difference(self) -> None:
        decision = BudgetDecision(
            permitted=False,
            breached="max_cost",
            limit=Decimal("0.50"),
            consumed=Decimal("2.00"),
            remaining=Decimal(0),
        )
        assert decision.overshoot == Decimal("1.50")

    def test_a_ceiling_that_was_reached_but_not_passed_overshot_nothing(self) -> None:
        decision = BudgetDecision(
            permitted=False,
            breached="max_cost",
            limit=Decimal("0.50"),
            consumed=Decimal("0.50"),
            remaining=Decimal(0),
        )
        assert decision.overshoot == Decimal(0)


class TestAStreamThatPassesTheCeiling:
    async def test_the_stream_ends_with_a_typed_error_rather_than_stopping(self) -> None:
        clock = FakeClock()
        limit = budget(clock, max_output_tokens=10)
        seen: list[StreamEvent] = []
        with pytest.raises(BudgetExceededError):
            await _drain(budgeted_stream(_stream(50), limit), seen)
        assert [type(event) for event in seen] == [TextDelta]

    async def test_a_stream_inside_the_ceiling_passes_every_event_through(self) -> None:
        clock = FakeClock()
        limit = budget(clock, max_output_tokens=1_000)
        seen = [event async for event in budgeted_stream(_stream(50), limit)]
        assert len(seen) == 2
        assert limit.spent.usage.output_tokens == 50

    async def test_a_priced_stream_is_charged_the_unbilled_part_of_each_total(self) -> None:
        """Cost arrives as a running total too, and billing it twice is the easy mistake."""
        clock = FakeClock()
        limit = budget(clock, max_cost=Decimal("10.00"), currency="USD")
        async for _ in budgeted_stream(_priced_growing(), limit):
            pass
        assert limit.spent.usage.cost is not None
        assert limit.spent.usage.cost.total == Decimal("0.40")

    async def test_an_iterator_that_is_not_a_generator_still_passes_through(self) -> None:
        """Not every provider hands back something with an aclose to call."""
        clock = FakeClock()
        limit = budget(clock, max_output_tokens=1_000)
        seen = [event async for event in budgeted_stream(_ByHand(), limit)]
        assert len(seen) == 1

    async def test_the_stream_is_charged_once_however_many_totals_it_reports(self) -> None:
        clock = FakeClock()
        limit = budget(clock, max_output_tokens=1_000)
        async for _ in budgeted_stream(_growing(), limit):
            pass
        assert limit.spent.usage.output_tokens == 40


async def _drain(events: AsyncIterator[StreamEvent], seen: list[StreamEvent]) -> None:
    """Consume `events` into `seen`, so what arrived before the refusal is still readable."""
    async for event in events:
        seen.append(event)


async def _stream(output_tokens: int) -> AsyncIterator[StreamEvent]:
    yield TextDelta(text="Kyoto")
    yield UsageDelta(usage=Usage(input_tokens=10, output_tokens=output_tokens))


class _ByHand:
    """An async iterator written the long way, with no generator machinery behind it."""

    def __aiter__(self) -> _ByHand:
        return self

    async def __anext__(self) -> StreamEvent:
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return UsageDelta(usage=Usage(input_tokens=10, output_tokens=5))

    _done = False


async def _priced_growing() -> AsyncIterator[StreamEvent]:
    """A running total that carries money as well as tokens."""
    yield UsageDelta(
        usage=Usage(
            input_tokens=10, output_tokens=20, cost=Cost(input=Decimal("0.10"), currency="USD")
        )
    )
    yield UsageDelta(
        usage=Usage(
            input_tokens=10, output_tokens=40, cost=Cost(input=Decimal("0.40"), currency="USD")
        )
    )


async def _growing() -> AsyncIterator[StreamEvent]:
    """A vendor reporting a running total, where later events replace earlier ones."""
    yield UsageDelta(usage=Usage(input_tokens=10, output_tokens=20))
    yield TextDelta(text="Kyoto")
    yield UsageDelta(usage=Usage(input_tokens=10, output_tokens=40))
