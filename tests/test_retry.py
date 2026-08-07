"""Retrying what deserves a second attempt, and nothing else.

The two failures this file exists to prevent: retrying a payment because the response
timed out, and every process in a fleet retrying a blip on the same schedule so the
recovering provider is knocked over again.
"""

from __future__ import annotations

import asyncio
from random import Random

import pytest

from tesserix_adk.core import (
    Agent,
    BudgetExceededError,
    BudgetLimits,
    CapabilityError,
    DeadlineConfig,
    GuardrailViolationError,
    ProviderError,
    ProviderTimeoutError,
    RetryConfig,
    Run,
    RunEventKind,
    RunState,
    SchemaViolationError,
    ToolCall,
    ToolExecutionError,
    ToolFailurePolicy,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, CancellationToken, ModelResponse, RetryPlan
from tesserix_adk.testing import (
    FakeBudgetPolicy,
    FakeClock,
    FakeToolRegistry,
    ScriptedProvider,
)


class TestWhatDeservesASecondAttempt:
    def test_a_timeout_is_retryable(self) -> None:
        assert ProviderTimeoutError("no answer in 30s").retryable is True

    def test_a_transport_fault_with_no_status_is_retryable(self) -> None:
        """A rejected request always carries a status; a reset connection cannot."""
        assert ProviderError("connection reset").retryable is True

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_a_transient_status_is_retryable(self, status: int) -> None:
        assert ProviderError("upstream", status=status).retryable is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_a_rejected_request_is_not_retryable(self, status: int) -> None:
        """Sending the same malformed request again gets the same 400, at the same price."""
        assert ProviderError("bad request", status=status).retryable is False

    @pytest.mark.parametrize(
        "error",
        [
            CapabilityError("no vision"),
            GuardrailViolationError("refused"),
            BudgetExceededError("over"),
            SchemaViolationError("not the declared shape"),
            ToolExecutionError("charge_card failed"),
        ],
    )
    def test_a_decision_is_never_retryable(self, error: Exception) -> None:
        """These are answers, not faults. Asking again gets the same answer."""
        assert getattr(error, "retryable") is False  # noqa: B009

    def test_an_unrelated_exception_is_not_retryable(self) -> None:
        assert RetryPlan(RetryConfig(max_attempts=3)).retryable(ValueError("nope")) is False


class TestDeclaringARetryPolicy:
    def test_nothing_is_retried_by_default(self) -> None:
        """A retry is a second charge on someone's account; the kit does not assume it."""
        assert RetryConfig().max_attempts == 1

    def test_zero_attempts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one attempt"):
            RetryConfig(max_attempts=0)

    def test_a_negative_delay_is_refused(self) -> None:
        with pytest.raises(ValueError, match="base_delay_seconds"):
            RetryConfig(base_delay_seconds=-1)

    def test_a_shrinking_backoff_is_refused(self) -> None:
        """A multiplier below 1 retries faster each time, which is the storm, not the cure."""
        with pytest.raises(ValueError, match="multiplier"):
            RetryConfig(multiplier=0.5)

    def test_an_agent_may_declare_its_own(self) -> None:
        declared = Agent(
            name="planner",
            instructions="Plan trips.",
            model="claude-sonnet-5",
            free_text=True,
            retry=RetryConfig(max_attempts=3),
        )
        assert declared.retry is not None
        assert declared.retry.max_attempts == 3


def drawn(plan: RetryPlan, attempt: int) -> float:
    delay = plan.delay_for(attempt)
    assert delay is not None
    return delay


class TestTheBackoff:
    def test_a_delay_is_drawn_from_the_full_window(self) -> None:
        """Full jitter, not a fixed schedule: aligned backoffs are a second outage."""
        plan = RetryPlan(RetryConfig(max_attempts=4, base_delay_seconds=1), random=Random(7))
        assert plan.delay_for(1) != RetryPlan(
            RetryConfig(max_attempts=4, base_delay_seconds=1), random=Random(8)
        ).delay_for(1)

    def test_a_seeded_source_is_deterministic(self) -> None:
        def plan() -> RetryPlan:
            return RetryPlan(RetryConfig(max_attempts=4, base_delay_seconds=1), random=Random(7))

        assert [plan().delay_for(n) for n in (1, 2, 3)] == [plan().delay_for(n) for n in (1, 2, 3)]

    def test_the_window_widens_with_each_attempt(self) -> None:
        config = RetryConfig(max_attempts=5, base_delay_seconds=1, multiplier=2)
        plan = RetryPlan(config, random=Random(1))
        assert drawn(plan, 1) <= 1
        assert drawn(plan, 2) <= 2
        assert drawn(plan, 3) <= 4

    def test_the_window_stops_widening_at_the_cap(self) -> None:
        config = RetryConfig(
            max_attempts=20, base_delay_seconds=1, multiplier=2, max_delay_seconds=5
        )
        plan = RetryPlan(config, random=Random(1))
        assert all(drawn(plan, n) <= 5 for n in range(1, 20))

    def test_a_provider_that_names_a_time_is_believed_over_the_backoff(self) -> None:
        plan = RetryPlan(RetryConfig(max_attempts=3, base_delay_seconds=1), random=Random(7))
        assert plan.delay_for(1, retry_after=2.5) == 2.5

    def test_a_hostile_retry_after_is_refused_rather_than_obeyed(self) -> None:
        """Waiting an hour stalls the run; retrying sooner ignores a real quota. Stop instead."""
        plan = RetryPlan(RetryConfig(max_attempts=3, max_retry_after_seconds=60))
        assert plan.delay_for(1, retry_after=3600) is None

    def test_the_last_attempt_schedules_nothing(self) -> None:
        assert RetryPlan(RetryConfig(max_attempts=2)).delay_for(2) is None


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


def answer() -> ModelResponse:
    return ModelResponse(content="done", usage=Usage(input_tokens=10, output_tokens=5))


def calling(tool: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name=tool, arguments={}),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


THREE = RetryConfig(max_attempts=3, base_delay_seconds=1, multiplier=2)


class TestRetryingAModelCall:
    async def test_a_transient_fault_is_recovered_without_the_caller_seeing_it(self) -> None:
        clock = FakeClock()
        provider = ScriptedProvider(
            ProviderError("upstream", status=503),
            ProviderError("upstream", status=503),
            answer(),
        )
        runner = AgentRunner(provider=provider, clock=clock, retry=THREE, jitter=Random(7))

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert len(provider.requests) == 3

    async def test_the_delays_are_exactly_what_the_seeded_source_drew(self) -> None:
        """Deterministic by injection: the suite asserts the schedule, it does not wait it out."""
        clock = FakeClock()
        provider = ScriptedProvider(
            ProviderError("upstream", status=503),
            ProviderError("upstream", status=503),
            answer(),
        )
        runner = AgentRunner(provider=provider, clock=clock, retry=THREE, jitter=Random(7))

        await runner.run(agent(), "plan a trip", tenant="acme")

        expected = Random(7)
        assert clock.slept == [expected.uniform(0, 1), expected.uniform(0, 2)]

    async def test_every_attempt_is_on_the_record(self) -> None:
        provider = ScriptedProvider(ProviderError("upstream", status=503), answer())
        runner = AgentRunner(provider=provider, clock=FakeClock(), retry=THREE)

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert kinds(run).count(RunEventKind.MODEL_CALL) == 2
        assert kinds(run).count(RunEventKind.ATTEMPT_FAILED) == 1

    async def test_a_rejected_request_is_not_sent_again(self) -> None:
        """A 400 retried three times is the same 400, three times the price."""
        provider = ScriptedProvider(ProviderError("bad request", status=400))
        runner = AgentRunner(provider=provider, clock=FakeClock(), retry=THREE)

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.FAILED
        assert len(provider.requests) == 1
        assert "not retryable" in details(run, RunEventKind.ATTEMPT_FAILED)[0]

    async def test_exhausted_attempts_carry_the_whole_history_not_just_the_last(self) -> None:
        provider = ScriptedProvider(
            ProviderError("first", status=503),
            ProviderError("second", status=502),
            ProviderError("third", status=500),
        )
        runner = AgentRunner(provider=provider, clock=FakeClock(), retry=THREE)

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.FAILED
        recorded = " ".join(details(run, RunEventKind.ATTEMPT_FAILED))
        assert "first" in recorded
        assert "second" in recorded
        assert "third" in recorded

    async def test_a_provider_that_names_a_time_is_waited_for(self) -> None:
        clock = FakeClock()
        provider = ScriptedProvider(
            ProviderError("slow down", status=429, retry_after=2.5), answer()
        )
        runner = AgentRunner(provider=provider, clock=clock, retry=THREE)

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert clock.slept == [2.5]

    async def test_a_quota_dressed_as_a_rate_limit_escalates_rather_than_stalling(self) -> None:
        clock = FakeClock()
        provider = ScriptedProvider(
            ProviderError("come back in an hour", status=429, retry_after=3600), answer()
        )
        runner = AgentRunner(
            provider=provider,
            clock=clock,
            retry=RetryConfig(max_attempts=3, max_retry_after_seconds=60),
        )

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.FAILED
        assert clock.slept == []
        assert "longer than" in details(run, RunEventKind.ATTEMPT_FAILED)[0]

    async def test_a_retry_never_outlives_the_run_deadline(self) -> None:
        """A backoff that would land past the deadline is not a backoff, it is a stall."""
        clock = FakeClock()  # `slept` also holds the deadline watcher, so count calls instead
        provider = ScriptedProvider(
            ProviderError("slow down", status=429, retry_after=30), answer()
        )
        runner = AgentRunner(
            provider=provider,
            clock=clock,
            retry=THREE,
            deadlines=DeadlineConfig(run_seconds=5),
        )

        run = await runner.run(agent(), "plan a trip", tenant="acme")

        assert run.state is RunState.FAILED
        assert len(provider.requests) == 1
        assert "no time left" in details(run, RunEventKind.ATTEMPT_FAILED)[0]

    async def test_a_run_cancelled_during_a_backoff_stops_there(self) -> None:
        clock = FakeClock(auto_advance=False)
        provider = ScriptedProvider(ProviderError("upstream", status=503), answer())
        runner = AgentRunner(provider=provider, clock=clock, retry=THREE)
        token = CancellationToken()

        task = asyncio.ensure_future(
            runner.run(agent(), "plan a trip", tenant="acme", cancellation=token)
        )
        await clock.wait_for_sleep(1)
        token.cancel("caller went away")

        run = await task
        assert run.state is RunState.CANCELLED
        assert len(provider.requests) == 1

    async def test_a_budget_ceiling_ends_the_run_rather_than_funding_more_attempts(self) -> None:
        """Every attempt reserves, so the budget bounds retries without knowing about them."""
        provider = ScriptedProvider(*[ProviderError("upstream", status=503)] * 3)
        budget = FakeBudgetPolicy(limit=8)
        runner = AgentRunner(provider=provider, clock=FakeClock(), retry=THREE, budget=budget)

        run = await runner.run(agent(budget=BudgetLimits()), "plan a trip", tenant="acme")

        assert run.state is RunState.BUDGET_EXHAUSTED
        assert len(provider.requests) == 1


class TestRetryingATool:
    async def test_an_idempotent_tool_is_tried_again(self) -> None:
        attempts: list[int] = []

        def search() -> str:
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("socket closed")
            return "found"

        provider = ScriptedProvider(calling("search"), answer())
        runner = AgentRunner(
            provider=provider,
            clock=FakeClock(),
            tools=FakeToolRegistry({"search": search}),
            retry=THREE,
        )

        run = await runner.run(
            agent(tools=("search",), idempotent_tools=("search",)), "plan", tenant="acme"
        )

        assert run.state is RunState.COMPLETED
        assert len(attempts) == 2

    async def test_a_payment_is_never_charged_twice(self) -> None:
        """The kit refuses to risk a duplicate transaction; it never invents success either."""
        charges: list[int] = []

        def charge_card() -> str:
            charges.append(1)
            raise ConnectionError("socket closed")

        provider = ScriptedProvider(calling("charge_card"))
        runner = AgentRunner(
            provider=provider,
            clock=FakeClock(),
            tools=FakeToolRegistry({"charge_card": charge_card}),
            retry=THREE,
        )

        run = await runner.run(
            agent(tools=("charge_card",), on_tool_error=ToolFailurePolicy.FAIL_RUN),
            "pay",
            tenant="acme",
        )

        assert run.state is RunState.FAILED
        assert charges == [1]
        assert "not declared idempotent" in details(run, RunEventKind.TOOL_ERROR)[0]
