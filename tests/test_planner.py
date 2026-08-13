"""A planner that reasons, an executor that acts, and a validated plan between them.

One agent that plans and acts in the same breath executes its own hallucinations: a step
is a sentence, the sentence becomes a call, and nothing between the two ever said the tool
existed. This file is the counter-argument. A plan is typed, every step names a registered
tool and carries arguments the tool declared, and the whole plan is checked — registry,
allowlist, scope, grant — before the first step touches anything.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, Field

from tesserix_adk.core import (
    Agent,
    ApprovalDeniedError,
    ConfigurationError,
    PlanValidationError,
    RunEventKind,
)
from tesserix_adk.core.autonomy import (
    ActionClass,
    ActionRegistry,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    Ceiling,
    InMemoryGrants,
)
from tesserix_adk.core.budget import BudgetLimits
from tesserix_adk.core.errors import AutonomyRefusedError, IndeterminateOutcomeError
from tesserix_adk.core.hooks import ApprovalDecision, ApprovalRecord
from tesserix_adk.core.idempotency import idempotency_key
from tesserix_adk.core.primitives import TextPart
from tesserix_adk.runtime import (
    AgentPlanner,
    AgentRunner,
    InMemoryPlanStore,
    MemoryIdempotencyStore,
    ModelResponse,
    Plan,
    PlanExecutor,
    PlanStep,
    ToolContract,
)
from tesserix_adk.runtime.delegation import Delegation, DelegationScope
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

HELD = frozenset({"search_flights", "book_flight", "notify", "refund"})


class Search(BaseModel):
    """What the search tool declared it takes."""

    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)


class Booking(BaseModel):
    """What the booking tool declared it takes."""

    flight: str = Field(min_length=1)
    seats: int = Field(ge=1)


class Refund(BaseModel):
    """What the refund tool declared it takes, money and all."""

    account: str = Field(min_length=1)
    amount: Decimal
    currency: str = "GBP"


class Note(BaseModel):
    """What the notify tool declared it takes."""

    text: str = Field(min_length=1)


class Approver:
    """A gate with a fixed answer, standing in for whoever actually decides."""

    def __init__(self, *, granted: bool = True) -> None:
        self._granted = granted
        self.asked: list[ApprovalRecord] = []

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        self.asked.append(record)
        return ApprovalDecision(
            record_id=record.id,
            granted=self._granted,
            decided_by="ada",
            reason="reviewed",
        )


class Planners:
    """A planner that hands back what a test scripted, refusal feedback included."""

    def __init__(self, *plans: Plan | None) -> None:
        self._plans = list(plans)
        self.asked: list[str] = []

    async def plan(self, task: str, *, feedback: str = "") -> Plan:
        self.asked.append(feedback)
        given = self._plans.pop(0) if len(self._plans) > 1 else self._plans[0]
        if given is None:
            raise AssertionError("the executor asked for a plan it should not have asked for")
        return given.model_copy(update={"goal": task})


def contracts() -> tuple[ToolContract, ...]:
    return (
        ToolContract(tool="search_flights", accepts=Search),
        ToolContract(tool="book_flight", accepts=Booking, irreversible=True),
        ToolContract(tool="refund", accepts=Refund),
        ToolContract(tool="notify", accepts=Note),
    )


def registry(**overrides: Any) -> FakeToolRegistry:
    tools: dict[str, Any] = {
        "search_flights": lambda origin, destination: f"{origin}->{destination}: BA117",
        "book_flight": lambda flight, seats: f"booked {seats} on {flight}",
        "refund": lambda account, amount, currency="GBP": f"{amount} {currency} to {account}",
        "notify": lambda text: f"told them: {text}",
    }
    return FakeToolRegistry({**tools, **overrides})


def agent(name: str = "courier", *tools: str, **overrides: object) -> Agent[Any]:
    fields: dict[str, object] = {
        "name": name,
        "instructions": f"You are {name}.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": tools or tuple(sorted(HELD)),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def delegation(tools: frozenset[str] = HELD) -> Delegation:
    return Delegation.root(
        run_id="run_1",
        tenant="acme",
        agent="courier",
        user="ada",
        scope=DelegationScope(tools=tools),
    )


def ladder(
    *,
    level: AutonomyLevel = AutonomyLevel.ACT_WITHIN_LIMITS,
    ceiling: Decimal = Decimal("500"),
    clock: FakeClock | None = None,
) -> AutonomyLadder:
    """A ladder that knows refunds are money and knows what has been granted about them."""
    return AutonomyLadder(
        ActionRegistry(
            {
                "refund": ActionClass(
                    name="payment.refund", amount_field="amount", currency_field="currency"
                )
            }
        ),
        grants=InMemoryGrants(
            [
                AutonomyGrant(
                    id="grant_1",
                    tenant="acme",
                    action_class="payment.refund",
                    level=level,
                    granted_by="registrar",
                    issued_at=0.0,
                    expires_at=9_000.0,
                    ceiling=Ceiling(amount=ceiling, currency="GBP", window_seconds=3_600.0)
                    if level is AutonomyLevel.ACT_WITHIN_LIMITS
                    else None,
                )
            ]
        ),
        clock=clock or FakeClock(),
    )


def executor(
    *,
    tools: FakeToolRegistry | None = None,
    acting: Agent[Any] | None = None,
    scope: frozenset[str] = HELD,
    approvals: Approver | None = None,
    autonomy: AutonomyLadder | None = None,
    plans: InMemoryPlanStore | None = None,
    idempotency: MemoryIdempotencyStore | None = None,
    max_steps: int = 8,
    max_replans: int = 2,
) -> PlanExecutor:
    return PlanExecutor(
        tools or registry(),
        contracts(),
        agent=acting or agent(),
        delegation=delegation(scope),
        approvals=approvals,
        autonomy=autonomy,
        plans=plans,
        idempotency=idempotency,
        clock=FakeClock(),
        max_steps=max_steps,
        max_replans=max_replans,
    )


def step(id: str = "s1", tool: str = "search_flights", **arguments: Any) -> PlanStep:  # noqa: A002 — the field is called id
    return PlanStep(
        id=id,
        tool=tool,
        arguments=arguments or {"origin": "LHR", "destination": "JFK"},
    )


def booking() -> PlanStep:
    return step("s1", "book_flight", flight="BA117", seats=1)


def plan(*steps: PlanStep, goal: str = "get them to New York") -> Plan:
    return Plan(goal=goal, steps=steps or (step(),))


class TestWhatAPlanIs:
    """A plan is data with a shape, not a paragraph the executor reads charitably."""

    def test_a_step_names_a_tool_and_carries_arguments(self) -> None:
        assert step().tool == "search_flights"

    def test_two_steps_cannot_answer_to_one_id(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            Plan(goal="go", steps=(step("s1"), step("s1", "notify", text="done")))

    def test_a_step_cannot_wait_for_itself(self) -> None:
        with pytest.raises(ValueError, match="itself"):
            PlanStep(id="s1", tool="notify", arguments={"text": "hi"}, depends_on=("s1",))

    def test_a_plan_says_which_step_answers_to_a_name(self) -> None:
        assert plan().step("s1") is not None
        assert plan().step("s404") is None


class TestValidatingBeforeAnythingRuns:
    """Every check happens before the first step, or it happens after the damage."""

    def test_a_tool_the_registry_never_heard_of_is_refused(self) -> None:
        with pytest.raises(PlanValidationError) as refused:
            executor().validate(plan(step("s1", "wire_transfer", text="oops")))
        assert refused.value.reason == "unknown_tool"
        assert refused.value.tool == "wire_transfer"
        assert refused.value.step == "s1"

    def test_a_tool_this_agent_may_not_call_is_refused(self) -> None:
        acting = agent("courier", "search_flights", "notify")
        with pytest.raises(PlanValidationError) as refused:
            executor(acting=acting).validate(
                plan(step("s1", "book_flight", flight="BA117", seats=1))
            )
        assert refused.value.reason == "not_allowed"

    def test_a_tool_outside_the_delegated_scope_is_refused(self) -> None:
        narrow = frozenset({"search_flights", "notify"})
        with pytest.raises(PlanValidationError) as refused:
            executor(scope=narrow).validate(
                plan(step("s1", "book_flight", flight="BA117", seats=1))
            )
        assert refused.value.reason == "not_allowed"

    def test_arguments_the_tool_did_not_declare_are_refused(self) -> None:
        with pytest.raises(PlanValidationError) as refused:
            executor().validate(plan(step("s1", "book_flight", flight="BA117", seats=1, seat="9A")))
        assert refused.value.reason == "arguments"
        assert refused.value.violations == ("seat",)

    def test_arguments_that_violate_the_schema_carry_the_payload_that_produced_them(self) -> None:
        broken = step("s1", "book_flight", flight="BA117", seats=0)
        with pytest.raises(PlanValidationError) as refused:
            executor().validate(plan(broken))
        assert refused.value.reason == "arguments"
        assert refused.value.violations == ("seats",)
        assert refused.value.payload == {"flight": "BA117", "seats": 0}

    def test_an_argument_of_the_wrong_type_is_refused_rather_than_coerced(self) -> None:
        with pytest.raises(PlanValidationError) as refused:
            executor().validate(plan(step("s1", "book_flight", flight="BA117", seats="2")))
        assert refused.value.violations == ("seats",)

    def test_a_plan_longer_than_the_ceiling_is_refused_rather_than_truncated(self) -> None:
        long = plan(*(step(f"s{index}", "notify", text="hi") for index in range(4)))
        with pytest.raises(PlanValidationError) as refused:
            executor(max_steps=3).validate(long)
        assert refused.value.reason == "too_long"

    def test_a_step_waiting_on_a_step_nobody_planned_is_refused(self) -> None:
        stray = PlanStep(id="s2", tool="notify", arguments={"text": "hi"}, depends_on=("s9",))
        with pytest.raises(PlanValidationError) as refused:
            executor().validate(plan(step(), stray))
        assert refused.value.reason == "dependency"

    def test_steps_waiting_on_each_other_are_refused_before_they_deadlock(self) -> None:
        first = PlanStep(id="s1", tool="notify", arguments={"text": "a"}, depends_on=("s2",))
        second = PlanStep(id="s2", tool="notify", arguments={"text": "b"}, depends_on=("s1",))
        with pytest.raises(PlanValidationError) as refused:
            executor().validate(plan(first, second))
        assert refused.value.reason == "cycle"
        assert set(refused.value.violations) == {"s1", "s2"}

    def test_a_plan_with_no_steps_is_a_planner_failure_rather_than_a_no_op(self) -> None:
        with pytest.raises(PlanValidationError) as refused:
            executor().validate(Plan(goal="do nothing"))
        assert refused.value.reason == "empty"

    def test_a_valid_plan_comes_back_as_it_was_written(self) -> None:
        written = plan(step(), step("s2", "notify", text="found one"))
        assert executor().validate(written) == written

    def test_a_contract_for_a_tool_nothing_registers_is_a_wiring_mistake(self) -> None:
        with pytest.raises(ConfigurationError, match="wire_transfer"):
            PlanExecutor(
                registry(),
                (*contracts(), ToolContract(tool="wire_transfer", accepts=Note)),
                agent=agent(),
                delegation=delegation(),
            )

    def test_an_allowed_tool_with_no_contract_could_not_be_checked(self) -> None:
        with pytest.raises(ConfigurationError, match="notify"):
            PlanExecutor(
                registry(),
                contracts()[:3],
                agent=agent(),
                delegation=delegation(),
            )


class TestNothingRunsBeforeTheWholePlanIsValid:
    """A plan that fails validation halfway is a plan that half happened."""

    async def test_no_step_runs_when_a_later_step_is_invalid(self) -> None:
        tools = registry()
        broken = plan(step(), step("s2", "book_flight", flight="BA117", seats=0))
        with pytest.raises(PlanValidationError):
            await executor(tools=tools).execute(broken)
        assert tools.calls == []

    async def test_the_refusal_lands_on_the_record(self) -> None:
        acting = executor()
        with pytest.raises(PlanValidationError):
            await acting.execute(plan(step("s1", "wire_transfer", text="oops")))
        assert [event.kind for event in acting.events] == [RunEventKind.PLAN_REFUSED]


class TestExecuting:
    """Deterministic code runs the plan; nothing here asks a model anything."""

    async def test_every_step_runs_and_the_outcome_is_recorded_under_its_id(self) -> None:
        done = await executor().execute(plan(step(), step("s2", "notify", text="found one")))
        assert done.outcomes["s1"] == "LHR->JFK: BA117"
        assert done.outcomes["s2"] == "told them: found one"

    async def test_a_step_runs_after_what_it_waits_for(self) -> None:
        tools = registry()
        first = PlanStep(id="s1", tool="notify", arguments={"text": "second"}, depends_on=("s2",))
        second = PlanStep(id="s2", tool="notify", arguments={"text": "first"})
        await executor(tools=tools).execute(plan(first, second))
        assert [name for name, _ in tools.calls] == ["notify", "notify"]
        assert [call["text"] for _, call in tools.calls] == ["first", "second"]

    async def test_the_tool_is_called_with_what_the_planner_wrote(self) -> None:
        tools = registry()
        await executor(tools=tools).execute(plan(step("s1", "notify", text="hi")))
        assert tools.calls == [("notify", {"text": "hi"})]

    async def test_every_step_lands_on_the_record(self) -> None:
        acting = executor()
        await acting.execute(plan(step(), step("s2", "notify", text="found one")))
        assert [event.kind for event in acting.events] == [
            RunEventKind.PLANNED,
            RunEventKind.STEP_EXECUTED,
            RunEventKind.STEP_EXECUTED,
        ]

    async def test_a_finished_plan_says_so(self) -> None:
        done = await executor().execute(plan())
        assert done.complete
        assert done.results[0].step_id == "s1"

    async def test_an_outcome_that_is_not_text_is_recorded_as_json(self) -> None:
        tools = registry(notify=lambda text: {"told": text, "at": 1})
        done = await executor(tools=tools).execute(plan(step("s1", "notify", text="hi")))
        assert done.outcomes["s1"] == '{"at": 1, "told": "hi"}'


class TestRunningAStepOnlyOnce:
    """A step with a key runs once; a repeat returns what the first one returned."""

    async def test_a_step_that_already_ran_returns_what_it_returned(self) -> None:
        tools = registry()
        acting = executor(tools=tools, idempotency=MemoryIdempotencyStore())
        await acting.execute(plan(step("s1", "notify", text="hi")))
        again = await acting.execute(plan(step("s1", "notify", text="hi")))
        assert again.results[0].replayed
        assert again.outcomes["s1"] == "told them: hi"
        assert len(tools.calls) == 1

    async def test_a_step_another_caller_holds_is_indeterminate_rather_than_repeated(self) -> None:
        store = MemoryIdempotencyStore()
        held = idempotency_key(
            tenant="acme", run_id="run_1", tool="notify", arguments={"text": "hi"}
        )
        assert held is not None
        await store.begin(held, tenant="acme", ttl_seconds=60.0)
        with pytest.raises(IndeterminateOutcomeError, match="nobody can say yet"):
            await executor(idempotency=store).execute(plan(step("s1", "notify", text="hi")))

    async def test_a_step_that_failed_leaves_its_key_free_for_the_next_attempt(self) -> None:
        store = MemoryIdempotencyStore()
        broken = executor(tools=registry(notify=_raises), idempotency=store)
        with pytest.raises(RuntimeError, match="went down"):
            await broken.execute(plan(step("s1", "notify", text="hi")))
        done = await executor(idempotency=store).execute(plan(step("s1", "notify", text="hi")))
        assert done.outcomes["s1"] == "told them: hi"
        assert not done.results[0].replayed


class TestWhatTouchesTheWorld:
    """An irreversible step is cleared by a person or by a grant, never by confidence."""

    async def test_an_irreversible_step_waits_for_a_person(self) -> None:
        gate = Approver()
        await executor(approvals=gate).execute(
            plan(step("s1", "book_flight", flight="BA1", seats=1))
        )
        assert [record.tool_name for record in gate.asked] == ["book_flight"]

    async def test_a_reversible_step_asks_nobody(self) -> None:
        gate = Approver()
        await executor(approvals=gate).execute(plan(step()))
        assert gate.asked == []

    async def test_a_denied_step_stops_the_plan_before_it_runs(self) -> None:
        tools = registry()
        gate = Approver(granted=False)
        booking = plan(step("s1", "book_flight", flight="BA1", seats=1))
        with pytest.raises(ApprovalDeniedError):
            await executor(tools=tools, approvals=gate).execute(booking)
        assert tools.calls == []

    async def test_an_irreversible_step_with_nobody_to_ask_is_a_wiring_mistake(self) -> None:
        with pytest.raises(ConfigurationError, match="no approval gate"):
            await executor().execute(plan(step("s1", "book_flight", flight="BA1", seats=1)))

    async def test_a_plan_is_cleared_in_full_before_its_first_step_runs(self) -> None:
        tools = registry()
        gate = Approver(granted=False)
        both = plan(
            step("s1", "notify", text="starting"),
            step("s2", "book_flight", flight="BA1", seats=1),
        )
        with pytest.raises(ApprovalDeniedError):
            await executor(tools=tools, approvals=gate).execute(both)
        assert tools.calls == []

    async def test_money_within_a_grant_needs_no_person(self) -> None:
        gate = Approver()
        refund = plan(step("s1", "refund", account="ac_9", amount=Decimal("100")))
        await executor(approvals=gate, autonomy=ladder()).execute(refund)
        assert gate.asked == []

    async def test_money_over_the_ceiling_goes_to_a_person(self) -> None:
        gate = Approver()
        refund = plan(step("s1", "refund", account="ac_9", amount=Decimal("900")))
        await executor(approvals=gate, autonomy=ladder()).execute(refund)
        assert [record.tool_name for record in gate.asked] == ["refund"]

    async def test_an_action_no_grant_could_permit_is_refused_outright(self) -> None:
        issuing = AutonomyLadder(
            ActionRegistry({"refund": ActionClass(name="autonomy.grant")}),
            grants=InMemoryGrants(),
            clock=FakeClock(),
        )
        refund = plan(step("s1", "refund", account="ac_9", amount=Decimal("1")))
        with pytest.raises(AutonomyRefusedError):
            await executor(approvals=Approver(), autonomy=issuing).execute(refund)

    async def test_an_escalation_lands_on_the_record(self) -> None:
        acting = executor(approvals=Approver(), autonomy=ladder())
        refund = plan(step("s1", "refund", account="ac_9", amount=Decimal("900")))
        await acting.execute(refund)
        kinds = [event.kind for event in acting.events]
        assert RunEventKind.AUTONOMY_ESCALATED in kinds
        assert RunEventKind.APPROVAL_GRANTED in kinds


class TestWhatThePlannerMayDo:
    """A planner reasons. Anything that could act is not a planner."""

    def test_a_planner_holding_tools_is_refused(self) -> None:
        planning = Agent(
            name="planner",
            instructions="Plan the trip.",
            model="claude-sonnet-5",
            output_type=Plan,
            tools=("book_flight",),
        )
        with pytest.raises(ConfigurationError, match="dispatch"):
            AgentPlanner(runner(), planning, delegation=delegation())

    def test_a_planner_that_answers_in_prose_is_refused(self) -> None:
        prose = Agent(
            name="planner", instructions="Plan the trip.", model="claude-sonnet-5", free_text=True
        )
        with pytest.raises(ConfigurationError, match="Plan"):
            # Typed away at compile time; the check exists for the caller who is not typed.
            AgentPlanner(runner(), prose, delegation=delegation())  # type: ignore[arg-type]

    async def test_the_plan_is_what_the_planner_returned(self) -> None:
        written = plan(step("s1", "notify", text="hi"))
        planner = AgentPlanner(
            runner(ModelResponse(content=written.model_dump_json())),
            planning_agent(),
            delegation=delegation(),
        )
        assert (await planner.plan("tell them")).steps[0].tool == "notify"

    async def test_a_planner_run_that_produced_nothing_is_a_refusal_rather_than_a_no_op(
        self,
    ) -> None:
        starved = planning_agent().model_copy(update={"budget": BudgetLimits(max_input_tokens=1)})
        planner = AgentPlanner(
            runner(ModelResponse(content="{}")), starved, delegation=delegation()
        )
        with pytest.raises(PlanValidationError) as refused:
            await planner.plan("tell them")
        assert refused.value.reason == "empty"

    async def test_feedback_reaches_the_planner_with_what_it_got_wrong(self) -> None:
        written = plan(step("s1", "notify", text="hi"))
        provider = ScriptedProvider(ModelResponse(content=written.model_dump_json()))
        planner = AgentPlanner(runner(provider=provider), planning_agent(), delegation=delegation())
        await planner.plan("tell them", feedback="seats: greater than or equal to 1")
        assert "seats" in _asked(provider)


class TestReplanning:
    """A planner allowed to try again forever is a loop with a model in it."""

    async def test_a_refused_plan_is_planned_again(self) -> None:
        good = plan(step("s1", "notify", text="hi"))
        planner = Planners(plan(step("s1", "wire_transfer", text="oops")), good)
        assert (await executor().planned(planner, "tell them")).steps[0].tool == "notify"

    async def test_the_second_attempt_is_told_what_the_first_got_wrong(self) -> None:
        planner = Planners(plan(step("s1", "wire_transfer", text="oops")), plan(step()))
        await executor().planned(planner, "tell them")
        assert planner.asked[0] == ""
        assert "wire_transfer" in planner.asked[1]

    async def test_a_planner_that_keeps_getting_it_wrong_is_capped(self) -> None:
        planner = Planners(plan(step("s1", "wire_transfer", text="oops")))
        with pytest.raises(PlanValidationError) as refused:
            await executor(max_replans=2).planned(planner, "tell them")
        assert refused.value.reason == "replan"
        assert refused.value.attempts == 3
        assert len(planner.asked) == 3

    async def test_every_replan_says_why_it_happened(self) -> None:
        acting = executor(max_replans=1)
        planner = Planners(plan(step("s1", "wire_transfer", text="oops")))
        with pytest.raises(PlanValidationError):
            await acting.planned(planner, "tell them")
        replans = [event for event in acting.events if event.kind is RunEventKind.REPLANNED]
        assert [event.detail for event in replans] == ["unknown_tool"]

    async def test_a_plan_that_validates_first_time_is_not_replanned(self) -> None:
        acting = executor()
        await acting.planned(Planners(plan()), "tell them")
        assert [event.kind for event in acting.events] == [RunEventKind.PLANNED]


class TestResuming:
    """Execution that died halfway carries on, and does not do the done part again."""

    async def test_the_plan_is_written_down_before_the_first_step(self) -> None:
        store = InMemoryPlanStore()
        await executor(plans=store).execute(plan())
        record = await store.latest("run_1", tenant="acme")
        assert record is not None
        assert record.plan.steps[0].id == "s1"

    async def test_a_step_that_failed_leaves_what_finished_on_the_record(self) -> None:
        store = InMemoryPlanStore()
        tools = registry(notify=_raises)
        both = plan(step(), step("s2", "notify", text="hi"))
        with pytest.raises(RuntimeError):
            await executor(tools=tools, plans=store).execute(both)
        record = await store.latest("run_1", tenant="acme")
        assert record is not None
        assert [result.step_id for result in record.results] == ["s1"]

    async def test_resuming_runs_what_is_left_and_nothing_else(self) -> None:
        store = InMemoryPlanStore()
        failing = registry(notify=_raises)
        both = plan(step(), step("s2", "notify", text="hi"))
        with pytest.raises(RuntimeError):
            await executor(tools=failing, plans=store).execute(both)

        working = registry()
        done = await executor(tools=working, plans=store).resume()
        assert [name for name, _ in working.calls] == ["notify"]
        assert done.outcomes["s1"] == "LHR->JFK: BA117"
        assert done.complete

    async def test_a_resume_revalidates_against_the_schema_as_it_is_now(self) -> None:
        store = InMemoryPlanStore()
        failing = registry(notify=_raises)
        both = plan(step("s1", "notify", text="hi"), step("s2", "notify", text="bye"))
        with pytest.raises(RuntimeError):
            await executor(tools=failing, plans=store).execute(both)

        moved = PlanExecutor(
            registry(),
            (*contracts()[:3], ToolContract(tool="notify", accepts=Search)),
            agent=agent(),
            delegation=delegation(),
            plans=store,
            clock=FakeClock(),
        )
        with pytest.raises(PlanValidationError) as refused:
            await moved.resume()
        assert refused.value.reason == "arguments"

    async def test_resuming_a_run_nothing_wrote_down_is_a_wiring_mistake(self) -> None:
        with pytest.raises(ConfigurationError, match="nothing to resume"):
            await executor(plans=InMemoryPlanStore()).resume()

    async def test_an_executor_with_nowhere_to_write_cannot_resume(self) -> None:
        with pytest.raises(ConfigurationError, match="no plan store"):
            await executor().resume()

    async def test_forgetting_a_plan_leaves_nothing_behind(self) -> None:
        store = InMemoryPlanStore()
        await executor(plans=store).execute(plan())
        await store.forget("run_1", tenant="acme")
        assert await store.latest("run_1", tenant="acme") is None


def _asked(provider: ScriptedProvider) -> str:
    """Everything the planning run put in front of the model, as one string."""
    return "\n".join(
        part.text
        for message in provider.requests[0].messages
        for part in message.content
        if isinstance(part, TextPart)
    )


def _raises(**_: object) -> str:
    raise RuntimeError("the tool went down")


def planning_agent() -> Agent[Plan]:
    return Agent(
        name="planner",
        instructions="Plan the trip.",
        model="claude-sonnet-5",
        output_type=Plan,
    )


def runner(*responses: ModelResponse, provider: ScriptedProvider | None = None) -> AgentRunner:
    return AgentRunner(
        provider=provider or ScriptedProvider(*responses),
        clock=FakeClock(),
        tools=registry(),
    )
