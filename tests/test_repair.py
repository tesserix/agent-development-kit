"""Repair: a nearly-correct answer gets one bounded chance to be corrected, or fails.

The two failures this file exists to prevent are opposite. One is coercion — filling a
missing field with a default, dropping an unknown one, casting a type — which returns an
object the model never produced. The other is the blind retry: the same prompt sent again
in the hope of a different answer, charged to the caller, with the model never told what
was wrong. Repair is neither: the exact validation failure goes back, a bounded number of
times, and running out is a loud failure rather than a best-effort object.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from tesserix_adk.core import (
    Agent,
    RepairConfig,
    Run,
    RunEventKind,
    RunState,
    SchemaViolationError,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse, OutputContract
from tesserix_adk.testing import FakeBudgetPolicy, FakeClock, ScriptedProvider

BAD = '{"destination": "Kyoto"}'
HALF = '{"nights": 4}'
GOOD = '{"destination": "Kyoto", "nights": 4}'
PLAN = {"destination": "Kyoto", "nights": 4}


class TripPlan(BaseModel):
    """A trip the model proposes.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


def agent(**overrides: object) -> Agent[TripPlan]:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "model": "claude-sonnet-5",
        "output_type": TripPlan,
        "repair": RepairConfig(),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str) -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def runner(*responses: ModelResponse, **overrides: object) -> AgentRunner:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, structured=True),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


async def start[OutputT: BaseModel](runner_: AgentRunner, agent_: Agent[OutputT]) -> Run[OutputT]:
    return await runner_.run(agent_, "plan a trip", tenant="acme", run_id="run_1")


def kinds(run: Run[Any], kind: RunEventKind) -> list[str]:
    return [record.detail or "" for record in run.events if record.kind is kind]


def sent(provider: ScriptedProvider, index: int) -> str:
    return "\n".join(
        part.text
        for message in provider.requests[index].messages
        for part in message.content
        if hasattr(part, "text")
    )


class TestARepairBudgetIsDeclared:
    def test_an_agent_repairs_nothing_unless_it_says_so(self) -> None:
        """Fail-closed: a second charge on someone's account is asked for, never assumed."""
        assert (
            Agent(
                name="planner",
                instructions="Plan trips.",
                model="claude-sonnet-5",
                output_type=TripPlan,
            ).repair
            is None
        )

    def test_the_declared_default_is_small(self) -> None:
        assert RepairConfig().max_attempts == 2

    def test_a_budget_of_zero_is_refused_as_a_way_of_saying_off(self) -> None:
        with pytest.raises(ValidationError, match="enabled"):
            RepairConfig(max_attempts=0)

    def test_repair_is_turned_off_explicitly_rather_than_by_deletion(self) -> None:
        assert RepairConfig(enabled=False).enabled is False


class TestTheFailureItselfGoesBack:
    async def test_a_correctable_answer_is_corrected_and_the_run_completes(self) -> None:
        run = await start(runner(answer(BAD), answer(GOOD)), agent())
        assert run.state is RunState.COMPLETED
        assert run.output == TripPlan.model_validate(PLAN)

    async def test_exactly_one_repair_is_recorded(self) -> None:
        run = await start(runner(answer(BAD), answer(GOOD)), agent())
        assert len(kinds(run, RunEventKind.REPAIR_REQUESTED)) == 1

    async def test_the_recorded_repair_names_the_field_that_failed(self) -> None:
        run = await start(runner(answer(BAD), answer(GOOD)), agent())
        assert "nights" in kinds(run, RunEventKind.REPAIR_REQUESTED)[0]

    async def test_the_recorded_repair_names_which_attempt_it_is(self) -> None:
        run = await start(runner(answer(BAD), answer(GOOD)), agent())
        assert "1 of 2" in kinds(run, RunEventKind.REPAIR_REQUESTED)[0]

    async def test_the_failing_path_is_fed_back_to_the_model(self) -> None:
        provider = ScriptedProvider(answer(BAD), answer(GOOD), structured=True)
        await start(runner(provider=provider), agent())
        assert "nights" in sent(provider, 1)

    async def test_what_was_wrong_with_it_is_fed_back_to_the_model(self) -> None:
        provider = ScriptedProvider(answer(BAD), answer(GOOD), structured=True)
        await start(runner(provider=provider), agent())
        assert "Field required" in sent(provider, 1)

    async def test_the_schema_goes_back_with_it(self) -> None:
        provider = ScriptedProvider(answer(BAD), answer(GOOD), structured=True)
        await start(runner(provider=provider), agent())
        assert "destination" in sent(provider, 1)

    async def test_the_correction_invents_no_value_for_the_failing_field(self) -> None:
        """The kit reports what was wrong. Supplying the answer would be coercion by prompt."""
        contract = OutputContract.of(TripPlan)
        with pytest.raises(SchemaViolationError) as raised:
            contract.parse(BAD)
        assert "nights" in contract.repair_prompt(raised.value)


class TestRunningOutIsLoud:
    async def test_an_answer_that_never_validates_fails_the_run(self) -> None:
        run = await start(runner(answer(BAD), answer(HALF), answer("still not")), agent())
        assert run.state is RunState.FAILED

    async def test_no_best_effort_object_is_returned(self) -> None:
        run = await start(runner(answer(BAD), answer(HALF), answer("still not")), agent())
        assert run.output is None

    async def test_the_budget_bounds_the_attempts(self) -> None:
        provider = ScriptedProvider(answer(BAD), answer(HALF), answer("still not"), structured=True)
        await start(runner(provider=provider), agent())
        assert len(provider.requests) == 3

    async def test_every_attempt_is_recorded_as_a_violation(self) -> None:
        run = await start(runner(answer(BAD), answer(HALF), answer("still not")), agent())
        assert len(kinds(run, RunEventKind.SCHEMA_VIOLATION)) == 3

    async def test_the_last_attempts_raw_output_is_what_the_failure_reports(self) -> None:
        run = await start(runner(answer(BAD), answer(HALF), answer("still not")), agent())
        assert "not valid JSON" in kinds(run, RunEventKind.SCHEMA_VIOLATION)[-1]


class TestRepairIsPaidFor:
    async def test_the_repair_attempts_tokens_are_on_the_runs_usage(self) -> None:
        run = await start(runner(answer(BAD), answer(GOOD)), agent())
        assert run.usage.input_tokens == 20
        assert run.usage.output_tokens == 10

    async def test_the_repair_attempt_is_charged_to_the_budget(self) -> None:
        budget = FakeBudgetPolicy()
        await start(runner(answer(BAD), answer(GOOD), budget=budget), agent())
        assert budget.spent == 30

    async def test_repair_cannot_spend_past_the_ceiling(self) -> None:
        """A repair the budget will not pay for stops the run rather than proceeding."""
        budget = FakeBudgetPolicy(limit=20)
        run = await start(runner(answer(BAD), answer(GOOD), budget=budget), agent())
        assert run.state is RunState.BUDGET_EXHAUSTED


class TestRepairTurnedOff:
    async def test_an_undeclared_policy_makes_the_first_violation_terminal(self) -> None:
        run = await start(
            runner(answer(BAD), answer(GOOD)),
            agent(repair=None),
        )
        assert run.state is RunState.FAILED
        assert not kinds(run, RunEventKind.REPAIR_REQUESTED)

    async def test_an_explicitly_disabled_policy_makes_it_terminal_too(self) -> None:
        run = await start(
            runner(answer(BAD), answer(GOOD)),
            agent(repair=RepairConfig(enabled=False)),
        )
        assert run.state is RunState.FAILED
        assert not kinds(run, RunEventKind.REPAIR_REQUESTED)

    async def test_nothing_is_repaired_for_an_agent_that_answers_in_prose(self) -> None:
        run = await start(
            runner(answer("Kyoto, four nights.")),
            Agent(
                name="chatter",
                instructions="Chat.",
                model="claude-sonnet-5",
                free_text=True,
                repair=RepairConfig(),
            ),
        )
        assert run.state is RunState.COMPLETED
        assert not kinds(run, RunEventKind.REPAIR_REQUESTED)


class TestAConstraintNothingCanSatisfy:
    async def test_the_same_failure_twice_is_abandoned_rather_than_retried_out(self) -> None:
        """Told exactly what was wrong and answering identically means the ask is impossible."""
        run = await start(runner(answer(BAD), answer(BAD)), agent(repair=RepairConfig()))
        assert run.state is RunState.FAILED
        assert len(kinds(run, RunEventKind.REPAIR_REQUESTED)) == 1

    async def test_it_is_reported_as_a_defect_in_the_declaration(self) -> None:
        run = await start(runner(answer(BAD), answer(BAD)), agent(repair=RepairConfig()))
        assert "cannot be satisfied" in kinds(run, RunEventKind.REPAIR_ABANDONED)[0]

    async def test_a_different_failure_is_still_worth_repairing(self) -> None:
        run = await start(
            runner(answer(BAD), answer(HALF), answer(GOOD)),
            agent(repair=RepairConfig(max_attempts=3)),
        )
        assert run.state is RunState.COMPLETED
        assert run.output == TripPlan.model_validate(PLAN)


class TestNothingIsCoerced:
    async def test_a_missing_field_is_never_defaulted(self) -> None:
        run = await start(runner(answer(BAD), answer(HALF), answer(BAD)), agent())
        assert run.output is None

    async def test_a_value_of_the_wrong_type_is_never_cast_into_shape(self) -> None:
        wrong = json.dumps({"destination": "Kyoto", "nights": "four"})
        run = await start(runner(answer(wrong), answer(wrong)), agent())
        assert run.state is RunState.FAILED
        assert run.output is None

    async def test_a_repaired_answer_is_the_models_own_and_is_carried_as_the_declared_type(
        self,
    ) -> None:
        run = await start(runner(answer(BAD), answer(f"```json\n{GOOD}\n```")), agent())
        assert run.output == TripPlan.model_validate(PLAN)
