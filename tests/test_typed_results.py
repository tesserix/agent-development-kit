"""Typed results: `Agent[TripPlan]` runs to a `Run[TripPlan]`, and the checker knows it.

Validation that a type checker cannot see is validation a caller still has to cast around.
The assertions here are of two kinds. `assert_type` states what the checker must infer and
is a no-op at runtime; a `type: ignore[code]` on a line of deliberate misuse states that
the checker must reject it — `warn_unused_ignores` fails the build if it stops doing so.
Both are checked by `mypy --strict` in `make check`, so a widened signature fails there
rather than in a consumer's editor.
"""

from __future__ import annotations

from typing import assert_type

import pytest
from pydantic import BaseModel, ValidationError

from tesserix_adk.core import Agent, NoOutput, Run, RunState, Usage
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

PLAN = {"destination": "Kyoto", "nights": 4}


class TripPlan(BaseModel):
    """A trip the model proposes.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


class Invoice(BaseModel):
    """Something an itinerary is not.

    Args:
        total: What is owed.
    """

    total: int


def planner() -> Agent[TripPlan]:
    return Agent(
        name="planner",
        instructions="Plan trips.",
        model="claude-sonnet-5",
        output_type=TripPlan,
    )


def chatter() -> Agent[NoOutput]:
    return Agent(name="chatter", instructions="Chat.", model="claude-sonnet-5", free_text=True)


async def go[OutputT: BaseModel](agent: Agent[OutputT], answer: str) -> Run[OutputT]:
    provider = ScriptedProvider(
        ModelResponse(content=answer, usage=Usage(input_tokens=10, output_tokens=5)),
        structured=True,
    )
    runner = AgentRunner(provider=provider, clock=FakeClock())
    return await runner.run(agent, "plan a trip", tenant="acme", run_id="run_1")


class TestTheDeclaredTypeSurvivesTheRun:
    async def test_the_answer_comes_back_as_the_declared_type(self) -> None:
        run = await go(planner(), '{"destination": "Kyoto", "nights": 4}')
        assert run.output == TripPlan(destination="Kyoto", nights=4)

    async def test_the_checker_knows_which_type_it_is(self) -> None:
        run = await go(planner(), '{"destination": "Kyoto", "nights": 4}')
        assert_type(run.output, "TripPlan | None")
        assert run.state is RunState.COMPLETED

    async def test_the_fields_are_reachable_without_a_cast(self) -> None:
        run = await go(planner(), '{"destination": "Kyoto", "nights": 4}')
        assert run.output is not None
        assert run.output.nights == 4

    async def test_a_prose_agent_carries_no_output_at_all(self) -> None:
        run = await go(chatter(), "Kyoto, four nights.")
        assert_type(run.output, "NoOutput | None")
        assert run.output is None

    async def test_a_run_from_the_loop_checkpoints_its_answer(self) -> None:
        """A run whose class forgot the parameter serialises the answer away to `{}`."""
        run = await go(planner(), '{"destination": "Kyoto", "nights": 4}')
        restored = Run[TripPlan].model_validate_json(run.model_dump_json())
        assert restored.output == TripPlan(destination="Kyoto", nights=4)


class TestMisuseIsRejectedBeforeItRuns:
    """Each ignore states a rejection the checker must keep making."""

    async def test_the_output_of_one_agent_is_not_the_output_of_another(self) -> None:
        run = await go(planner(), '{"destination": "Kyoto", "nights": 4}')
        wrong: Invoice | None = run.output  # type: ignore[assignment]
        assert wrong is not None

    async def test_a_prose_agent_is_not_a_typed_one(self) -> None:
        typed: Agent[TripPlan] = chatter()  # type: ignore[assignment]
        assert typed.output_type is None

    def test_a_run_of_one_type_is_not_a_run_of_another(self) -> None:
        run: Run[TripPlan] = Run[Invoice](
            id="run_1",
            tenant="acme",
            agent_name="planner",
            agent_version="1.0.0",
            model="claude-sonnet-5",
        )  # type: ignore[assignment]
        assert run.output is None


class TestTheParameterSurvivesSerialisation:
    def run_with_output(self) -> Run[TripPlan]:
        return Run[TripPlan](
            id="run_1",
            tenant="acme",
            agent_name="planner",
            agent_version="1.0.0",
            model="claude-sonnet-5",
        ).with_output(TripPlan(destination="Kyoto", nights=4))

    def test_a_checkpointed_run_rehydrates_as_the_same_type(self) -> None:
        restored = Run[TripPlan].model_validate_json(self.run_with_output().model_dump_json())
        assert restored.output == TripPlan(destination="Kyoto", nights=4)

    def test_the_rehydrated_output_is_an_instance_rather_than_a_payload(self) -> None:
        restored = Run[TripPlan].model_validate_json(self.run_with_output().model_dump_json())
        assert_type(restored.output, "TripPlan | None")
        assert isinstance(restored.output, TripPlan)

    def test_rehydrating_as_the_wrong_type_is_refused_rather_than_dropped(self) -> None:
        """The parameter is load-bearing: a payload that is not that type does not fit."""
        with pytest.raises(ValidationError):
            Run[Invoice].model_validate_json(self.run_with_output().model_dump_json())

    def test_rehydrating_without_the_parameter_is_refused_rather_than_dropped(self) -> None:
        """Nothing on the wire says which type it was, so an unparameterised read is a guess."""
        with pytest.raises(ValidationError):
            Run.model_validate_json(self.run_with_output().model_dump_json())

    def test_an_output_of_the_wrong_type_is_refused_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            Run[TripPlan](
                id="run_1",
                tenant="acme",
                agent_name="planner",
                agent_version="1.0.0",
                model="claude-sonnet-5",
                output=Invoice(total=1),  # type: ignore[arg-type]
            )


class TestDownstreamProductsInheritTheTypes:
    def test_the_package_is_marked_as_typed(self) -> None:
        """`py.typed` is what makes any of the above visible to a consumer's checker."""
        from importlib.resources import files

        assert files("tesserix_adk").joinpath("py.typed").is_file()
