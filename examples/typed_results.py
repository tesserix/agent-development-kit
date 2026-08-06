"""The declared answer type is what comes back, and what a checkpoint rehydrates as.

Three scenarios: a typed agent's fields reached without a cast, a prose agent carrying no
answer at all, and a round trip that needs its type parameter named. Run it with
`python examples/typed_results.py`, and read it with `mypy --strict` — the point is as much
what a checker infers here as what the prints say.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ValidationError

from tesserix_adk.core import Agent, NoOutput, Run, Usage
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider

PLAN = '{"destination": "Kyoto", "nights": 4}'


class TripPlan(BaseModel):
    """A trip proposed for a traveller.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


async def go[OutputT: BaseModel](agent: Agent[OutputT], answer: str) -> Run[OutputT]:
    """Drive one agent against a scripted provider and hand its run back, typed."""
    provider = ScriptedProvider(
        ModelResponse(content=answer, usage=Usage(input_tokens=10, output_tokens=5)),
        capabilities=CAPABLE.declaring(structured_output=True),
    )
    return await AgentRunner(provider=provider, clock=FakeClock()).run(
        agent, "plan a trip", tenant="acme"
    )


async def typed() -> None:
    """`Agent[TripPlan]` runs to `Run[TripPlan]`, so the fields are just there."""
    planner: Agent[TripPlan] = Agent(
        name="planner",
        instructions="Plan trips.",
        model="claude-sonnet-5",
        output_type=TripPlan,
    )
    finished = await go(planner, PLAN)

    print("output:   ", finished.output)  # noqa: T201
    print("nights:   ", finished.output.nights if finished.output else None)  # noqa: T201


async def prose() -> None:
    """An agent that answers in prose declares no answer type and carries none."""
    chatter: Agent[NoOutput] = Agent(
        name="chatter", instructions="Chat.", model="claude-sonnet-5", free_text=True
    )
    finished = await go(chatter, "Kyoto, four nights.")

    print("output:   ", finished.output)  # noqa: T201


async def rehydrated() -> None:
    """A checkpoint is JSON, and JSON does not say which type the answer was."""
    stored = (await go(planner_agent(), PLAN)).model_dump_json()

    print("restored: ", Run[TripPlan].model_validate_json(stored).output)  # noqa: T201
    try:
        Run.model_validate_json(stored)
    except ValidationError:
        print("unnamed:   refused")  # noqa: T201


def planner_agent() -> Agent[TripPlan]:
    """The typed agent again, so the round trip reads without repeating its fields."""
    return Agent(
        name="planner",
        instructions="Plan trips.",
        model="claude-sonnet-5",
        output_type=TripPlan,
    )


async def main() -> None:
    """Run every scenario in order."""
    await typed()
    await prose()
    await rehydrated()


if __name__ == "__main__":
    asyncio.run(main())
