"""An answer that nearly validated is corrected once, or fails loudly.

Four scenarios: a correctable answer being corrected, what actually goes back to the model,
a budget running out without a best-effort object, and an ask nothing can satisfy being
abandoned rather than retried out. Run it with `python examples/repair.py`.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from tesserix_adk.core import Agent, RepairConfig, Run, RunEventKind, RunState, Usage
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

BAD = '{"destination": "Kyoto"}'
HALF = '{"nights": 4}'
GOOD = '{"destination": "Kyoto", "nights": 4}'


class TripPlan(BaseModel):
    """A trip proposed for a traveller.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


def planner(**overrides: object) -> Agent:
    """The same agent throughout, so only the thing being shown differs."""
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "model": "claude-sonnet-5",
        "output_type": TripPlan,
        "repair": RepairConfig(),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


async def run(agent: Agent, *answers: str) -> tuple[Run, ScriptedProvider]:
    """Drive the loop against a scripted provider and hand back what it was sent."""
    provider = ScriptedProvider(
        *(
            ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))
            for text in answers
        ),
        structured=True,
    )
    runner = AgentRunner(provider=provider, clock=FakeClock())
    return await runner.run(agent, "plan a trip", tenant="acme"), provider


def details(finished: Run, kind: RunEventKind) -> list[str]:
    """Every recorded detail of one kind, in order."""
    return [record.detail or "" for record in finished.events if record.kind is kind]


async def corrected() -> None:
    """A missing field named back to the model comes back filled by the model."""
    finished, _ = await run(planner(), BAD, GOOD)

    print("state:    ", finished.state is RunState.COMPLETED)  # noqa: T201
    print("output:   ", finished.output)  # noqa: T201
    print("repaired: ", details(finished, RunEventKind.REPAIR_REQUESTED)[0])  # noqa: T201
    print("charged:  ", finished.usage.input_tokens)  # noqa: T201


async def fed_back() -> None:
    """What the correction contains: the failure, the schema, and no invented value."""
    _, provider = await run(planner(), BAD, GOOD)
    correction = "\n".join(
        part.text
        for message in provider.requests[1].messages
        for part in message.content
        if hasattr(part, "text")
    )

    print("names it: ", "nights: Field required" in correction)  # noqa: T201
    print("invents:  ", 'nights": 4' in correction)  # noqa: T201


async def exhausted() -> None:
    """Running out is a failure, never a half-populated object."""
    finished, _ = await run(planner(), BAD, HALF, "not json at all")

    print("failed:   ", finished.state is RunState.FAILED)  # noqa: T201
    print("output:   ", finished.output)  # noqa: T201
    print("attempts: ", len(details(finished, RunEventKind.SCHEMA_VIOLATION)))  # noqa: T201


async def futile() -> None:
    """The identical failure after being told what it was is a defect in the declaration."""
    finished, _ = await run(planner(), BAD, BAD)

    print("abandoned:", bool(details(finished, RunEventKind.REPAIR_ABANDONED)))  # noqa: T201
    print("attempts: ", len(details(finished, RunEventKind.REPAIR_REQUESTED)))  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await corrected()
    await fed_back()
    await exhausted()
    await futile()


if __name__ == "__main__":
    asyncio.run(main())
