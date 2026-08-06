"""An agent answers in a declared shape, and the runtime proves it before finishing.

Four scenarios: a validated answer arriving as data, an answer that does not validate
failing instead of completing, the same type sent to a provider that cannot enforce a
schema, and the explicit free-text opt-out. Run it with
`python examples/structured_output.py`.
"""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, ValidationError

from tesserix_adk.core import Agent, Run, RunEventKind, RunState, Usage
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider


class TripPlan(BaseModel):
    """A trip proposed for a traveller.

    Args:
        destination: Where the traveller goes.
        nights: How long they stay.
    """

    destination: str
    nights: int


PLAN = {"destination": "Kyoto", "nights": 4}


def planner(**overrides: object) -> Agent:
    """The same agent throughout, so only the thing being shown differs."""
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "model": "claude-sonnet-5",
        "output_type": TripPlan,
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


async def run(agent: Agent, content: str, *, native: bool) -> tuple[Run, ScriptedProvider]:
    """Drive one turn against a scripted provider and hand back what it was sent."""
    provider = ScriptedProvider(
        ModelResponse(content=content, usage=Usage(input_tokens=10, output_tokens=5)),
        capabilities=CAPABLE.declaring(structured_output=native),
    )
    runner = AgentRunner(provider=provider, clock=FakeClock())
    return await runner.run(agent, "plan a trip", tenant="acme"), provider


async def validated() -> None:
    """The caller gets an object, not a string to parse."""
    finished, _ = await run(planner(), json.dumps(PLAN), native=True)

    print("state:   ", finished.state is RunState.COMPLETED)  # noqa: T201
    print("output:  ", finished.output)  # noqa: T201


async def refused() -> None:
    """Prose where an object was asked for fails the run rather than completing it."""
    finished, _ = await run(planner(), "Kyoto, four nights.", native=True)

    violation = next(e for e in finished.events if e.kind is RunEventKind.SCHEMA_VIOLATION)
    print("failed:  ", finished.state is RunState.FAILED)  # noqa: T201
    print("output:  ", finished.output)  # noqa: T201
    print("recorded:", (violation.detail or "").split(";")[0])  # noqa: T201


async def fallback() -> None:
    """A provider that cannot enforce a schema is told the schema instead."""
    _, provider = await run(planner(), json.dumps(PLAN), native=False)
    sent = "\n".join(
        part.text
        for message in provider.requests[0].messages
        for part in message.content
        if hasattr(part, "text")
    )

    print("in prompt:", "destination" in sent)  # noqa: T201
    print("hashed:  ", provider.requests[0].output_schema_hash is not None)  # noqa: T201


async def free_text() -> None:
    """Prose is a declaration, never something reached by omitting one."""
    finished, _ = await run(
        Agent(name="chatter", instructions="Chat.", model="claude-sonnet-5", free_text=True),
        "Kyoto, four nights.",
        native=True,
    )

    print("state:   ", finished.state is RunState.COMPLETED)  # noqa: T201
    print("output:  ", finished.output)  # noqa: T201

    try:
        Agent(name="chatter", instructions="Chat.", model="claude-sonnet-5")
    except ValidationError as refusal:
        print("omitted: ", "free_text" in str(refusal))  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await validated()
    await refused()
    await fallback()
    await free_text()


if __name__ == "__main__":
    asyncio.run(main())
