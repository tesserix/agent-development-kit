"""What happens to a run when the vendor it was routed to will not answer.

Four scenarios: a rate-limited primary handing the run to the secondary with both attempts
on the record; a bad key ending the run rather than shopping it around; every candidate
refusing and the one error naming them all; and a fallback refused because a tool that is
not idempotent has already run.

Run it with `python examples/fallback.py`. Scripted providers stand in for the vendors, so
nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    CHEAP,
    Agent,
    AuthenticationError,
    ModelCapabilities,
    ModelSpec,
    ProviderUnavailableError,
    RateLimitError,
    RetryConfig,
    RunEventKind,
    ToolCall,
    Usage,
)
from tesserix_adk.models.routing import RoutingRule, RoutingTable, TableRouter
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=200_000)

TABLE = RoutingTable(
    rules=(
        RoutingRule(
            task_class=CHEAP,
            candidates=(
                ModelSpec(provider="openai", model="gpt-4o-mini", capabilities=CAPABLE),
                ModelSpec(provider="anthropic", model="claude-haiku-4-5", capabilities=CAPABLE),
            ),
        ),
    )
)


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    """An answer any of these vendors could have given."""
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def runner(fleet: dict[str, ScriptedProvider], **overrides: object) -> AgentRunner:
    """A runner wired to the table above and to every vendor in the fleet."""
    fields: dict[str, object] = {
        "provider": next(iter(fleet.values())),
        "providers": fleet,
        "router": TableRouter(TABLE),
        "retry": RetryConfig(max_attempts=1),
        "clock": FakeClock(),
    }
    return AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]


def fleet_of(**scripts: list[ModelResponse | BaseException]) -> dict[str, ScriptedProvider]:
    """Vendors that answer, or fail, exactly as scripted."""
    return {
        name: ScriptedProvider(*script, name=name, capabilities=CAPABLE)
        for name, script in scripts.items()
    }


def planner(**overrides: object) -> Agent:
    """An agent that names a task class rather than a model."""
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "task_class": CHEAP,
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def ended(run: object) -> str:
    """Why the run stopped, as recorded on it."""
    return next(
        event.detail or ""
        for event in reversed(run.events)  # type: ignore[attr-defined]
        if event.kind is RunEventKind.TERMINATED
    )


async def the_secondary_finishes_what_the_primary_would_not() -> None:
    """A rate-limited primary hands the run on, and the run says who finished it."""
    fleet = fleet_of(
        openai=[RateLimitError("slow down")],
        anthropic=[answer()],
    )
    run = await runner(fleet).run(planner(), "plan a trip", tenant="acme")
    moved = [event for event in run.events if event.kind is RunEventKind.MODEL_FELL_BACK]
    print(f"answered by   {run.model}, {moved[0].detail}")  # noqa: T201


async def a_bad_key_is_not_shopped_around() -> None:
    """A second vendor will not fix the first one's key, so the run ends there."""
    fleet = fleet_of(openai=[AuthenticationError("bad key")], anthropic=[answer()])
    run = await runner(fleet).run(planner(), "plan a trip", tenant="acme")
    print(f"terminal      {ended(run)} (secondary asked: {bool(fleet['anthropic'].requests)})")  # noqa: T201


async def every_attempt_is_in_the_failure() -> None:
    """When everyone refuses, the last refusal alone is not the story."""
    fleet = fleet_of(
        openai=[RateLimitError("slow down")],
        anthropic=[ProviderUnavailableError("overloaded", status=529)],
    )
    run = await runner(fleet).run(planner(), "plan a trip", tenant="acme")
    print(f"exhausted     {ended(run)}")  # noqa: T201


async def a_charge_that_must_not_happen_twice() -> None:
    """A tool that is not declared idempotent blocks the fallback rather than repeating."""
    registry = FakeToolRegistry({"charge": lambda **_: "charged"})
    fleet = fleet_of(
        openai=[
            ModelResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="charge", arguments={"amount": 100}),),
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            RateLimitError("slow down"),
        ],
        anthropic=[answer()],
    )
    run = await runner(fleet, tools=registry).run(
        planner(tools=("charge",)), "pay the invoice", tenant="acme"
    )
    print(f"refused       {ended(run)} (charged {len(registry.calls)}x)")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await the_secondary_finishes_what_the_primary_would_not()
    await a_bad_key_is_not_shopped_around()
    await every_attempt_is_in_the_failure()
    await a_charge_that_must_not_happen_twice()


if __name__ == "__main__":
    asyncio.run(main())
