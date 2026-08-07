"""What the run loop does when a ceiling is reached with work still in flight.

Four scenarios: a run stopped mid-loop by its iteration ceiling; a call refused before
dispatch because the tokens would not fit; a tool whose side effect outlived the run and is
named for compensation; and a stream held to the same ceiling as everything else.

Run it with `python examples/budget_enforcement.py`. A scripted provider stands in for the
vendor, so nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from tesserix_adk.core import (
    Agent,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    ModelCapabilities,
    Run,
    RunBudget,
    RunEventKind,
    ScopedLimits,
    StreamEvent,
    ToolCall,
    Usage,
    UsageDelta,
    most_restrictive,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse, budgeted_stream
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


def booking() -> ModelResponse:
    """A response that asks for a tool, so the loop keeps going and keeps spending."""
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id="call_1", name="book_seat", arguments={}),),
        usage=Usage(input_tokens=120, output_tokens=20),
    )


def planner(**overrides: object) -> Agent[Any]:
    """The agent every scenario runs."""
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "scripted",
        "tools": ("book_seat",),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def held_to(clock: FakeClock, **limits: object) -> RunBudget:
    """A run-scoped ceiling, stated in one place."""
    return RunBudget(
        resolved=most_restrictive(
            ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(**limits))  # type: ignore[arg-type]
        ),
        clock=clock,
    )


def runner(clock: FakeClock, calls: int, **limits: object) -> AgentRunner:
    """A runner whose model keeps booking seats until something stops it."""
    return AgentRunner(
        provider=ScriptedProvider(
            *[booking() for _ in range(calls)], name="scripted", capabilities=CAPABLE
        ),
        clock=clock,
        budget=held_to(clock, **limits),
        tools=FakeToolRegistry({"book_seat": lambda: {"seat": "12A"}}),
    )


async def a_loop_stopped_where_it_stood() -> None:
    """Forty iterations nobody expected is what a per-iteration check is for."""
    clock = FakeClock()
    run = await runner(clock, 5, max_iterations=2).run(planner(), "Book it", tenant="acme")
    print("=== a loop that ran out of iterations")  # noqa: T201
    print(f"state          {run.state}")  # noqa: T201
    print(f"answer         {run.output}")  # noqa: T201
    print(f"why            {_detail(run, RunEventKind.BUDGET_EXCEEDED)}")  # noqa: T201


async def a_call_that_was_never_made() -> None:
    """The tokens were estimated first, so this one cost nothing at all."""
    clock = FakeClock()
    provider = ScriptedProvider(booking(), name="scripted", capabilities=CAPABLE)
    run = await AgentRunner(
        provider=provider,
        clock=clock,
        budget=held_to(clock, max_input_tokens=2),
        tools=FakeToolRegistry({"book_seat": lambda: {"seat": "12A"}}),
    ).run(planner(), "Book it", tenant="acme")
    print("\n=== a call refused before dispatch")  # noqa: T201
    print(f"state          {run.state}")  # noqa: T201
    print(f"vendor asked   {len(provider.requests)} times")  # noqa: T201


async def a_seat_that_is_still_booked() -> None:
    """The runtime never unbooks it; it says who has to."""
    clock = FakeClock()
    run = await runner(clock, 3, max_tool_calls=1).run(planner(), "Book it", tenant="acme")
    outstanding = [
        event for event in run.events if event.kind is RunEventKind.COMPENSATION_REQUIRED
    ]
    print("\n=== a side effect that outlived the run")  # noqa: T201
    print(f"state          {run.state}")  # noqa: T201
    for event in outstanding:
        print(f"compensate     {event.name}: {event.detail}")  # noqa: T201


async def a_stream_held_to_the_same_ceiling() -> None:
    """A ceiling passed mid-stream ends the stream, rather than letting it trail off."""
    clock = FakeClock()
    limit = held_to(clock, max_output_tokens=10)
    seen = 0
    print("\n=== a stream that passed the ceiling")  # noqa: T201
    try:
        async for _ in budgeted_stream(_answering(), limit):
            seen += 1
    except BudgetExceededError as exceeded:
        print(f"events seen    {seen}")  # noqa: T201
        print(f"stopped by     {exceeded}")  # noqa: T201


async def _answering() -> AsyncIterator[StreamEvent]:
    """A vendor reporting its running total as the answer is written."""
    yield UsageDelta(usage=Usage(input_tokens=10, output_tokens=5))
    yield UsageDelta(usage=Usage(input_tokens=10, output_tokens=50))


def _detail(run: Run[Any], kind: RunEventKind) -> str:
    """The note on the first event of `kind`, which is where the reason is written."""
    return next((event.detail or "" for event in run.events if event.kind is kind), "")


async def main() -> None:
    """Run the four scenarios in order."""
    await a_loop_stopped_where_it_stood()
    await a_call_that_was_never_made()
    await a_seat_that_is_still_booked()
    await a_stream_held_to_the_same_ceiling()


if __name__ == "__main__":
    asyncio.run(main())
