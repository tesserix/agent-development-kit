"""Bound the shape of a run: how deep, how wide, and how often the same thing.

Three runs that each hit a different cap, and one that shows a cap narrowing rather than
widening. Every one of them ends in a terminal state with a named reason, because a run
that stops without saying which bound it hit is a run nobody can tune.

Run it with `python examples/loops.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    LoopConfig,
    Run,
    RunContext,
    RunEventKind,
    TenantContext,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeToolRegistry, ScriptedProvider


def agent(**overrides: object) -> Agent:
    """The same researcher throughout; each run changes only the caps in play."""
    fields: dict[str, object] = {
        "name": "researcher",
        "instructions": "Answer from sources. Cite the page.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": ("search",),
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def fanning_out(count: int) -> ModelResponse:
    """One turn asking for `count` searches at once."""
    return ModelResponse(
        tool_calls=tuple(
            ToolCall(id=f"call_{n}", name="search", arguments={"page": n}) for n in range(count)
        ),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def asking_again() -> ModelResponse:
    """The same request, argument for argument — the shape a wedged loop takes."""
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name="search", arguments={"q": "kyoto"}),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def tools() -> FakeToolRegistry:
    """One search tool, the same in every run here."""
    return FakeToolRegistry({"search": lambda **_: "a result"})


def report(title: str, run: Run) -> None:
    """Print how the run ended and which cap said so."""
    print(f"\n{title}")  # noqa: T201
    print(f"  state: {run.state}")  # noqa: T201
    for event in run.events:
        if event.kind in _INTERESTING:
            print(f"  {event.kind}: {event.detail}")  # noqa: T201


_INTERESTING = frozenset(
    {
        RunEventKind.FAN_OUT_REFUSED,
        RunEventKind.REPEAT_DETECTED,
        RunEventKind.DEPTH_EXCEEDED,
        RunEventKind.TERMINATED,
    }
)


async def a_turn_too_wide_is_refused_whole() -> None:
    """Six searches against a cap of three: none of them run.

    Half a fan-out is a set of side effects nobody chose.
    """
    provider = ScriptedProvider(fanning_out(6))
    runner = AgentRunner(
        provider=provider, tools=tools(), loop=LoopConfig(max_tool_calls_per_turn=3)
    )

    run = await runner.run(agent(), "Everything about Kyoto.", tenant="acme")
    report("a turn wider than its cap", run)
    print(f"  tool calls made: {len(run.tool_calls)}")  # noqa: T201


async def the_same_request_forever_is_not_progress() -> None:
    """The same arguments, over and over, until the cap says it is not progress.

    A tool declared idempotent is exempt; this one is not.
    """
    provider = ScriptedProvider(*[asking_again() for _ in range(6)])
    runner = AgentRunner(provider=provider, tools=tools(), loop=LoopConfig(max_repeated_calls=2))

    run = await runner.run(agent(), "Find the timetable.", tenant="acme")
    report("the same request, over and over", run)


async def a_chain_of_agents_cannot_outrun_its_ceiling() -> None:
    """A run started from a parent context carries the depth down the chain.

    Past the ceiling it fails closed, before a prompt is assembled or a token is spent.
    """
    provider = ScriptedProvider()
    runner = AgentRunner(provider=provider, tools=tools(), loop=LoopConfig(max_depth=2))
    deep = RunContext(run_id="run_parent", tenant=TenantContext(tenant="acme"), depth=2)

    run = await runner.run(agent(), "Delegate once more.", tenant="acme", parent=deep)
    report("one agent calling another, too far down", run)
    print(f"  model calls: {len(provider.requests)}")  # noqa: T201


async def an_agent_narrows_a_cap_but_never_widens_it() -> None:
    """The agent asks for twelve; the runner allows three, and three wins.

    An agent cannot vote itself more rope than the runner allows.
    """
    provider = ScriptedProvider(fanning_out(6))
    runner = AgentRunner(
        provider=provider, tools=tools(), loop=LoopConfig(max_tool_calls_per_turn=3)
    )

    run = await runner.run(
        agent(loop=LoopConfig(max_tool_calls_per_turn=12)), "Everything at once.", tenant="acme"
    )
    report("an agent asking for more rope", run)


async def main() -> None:
    """Four runs, four bounds, no run that cannot be stopped."""
    await a_turn_too_wide_is_refused_whole()
    await the_same_request_forever_is_not_progress()
    await a_chain_of_agents_cannot_outrun_its_ceiling()
    await an_agent_narrows_a_cap_but_never_widens_it()


if __name__ == "__main__":
    asyncio.run(main())
