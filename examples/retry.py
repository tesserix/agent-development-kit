"""Try again where a second attempt could work, and stop where it could not.

Three runs: a blip the caller never sees, a rejected request that is not sent twice, and a
tool that is only retried because the agent declared it safe to repeat. Nothing here sleeps
— the clock and the jitter source are injected, so the schedule is printed rather than
waited out.

Run it with `python examples/retry.py`.
"""

from __future__ import annotations

import asyncio
from random import Random

from tesserix_adk.core import (
    Agent,
    ProviderError,
    RetryConfig,
    Run,
    RunEventKind,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider

THRICE = RetryConfig(max_attempts=3, base_delay_seconds=1, multiplier=2)


def agent(**overrides: object) -> Agent:
    """The same planner throughout; each run changes only what it declares about retrying."""
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips. Cite the timetable before recommending a leg.",
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer() -> ModelResponse:
    """What the provider eventually says, once it is answering at all."""
    return ModelResponse(
        content="Kyoto, four nights.", usage=Usage(input_tokens=10, output_tokens=5)
    )


def report(title: str, run: Run, *, slept: list[float]) -> None:
    """Print how the run ended, every attempt it took, and what it waited in between."""
    print(f"\n{title}")  # noqa: T201
    print(f"  state:  {run.state}")  # noqa: T201
    for event in run.events:
        if event.kind in _INTERESTING:
            print(f"  {event.kind}: {event.detail}")  # noqa: T201
    print(f"  waited: {[round(second, 2) for second in slept]}")  # noqa: T201


_INTERESTING = frozenset(
    {
        RunEventKind.ATTEMPT_FAILED,
        RunEventKind.TOOL_ERROR,
        RunEventKind.TERMINATED,
    }
)


async def a_blip_the_caller_never_sees() -> None:
    """Two 503s and an answer: the run completes, and the failed attempts stay in the record."""
    clock = FakeClock()
    provider = ScriptedProvider(
        ProviderError("upstream unavailable", status=503),
        ProviderError("upstream unavailable", status=503),
        answer(),
    )
    runner = AgentRunner(provider=provider, clock=clock, retry=THRICE, jitter=Random(7))

    run = await runner.run(agent(), "Four nights near Kyoto.", tenant="acme")
    report("a blip, recovered", run, slept=clock.slept)


async def a_rejected_request_is_not_sent_twice() -> None:
    """A 400 is an answer, not a fault. Sending it again gets the same answer, at the same price."""
    clock = FakeClock()
    provider = ScriptedProvider(ProviderError("malformed tool schema", status=400))
    runner = AgentRunner(provider=provider, clock=clock, retry=THRICE, jitter=Random(7))

    run = await runner.run(agent(), "Four nights near Kyoto.", tenant="acme")
    report("a rejected request", run, slept=clock.slept)
    print(f"  calls:  {len(provider.requests)}")  # noqa: T201


async def only_a_declared_tool_is_repeated() -> None:
    """`charge_card` is not in `idempotent_tools`, so a failure after dispatch is not repeated."""
    clock = FakeClock()
    provider = ScriptedProvider(
        ModelResponse(
            tool_calls=(ToolCall(id="call_1", name="charge_card", arguments={}),),
            usage=Usage(input_tokens=10, output_tokens=5),
        ),
        answer(),
    )
    tools = FakeToolRegistry({"charge_card": _fails})
    runner = AgentRunner(
        provider=provider, tools=tools, clock=clock, retry=THRICE, jitter=Random(7)
    )

    run = await runner.run(agent(tools=("charge_card",)), "Pay for the hotel.", tenant="acme")
    report("a tool nobody declared safe to repeat", run, slept=clock.slept)


def _fails() -> str:
    raise RuntimeError("gateway timeout after the charge was submitted")


async def main() -> None:
    """One recovered fault, one refused retry, one side effect nobody repeats."""
    await a_blip_the_caller_never_sees()
    await a_rejected_request_is_not_sent_twice()
    await only_a_declared_tool_is_repeated()


if __name__ == "__main__":
    asyncio.run(main())
