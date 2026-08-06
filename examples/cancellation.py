"""Stop work nobody is waiting for: a caller that walks away, and a call that overruns.

Both runs come back as a `Run` in `CANCELLED`, carrying what was spent and how far they
got — a cancelled run still has to be accountable. The provider here stalls on purpose, so
neither run depends on a network being slow.

Run it with `python examples/cancellation.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import Agent, DeadlineConfig, Run, RunEventKind
from tesserix_adk.runtime import AgentRunner, CancellationToken
from tesserix_adk.testing import StallingProvider

AGENT = Agent(
    name="planner",
    instructions="Plan trips. Cite the timetable before recommending a leg.",
    model="claude-sonnet-5",
)


def report(title: str, run: Run) -> None:
    """Print the terminal state and why it was reached."""
    print(f"\n{title}")  # noqa: T201
    print(f"  state:  {run.state}")  # noqa: T201
    for event in run.events:
        if event.kind in _INTERESTING:
            print(f"  {event.kind}: {event.detail}")  # noqa: T201


_INTERESTING = frozenset(
    {
        RunEventKind.CANCELLATION_REQUESTED,
        RunEventKind.DEADLINE_EXCEEDED,
        RunEventKind.WORK_ORPHANED,
        RunEventKind.TERMINATED,
    }
)


async def caller_walks_away() -> None:
    """An HTTP client disconnects, so the handler cancels the token it passed in."""
    provider = StallingProvider()
    runner = AgentRunner(provider=provider)
    token = CancellationToken()

    task = asyncio.ensure_future(
        runner.run(AGENT, "Four nights near Kyoto.", tenant="acme", cancellation=token)
    )
    await provider.entered.wait()
    token.cancel("client disconnected")

    report("caller walked away", await task)


async def the_call_overruns() -> None:
    """A ceiling the caller declared, enforced against a model call that will not return."""
    runner = AgentRunner(
        provider=StallingProvider(),
        # Nothing is bounded by default: a ceiling the kit invented would kill good runs
        # on slow hardware. Declare the ones the deployment can actually stand.
        deadlines=DeadlineConfig(model_call_seconds=0.1, grace_seconds=0.1),
    )

    report(
        "the model call overran",
        await runner.run(AGENT, "Four nights near Kyoto.", tenant="acme"),
    )


async def main() -> None:
    """Both ways a run stops early."""
    await caller_walks_away()
    await the_call_overruns()


if __name__ == "__main__":
    asyncio.run(main())
