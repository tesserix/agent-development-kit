"""One trace over a supervisor, three workers and a peer, and a total that admits its gaps.

Four scenarios: the whole run drawn as a tree with the money on every node; a worker that
crashed before reporting, so the total says it is a lower bound; a peer billing in euros,
summed only through a rate somebody recorded; and a wide fan-out whose trace was sampled
away but whose cost was not.

Run it with `python examples/multi_agent_trace.py`. A scripted provider stands in for the
vendor, so nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from tesserix_adk.core import Agent, Cost, ModelCapabilities, Run, Usage
from tesserix_adk.observability import (
    COST,
    Pattern,
    Rate,
    TraceContext,
    node_of,
    peer_node,
    record_tree,
    render,
    tree,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeMeter, FakeTracer, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


def priced(amount: str, currency: str = "USD") -> Usage:
    """What one call consumed, at a price the model layer already worked out."""
    return Usage(
        input_tokens=120, output_tokens=30, cost=Cost(input=Decimal(amount), currency=currency)
    )


async def a_run(run_id: str, agent: str, amount: str) -> Run[Any]:
    """One participant that ran here, answered by a scripted vendor."""
    return await AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(content="Kyoto, four nights.", usage=priced(amount)),
            name="scripted",
            capabilities=CAPABLE,
        ),
        clock=FakeClock(),
    ).run(
        Agent(name=agent, instructions="Plan trips.", free_text=True, model="scripted-1"),
        "where to?",
        tenant="acme",
        user="ada",
        run_id=run_id,
    )


async def one_run() -> None:
    """A supervisor, three workers and a remote peer, on one trace."""
    supervisor = await a_run("run_1", "planner", "0.20")
    root = TraceContext.root_of(supervisor)
    nodes = [node_of(supervisor, root)]
    for index in range(3):
        worker = await a_run(f"w{index}", f"worker{index}", "0.10")
        nodes.append(
            node_of(
                worker,
                root.child(
                    run_id=f"w{index}",
                    agent=f"worker{index}",
                    pattern=Pattern.FAN_OUT,
                    branch=f"leg{index}",
                ),
            )
        )
    nodes.append(
        peer_node(
            root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
            usage=priced("0.05"),
            cost=Cost(input=Decimal("0.05"), currency="USD"),
            started_at=1000.0,
            ended_at=1003.0,
        )
    )

    print("\n=== the whole run")  # noqa: T201
    print(render(tree(nodes)), end="")  # noqa: T201


async def a_gap() -> None:
    """A worker that crashed before reporting is named, never counted as zero."""
    supervisor = await a_run("run_1", "planner", "0.20")
    root = TraceContext.root_of(supervisor)
    crashed = peer_node(root.child(run_id="w0", agent="worker0"))

    print("\n=== one worker never reported")  # noqa: T201
    print(render(tree([node_of(supervisor, root), crashed])), end="")  # noqa: T201


async def another_currency() -> None:
    """A euro-billing peer reaches the total only through a rate somebody recorded."""
    supervisor = await a_run("run_1", "planner", "0.20")
    root = TraceContext.root_of(supervisor)
    peer = peer_node(
        root.child(run_id="p1", agent="peer", pattern=Pattern.PEER),
        usage=priced("2.00", "EUR"),
        cost=Cost(input=Decimal("2.00"), currency="EUR"),
        rate=Rate(
            source="treasury",
            recorded_at=1000.0,
            multiplier=Decimal("1.10"),
            of_currency="EUR",
            to_currency="USD",
        ),
    )

    print("\n=== a peer that bills in euros")  # noqa: T201
    print(render(tree([node_of(supervisor, root), peer])), end="")  # noqa: T201


async def sampled_away() -> None:
    """The money never travels on a span, so a sampler cannot drop it."""
    supervisor = await a_run("run_1", "planner", "0.20")
    root = TraceContext.root_of(supervisor)
    nodes = [node_of(supervisor, root)]
    for index in range(8):
        worker = await a_run(f"w{index}", "worker", "0.01")
        nodes.append(
            node_of(
                worker,
                root.child(
                    run_id=f"w{index}",
                    agent="worker",
                    pattern=Pattern.FAN_OUT,
                    branch=f"leg{index}",
                ),
            )
        )
    meter, tracer = FakeMeter(), FakeTracer()
    record_tree(tree(nodes), meter=meter, tracer=tracer, sampled=False)

    counted = [point for point in meter.points if point.name == COST]
    print("\n=== the trace was sampled away")  # noqa: T201
    print(f"spans exported  {len(tracer.recorded)}")  # noqa: T201
    print(f"cost counted    {sum(point.value for point in counted):.2f} over {len(counted)}")  # noqa: T201


async def main() -> None:
    """Run the four scenarios in order."""
    await one_run()
    await a_gap()
    await another_currency()
    await sampled_away()


if __name__ == "__main__":
    asyncio.run(main())
