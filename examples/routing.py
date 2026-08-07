"""An agent that names the job, and the table that decides which model does it.

Five scenarios: one agent answered by two different vendors on two tables and no code
change between them; a narrower rule escalating one agent; an agent that needs vision not
being routed to a model without it; a class nobody routed refusing rather than downgrading;
and a table read from TOML, validated as it is read.

Run it with `python examples/routing.py`. Scripted providers stand in for the vendors, so
nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from tesserix_adk.core import (
    CHEAP,
    REASONING,
    Agent,
    Capability,
    ModelCapabilities,
    ModelSpec,
    NoEligibleModelError,
    RunEventKind,
    Usage,
)
from tesserix_adk.models.routing import RoutingRule, RoutingTable, TableRouter, routing_table
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

SEES = ModelCapabilities(
    tool_calling=True, streaming=True, vision=True, context_window_tokens=200_000
)
BLIND = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=128_000)

TABLE_TOML = """
version = 1

[[rules]]
task_class = "cheap"

[[rules.candidates]]
provider = "openai"
model = "gpt-4o-mini"
capabilities = { tool_calling = true, streaming = true, context_window_tokens = 128000 }
"""


def spec(ref: str, capabilities: ModelCapabilities = SEES) -> ModelSpec:
    """A candidate written as `provider:model`, with what it declares it can do."""
    provider, _, model = ref.partition(":")
    return ModelSpec(provider=provider, model=model, capabilities=capabilities)


def fleet() -> dict[str, ScriptedProvider]:
    """One scripted provider per vendor the table may choose."""
    answer = ModelResponse(
        content="Kyoto, four nights.", usage=Usage(input_tokens=10, output_tokens=5)
    )
    return {
        name: ScriptedProvider(answer, name=name, capabilities=SEES)
        for name in ("openai", "anthropic")
    }


def runner(table: RoutingTable, vendors: dict[str, ScriptedProvider]) -> AgentRunner:
    """A runner that resolves a task class against `table` before every run."""
    return AgentRunner(
        provider=next(iter(vendors.values())),
        providers=vendors,
        router=TableRouter(table),
        clock=FakeClock(),
    )


def planner(**overrides: object) -> Agent:
    """An agent that names a task class rather than a model."""
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "task_class": CHEAP,
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


async def the_table_decides_not_the_code() -> None:
    """The same agent, two tables, two vendors. Nothing in the agent moved."""
    for chosen in ("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"):
        vendors = fleet()
        table = RoutingTable(rules=(RoutingRule(task_class=CHEAP, candidates=(spec(chosen),)),))
        run = await runner(table, vendors).run(planner(), "plan a trip", tenant="acme")
        called = [name for name, vendor in vendors.items() if vendor.requests]
        print(f"cheap -> {run.model:<20}", f"called={called}")  # noqa: T201


async def one_agent_escalated() -> None:
    """A rule naming a tenant and an agent beats the general one, for that agent only."""
    table = RoutingTable(
        rules=(
            RoutingRule(task_class=CHEAP, candidates=(spec("openai:gpt-4o-mini", BLIND),)),
            RoutingRule(
                task_class=CHEAP,
                tenant="acme",
                agent="planner",
                candidates=(spec("anthropic:claude-sonnet-4-5"),),
            ),
        )
    )
    router = TableRouter(table)
    for agent in ("planner", "summariser"):
        decision = router.resolve(CHEAP, tenant="acme", agent=agent)
        print(f"{agent:<12}", decision.explain())  # noqa: T201


async def a_model_that_cannot_do_the_work() -> None:
    """An agent that sends images is not routed to a model that cannot read them."""
    table = RoutingTable(
        rules=(RoutingRule(task_class=CHEAP, candidates=(spec("openai:gpt-4o-mini", BLIND),)),)
    )
    try:
        await runner(table, fleet()).run(
            planner(requires={Capability.VISION}), "describe the exhibit", tenant="acme"
        )
    except NoEligibleModelError as refused:
        print("vision      ", f"unsatisfied={refused.unsatisfied}")  # noqa: T201


async def nothing_falls_back() -> None:
    """A class nobody routed fails. A cheaper model is not a cheaper way to do the job."""
    table = RoutingTable(
        rules=(RoutingRule(task_class=CHEAP, candidates=(spec("openai:gpt-4o-mini"),)),)
    )
    try:
        await runner(table, fleet()).run(
            planner(task_class=REASONING), "prove the lemma", tenant="acme"
        )
    except NoEligibleModelError as refused:
        print("reasoning   ", refused.task_class, "not routed")  # noqa: T201


async def read_from_configuration() -> None:
    """The table an operator edits, and the event the run records for it."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "routing.toml"
        path.write_text(TABLE_TOML)
        vendors = fleet()
        run = await runner(routing_table(path), vendors).run(
            planner(), "plan a trip", tenant="acme"
        )
    routed = [event for event in run.events if event.kind is RunEventKind.MODEL_ROUTED]
    print("from toml   ", routed[0].detail)  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await the_table_decides_not_the_code()
    await one_agent_escalated()
    await a_model_that_cannot_do_the_work()
    await nothing_falls_back()
    await read_from_configuration()


if __name__ == "__main__":
    asyncio.run(main())
