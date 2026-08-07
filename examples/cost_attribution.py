"""Answering 'who spent this' with a query instead of an investigation.

Three scenarios: two tenants and two models broken down for chargeback; a consumer
attribute carrying an email address and a key, scrubbed before export; and a run whose
trace was sampled away but whose spend was not.

Run it with `python examples/cost_attribution.py`. A scripted provider stands in for the
vendor, so nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from tesserix_adk.core import Agent, Cost, ModelCapabilities, Run, Usage
from tesserix_adk.observability import Dimensions, Redactor, record_spend, spend_of, totals_by
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeMeter, FakeTracer, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)


def priced(amount: str) -> Usage:
    """What one call consumed, at a price the model layer already worked out."""
    return Usage(
        input_tokens=120, output_tokens=30, cost=Cost(input=Decimal(amount), currency="USD")
    )


async def a_run(tenant: str, agent: str, model: str, amount: str) -> Run[Any]:
    """One run by one agent for one tenant, answered by a scripted vendor."""
    return await AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(content="Kyoto, four nights.", usage=priced(amount)),
            name="scripted",
            capabilities=CAPABLE,
        ),
        clock=FakeClock(),
    ).run(
        Agent(name=agent, instructions="Plan trips.", free_text=True, model=model),
        "where to?",
        tenant=tenant,
        user="ada",
    )


async def chargeback() -> None:
    """Spend split by whichever dimensions the question is about."""
    runs = [
        await a_run("acme", "planner", "scripted-1", "0.20"),
        await a_run("acme", "researcher", "scripted-2", "0.30"),
        await a_run("globex", "planner", "scripted-1", "0.50"),
    ]
    records = tuple(record for run in runs for record in spend_of(run))

    print("\n=== by tenant")  # noqa: T201
    for (tenant,), total in totals_by(records, "tenant").items():
        print(f"{tenant:<10} {total.cost.total} {total.cost.currency} over {total.calls} calls")  # noqa: T201

    print("\n=== by agent and model")  # noqa: T201
    for (agent, model), total in totals_by(records, "agent", "model").items():
        print(f"{agent:<12} {model:<12} {total.cost.total} {total.cost.currency}")  # noqa: T201


async def what_never_leaves_the_process() -> None:
    """A consumer attribute carrying an address and a key, and the record of the drop."""
    tracer, meter = FakeTracer(), FakeMeter()
    record_spend(
        await a_run("acme", "planner", "scripted-1", "0.20"),
        tracer=tracer,
        meter=meter,
        redactor=Redactor(extra_patterns=(r"CASE-\d+",)),
        extra={"requested_by": "ada@example.com", "ref": "CASE-4471", "team": "platform"},
    )
    print("\n=== what was exported")  # noqa: T201
    for recorded in tracer.recorded:
        interesting = {
            key: value
            for key, value in recorded.attributes.items()
            if key in {"requested_by", "ref", "team", "adk.tenant", "adk.cost"}
            or key.endswith("redacted_keys")
        }
        print(f"{recorded.kind:<6} {recorded.name:<16} {interesting}")  # noqa: T201


async def spend_that_outlives_its_trace() -> None:
    """The trace was sampled away; the money still has to be counted."""
    tracer, meter = FakeTracer(), FakeMeter()
    record_spend(
        await a_run("globex", "planner", "scripted-1", "0.50"),
        tracer=tracer,
        meter=meter,
        dimensions=Dimensions(tenants=frozenset({"acme"})),
        sampled=False,
    )
    print("\n=== sampled away")  # noqa: T201
    print(f"spans exported {len(tracer.recorded)}")  # noqa: T201
    print(f"cost counted   {meter.total('adk.cost')}")  # noqa: T201
    print(f"under tenant   {meter.points[0].dimensions['tenant']}")  # noqa: T201


async def main() -> None:
    """Run the three scenarios in order."""
    await chargeback()
    await what_never_leaves_the_process()
    await spend_that_outlives_its_trace()


if __name__ == "__main__":
    asyncio.run(main())
