"""Whether prompt caching is actually working, as a number rather than a belief.

Four scenarios: a stable prefix across three turns and the ratio it earns; the same
workload with a prefix that shifts every turn; a group that sent nothing, which is not the
same finding as a group that missed every time; and the two counters a metric store divides.

Run it with `python examples/cache_hit_ratio.py`. A scripted provider stands in for the
vendor, so nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from tesserix_adk.core import Agent, Cost, ModelCapabilities, Run, Usage
from tesserix_adk.observability import (
    CACHED_TOKENS,
    INPUT_TOKENS,
    record_spend,
    spend_of,
    totals_by,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeMeter, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)
PREFIX_TOKENS = 8_000


def turn(cached: int) -> Usage:
    """One turn over an 8k-token file, of which `cached` was served from the cache."""
    return Usage(
        input_tokens=PREFIX_TOKENS,
        cached_tokens=cached,
        output_tokens=40,
        cost=Cost(input=Decimal("0.02"), cache_read=Decimal("0.001"), currency="USD"),
    )


async def a_run(agent: str, *turns: Usage) -> Run[Any]:
    """One run answered by a scripted vendor reporting the usage it was given."""
    return await AgentRunner(
        provider=ScriptedProvider(
            *(ModelResponse(content="page 12.", usage=one) for one in turns),
            name="scripted",
            capabilities=CAPABLE,
        ),
        clock=FakeClock(),
    ).run(
        Agent(name=agent, instructions="Answer from the file.", free_text=True, model="llama-8b"),
        "what does page 12 say?",
        tenant="acme",
        user="ada",
    )


async def a_stable_prefix() -> None:
    """The first turn evaluates the file; the rest read it back."""
    run = await a_run("stable", turn(cached=0))
    later = await a_run("stable", turn(cached=PREFIX_TOKENS - 200))
    records = tuple(record for one in (run, later) for record in spend_of(one))
    (total,) = totals_by(records, "tenant").values()
    print(f"stable prefix: {total.hit_ratio:.0%} of {total.input_tokens} tokens cached")  # noqa: T201


async def a_prefix_that_moves() -> None:
    """A byte of drift in the prefix and every turn pays prefill again."""
    run = await a_run("drifting", turn(cached=0))
    later = await a_run("drifting", turn(cached=0))
    records = tuple(record for one in (run, later) for record in spend_of(one))
    (total,) = totals_by(records, "tenant").values()
    print(  # noqa: T201
        f"drifting prefix: {total.hit_ratio:.0%} cached,",
        f"measured {total.measured} — the cache missed, it was not absent",
    )


async def nothing_sent_at_all() -> None:
    """Zero percent and no data are different findings; a dashboard must tell them apart."""
    run = await a_run("silent", Usage(input_tokens=0, output_tokens=0, cost=Cost.nothing()))
    (total,) = totals_by(spend_of(run), "tenant").values()
    print(f"nothing sent: ratio {total.hit_ratio}, measured {total.measured}")  # noqa: T201


async def what_the_metric_store_gets() -> None:
    """A ratio counter cannot be re-aggregated; two counters divide at any grouping."""
    meter = FakeMeter()
    record_spend(await a_run("metered", turn(cached=6_000)), meter=meter)
    read, sent = meter.total(CACHED_TOKENS), meter.total(INPUT_TOKENS)
    print(f"{CACHED_TOKENS} / {INPUT_TOKENS} = {read / sent:.0%}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await a_stable_prefix()
    await a_prefix_that_moves()
    await nothing_sent_at_all()
    await what_the_metric_store_gets()


if __name__ == "__main__":
    asyncio.run(main())
