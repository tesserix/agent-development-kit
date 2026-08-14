"""The three numbers that decide whether CPU inference feels usable.

Four scenarios: a streamed run timed as a user experiences it; a cold start next to a warm
one; a provider that says nothing about its cache; and what a broken prefix does to the
numbers the benchmark gate watches.

Run it with `python examples/latency_objectives.py`. The clock is a fake so the output is
the same on every machine — a real run passes no clock at all.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.observability import CacheHits, RunTimer
from tesserix_adk.testing import FakeClock, FakeMeter

PREFILL_PER_SECOND = 420.0
DECODE_PER_SECOND = 18.0


def spent(clock: FakeClock, timer: RunTimer, *, input_tokens: int, cached: int | None) -> None:
    """Advance the clock the way the target CPU would spend it on this run."""
    uncached = input_tokens - (cached or 0)
    clock.advance(uncached / PREFILL_PER_SECOND)
    timer.first_token()
    clock.advance(200 / DECODE_PER_SECOND)


def one_run(*, input_tokens: int, cached: int | None, cold: bool) -> None:
    """Time one streamed run and print what an operator would read."""
    clock = FakeClock()
    timer = RunTimer(clock=clock, cold=cold)
    spent(clock, timer, input_tokens=input_tokens, cached=cached)

    report = timer.finished(
        output_tokens=200, hits=CacheHits(input_tokens=input_tokens, cached_tokens=cached)
    )
    print(report.render())  # noqa: T201


async def a_cold_start_and_a_warm_turn() -> None:
    """Averaging these together would hide both."""
    one_run(input_tokens=4_000, cached=0, cold=True)
    one_run(input_tokens=4_000, cached=3_400, cold=False)


async def an_unknown_ratio_stays_unknown() -> None:
    """A provider that reports nothing is not a provider with a cold cache."""
    one_run(input_tokens=4_000, cached=None, cold=False)


async def what_reaches_the_collector() -> None:
    """Metrics, not only spans: a sampled-away trace still leaves the numbers behind."""
    meter = FakeMeter()
    clock = FakeClock()
    timer = RunTimer(clock=clock)
    spent(clock, timer, input_tokens=4_000, cached=3_400)

    timer.finished(output_tokens=200, hits=CacheHits(input_tokens=4_000, cached_tokens=3_400)).emit(
        meter, model="llama-3.1-8b-instruct"
    )

    for point in meter.points:
        print(f"{point.name} {point.value:.3f} {point.dimensions}")  # noqa: T201


async def a_broken_prefix_moves_two_numbers() -> None:
    """What the benchmark gate catches: a volatile value near the front of the prompt."""
    print("stable prefix:")  # noqa: T201
    one_run(input_tokens=4_000, cached=3_400, cold=False)
    print("a timestamp added to the system prompt:")  # noqa: T201
    one_run(input_tokens=4_000, cached=0, cold=False)


async def main() -> None:
    """Run every scenario in order."""
    await a_cold_start_and_a_warm_turn()
    await an_unknown_ratio_stays_unknown()
    await what_reaches_the_collector()
    await a_broken_prefix_moves_two_numbers()


if __name__ == "__main__":
    asyncio.run(main())
