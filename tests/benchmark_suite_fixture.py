"""A two-scenario suite the benchmark tool's tests point at instead of the real one."""

from __future__ import annotations

from tesserix_adk.testing.benchmarks import Scenario

SLOWNESS = {"factor": 1}


async def _work() -> None:
    """Work whose cost the test controls, so a regression can be injected."""
    sum(range(200 * SLOWNESS["factor"]))


def scenarios() -> tuple[Scenario, ...]:
    """The scenarios this suite measures."""
    return (Scenario(name="counting", run=_work, iterations=2, warmup=1, rounds=2),)
