"""Measuring a scenario, and what the harness refuses to conclude from it.

A gate that fails on a noisy machine gets switched off, and one that passes anything a
noisy machine produces defends nothing. This walks the middle: a regression larger than the
measured noise is named, a delta the noise could have produced is reported as inconclusive,
and a change too small to be worth a percentage is left alone.

Everything here is local arithmetic — no provider, no network, no key. Run it with
`python examples/benchmarks.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.testing.benchmarks import (
    Baseline,
    Measurement,
    Metric,
    Scenario,
    compare,
    measure,
)

SIZE = 20_000


async def assemble() -> None:
    """The work under measurement: cheap, deterministic, allocating something."""
    paragraphs = [f"paragraph {index}" for index in range(SIZE)]
    joined = "\n".join(paragraphs)
    del joined


def recorded(**values: float) -> Baseline:
    """A committed baseline for this scenario on this interpreter."""
    return Baseline(
        {("assemble", "3.13"): {Metric(name): value for name, value in values.items()}},
        {("assemble", "3.13"): 5},
    )


def taken(spread: float, **values: float) -> Measurement:
    """A measurement with the run's own noise stated alongside it."""
    return Measurement(
        scenario="assemble",
        python="3.13",
        values={Metric(name): value for name, value in values.items()},
        spread=spread,
        rounds=5,
        iterations=5,
    )


async def main() -> None:
    """Measure once, then show what the same numbers are judged to mean."""
    result = await measure(Scenario(name="assemble", run=assemble, iterations=5, rounds=3))
    print("measured")  # noqa: T201
    for metric, value in sorted(result.values.items()):
        print(f"  {metric.value:<14} {value:.6g}")  # noqa: T201
    print(f"  spread         {result.spread:.1%}")  # noqa: T201

    print("\nverdicts")  # noqa: T201
    for label, one, against in (
        ("steady run, unchanged", taken(0.01, tokens=420.0), recorded(tokens=420.0)),
        ("one more token", taken(0.01, tokens=421.0), recorded(tokens=420.0)),
        ("40% slower, quiet runner", taken(0.01, latency_p95=1.4), recorded(latency_p95=1.0)),
        ("20% slower, noisy runner", taken(0.30, latency_p95=1.2), recorded(latency_p95=1.0)),
        ("two blocks became three", taken(0.01, allocations=3.0), recorded(allocations=2.0)),
        ("nothing recorded yet", taken(0.01, peak_bytes=1e6), recorded(tokens=420.0)),
    ):
        report = compare((one,), against)
        print(f"  {label:<26} {report.comparisons[0].verdict.value}")  # noqa: T201

    noisy = compare((taken(0.30, latency_p95=1.2),), recorded(latency_p95=1.0))
    print(f"\nwhy inconclusive: {noisy.comparisons[0].reason}")  # noqa: T201
    print(f"exit code: {noisy.exit_code}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
