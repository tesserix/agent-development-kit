"""Run the benchmark suite and judge it against the committed baseline.

`python -m tools.benchmark` (or `make bench`) measures and compares; `--write` records a
new baseline, which is a reviewed commit rather than something a check run does for you.

Exit codes are the point of the command: 0 held, 1 regressed, 2 the run was too noisy to
say, 3 the suite could not be loaded. CI fails on 1 and warns on 2, so a shared runner
having a bad afternoon does not block a merge and does not quietly pass a regression.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.testing.benchmarks import (
    Measurement,
    Metric,
    Scenario,
    compare,
    load_baseline,
    run_suite,
    write_baseline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = "benchmarks.suite"
DEFAULT_BASELINE = ROOT / "benchmarks" / "baseline.json"


def scenarios_from(module: str) -> tuple[Scenario, ...]:
    """Load a suite module and return the scenarios it names.

    Args:
        module: Importable module exposing `scenarios()`.

    Returns:
        The scenarios, in the order the suite declares them.

    Raises:
        ValueError: If the module exposes no `scenarios()`. A typo that read as an empty
            run would report green while measuring nothing.
        ImportError: If the module cannot be imported.
    """
    loaded = importlib.import_module(module)
    naming = getattr(loaded, "scenarios", None)
    if naming is None:
        raise ValueError(f"{module} exposes no scenarios()")
    return tuple(naming())


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, then either record or judge. See the module docstring for exit codes."""
    parsed = _parser().parse_args(argv)
    try:
        scenarios = scenarios_from(parsed.suite)
    except (ImportError, ValueError) as unloadable:
        print(f"cannot load suite {parsed.suite}: {unloadable}", file=sys.stderr)  # noqa: T201
        return 3

    scenarios = tuple(_sized(one, parsed) for one in scenarios)
    measurements = asyncio.run(run_suite(scenarios))
    baseline = Path(parsed.baseline)
    if parsed.write:
        write_baseline(baseline, [_only(one, parsed.only) for one in measurements])
        print(f"recorded {len(measurements)} scenarios in {baseline}")  # noqa: T201
        return 0

    report = compare(measurements, load_baseline(baseline))
    print(report.render())  # noqa: T201
    return report.exit_code


def _only(measurement: Measurement, wanted: str | None) -> Measurement:
    """The measurement narrowed to the metrics being recorded, where a subset was asked for."""
    if not wanted:
        return measurement
    keep = {Metric(name.strip()) for name in wanted.split(",")}
    return replace(
        measurement,
        values={metric: value for metric, value in measurement.values.items() if metric in keep},
    )


def _sized(scenario: Scenario, parsed: argparse.Namespace) -> Scenario:
    """The scenario as the suite declared it, cut down where a local run asked for less."""
    return replace(
        scenario,
        rounds=parsed.rounds if parsed.rounds is not None else scenario.rounds,
        iterations=parsed.iterations if parsed.iterations is not None else scenario.iterations,
    )


def _parser() -> argparse.ArgumentParser:
    """The command line, small enough that a contributor can guess it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="module exposing scenarios()")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="baseline file")
    parser.add_argument(
        "--write", action="store_true", help="record a new baseline instead of checking"
    )
    parser.add_argument("--rounds", type=int, help="override the rounds each scenario runs")
    parser.add_argument("--iterations", type=int, help="override the iterations per round")
    # A committed baseline holds the metrics that travel: wall clock on a shared runner
    # does not, and gating on it is how a performance job gets switched off.
    parser.add_argument("--only", help="comma-separated metrics to record, e.g. tokens,allocations")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
