"""A benchmark harness whose verdicts survive a shared runner.

Two failure modes kill a performance gate. One fails on a noisy machine and is disabled
within a fortnight; the other passes anything a noisy machine produces and defends
nothing. This one measures the noise alongside the numbers and refuses to call a delta the
noise could have produced, while a regression larger than the noise is named with its
scenario, metric and percentage.

Numbers are only ever compared against a committed baseline for the same scenario on the
same interpreter, and a baseline moves in a reviewed commit rather than by a run
overwriting it — a harness that re-records what it just measured ratchets performance
downwards and calls it green.
"""

from __future__ import annotations

import gc
import json
import math
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
    from pathlib import Path

__all__ = [
    "DEFAULT_FLOORS",
    "DEFAULT_LIMITS",
    "Baseline",
    "BenchmarkReport",
    "Comparison",
    "Measurement",
    "Metric",
    "Scenario",
    "Thresholds",
    "Verdict",
    "compare",
    "load_baseline",
    "measure",
    "run_suite",
    "write_baseline",
]

BASELINE_VERSION = 1


class Metric(StrEnum):
    """What a scenario is measured on."""

    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    THROUGHPUT = "throughput"
    PEAK_BYTES = "peak_bytes"
    ALLOCATIONS = "allocations"
    TOKENS = "tokens"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"  # noqa: S105 — a metric name, not a secret
    TOKENS_PER_SECOND = "tokens_per_second"
    CACHE_HIT_RATIO = "cache_hit_ratio"

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger number is an improvement."""
        return self in _HIGHER_IS_BETTER

    @property
    def scales_with_iterations(self) -> bool:
        """Whether the value depends on how many iterations a round ran.

        Everything else is per iteration or per second; peak memory is the high-water mark
        of a whole round, so a shortened local run cannot be judged against a full baseline.
        """
        return self is Metric.PEAK_BYTES


# Tokens have no tolerance: prompt assembly adding one is a change somebody made, never
# machine noise, and every consumer pays for it on every call.
DEFAULT_LIMITS: Mapping[Metric, float] = {
    Metric.LATENCY_P50: 0.10,
    Metric.LATENCY_P95: 0.10,
    Metric.LATENCY_P99: 0.15,
    Metric.THROUGHPUT: 0.10,
    Metric.PEAK_BYTES: 0.10,
    Metric.ALLOCATIONS: 0.05,
    Metric.TOKENS: 0.0,
    Metric.TIME_TO_FIRST_TOKEN: 0.10,
    Metric.TOKENS_PER_SECOND: 0.10,
    Metric.CACHE_HIT_RATIO: 0.05,
}


# A percentage of a very small number is not a measurement. A run that leaves two blocks
# behind and then three has not regressed by fifty percent, so each metric also carries the
# absolute change below which it is not worth a verdict.
DEFAULT_FLOORS: Mapping[Metric, float] = {
    Metric.LATENCY_P50: 50e-6,
    Metric.LATENCY_P95: 50e-6,
    Metric.LATENCY_P99: 50e-6,
    Metric.THROUGHPUT: 0.0,
    Metric.PEAK_BYTES: 4096.0,
    Metric.ALLOCATIONS: 1.0,
    Metric.TOKENS: 0.0,
    Metric.TIME_TO_FIRST_TOKEN: 5e-3,
    Metric.TOKENS_PER_SECOND: 0.5,
    Metric.CACHE_HIT_RATIO: 0.02,
}


# Read by `Metric.higher_is_better`, and kept beside the enum so adding a metric means
# deciding which way it runs rather than inheriting the wrong default.
_HIGHER_IS_BETTER = frozenset({Metric.THROUGHPUT, Metric.TOKENS_PER_SECOND, Metric.CACHE_HIT_RATIO})


class Verdict(StrEnum):
    """What comparing one metric against its baseline concluded."""

    WITHIN = "within"
    IMPROVED = "improved"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"
    UNRECORDED = "unrecorded"


@dataclass(frozen=True, slots=True)
class Scenario:
    """One measured piece of work, and how many times to do it.

    Args:
        name: What the scenario is called, in the baseline and in the report.
        run: The work, awaited once per iteration. It must not touch the network.
        iterations: Measured repeats per round.
        warmup: Repeats before each round that are run but not timed.
        rounds: How many times the whole measured block is repeated, which is where the
            variance estimate comes from.
        drop_slowest: Discard the slowest round where there are at least three, so one
            stall on a shared runner does not become the result.
        tokens: Cumulative tokens the scenario has consumed, read either side of the last
            measured round. `None` where the scenario has no token cost to report.
        observed: Numbers the scenario measured itself — time to first token, sustained
            rate, cache hit ratio — read after the last measured round. The harness cannot
            time a first token from the outside, so a scenario that cares reports it.

    Raises:
        ValueError: If iterations or rounds is below one, or warmup is negative.
    """

    name: str
    run: Callable[[], Coroutine[object, object, object]]
    iterations: int = 50
    warmup: int = 5
    rounds: int = 5
    drop_slowest: bool = True
    tokens: Callable[[], int] | None = None
    observed: Callable[[], Mapping[Metric, float]] | None = None

    def __post_init__(self) -> None:
        """Refuse a configuration that would produce a number nobody can read."""
        if self.iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {self.iterations}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be at least 1, got {self.rounds}")
        if self.warmup < 0:
            raise ValueError(f"warmup must not be negative, got {self.warmup}")


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one scenario measured, on one interpreter, in one run.

    Args:
        scenario: The scenario's name.
        python: The interpreter the numbers came from, as `major.minor`.
        values: The metric values, per iteration except throughput and peak memory.
        spread: Relative standard deviation of the kept rounds — the run's own noise.
        rounds: How many rounds were kept.
        iterations: Measured iterations per round.
    """

    scenario: str
    python: str
    values: Mapping[Metric, float]
    spread: float
    rounds: int
    iterations: int


@dataclass(frozen=True, slots=True)
class Thresholds:
    """How much worse a metric may get before it is called a regression.

    Args:
        limits: Allowed fractional worsening per metric; anything unlisted uses
            `DEFAULT_LIMITS`.
        floors: Absolute change per metric below which no verdict is drawn, so a metric
            whose value is near zero is not judged on a percentage of nothing. Anything
            unlisted uses `DEFAULT_FLOORS`.
        noise_ceiling: Relative spread above which a delta the noise could have produced
            is reported as inconclusive rather than judged.

    Raises:
        ValueError: If a limit or floor is negative, or the ceiling is outside zero to one.
    """

    limits: Mapping[Metric, float] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    floors: Mapping[Metric, float] = field(default_factory=lambda: dict(DEFAULT_FLOORS))
    noise_ceiling: float = 0.10

    def __post_init__(self) -> None:
        """Refuse a threshold that would make the gate meaningless."""
        for metric, limit in self.limits.items():
            if limit < 0:
                raise ValueError(f"threshold for {metric} must not be negative, got {limit}")
        for metric, floor in self.floors.items():
            if floor < 0:
                raise ValueError(f"floor for {metric} must not be negative, got {floor}")
        if not 0.0 <= self.noise_ceiling <= 1.0:
            raise ValueError(f"noise_ceiling must be between 0 and 1, got {self.noise_ceiling}")

    def limit(self, metric: Metric) -> float:
        """The allowed worsening for `metric`, falling back to the shipped default."""
        return self.limits.get(metric, DEFAULT_LIMITS[metric])

    def floor(self, metric: Metric) -> float:
        """The absolute change `metric` must exceed to be worth a verdict."""
        return self.floors.get(metric, DEFAULT_FLOORS[metric])


@dataclass(frozen=True, slots=True)
class Comparison:
    """One metric of one scenario, measured against its baseline.

    Args:
        scenario: The scenario's name.
        metric: Which metric this is.
        baseline: The recorded value, or `None` where nothing was recorded.
        measured: What this run produced.
        delta: Signed fraction, positive meaning worse, whichever direction the metric runs.
        verdict: What the comparison concluded.
        reason: Why, where the verdict needs one.
    """

    scenario: str
    metric: Metric
    baseline: float | None
    measured: float
    delta: float
    verdict: Verdict
    reason: str = ""

    @property
    def percentage(self) -> str:
        """The delta as a signed percentage, or `n/a` where there is nothing to compare."""
        if self.baseline is None:
            return "n/a"
        if math.isinf(self.delta):
            return "+inf%"
        return f"{self.delta:+.1%}"


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Every comparison from one run, and the verdict a process exits with.

    Args:
        comparisons: One per metric per scenario, in the order they were measured.
    """

    comparisons: tuple[Comparison, ...]

    @property
    def regressions(self) -> tuple[Comparison, ...]:
        """The metrics that got worse by more than their threshold and more than the noise."""
        return tuple(one for one in self.comparisons if one.verdict is Verdict.REGRESSED)

    @property
    def inconclusive(self) -> tuple[Comparison, ...]:
        """The metrics whose delta the run's own variance could have produced."""
        return tuple(one for one in self.comparisons if one.verdict is Verdict.INCONCLUSIVE)

    @property
    def unrecorded(self) -> tuple[Comparison, ...]:
        """The metrics with no committed baseline to compare against."""
        return tuple(one for one in self.comparisons if one.verdict is Verdict.UNRECORDED)

    @property
    def exit_code(self) -> int:
        """0 where everything held, 1 on a regression, 2 where the run was inconclusive."""
        if self.regressions:
            return 1
        return 2 if self.inconclusive else 0

    def render(self) -> str:
        """The report a maintainer reads, naming scenario, metric and delta."""
        lines = [
            f"{one.scenario} {one.metric.value}: {one.verdict.value} "
            f"({one.percentage}, measured {one.measured:.6g})"
            + (f" — {one.reason}" if one.reason else "")
            for one in self.comparisons
        ]
        held = len(self.comparisons) - len(self.regressions) - len(self.inconclusive)
        lines.append(
            f"{len(self.regressions)} regressed, {len(self.inconclusive)} inconclusive, "
            f"{held} within threshold"
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Baseline:
    """The recorded numbers, keyed by scenario and interpreter.

    Args:
        entries: `(scenario, python)` to the metric values recorded for it.
        sizes: `(scenario, python)` to the measured iterations behind those values, so a
            metric that scales with them is not compared across two different sizes.
    """

    entries: Mapping[tuple[str, str], Mapping[Metric, float]]
    sizes: Mapping[tuple[str, str], int] = field(default_factory=dict)

    def size(self, scenario: str, python: str) -> int | None:
        """The iterations this pair was recorded at, or `None` where it is not known."""
        return self.sizes.get((scenario, python))

    def value(self, scenario: str, python: str, metric: Metric) -> float | None:
        """The recorded value, or `None` where this pair has none."""
        return self.entries.get((scenario, python), {}).get(metric)


async def measure(
    scenario: Scenario,
    *,
    timer: Callable[[], float] = time.perf_counter,
    python: str | None = None,
) -> Measurement:
    """Run one scenario and report what it cost.

    Timings are taken with allocation tracing off, and memory is measured in a separate
    round afterwards, since tracing every allocation costs more than most of what is being
    measured here.

    Args:
        scenario: What to run.
        timer: The clock, injectable so a test can state the durations it expects.
        python: The interpreter to record, defaulting to the running one.

    Returns:
        What the scenario measured, with the run's own variance alongside it.
    """
    rounds: list[list[float]] = []
    spent: int | None = None
    for _ in range(scenario.rounds):
        durations_of_round, spent = await _round(scenario, timer)
        rounds.append(durations_of_round)

    kept = _kept(rounds, drop_slowest=scenario.drop_slowest)
    durations = sorted(one for held in kept for one in held)
    elapsed = sum(one for held in kept for one in held)
    values: dict[Metric, float] = {
        Metric.LATENCY_P50: _percentile(durations, 0.50),
        Metric.LATENCY_P95: _percentile(durations, 0.95),
        Metric.LATENCY_P99: _percentile(durations, 0.99),
        Metric.THROUGHPUT: len(durations) / elapsed if elapsed > 0 else math.inf,
    }
    peak, allocations = await _memory(scenario)
    values[Metric.PEAK_BYTES] = float(peak)
    values[Metric.ALLOCATIONS] = allocations / scenario.iterations
    if spent is not None:
        values[Metric.TOKENS] = spent / scenario.iterations
    if scenario.observed is not None:
        values.update(scenario.observed())
    return Measurement(
        scenario=scenario.name,
        python=python or f"{sys.version_info.major}.{sys.version_info.minor}",
        values=values,
        spread=_spread(kept),
        rounds=len(kept),
        iterations=scenario.iterations,
    )


async def run_suite(
    scenarios: Sequence[Scenario],
    *,
    timer: Callable[[], float] = time.perf_counter,
    python: str | None = None,
) -> tuple[Measurement, ...]:
    """Measure every scenario in order.

    Args:
        scenarios: What to run. Names must be distinct, since a baseline is keyed by name.
        timer: The clock, passed through to `measure`.
        python: The interpreter to record, passed through to `measure`.

    Returns:
        One measurement per scenario.

    Raises:
        ValueError: If two scenarios share a name.
    """
    seen: set[str] = set()
    for one in scenarios:
        if one.name in seen:
            raise ValueError(f"two scenarios are named {one.name!r}")
        seen.add(one.name)
    return tuple([await measure(one, timer=timer, python=python) for one in scenarios])


def compare(
    measurements: Sequence[Measurement],
    baseline: Baseline,
    thresholds: Thresholds | None = None,
) -> BenchmarkReport:
    """Judge each measurement against the baseline recorded for its own interpreter.

    Args:
        measurements: What this run produced.
        baseline: What was committed.
        thresholds: How much movement is allowed, defaulting to the shipped limits.

    Returns:
        Every comparison, including the metrics nothing was recorded for.
    """
    against = thresholds or Thresholds()
    comparisons = [
        _judged(one, metric, value, baseline, against)
        for one in measurements
        for metric, value in one.values.items()
    ]
    return BenchmarkReport(tuple(comparisons))


def load_baseline(path: Path) -> Baseline:
    """Read the committed baseline, treating an absent file as nothing recorded yet.

    Args:
        path: Where the baseline lives.

    Returns:
        What was recorded, or an empty baseline.

    Raises:
        ValueError: If the file is not a baseline document. A malformed file read as empty
            would silently pass every metric it was supposed to defend.
    """
    if not path.exists():
        return Baseline({})
    held = json.loads(path.read_text())
    if not isinstance(held, dict) or "scenarios" not in held:
        raise ValueError(f"{path} is not a baseline document")
    entries: dict[tuple[str, str], Mapping[Metric, float]] = {}
    sizes: dict[tuple[str, str], int] = {}
    for scenario, interpreters in held["scenarios"].items():
        for python, recorded in interpreters.items():
            entries[scenario, python] = {
                Metric(name): float(value) for name, value in recorded["metrics"].items()
            }
            sizes[scenario, python] = int(recorded["iterations"])
    return Baseline(entries, sizes)


def write_baseline(path: Path, measurements: Iterable[Measurement]) -> None:
    """Record `measurements` as the baseline, merging with what is already there.

    Only what was measured is replaced, so re-recording one interpreter does not erase
    another's numbers. This is deliberately never called by a check run: a baseline that
    updates itself ratchets performance downwards without anybody reviewing it.

    Args:
        path: Where the baseline lives.
        measurements: What to record.
    """
    held: dict[str, dict[str, dict[str, object]]] = {}
    if path.exists():
        held = json.loads(path.read_text()).get("scenarios", {})
    for one in measurements:
        held.setdefault(one.scenario, {})[one.python] = {
            "rounds": one.rounds,
            "iterations": one.iterations,
            "spread": round(one.spread, 4),
            "metrics": {
                metric.value: round(value, 6) for metric, value in sorted(one.values.items())
            },
        }
    document = {
        "version": BASELINE_VERSION,
        "scenarios": {name: held[name] for name in sorted(held)},
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")


async def _round(scenario: Scenario, timer: Callable[[], float]) -> tuple[list[float], int | None]:
    """One round: the warm-up nobody measures, then the iterations that count.

    Tokens are bracketed around the measured iterations alone, so the warm-up's own
    consumption is not billed to the round it was warming up for.
    """
    for _ in range(scenario.warmup):
        await scenario.run()
    before = scenario.tokens() if scenario.tokens is not None else None
    durations: list[float] = []
    for _ in range(scenario.iterations):
        started = timer()
        await scenario.run()
        durations.append(timer() - started)
    spent = None if scenario.tokens is None or before is None else scenario.tokens() - before
    return durations, spent


async def _memory(scenario: Scenario) -> tuple[int, float]:
    """Peak bytes and allocation count, measured apart from the timings, over several rounds.

    Tracing starts after the warm-up and a collection runs before the count, so neither a
    cache built on the first pass nor garbage nobody has swept is billed to the scenario.
    The median across rounds is taken because a block count on a small scenario moves by a
    block or two between runs, and on a mean of ten that reads as a ten-percent regression.
    """
    peaks: list[int] = []
    counts: list[float] = []
    for _ in range(scenario.rounds):
        for _ in range(scenario.warmup):
            await scenario.run()
        tracemalloc.start()
        try:
            for _ in range(scenario.iterations):
                await scenario.run()
            gc.collect()
            _, peak = tracemalloc.get_traced_memory()
            live = sum(one.count for one in tracemalloc.take_snapshot().statistics("filename"))
        finally:
            tracemalloc.stop()
        peaks.append(peak)
        counts.append(float(live))
    return round(statistics.median(peaks)), statistics.median(counts)


def _kept(rounds: Sequence[Sequence[float]], *, drop_slowest: bool) -> list[list[float]]:
    """Every round, minus the slowest where there are enough rounds to spare one."""
    held = [list(one) for one in rounds]
    if drop_slowest and len(held) >= 3:
        held.remove(max(held, key=sum))
    return held


def _percentile(sorted_durations: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, which needs no interpolation to explain in a report."""
    rank = max(1, math.ceil(fraction * len(sorted_durations)))
    return sorted_durations[rank - 1]


def _spread(rounds: Sequence[Sequence[float]]) -> float:
    """How far the rounds disagreed with each other, relative to their mean."""
    means = [statistics.fmean(one) for one in rounds]
    average = statistics.fmean(means)
    if len(means) < 2 or average == 0:
        return 0.0
    return statistics.pstdev(means) / average


def _judged(
    one: Measurement,
    metric: Metric,
    value: float,
    baseline: Baseline,
    thresholds: Thresholds,
) -> Comparison:
    """One metric against its baseline, with the run's own noise taken into account."""
    recorded = baseline.value(one.scenario, one.python, metric)
    if recorded is None:
        return Comparison(
            one.scenario, metric, None, value, 0.0, Verdict.UNRECORDED, "no baseline recorded"
        )
    delta = _delta(recorded, value, higher_is_better=metric.higher_is_better)
    size = baseline.size(one.scenario, one.python)
    if metric.scales_with_iterations and size is not None and size != one.iterations:
        return Comparison(
            one.scenario,
            metric,
            recorded,
            value,
            delta,
            Verdict.INCONCLUSIVE,
            f"measured over {one.iterations} iterations against a baseline recorded over "
            f"{size}; run the scenario at its declared size to judge this one",
        )
    limit = thresholds.limit(metric)
    if abs(delta) <= limit or abs(value - recorded) <= thresholds.floor(metric):
        return Comparison(one.scenario, metric, recorded, value, delta, Verdict.WITHIN)
    if one.spread > thresholds.noise_ceiling and abs(delta) <= one.spread:
        return Comparison(
            one.scenario,
            metric,
            recorded,
            value,
            delta,
            Verdict.INCONCLUSIVE,
            f"a spread of {one.spread:.1%} exceeds the {thresholds.noise_ceiling:.1%} ceiling "
            f"and covers the {abs(delta):.1%} delta; needs a quieter runner or more rounds "
            f"than {one.rounds} to separate them",
        )
    verdict = Verdict.REGRESSED if delta > 0 else Verdict.IMPROVED
    return Comparison(one.scenario, metric, recorded, value, delta, verdict)


def _delta(recorded: float, value: float, *, higher_is_better: bool) -> float:
    """The signed movement, positive meaning worse whichever way the metric runs."""
    worse = (recorded - value) if higher_is_better else (value - recorded)
    if recorded == 0:
        return 0.0 if worse == 0 else math.inf * math.copysign(1.0, worse)
    return worse / abs(recorded)
