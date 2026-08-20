"""What a change did to answers, to spend and to tail latency, in one report.

Quality is argued about in review and cost is discovered on the bill, so a prompt that
answers better while tripling spend reads as an unqualified win right up until the invoice
arrives. Cost per case and p95 latency are metrics here on the same footing as correctness:
same protocol, same aggregation, same thresholds, same report.

Three rules keep the numbers honest. A value nobody could compute is `unknown` and never
zero, because a self-hosted model with no price data is not free and averaging it as free
understates every suite it appears in. A metric that raised is an errored case with its
traceback, never a passing zero. And a threshold that could not be evaluated fails, because
a gate that clears itself when the number is missing is not a gate.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. Custom
metrics are consumer-owned code against `Metric`, so that protocol moves only under the
deprecation policy in `docs/versioning.md`.
"""

from __future__ import annotations

import math
import re
import statistics
import traceback
from collections.abc import Mapping, Sequence  # noqa: TC003 — read at runtime by the report
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from tesserix_adk.core.cost import CostConfidence
from tesserix_adk.core.run import RunState

if TYPE_CHECKING:
    from tesserix_adk.core.run import Run
    from tesserix_adk.evals.dataset import EvalCase, EvalSuite
    from tesserix_adk.evals.suite import SuiteResult

__all__ = [
    "Aggregate",
    "CacheHitRate",
    "CostPerCase",
    "ExactMatch",
    "Groundedness",
    "LatencyMs",
    "Metric",
    "MetricFailure",
    "MetricReport",
    "MetricValue",
    "RefusalRate",
    "SchemaValidity",
    "Threshold",
    "ThresholdResult",
    "TokensIn",
    "TokensOut",
    "ToolSequenceMatch",
    "measure",
]

Verdict = Literal["pass", "warn", "fail"]

_RELIABLE_SAMPLE = 5
"""Below this many scored cases an interval says more about luck than about the change."""

_Z95 = 1.96

_CITATION = re.compile(r"\[([^\[\]]+)\]")

_REFUSALS = ("i cannot", "i can't", "i am unable", "i'm unable", "i won't", "i will not")


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One metric's answer for one case, or its reason for not having one.

    Args:
        value: The number, or `None` where nothing could compute it.
        reason: Why there is no number. Empty when there is one.
        unit: What the number counts, for the report's own column.
        currency: ISO 4217 code where the value is money, so two providers are never added
            together without anyone noticing.
    """

    value: float | None = None
    reason: str = ""
    unit: str = ""
    currency: str = ""

    @property
    def known(self) -> bool:
        """Whether this carries a number. An unknown is not a zero.

        Example:
            >>> MetricValue(reason="nothing priced this model").known
            False
        """
        return self.value is not None


@runtime_checkable
class Metric(Protocol):
    """One thing a suite measures, built in or consumer-written.

    Implementations are ordinary objects with three names. `compute` is called once per
    case that produced a run; it returns an unknown rather than raising for a case it
    cannot judge, and anything it does raise errors that case rather than scoring it.
    """

    @property
    def name(self) -> str:
        """What the report calls this metric. Two metrics may not share one."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """Which direction is an improvement, which is what a gate compares against."""
        ...

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:
        """Measure `run` against what `case` asked for."""
        ...


def _answer(run: Run[Any]) -> str:
    """The assistant's last text, which is what a reader would call the answer."""
    for message in reversed(run.messages):
        if message.role != "assistant":
            continue
        return "".join(getattr(part, "text", "") for part in message.content)
    return ""


def _unfinished(run: Run[Any]) -> MetricValue | None:
    """An unknown for a run that never produced an answer to judge, otherwise `None`."""
    if run.state is RunState.COMPLETED:
        return None
    return MetricValue(reason=f"the run was {run.state}, so there is no answer to judge")


@dataclass(frozen=True, slots=True)
class ExactMatch:
    """Whether the answer is the one the case declared, ignoring case and outer spacing."""

    name: str = "exact_match"
    higher_is_better: bool = True

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:
        """Score `1.0` on a match and `0.0` otherwise; unknown when the case declared none."""
        unfinished = _unfinished(run)
        if unfinished is not None:
            return unfinished
        if case.expected is None:
            return MetricValue(reason="the case declares no expected answer")
        matched = _answer(run).strip().casefold() == case.expected.strip().casefold()
        return MetricValue(value=float(matched))


@dataclass(frozen=True, slots=True)
class SchemaValidity:
    """Whether the run produced the structured output its agent declared."""

    name: str = "schema_validity"
    higher_is_better: bool = True

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
        """Score `1.0` where validated output arrived, `0.0` where a finished run had none."""
        unfinished = _unfinished(run)
        if unfinished is not None:
            return unfinished
        return MetricValue(value=float(run.output is not None))


@dataclass(frozen=True, slots=True)
class ToolSequenceMatch:
    """Whether the tools ran in the order the case declared in `expected_tools`."""

    name: str = "tool_sequence_match"
    higher_is_better: bool = True

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:
        """Score `1.0` on the exact sequence; unknown when the case declares no sequence."""
        unfinished = _unfinished(run)
        if unfinished is not None:
            return unfinished
        if not case.expected_tools:
            return MetricValue(reason="the case declares no expected tool sequence")
        called = tuple(call.name for call in run.tool_calls)
        return MetricValue(value=float(called == tuple(case.expected_tools)))


@dataclass(frozen=True, slots=True)
class Groundedness:
    """The share of the answer's citations that name a source the case declared.

    This measures citation discipline, not semantic support: an answer citing a real source
    for a claim that source does not make scores well here. The judge story covers support.
    """

    name: str = "groundedness"
    higher_is_better: bool = True

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:
        """Score the cited fraction; unknown when the answer cites nothing at all."""
        unfinished = _unfinished(run)
        if unfinished is not None:
            return unfinished
        cited = _CITATION.findall(_answer(run))
        if not cited:
            return MetricValue(reason="the answer cites nothing, so there is nothing to check")
        declared = set(case.expected_sources)
        return MetricValue(value=sum(one in declared for one in cited) / len(cited))


@dataclass(frozen=True, slots=True)
class RefusalRate:
    """Whether the agent declined. Lower is better, and a refused case is still measured."""

    name: str = "refusal_rate"
    higher_is_better: bool = False

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
        """Score `1.0` where a guardrail refused or the answer reads as a refusal."""
        unfinished = _unfinished(run)
        if unfinished is not None:
            return unfinished
        answer = _answer(run).casefold()
        return MetricValue(value=float(any(phrase in answer for phrase in _REFUSALS)))


def _reported_usage(run: Run[Any]) -> MetricValue | None:
    """An unknown where the provider reported no usage at all, otherwise `None`."""
    if run.usage.input_tokens or run.usage.output_tokens:
        return None
    return MetricValue(reason="the provider reported no usage for this run")


@dataclass(frozen=True, slots=True)
class TokensIn:
    """Prompt tokens, cached ones included, as the provider counted them."""

    name: str = "tokens_in"
    higher_is_better: bool = False

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
        """Return the input tokens; unknown where the response carried no usage."""
        return _reported_usage(run) or MetricValue(
            value=float(run.usage.input_tokens), unit="tokens"
        )


@dataclass(frozen=True, slots=True)
class TokensOut:
    """Generated tokens, as the provider counted them."""

    name: str = "tokens_out"
    higher_is_better: bool = False

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
        """Return the output tokens; unknown where the response carried no usage."""
        return _reported_usage(run) or MetricValue(
            value=float(run.usage.output_tokens), unit="tokens"
        )


@dataclass(frozen=True, slots=True)
class CacheHitRate:
    """The share of prompt tokens the provider served from its own cache."""

    name: str = "cache_hit_rate"
    higher_is_better: bool = True

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
        """Return cached over input tokens; unknown where nothing was sent to be cached."""
        if not run.usage.input_tokens:
            return MetricValue(reason="the run sent no prompt tokens to hit a cache with")
        return MetricValue(value=run.usage.cached_tokens / run.usage.input_tokens)


@dataclass(frozen=True, slots=True)
class CostPerCase:
    """What one case cost, in the currency the provider priced it in."""

    name: str = "cost_per_case"
    higher_is_better: bool = False

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
        """Return the run's cost; unknown where nothing priced the model.

        A self-hosted vLLM or Ollama deployment has no price list, and reporting it as zero
        would let a migration to it read as free rather than as unmeasured.
        """
        cost = run.usage.cost
        if cost is None or cost.confidence is CostConfidence.UNKNOWN:
            return MetricValue(reason="nothing priced this model, so the cost is unknown")
        return MetricValue(value=float(cost.total), unit="cost", currency=cost.currency)


@dataclass(frozen=True, slots=True)
class LatencyMs:
    """Wall-clock milliseconds from the run starting to the run ending."""

    name: str = "latency_ms"
    higher_is_better: bool = False

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the Metric protocol's shape
        """Return the elapsed milliseconds; unknown where the run recorded no timings."""
        if run.started_at is None or run.ended_at is None:
            return MetricValue(reason="the run recorded no start or end, so it has no latency")
        return MetricValue(value=(run.ended_at - run.started_at) * 1000.0, unit="ms")


@dataclass(frozen=True, slots=True)
class Aggregate:
    """One metric across a set of cases, with the sample it rests on shown.

    Args:
        metric: The metric's name.
        n: How many cases produced a value.
        unknown: How many could not, counted apart rather than averaged in as zero.
        mean: The average, or `None` where nothing was measured.
        p50: The median value.
        p95: The tail, which is what a user waiting notices.
        ci_low: Lower bound of the 95% interval, `None` for a sample too small to have one.
        ci_high: Upper bound.
        reliable: Whether the sample is large enough for the interval to mean anything.
        unit: The metric's unit, for the report's column.
        currency: The currency, where the metric is money.
        note: Why a number is missing or an interval absent, in one line.
    """

    metric: str
    n: int
    unknown: int
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    reliable: bool = False
    unit: str = ""
    currency: str = ""
    note: str = ""

    def line(self) -> str:
        """One row of the human-readable table.

        Example:
            >>> Aggregate(metric="tokens_in", n=1, unknown=0, mean=8.0).line()
            'tokens_in            8.0000                                             n=1 unknown=0'
        """
        shown = "unknown" if self.mean is None else f"{self.mean:.4f}"
        tail = "" if self.p95 is None else f"p50={self.p50:.4f} p95={self.p95:.4f}"
        unit = self.currency or self.unit
        return (
            f"{self.metric:<20} {shown:<12} {unit:<8} {tail:<28} "
            f"n={self.n} unknown={self.unknown}{' ' + self.note if self.note else ''}"
        )


@dataclass(frozen=True, slots=True)
class Threshold:
    """A limit one metric has to stay inside for the report to pass.

    Args:
        metric: The metric's name, which must be one the report computed.
        maximum: The most the metric may be, for cost, latency and refusals.
        minimum: The least it may be, for correctness.
        warn_within: A margin before the limit reported as `warn` rather than `fail`, so a
            metric drifting towards its ceiling is visible before it breaches.

    Raises:
        ValueError: Neither bound was given, which would decide nothing.
    """

    metric: str
    maximum: float | None = None
    minimum: float | None = None
    warn_within: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a threshold with no bound, which would report on nothing."""
        if self.maximum is None and self.minimum is None:
            raise ValueError(
                f"threshold on {self.metric!r} declares neither a maximum nor a minimum"
            )

    def judge(self, value: float | None) -> ThresholdResult:
        """Decide `pass`, `warn` or `fail` for one aggregated value.

        An unaggregated metric fails: a threshold that clears itself when the number is
        missing would pass the run it exists to stop.
        """
        limit = cast("float", self.maximum if self.maximum is not None else self.minimum)
        if value is None:
            return ThresholdResult(
                metric=self.metric,
                verdict="fail",
                value=None,
                limit=limit,
                reason=f"nothing measured {self.metric}, so its threshold cannot be cleared",
            )
        if self.maximum is not None and value > self.maximum:
            return self._breached(value, self.maximum, "above")
        if self.minimum is not None and value < self.minimum:
            return self._breached(value, self.minimum, "below")
        if self._near(value):
            return ThresholdResult(
                metric=self.metric,
                verdict="warn",
                value=value,
                limit=limit,
                reason=f"{self.metric} is within {self.warn_within:g} of its limit",
            )
        return ThresholdResult(
            metric=self.metric,
            verdict="pass",
            value=value,
            limit=limit,
            reason=f"{self.metric} is {value:g}, inside its limit of {limit:g}",
        )

    def _near(self, value: float) -> bool:
        """Whether `value` sits inside the warning margin on either declared bound."""
        if not self.warn_within:
            return False
        if self.maximum is not None and value > self.maximum - self.warn_within:
            return True
        return self.minimum is not None and value < self.minimum + self.warn_within

    def _breached(self, value: float, limit: float, side: str) -> ThresholdResult:
        """The failing verdict, worded so the report names the number and the limit."""
        return ThresholdResult(
            metric=self.metric,
            verdict="fail",
            value=value,
            limit=limit,
            reason=f"{self.metric} is {value:g}, {side} its limit of {limit:g}",
        )


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """What one threshold decided, and about which number.

    Args:
        metric: The metric judged.
        verdict: `pass`, `warn` or `fail`.
        value: The aggregated value, or `None` where there was none.
        limit: The bound it was judged against.
        reason: Why, in the words the report prints.
    """

    metric: str
    verdict: Verdict
    value: float | None
    limit: float | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MetricFailure:
    """A metric that raised on a case, kept whole rather than scored as zero.

    Args:
        case_id: The case being measured.
        metric: The metric that raised.
        traceback: The formatted traceback, so a consumer-owned metric is debuggable.
    """

    case_id: str
    metric: str
    traceback: str


@dataclass(frozen=True, slots=True)
class MetricReport:
    """Every metric over a suite: aggregates, per-tag breakdowns, verdicts and breakages.

    Args:
        suite: The suite measured.
        version: The dataset version, since numbers only compare within one.
        aggregates: One per metric, over every case that produced a value.
        by_tag: The same aggregates restricted to each tag the dataset uses.
        verdicts: One per declared threshold.
        failures: Metrics that raised, with their tracebacks.
        unscored: Cases that produced no run, so no metric could see them.
        values: Every known per-case value, by metric and then by case id. An unknown is
            absent rather than zero, so a regression gate can name the cases that moved.
    """

    suite: str
    version: str
    aggregates: Mapping[str, Aggregate]
    by_tag: Mapping[str, Mapping[str, Aggregate]] = field(default_factory=dict)
    verdicts: tuple[ThresholdResult, ...] = ()
    failures: tuple[MetricFailure, ...] = ()
    unscored: tuple[str, ...] = ()
    values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether nothing breached a threshold and no metric raised."""
        breached = any(result.verdict == "fail" for result in self.verdicts)
        return not breached and not self.failures

    @property
    def exit_code(self) -> int:
        """`0` when the report passes, `1` otherwise — what CI reads."""
        return 0 if self.ok else 1

    def aggregate(self, metric: str, *, tag: str | None = None) -> Aggregate:
        """Return one metric's aggregate, over the whole suite or one tag.

        Raises:
            KeyError: No such metric, or no case carries that tag. An empty row would read
                as a measurement of zero cases rather than as a missing metric.
        """
        if tag is None:
            return self.aggregates[metric]
        return self.by_tag[tag][metric]

    def verdict(self, metric: str) -> ThresholdResult:
        """Return the threshold result for `metric`.

        Raises:
            KeyError: No threshold was declared for that metric.
        """
        for result in self.verdicts:
            if result.metric == metric:
                return result
        raise KeyError(f"no threshold was declared for {metric!r}")

    def table(self) -> str:
        """The whole report as a fixed-width table, one metric per row."""
        rows = [aggregate.line() for aggregate in self.aggregates.values()]
        rows.extend(f"{result.verdict.upper():<8} {result.reason}" for result in self.verdicts)
        return "\n".join(rows)

    def summary(self) -> str:
        """One line: what passed, and which metric did not."""
        broken = [failure.metric for failure in self.failures]
        breached = [result.metric for result in self.verdicts if result.verdict == "fail"]
        if not broken and not breached:
            return f"{self.suite} {self.version}: {len(self.aggregates)} metrics, all within limits"
        parts = []
        if breached:
            parts.append("breached " + ", ".join(breached))
        if broken:
            parts.append("raised " + ", ".join(broken))
        return f"{self.suite} {self.version}: " + "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """The machine-readable report the CI gate reads."""
        return {
            "suite": self.suite,
            "version": self.version,
            "ok": self.ok,
            "metrics": {
                name: {
                    "mean": aggregate.mean,
                    "p50": aggregate.p50,
                    "p95": aggregate.p95,
                    "ci_low": aggregate.ci_low,
                    "ci_high": aggregate.ci_high,
                    "n": aggregate.n,
                    "unknown": aggregate.unknown,
                    "unit": aggregate.unit,
                    "currency": aggregate.currency,
                    "reliable": aggregate.reliable,
                    "note": aggregate.note,
                }
                for name, aggregate in self.aggregates.items()
            },
            "by_tag": {
                tag: {name: aggregate.mean for name, aggregate in tagged.items()}
                for tag, tagged in self.by_tag.items()
            },
            "verdicts": {
                result.metric: {
                    "verdict": result.verdict,
                    "value": result.value,
                    "limit": result.limit,
                    "reason": result.reason,
                }
                for result in self.verdicts
            },
            "failures": [
                {"case_id": failure.case_id, "metric": failure.metric} for failure in self.failures
            ],
            "unscored": list(self.unscored),
            "values": {name: dict(cases) for name, cases in self.values.items()},
        }


def measure(
    suite: EvalSuite,
    result: SuiteResult,
    metrics: Sequence[Metric],
    *,
    thresholds: Sequence[Threshold] = (),
) -> MetricReport:
    """Compute every metric over every case that ran, and judge the declared thresholds.

    Args:
        suite: The dataset, which is where a case's expectations and tags live.
        result: What the runner produced. Cases with no run are reported as unscored.
        metrics: What to measure. Two metrics sharing a name are refused.
        thresholds: Limits to judge, each naming a metric in `metrics`.

    Returns:
        The report, whose `ok` is false when a threshold was breached or a metric raised.

    Raises:
        ValueError: Two metrics share a name, so a report row would silently be overwritten.
        KeyError: A threshold names a metric nobody computed.
    """
    named = _named(metrics)
    for threshold in thresholds:
        if threshold.metric not in named:
            raise KeyError(f"threshold names {threshold.metric!r}, which no metric computes")

    cases = {case.id: case for case in suite.cases}
    scored: dict[str, list[tuple[str, MetricValue]]] = {name: [] for name in named}
    failures: list[MetricFailure] = []
    unscored: list[str] = []

    for outcome in result.results:
        case = cases.get(outcome.case_id)
        if case is None or outcome.run is None:
            unscored.append(outcome.case_id)
            continue
        for name, metric in named.items():
            try:
                value = metric.compute(case, outcome.run)
            except Exception:
                failures.append(
                    MetricFailure(case_id=case.id, metric=name, traceback=traceback.format_exc())
                )
                continue
            scored[name].append((case.id, value))

    aggregates = {name: _aggregate(name, values) for name, values in scored.items()}
    tags = sorted({tag for case in suite.cases for tag in case.tags})
    by_tag = {
        tag: {
            name: _aggregate(name, [pair for pair in values if tag in cases[pair[0]].tags])
            for name, values in scored.items()
        }
        for tag in tags
    }
    verdicts = tuple(threshold.judge(aggregates[threshold.metric].mean) for threshold in thresholds)
    return MetricReport(
        suite=suite.name,
        version=suite.version,
        aggregates=aggregates,
        by_tag=by_tag,
        verdicts=verdicts,
        failures=tuple(failures),
        unscored=tuple(unscored),
        values={
            name: {case_id: value.value for case_id, value in measured if value.value is not None}
            for name, measured in scored.items()
        },
    )


def _named(metrics: Sequence[Metric]) -> dict[str, Metric]:
    """Index the metrics by name, refusing a name two of them answer to."""
    named: dict[str, Metric] = {}
    for metric in metrics:
        if metric.name in named:
            raise ValueError(f"two metrics are named {metric.name!r}")
        named[metric.name] = metric
    return named


def _aggregate(name: str, measured: Sequence[tuple[str, MetricValue]]) -> Aggregate:
    """Average what was measured, keeping the unknowns out of the mean and in the count."""
    known = [value for _, value in measured if value.known]
    unknown = len(measured) - len(known)
    unit = next((value.unit for value in known if value.unit), "")
    currencies = sorted({value.currency for value in known if value.currency})
    if not known:
        return Aggregate(metric=name, n=0, unknown=unknown, unit=unit, note="nothing was measured")
    if len(currencies) > 1:
        return Aggregate(
            metric=name,
            n=len(known),
            unknown=unknown,
            unit=unit,
            note=f"mixed currencies ({', '.join(currencies)}) cannot be averaged",
        )

    numbers = sorted(value.value or 0.0 for value in known)
    mean = statistics.fmean(numbers)
    ci_low, ci_high, reliable, note = _interval(numbers, mean)
    return Aggregate(
        metric=name,
        n=len(numbers),
        unknown=unknown,
        mean=mean,
        p50=_percentile(numbers, 0.50),
        p95=_percentile(numbers, 0.95),
        ci_low=ci_low,
        ci_high=ci_high,
        reliable=reliable,
        unit=unit,
        currency=currencies[0] if currencies else "",
        note=note,
    )


def _interval(
    numbers: Sequence[float], mean: float
) -> tuple[float | None, float | None, bool, str]:
    """The 95% interval, or a note saying the sample is too small to have one."""
    if len(numbers) < _RELIABLE_SAMPLE:
        return None, None, False, "sample too small for a confidence interval"
    margin = _Z95 * statistics.stdev(numbers) / math.sqrt(len(numbers))
    return mean - margin, mean + margin, True, ""


def _percentile(numbers: Sequence[float], share: float) -> float:
    """Nearest-rank percentile over already-sorted `numbers`."""
    rank = max(1, math.ceil(share * len(numbers)))
    return numbers[rank - 1]
