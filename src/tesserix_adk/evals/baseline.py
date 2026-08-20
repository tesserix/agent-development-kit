"""The numbers a change is measured against, and the gate that reads them.

A one-word prompt edit or a model bump can degrade an agent measurably and merge anyway,
because nothing in review compared it with the version already in production. The
regression is found later by users, by which time the commit that caused it is buried in a
week of other work.

So a suite's numbers are stored: per-metric aggregates and per-case values, with the
provenance that makes them mean something — agent version, prompt version, model, dataset
version and the cassettes they were replayed from. A candidate run is compared against that
artefact with a noise band, so ordinary variance is not reported as a regression, and the
cases that actually got worse are named rather than averaged away.

It fails closed. A missing baseline, one recorded on another dataset version, and a declared
metric nobody measured all refuse. A gate that passes because there was nothing to compare
against is not a gate, so the first run on a new suite is bootstrapped deliberately.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence  # noqa: TC003 — pydantic reads these at runtime
from pathlib import Path  # noqa: TC003 — pydantic reads it at runtime
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError, model_validator

from tesserix_adk.core.errors import BaselineUnusableError, ConfigurationError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.evals.gate import (  # noqa: TC001 — pydantic reads them at runtime
    Bypass,
    Tolerance,
)

if TYPE_CHECKING:
    from tesserix_adk.evals.metrics import Aggregate, MetricReport

__all__ = [
    "BASELINE_FORMAT",
    "Baseline",
    "BaselinePolicy",
    "BaselineReport",
    "CaseRegression",
    "Delta",
    "MetricSnapshot",
    "Provenance",
    "compare",
    "promote",
]

BASELINE_FORMAT = 1
"""The artefact's format. A file without it is not a baseline this kit reads."""

BOOTSTRAP = "record one with `python -m tesserix_adk.cli.evals bootstrap`"
"""What the refusals tell a consumer to run, named once so they cannot disagree."""


class Provenance(AdkModel):
    """What produced a set of numbers, which is what makes them comparable to another set.

    Args:
        suite: The dataset's name.
        dataset_version: Its version. Numbers only compare within one, so this is the field
            a comparison refuses to cross.
        agent_version: The agent measured.
        prompt_version: The prompt it ran.
        model: The model that answered.
        cassettes: A digest of the recordings replayed, so a re-record is visible as a
            candidate explanation for a number that moved.
    """

    suite: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    agent_version: str = ""
    prompt_version: str = ""
    model: str = ""
    cassettes: str = ""

    def differences(self, other: Provenance) -> tuple[str, ...]:
        """Every field where these two runs differ, in the order a reviewer reads them.

        Example:
            >>> one = Provenance(suite="s", dataset_version="1", model="a")
            >>> one.differences(one.model_copy(update={"model": "b"}))
            ('model',)
        """
        fields = (
            "suite",
            "dataset_version",
            "agent_version",
            "prompt_version",
            "model",
            "cassettes",
        )
        return tuple(name for name in fields if getattr(self, name) != getattr(other, name))


class MetricSnapshot(AdkModel):
    """One metric's aggregate, frozen into the artefact.

    Args:
        metric: Its name.
        mean: The average over scored cases, `None` where nothing was scored.
        p50: The median, `None` where nothing was scored.
        p95: The tail, `None` where nothing was scored.
        n: How many cases produced a value.
        unknown: How many could not be measured, kept because a metric that stopped being
            measurable is itself a change worth seeing.
        half_width: Half the 95% interval where the sample supported one. This is the
            statistical half of the noise band.
        unit: What the number counts.
        currency: ISO 4217 code, empty where the metric is not money.
    """

    metric: str = Field(min_length=1)
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    n: int = Field(default=0, ge=0)
    unknown: int = Field(default=0, ge=0)
    half_width: float = Field(default=0.0, ge=0.0)
    unit: str = ""
    currency: str = ""

    @classmethod
    def of(cls, aggregate: Aggregate) -> MetricSnapshot:
        """Freeze one aggregate, keeping its interval as the band it implies."""
        band = 0.0
        if aggregate.ci_low is not None and aggregate.ci_high is not None:
            band = (aggregate.ci_high - aggregate.ci_low) / 2.0
        return cls(
            metric=aggregate.metric,
            mean=aggregate.mean,
            p50=aggregate.p50,
            p95=aggregate.p95,
            n=aggregate.n,
            unknown=aggregate.unknown,
            half_width=band,
            unit=aggregate.unit,
            currency=aggregate.currency,
        )


class Baseline(AdkModel):
    """A suite's numbers and where they came from, as the file CI reads.

    Args:
        format: The artefact format, refused when it is not this kit's.
        provenance: What produced the numbers.
        metrics: One snapshot per metric measured.
        values: Every per-case value, by metric and then by case id, so a comparison can
            name the cases that regressed rather than only the average that moved.
        quarantined: Case ids known to flake. They are reported and never block.
    """

    format: int = BASELINE_FORMAT
    provenance: Provenance
    metrics: tuple[MetricSnapshot, ...] = ()
    values: Mapping[str, Mapping[str, float]] = {}
    quarantined: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _one_snapshot_per_metric(self) -> Baseline:
        """Refuse two snapshots for one metric, which would compare by ordering."""
        names = [snapshot.metric for snapshot in self.metrics]
        if len(names) != len(set(names)):
            raise ConfigurationError("a baseline holds one snapshot per metric")
        return self

    @classmethod
    def of(
        cls,
        report: MetricReport,
        *,
        provenance: Provenance,
        quarantined: Sequence[str] = (),
    ) -> Baseline:
        """Freeze a measured report into an artefact.

        Args:
            report: What `measure` produced.
            provenance: What produced it, which the report itself does not know.
            quarantined: Case ids to record as flaky.

        Returns:
            The artefact, ready to write.

        Raises:
            ConfigurationError: The report measured a different dataset version than the
                provenance claims, which would file a number under the wrong label.
        """
        if report.version != provenance.dataset_version:
            raise ConfigurationError(
                f"the report measured dataset {report.version!r} and the provenance says "
                f"{provenance.dataset_version!r}"
            )
        return cls(
            provenance=provenance,
            metrics=tuple(MetricSnapshot.of(aggregate) for aggregate in report.aggregates.values()),
            values={metric: dict(cases) for metric, cases in report.values.items()},
            quarantined=tuple(sorted(quarantined)),
        )

    def snapshot(self, metric: str) -> MetricSnapshot | None:
        """The snapshot for `metric`, or `None` where nothing measured it."""
        return next((each for each in self.metrics if each.metric == metric), None)

    def write(self, path: Path) -> None:
        """Write the artefact, creating the directory it lives in."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Baseline:
        """Read an artefact, refusing anything that is not one.

        Args:
            path: Where the baseline lives.

        Returns:
            The stored baseline.

        Raises:
            BaselineUnusableError: The file is absent, is not JSON, is not a baseline, or
                carries a format this kit does not read. Each names how to record one.
        """
        if not path.exists():
            raise BaselineUnusableError(
                f"no baseline at {path}", path=str(path), reason="missing", remedy=BOOTSTRAP
            )
        text = path.read_text(encoding="utf-8")
        try:
            document = json.loads(text)
        except ValueError as broken:
            raise BaselineUnusableError(
                f"{path} is not JSON", path=str(path), reason="format", remedy=BOOTSTRAP
            ) from broken
        if not isinstance(document, dict) or document.get("format") != BASELINE_FORMAT:
            raise BaselineUnusableError(
                f"{path} is not a format {BASELINE_FORMAT} baseline",
                path=str(path),
                reason="format",
                remedy=BOOTSTRAP,
            )
        try:
            return cls.model_validate_json(text)
        except ValidationError as invalid:
            raise BaselineUnusableError(
                f"{path} is not a baseline this kit can read",
                path=str(path),
                reason="format",
                remedy=BOOTSTRAP,
            ) from invalid


class BaselinePolicy(AdkModel):
    """What this project blocks a merge on, declared once and read by CI.

    Args:
        tolerances: One per metric the gate judges, each with the movement it allows and
            the noise band beyond it. A metric nobody declared is reported and decides
            nothing.
        warn_only: Metrics that report but never block, for the exploratory ones.
        quarantined: Case ids known to flake. They are measured and reported, and they are
            kept out of the comparison, because a flaky case failing every pull request
            gets the whole gate switched off.
        case_tolerance: How far one case may move before it is named in the report.
    """

    tolerances: tuple[Tolerance, ...] = ()
    warn_only: tuple[str, ...] = ()
    quarantined: tuple[str, ...] = ()
    case_tolerance: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _one_per_metric(self) -> BaselinePolicy:
        """Refuse two tolerances for one metric, which would decide by ordering."""
        names = [tolerance.metric for tolerance in self.tolerances]
        if len(names) != len(set(names)):
            raise ConfigurationError("a metric may declare only one tolerance")
        return self

    def blocks(self, metric: str) -> bool:
        """Whether this metric may fail the gate, as opposed to only reporting."""
        return metric not in self.warn_only


class Delta(AdkModel):
    """What one metric did between the baseline and the candidate.

    Args:
        metric: Its name.
        before: The baseline's mean, `None` where it measured nothing.
        after: The candidate's mean, `None` where it measured nothing.
        regression: How much worse the candidate is, negative where it improved.
        tolerance: What the policy allowed.
        band: The noise band beyond the tolerance, the wider of the declared one and the
            baseline's own interval.
        verdict: `pass`, `warn` where the move is inside the band or the metric only warns,
            `fail` otherwise.
        reason: What a reviewer needs to know, empty where the metric passed cleanly.
    """

    metric: str
    before: float | None = None
    after: float | None = None
    regression: float = 0.0
    tolerance: float = 0.0
    band: float = 0.0
    verdict: Literal["pass", "warn", "fail"] = "pass"
    reason: str = ""

    def line(self) -> str:
        """The row a summary table prints.

        Example:
            >>> Delta(metric="exact_match", before=1.0, after=0.9, regression=0.1).line()
            '| exact_match | 1 | 0.9 | +0.1 | 0 ± 0 | pass |  |'
        """
        before = "-" if self.before is None else f"{self.before:.4g}"
        after = "-" if self.after is None else f"{self.after:.4g}"
        return (
            f"| {self.metric} | {before} | {after} | {self.regression:+.4g} | "
            f"{self.tolerance:g} ± {self.band:.4g} | {self.verdict} | {self.reason} |"
        )


class CaseRegression(AdkModel):
    """One case that got worse, which is what a reviewer actually opens.

    Args:
        case_id: The case.
        metric: The metric that moved on it.
        before: What it scored on the baseline.
        after: What it scores now, `None` where nothing measured it this time.
        quarantined: Whether the case is known to flake, and so reported without blocking.
    """

    case_id: str
    metric: str
    before: float | None = None
    after: float | None = None
    quarantined: bool = False

    def artefact(self, base: str, suite: str) -> str:
        """Where this case's run artefacts are, given the run's artefact base."""
        return f"{base.rstrip('/')}/{suite}/{self.case_id}"


class BaselineReport(AdkModel):
    """The verdict, what moved, which cases moved it, and any override taken.

    Args:
        suite: The suite compared.
        baseline: The provenance compared against.
        candidate: The provenance of the run under review.
        deltas: One per declared metric, in policy order.
        regressions: Every case that got worse on a declared metric.
        moved: The provenance fields that changed, which is the list of causal candidates.
        override: The recorded exception, where one was taken.
        verdict: `pass`, `warn` or `fail`.
    """

    suite: str
    baseline: Provenance
    candidate: Provenance
    deltas: tuple[Delta, ...] = ()
    regressions: tuple[CaseRegression, ...] = ()
    moved: tuple[str, ...] = ()
    override: Bypass | None = None
    verdict: Literal["pass", "warn", "fail"] = "pass"

    @property
    def ok(self) -> bool:
        """Whether the merge may proceed."""
        return self.verdict != "fail"

    @property
    def exit_code(self) -> int:
        """`0` where the merge may proceed, `1` where it may not — what CI reads."""
        return 0 if self.ok else 1

    def failing(self) -> tuple[CaseRegression, ...]:
        """The regressions that block, quarantined cases taken out."""
        return tuple(case for case in self.regressions if not case.quarantined)

    def summary(self) -> str:
        """One line for a log, verdict first."""
        blocking = len(self.failing())
        moved = ", ".join(self.moved) or "nothing"
        return f"{self.suite}: {self.verdict.upper()}, {blocking} case(s) worse, {moved} changed"

    def comment(self, *, artefacts: str = "") -> str:
        """The pull-request comment: the metric table, the failing cases, and any override.

        Args:
            artefacts: The run's artefact base, where there is one. Each failing case is
                linked under it, because the case id alone is not enough to inspect it.
        """
        lines = [
            f"### Eval gate: {self.verdict.upper()}",
            "",
            f"`{self.suite}` on dataset `{self.candidate.dataset_version}`. "
            f"Changed: {', '.join(self.moved) or 'nothing'}.",
            "",
            "| metric | before | after | change | allowed | verdict | note |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *[delta.line() for delta in self.deltas],
        ]
        if self.regressions:
            lines += ["", "<details><summary>Cases that got worse</summary>", ""]
            lines += [self._case(case, artefacts) for case in self.regressions]
            lines += ["", "</details>"]
        if self.override is not None:
            ticket = f" ({self.override.incident})" if self.override.incident else ""
            lines += [
                "",
                f"> **Override** by @{self.override.by} on "
                f"{', '.join(self.override.metrics)}: {self.override.reason}{ticket}",
            ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        """The machine-readable verdict a workflow reads."""
        return {
            "suite": self.suite,
            "verdict": self.verdict,
            "ok": self.ok,
            "moved": list(self.moved),
            "deltas": [delta.model_dump() for delta in self.deltas],
            "regressions": [case.model_dump() for case in self.regressions],
            "override": None if self.override is None else self.override.model_dump(),
        }

    def _case(self, case: CaseRegression, artefacts: str) -> str:
        """One bullet: what the case scored, whether it is quarantined, where to look."""
        where = case.artefact(artefacts, self.suite) if artefacts else ""
        link = f" — [artefacts]({where})" if where else ""
        flake = " _(quarantined)_" if case.quarantined else ""
        before = "-" if case.before is None else f"{case.before:.4g}"
        after = "-" if case.after is None else f"{case.after:.4g}"
        return f"- `{case.case_id}` {case.metric}: {before} → {after}{flake}{link}"


def compare(
    baseline: Baseline,
    candidate: Baseline,
    *,
    policy: BaselinePolicy,
    override: Bypass | None = None,
) -> BaselineReport:
    """Compare a candidate run against the stored baseline and say whether it may merge.

    Args:
        baseline: The stored artefact.
        candidate: The run under review, measured on the same dataset.
        policy: What this project blocks on.
        override: A recorded exception. It excuses the metrics it names and appears in the
            comment, so an override is argued about in the pull request rather than hidden
            in config.

    Returns:
        The verdict, every declared metric's movement, and the cases behind it.

    Raises:
        BaselineUnusableError: The two runs are not comparable — a different suite, or a
            dataset edited in the same change, which makes every number a mixture of two
            causes.
    """
    _comparable(baseline, candidate)
    excused = set(override.metrics) if override else set()
    quarantined = set(policy.quarantined) | set(baseline.quarantined)
    deltas = tuple(
        _delta(tolerance, baseline, candidate, policy=policy, excused=excused, skip=quarantined)
        for tolerance in policy.tolerances
    )
    regressions = tuple(
        regression.model_copy(update={"quarantined": regression.case_id in quarantined})
        for tolerance in policy.tolerances
        for regression in _cases(tolerance, baseline, candidate, policy=policy)
    )
    verdicts = [delta.verdict for delta in deltas]
    verdict: Literal["pass", "warn", "fail"] = (
        "fail" if "fail" in verdicts else "warn" if "warn" in verdicts else "pass"
    )
    return BaselineReport(
        suite=candidate.provenance.suite,
        baseline=baseline.provenance,
        candidate=candidate.provenance,
        deltas=deltas,
        regressions=regressions,
        moved=baseline.provenance.differences(candidate.provenance),
        override=override,
        verdict=verdict,
    )


def promote(candidate: Baseline, path: Path) -> Path | None:
    """Store `candidate` as the new baseline, keeping the one it replaces.

    Args:
        candidate: The measured run being promoted, normally on merge to the default branch.
        path: Where the baseline lives.

    Returns:
        Where the previous baseline was kept, or `None` where there was not one.
    """
    previous: Path | None = None
    if path.exists():
        previous = path.with_suffix(".previous.json")
        previous.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    candidate.write(path)
    return previous


def _comparable(baseline: Baseline, candidate: Baseline) -> None:
    """Refuse the comparisons that cannot mean anything, naming the way out of each."""
    if baseline.provenance.suite != candidate.provenance.suite:
        raise BaselineUnusableError(
            f"the baseline measured {baseline.provenance.suite!r} and this run measured "
            f"{candidate.provenance.suite!r}",
            reason="suite",
            remedy=BOOTSTRAP,
        )
    if baseline.provenance.dataset_version != candidate.provenance.dataset_version:
        raise BaselineUnusableError(
            f"the baseline is on dataset {baseline.provenance.dataset_version!r} and this run "
            f"is on {candidate.provenance.dataset_version!r}: a dataset edit and an agent edit "
            f"in one change cannot be told apart",
            reason="dataset",
            remedy=BOOTSTRAP,
        )


def _mean(source: Baseline, metric: str, skip: set[str]) -> float | None:
    """The metric's mean, recomputed without the quarantined cases where they are known."""
    snapshot = source.snapshot(metric)
    values = source.values.get(metric, {})
    kept = [value for case_id, value in values.items() if case_id not in skip]
    if skip and values and kept:
        return sum(kept) / len(kept)
    return snapshot.mean if snapshot else None


def _delta(
    tolerance: Tolerance,
    baseline: Baseline,
    candidate: Baseline,
    *,
    policy: BaselinePolicy,
    excused: set[str],
    skip: set[str],
) -> Delta:
    """What one declared metric did, failing closed where nobody measured it."""
    before = _mean(baseline, tolerance.metric, skip)
    after = _mean(candidate, tolerance.metric, skip)
    blocks = policy.blocks(tolerance.metric) and tolerance.metric not in excused
    if before is None or after is None:
        missing = "the baseline" if before is None else "this run"
        return Delta(
            metric=tolerance.metric,
            before=before,
            after=after,
            tolerance=tolerance.tolerance,
            verdict="fail" if blocks else "warn",
            reason=f"{missing} has no value for it, so the gate cannot clear it",
        )
    snapshot = baseline.snapshot(tolerance.metric)
    band = max(tolerance.noise, snapshot.half_width if snapshot else 0.0)
    regression = tolerance.regression(before, after)
    verdict: Literal["pass", "warn", "fail"] = "pass"
    reason = ""
    if regression > tolerance.tolerance + band:
        verdict = "fail" if blocks else "warn"
        reason = f"worse by {regression:.4g}, beyond the {tolerance.tolerance:g} allowed"
    elif regression > tolerance.tolerance:
        verdict = "warn"
        reason = f"worse by {regression:.4g}, inside the noise band"
    if verdict != "pass" and not baseline.provenance.differences(candidate.provenance):
        reason += "; nothing about the run changed, so look at the price list or the provider"
    return Delta(
        metric=tolerance.metric,
        before=before,
        after=after,
        regression=regression,
        tolerance=tolerance.tolerance,
        band=band,
        verdict=verdict,
        reason=reason,
    )


def _cases(
    tolerance: Tolerance,
    baseline: Baseline,
    candidate: Baseline,
    *,
    policy: BaselinePolicy,
) -> tuple[CaseRegression, ...]:
    """Every case this metric got worse on, in dataset order."""
    after = candidate.values.get(tolerance.metric, {})
    worse: list[CaseRegression] = []
    for case_id, was in baseline.values.get(tolerance.metric, {}).items():
        now = after.get(case_id)
        if now is None:
            worse.append(CaseRegression(case_id=case_id, metric=tolerance.metric, before=was))
        elif tolerance.regression(was, now) > policy.case_tolerance:
            worse.append(
                CaseRegression(case_id=case_id, metric=tolerance.metric, before=was, after=now)
            )
    return tuple(worse)
