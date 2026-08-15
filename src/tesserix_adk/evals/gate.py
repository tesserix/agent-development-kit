"""The gate a prompt change has to pass before an alias moves to it.

Prompt edits ship on intuition: a reviewer reads the new wording, it reads well, and the
quality regression is found by users. Cost regressions are worse, because a longer prompt
raises the spend on every future run and nothing flags it — the graph moves a fortnight
later and nobody connects it to a wording change.

So a candidate version is measured against the version in production on the same dataset,
and the two sets of numbers are compared here. This module is the comparison and the
verdict: the dataset, the judge and the metric definitions belong to the eval framework, and
running the suite belongs to CI.

It fails closed. A dataset only partly scored, a judge that moved since the baseline, or a
declared metric nobody computed all refuse rather than pass, because a gate that guesses at
the missing half is a gate that passes the run it exists to stop.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from typing import Literal

from pydantic import model_validator

from tesserix_adk.core.errors import (
    ConfigurationError,
    EvalIncompleteError,
    IncomparableEvalError,
)
from tesserix_adk.core.models import AdkModel

__all__ = [
    "DEFAULT_POLICY",
    "Bypass",
    "GatePolicy",
    "GateReport",
    "Measured",
    "MetricMove",
    "Tolerance",
    "Verdict",
    "gate",
]

Verdict = Literal["pass", "fail", "repeat"]
Direction = Literal["higher_is_better", "lower_is_better"]


class Tolerance(AdkModel):
    """How far one metric may move before the change is a regression.

    Args:
        metric: The metric's name, as the suite reports it.
        direction: Which way is better.
        tolerance: How much worse the candidate may be, in the metric's own units. Zero
            means no regression at all.
        noise: A band beyond the tolerance where the verdict is `repeat` rather than
            `fail`, so an example that flakes across the line costs a rerun instead of a
            coin flip.
    """

    metric: str
    direction: Direction = "higher_is_better"
    tolerance: float = 0.0
    noise: float = 0.0

    def regression(self, before: float, after: float) -> float:
        """How much worse `after` is than `before`, negative where it improved."""
        return before - after if self.direction == "higher_is_better" else after - before


class GatePolicy(AdkModel):
    """What this project will accept, declared once and read by CI.

    Args:
        tolerances: One per metric the gate judges. A metric nobody declared is reported
            and does not decide anything.
        repeats: How many runs CI should average before trusting a `repeat` verdict.
    """

    tolerances: tuple[Tolerance, ...] = ()
    repeats: int = 3

    @model_validator(mode="after")
    def _one_per_metric(self) -> GatePolicy:
        """Refuse two tolerances for the same metric, which would decide by ordering."""
        names = [tolerance.metric for tolerance in self.tolerances]
        if len(names) != len(set(names)):
            raise ConfigurationError("a metric may declare only one tolerance")
        return self


DEFAULT_POLICY = GatePolicy(
    tolerances=(
        Tolerance(metric="task_success", tolerance=0.01),
        Tolerance(metric="schema_validity", tolerance=0.0),
        Tolerance(metric="judge_score", tolerance=0.05, noise=0.02),
        Tolerance(metric="p95_latency_ms", direction="lower_is_better", tolerance=250.0),
        Tolerance(metric="cost_per_run", direction="lower_is_better", tolerance=0.0005),
    )
)
"""The five the issue that created this module names, cost among them rather than beside them."""


class Measured(AdkModel):
    """One prompt version's numbers on one dataset.

    Args:
        prompt: The prompt's name.
        version: The concrete version measured, never an alias.
        digest: The version's content digest, so a result cannot be read as belonging to
            text it was not measured on.
        examples: How many examples the dataset holds.
        scored: How many were actually scored.
        metrics: The numbers, by metric name.
        variables: What the version declares, so a schema change invalidates the dataset
            rather than quietly comparing two different prompts.
        judge: What scored it, as an identifier that changes when the judge does.
    """

    prompt: str
    version: str
    digest: str = ""
    examples: int = 0
    scored: int = 0
    metrics: Mapping[str, float] = {}
    variables: tuple[str, ...] = ()
    judge: str = ""

    @property
    def coverage(self) -> float:
        """The share of the dataset that was scored. `1.0` or the gate refuses."""
        return self.scored / self.examples if self.examples else 0.0


class Bypass(AdkModel):
    """An exception taken deliberately, during an incident, with a name on it.

    Args:
        metrics: Which metrics it excuses. A bypass excusing everything says so by naming
            everything, rather than by being a flag.
        by: Who took it.
        reason: Why. Refused when empty, because the record is the point.
        incident: The ticket, where there is one.
    """

    metrics: tuple[str, ...]
    by: str
    reason: str = ""
    incident: str = ""

    @model_validator(mode="after")
    def _needs_a_reason(self) -> Bypass:
        """Refuse a bypass nobody justified."""
        if not self.by or not self.reason:
            raise ConfigurationError("a bypass records who took it and why")
        return self


class MetricMove(AdkModel):
    """What one metric did between the baseline and the candidate.

    Args:
        metric: Its name.
        before: The baseline's value.
        after: The candidate's value.
        regression: How much worse the candidate is, negative where it improved.
        tolerance: What was allowed.
        verdict: What this metric alone says.
        bypassed: Whether a recorded exception excuses it.
    """

    metric: str
    before: float = 0.0
    after: float = 0.0
    regression: float = 0.0
    tolerance: float = 0.0
    verdict: Verdict = "pass"
    bypassed: bool = False

    def __str__(self) -> str:
        """The line a CI summary prints."""
        excused = " (bypassed)" if self.bypassed else ""
        return (
            f"{self.metric}: {self.before:g} -> {self.after:g} "
            f"({self.verdict}, tolerance {self.tolerance:g}){excused}"
        )


class GateReport(AdkModel):
    """The verdict, what moved, and the record a rollback can read later.

    Args:
        prompt: The prompt's name.
        version: The candidate version.
        digest: The candidate's digest. A promotion is checked against this and not
            against a version label, so text edited under a label cannot inherit a pass.
        baseline: The version compared against.
        moves: One per declared metric, in policy order.
        verdict: `pass`, `fail`, or `repeat` where something landed in the noise band.
        bypassed: The metrics a recorded exception excused.
    """

    prompt: str
    version: str
    digest: str = ""
    baseline: str = ""
    moves: tuple[MetricMove, ...] = ()
    verdict: Verdict = "pass"
    bypassed: tuple[str, ...] = ()

    def permits(self, digest: str) -> bool:
        """Whether an alias may be promoted to the text with this digest.

        A promotion needs a passing result for the exact digest measured. A label is not
        enough: the same version number over edited text is a different prompt.
        """
        return self.verdict == "pass" and bool(digest) and digest == self.digest

    def summary(self) -> str:
        """What CI prints: the verdict first, then every metric that moved."""
        headline = f"{self.prompt}@{self.version} vs {self.baseline}: {self.verdict.upper()}"
        excused = f"\nbypassed: {', '.join(self.bypassed)}" if self.bypassed else ""
        return headline + "".join(f"\n  {move}" for move in self.moves) + excused

    def attributes(self) -> dict[str, str]:
        """The record, as attributes an alias store or a span can keep."""
        return {
            "adk.prompt": f"{self.prompt}@{self.version}",
            "adk.prompt_digest": self.digest,
            "adk.eval_verdict": self.verdict,
            "adk.eval_baseline": self.baseline,
        }


def gate(
    baseline: Measured,
    candidate: Measured,
    *,
    policy: GatePolicy = DEFAULT_POLICY,
    bypass: Bypass | None = None,
) -> GateReport:
    """Compare two measured versions and say whether the candidate may ship.

    Args:
        baseline: The version in production, measured on the dataset.
        candidate: The version being proposed, measured on the same dataset with the same
            judge and the same replayed inputs, so the prompt is the only difference.
        policy: What this project accepts.
        bypass: A recorded exception, where one was taken.

    Returns:
        The verdict, the movement in every declared metric, and the record a rollback
        command can read.

    Raises:
        EvalIncompleteError: Where either side scored less than the whole dataset. The
            gate never infers a score for an example nobody scored.
        IncomparableEvalError: Where the judge moved between the two runs, or the
            candidate changed the variables the dataset supplies. Neither comparison
            means anything, and a meaningless comparison must not read as a pass.

    Example:
        >>> before = Measured(prompt="p", version="1", examples=2, scored=2,
        ...                   metrics={"task_success": 0.9})
        >>> after = before.model_copy(update={"version": "2", "metrics": {"task_success": 0.6}})
        >>> policy = GatePolicy(tolerances=(Tolerance(metric="task_success"),))
        >>> gate(before, after, policy=policy).verdict
        'fail'
    """
    _comparable(baseline, candidate)
    excused = set(bypass.metrics) if bypass else set()
    moves = tuple(
        _move(tolerance, baseline, candidate, excused=excused) for tolerance in policy.tolerances
    )
    judged = [move.verdict for move in moves if not move.bypassed]
    verdict: Verdict = "fail" if "fail" in judged else "repeat" if "repeat" in judged else "pass"
    return GateReport(
        prompt=candidate.prompt,
        version=candidate.version,
        digest=candidate.digest,
        baseline=baseline.version,
        moves=moves,
        verdict=verdict,
        bypassed=tuple(move.metric for move in moves if move.bypassed),
    )


def _comparable(baseline: Measured, candidate: Measured) -> None:
    """Refuse the comparisons that cannot mean anything."""
    for side in (baseline, candidate):
        if side.coverage < 1.0:
            raise EvalIncompleteError(
                f"{side.prompt}@{side.version} scored {side.scored} of {side.examples} "
                f"examples; the gate does not infer the rest",
                prompt=side.prompt,
                version=side.version,
                coverage=side.coverage,
            )
    if baseline.judge != candidate.judge:
        raise IncomparableEvalError(
            f"the judge moved from {baseline.judge!r} to {candidate.judge!r}; "
            f"re-measure the baseline before comparing",
            reason="judge",
        )
    if baseline.variables != candidate.variables:
        raise IncomparableEvalError(
            f"{candidate.prompt}@{candidate.version} declares "
            f"{', '.join(candidate.variables) or 'nothing'} where the baseline declares "
            f"{', '.join(baseline.variables) or 'nothing'}; update the dataset with the change",
            reason="variables",
        )


def _move(
    tolerance: Tolerance, baseline: Measured, candidate: Measured, *, excused: set[str]
) -> MetricMove:
    """What one declared metric did, failing closed where nobody computed it."""
    bypassed = tolerance.metric in excused
    if tolerance.metric not in baseline.metrics or tolerance.metric not in candidate.metrics:
        return MetricMove(
            metric=tolerance.metric,
            tolerance=tolerance.tolerance,
            verdict="fail",
            bypassed=bypassed,
        )
    before = baseline.metrics[tolerance.metric]
    after = candidate.metrics[tolerance.metric]
    regression = tolerance.regression(before, after)
    verdict: Verdict = (
        "pass"
        if regression <= tolerance.tolerance
        else "repeat"
        if regression <= tolerance.tolerance + tolerance.noise
        else "fail"
    )
    return MetricMove(
        metric=tolerance.metric,
        before=before,
        after=after,
        regression=regression,
        tolerance=tolerance.tolerance,
        verdict=verdict,
        bypassed=bypassed,
    )
