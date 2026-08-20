"""Blocking a merge on a regression, and naming the cases a reviewer has to open.

Run it with `uv run python examples/eval_baseline.py`.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tesserix_adk.core.errors import BaselineUnusableError
from tesserix_adk.evals import (
    Baseline,
    BaselinePolicy,
    Bypass,
    MetricSnapshot,
    Provenance,
    Tolerance,
    compare,
    promote,
)

SHIPPED = Provenance(
    suite="refunds",
    dataset_version="2026-08-01",
    agent_version="1.0.0",
    prompt_version="7",
    model="claude-recorded",
    cassettes="a1b2c3",
)
CANDIDATE = SHIPPED.model_copy(update={"agent_version": "1.1.0", "prompt_version": "8"})

POLICY = BaselinePolicy(
    tolerances=(Tolerance(metric="exact_match", tolerance=0.02, noise=0.01),),
    quarantined=("case-49",),
)


def measured(correct: int, provenance: Provenance) -> Baseline:
    """Fifty cases, the first `correct` of them answered right."""
    values = {f"case-{index:02d}": float(index < correct) for index in range(50)}
    mean = sum(values.values()) / len(values)
    return Baseline(
        provenance=provenance,
        metrics=(MetricSnapshot(metric="exact_match", mean=mean, n=50, half_width=0.005),),
        values={"exact_match": values},
    )


def main() -> None:
    """Compare a prompt change against what is in production, then promote a good one."""
    baseline = measured(50, SHIPPED)
    candidate = measured(44, CANDIDATE)

    report = compare(baseline, candidate, policy=POLICY)
    print(report.summary())  # noqa: T201
    print(report.comment(artefacts="https://ci.example/run/12/"))  # noqa: T201
    print(f"blocking cases: {[case.case_id for case in report.failing()]}")  # noqa: T201

    excused = compare(
        baseline,
        candidate,
        policy=POLICY,
        override=Bypass(metrics=("exact_match",), by="sam", reason="provider incident INC-12"),
    )
    print(f"with a recorded override: {excused.verdict}, exit code {excused.exit_code}")  # noqa: T201

    edited = candidate.model_copy(
        update={"provenance": CANDIDATE.model_copy(update={"dataset_version": "2026-08-02"})}
    )
    try:
        compare(baseline, edited, policy=POLICY)
    except BaselineUnusableError as refused:
        print(f"a dataset edited in the same change is {refused.reason}: {refused}")  # noqa: T201

    with TemporaryDirectory() as folder:
        path = Path(folder) / "baseline.json"
        promote(baseline, path)
        kept = promote(measured(50, CANDIDATE), path)
        print(f"promoted, previous kept at {Path(kept).name if kept else 'nothing'}")  # noqa: T201

    print(f"exit code {report.exit_code}")  # noqa: T201


if __name__ == "__main__":
    main()
