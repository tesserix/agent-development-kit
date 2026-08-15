"""Gating a prompt change on what the eval suite measured.

Runs one candidate through the default policy four times: clean, cheaper but worse, better
but dearer, and dearer with the spend taken deliberately.

Run it with `python examples/prompt_gate.py`.
"""

from __future__ import annotations

from tesserix_adk.core import EvalIncompleteError
from tesserix_adk.evals import Bypass, Measured, gate

BASELINE = Measured(
    prompt="itinerary_system",
    version="4",
    digest="4d1f",
    examples=200,
    scored=200,
    metrics={
        "task_success": 0.91,
        "schema_validity": 1.0,
        "judge_score": 4.2,
        "p95_latency_ms": 1800.0,
        "cost_per_run": 0.012,
    },
    variables=("budget", "city"),
    judge="judge-v3",
)


def measured(**moved: float) -> Measured:
    """Version 5, measured on the same dataset with the named metrics moved."""
    return BASELINE.model_copy(
        update={"version": "5", "digest": "9c02", "metrics": {**BASELINE.metrics, **moved}}
    )


def main() -> None:
    """Four candidates, four verdicts."""
    print(gate(BASELINE, measured(task_success=0.93)).summary())  # noqa: T201

    print("\n" + gate(BASELINE, measured(task_success=0.78, cost_per_run=0.008)).summary())  # noqa: T201

    dearer = measured(task_success=0.96, cost_per_run=0.031)
    print("\n" + gate(BASELINE, dearer).summary())  # noqa: T201

    excused = gate(
        BASELINE,
        dearer,
        bypass=Bypass(
            metrics=("cost_per_run",),
            by="ada",
            reason="PLAT-102, worth the spend until the shorter rewrite lands",
            incident="PLAT-102",
        ),
    )
    print("\n" + excused.summary())  # noqa: T201
    print(f"promotable: {excused.permits('9c02')}")  # noqa: T201

    try:
        gate(BASELINE, measured().model_copy(update={"scored": 150}))
    except EvalIncompleteError as refused:
        print(f"\nrefused: {refused} ({refused.coverage:.0%} scored)")  # noqa: T201


if __name__ == "__main__":
    main()
