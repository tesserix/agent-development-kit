"""Measuring a change on answers, spend and latency at once — and failing it on cost.

Run it with `uv run python examples/eval_metrics.py`.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from tesserix_adk.core import Message, NoOutput, Run, RunState, TextPart, Usage
from tesserix_adk.core.cost import Cost
from tesserix_adk.core.tenancy import current_tenant
from tesserix_adk.evals import (
    CostPerCase,
    EvalCase,
    EvalSuite,
    ExactMatch,
    LatencyMs,
    MetricValue,
    SuiteRunner,
    Threshold,
    TokensIn,
    measure,
)

SUITE = EvalSuite(
    name="refunds",
    version="2026-08-01",
    cases=(
        EvalCase(id="late", input="my order never arrived", tenant="acme", expected="refunded"),
        EvalCase(id="damaged", input="it arrived broken", tenant="acme", expected="refunded"),
        EvalCase(id="unpriced", input="where is my parcel", tenant="beta", expected="tomorrow"),
    ),
)

ANSWERS = {"late": "refunded", "damaged": "refunded", "unpriced": "tomorrow"}


class AnswerLength:
    """A consumer-written metric: three names and a value, or a reason there is none."""

    name = "answer_length"
    higher_is_better = False

    def compute(self, case: EvalCase, run: Run[Any]) -> MetricValue:  # noqa: ARG002 — the protocol's shape
        """Count the characters the agent sent back."""
        text = "".join(getattr(part, "text", "") for part in run.messages[-1].content)
        return MetricValue(value=float(len(text)), unit="chars")


async def replay(case: EvalCase, *, run_id: str) -> Run[NoOutput]:
    """Answer from a recording. The self-hosted case is deliberately unpriced."""
    priced = None if case.id == "unpriced" else Cost(output=Decimal("0.05"), currency="USD")
    return Run[NoOutput](
        id=run_id,
        tenant=current_tenant().tenant,
        agent_name="support",
        agent_version="1.0.0",
        model="recorded",
        state=RunState.COMPLETED,
        messages=[Message(role="assistant", content=[TextPart(text=ANSWERS[case.id])])],
        usage=Usage(input_tokens=800, output_tokens=40, cost=priced),
        started_at=0.0,
        ended_at=0.9,
    )


async def main() -> None:
    """Run the suite, measure it on five metrics, and let cost decide the verdict."""
    outcome = await SuiteRunner(replay).run(SUITE)
    report = measure(
        SUITE,
        outcome,
        (ExactMatch(), CostPerCase(), LatencyMs(), TokensIn(), AnswerLength()),
        thresholds=(
            Threshold(metric="exact_match", minimum=0.9),
            Threshold(metric="cost_per_case", maximum=0.02, warn_within=0.005),
            Threshold(metric="latency_ms", maximum=2500.0),
        ),
    )

    print(report.table())  # noqa: T201
    print(report.summary())  # noqa: T201

    correct = report.aggregate("exact_match")
    cost = report.aggregate("cost_per_case")
    print(f"every answer was right ({correct.mean}), and it still failed on cost")  # noqa: T201
    print(f"one case had no price, so it is unknown rather than free: {cost.unknown}")  # noqa: T201
    print(f"the interval is withheld on {cost.n} cases: {cost.note}")  # noqa: T201
    print(f"exit code {report.exit_code}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
