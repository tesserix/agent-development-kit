"""Measuring three shipped guards against the corpus, and a bar one of them fails.

Run it with `uv run python examples/guard_testing.py`.
"""

from __future__ import annotations

import asyncio
import json

from tesserix_adk.core.content_policy import ContentSeverity, Thresholds
from tesserix_adk.guardrails import ContentFilterGuard, InjectionGuard, PIIGuard
from tesserix_adk.testing import (
    CORPUS_VERSION,
    GUARD_CORPUS,
    GuardFamily,
    GuardThresholds,
    assert_synthetic,
    measure,
    sampled,
)


async def main() -> None:
    """Report recall, false positives and latency, then show the gate refusing a drop."""
    assert_synthetic()
    print(f"corpus {CORPUS_VERSION}: {len(GUARD_CORPUS)} cases\n")  # noqa: T201

    measured = (
        (PIIGuard(tenant="acme"), {GuardFamily.PII}, GuardThresholds(recall=1.0)),
        (
            InjectionGuard(),
            {GuardFamily.INJECTION},
            GuardThresholds(recall=0.71, false_positives=0.3),
        ),
        (
            ContentFilterGuard(thresholds=Thresholds(default=ContentSeverity.MEDIUM)),
            {GuardFamily.POLICY},
            GuardThresholds(recall=0.75),
        ),
    )
    for guard, families, bar in measured:
        metrics = await measure(guard, families=families)
        print(json.dumps(metrics.as_dict()))  # noqa: T201
        print(f"  meets its bar: {not metrics.failures(bar)}\n")  # noqa: T201

    metrics = await measure(InjectionGuard(), families={GuardFamily.INJECTION})
    print("held to a bar it does not meet:")  # noqa: T201
    for reason in metrics.failures(GuardThresholds(recall=1.0)):
        print(f"  {reason}")  # noqa: T201

    subset = sampled(GUARD_CORPUS, 8, seed="9f2c1ab")
    same = sampled(GUARD_CORPUS, 8, seed="9f2c1ab")
    print(f"\nsampling is deterministic per commit: {subset == same}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
