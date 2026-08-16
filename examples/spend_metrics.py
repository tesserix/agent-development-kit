"""Three tenants, one unpriced model and a collector that is down.

Run it with `uv run python examples/spend_metrics.py`.
"""

from __future__ import annotations

from decimal import Decimal

from tesserix_adk.core import Cost, CountSource, Usage
from tesserix_adk.observability import (
    COST,
    TOKENS,
    UNKNOWN_USAGE,
    Attribution,
    Dimensions,
    ModelRate,
    Outcome,
    PricingTable,
    SpendMeter,
    SpendRecord,
    Step,
)


class Collector:
    """A metric store that keeps what it was told."""

    def __init__(self) -> None:
        self.counted: list[tuple[str, float, dict[str, str]]] = []

    def count(self, name: str, value: float, **dimensions: str) -> None:
        """Record one counter."""
        self.counted.append((name, value, dimensions))

    def total(self, name: str, **matching: str) -> float:
        """What one series adds up to, optionally within one dimension."""
        return sum(
            value
            for series, value, dimensions in self.counted
            if series == name and all(dimensions.get(k) == v for k, v in matching.items())
        )


def _record(tenant: str, model: str, usage: Usage) -> SpendRecord:
    """One metered model call."""
    return SpendRecord(
        attribution=Attribution(
            tenant=tenant,
            user="ada",
            agent="refunds",
            agent_version="3",
            definition="refunds@3",
            model=model,
            prompt_version="7",
            task_class="support",
            run_id=f"run-{tenant}",
        ),
        step=Step.MODEL,
        outcome=Outcome.ANSWERED,
        usage=usage,
    )


def main() -> None:
    """What the store ends up holding, and what it deliberately does not."""
    table = PricingTable(
        version="2026-08-01",
        rates={
            "gpt-5": ModelRate(input_per_million=Decimal("1.25"), output_per_million=Decimal("10")),
            "local-8b": ModelRate(
                input_per_million=Decimal("0.02"),
                output_per_million=Decimal("0.02"),
                source="self-hosted",
            ),
        },
    )
    collector = Collector()
    meter = SpendMeter(collector, pricing=table, dimensions=Dimensions(tenants=frozenset({"acme"})))

    billed = Usage(
        input_tokens=1000,
        output_tokens=200,
        extras={},
        cost=Cost(input=Decimal("0.00125"), output=Decimal("0.002")),
    )
    meter.record(_record("acme", "gpt-5", billed), seconds=0.8, key="run-acme/model/1")
    meter.record(_record("acme", "gpt-5", billed), seconds=0.8, key="run-acme/model/1")
    meter.record(_record("globex", "gpt-5", billed), seconds=1.1, key="run-globex/model/1")

    self_hosted = Usage(input_tokens=1_000_000, output_tokens=0, extras={})
    meter.record(_record("acme", "local-8b", self_hosted), key="run-acme/local/1")

    silent = Usage(input_tokens=0, output_tokens=0, extras={}, source=CountSource.HEURISTIC)
    meter.record(_record("acme", "gpt-5", silent), key="run-acme/model/2")

    print(f"tokens for acme: {collector.total(TOKENS, tenant='acme')}")  # noqa: T201
    print(f"tokens for everyone else: {collector.total(TOKENS, tenant='other')}")  # noqa: T201
    print(f"cost recorded: {collector.total(COST):.5f}")  # noqa: T201
    print(f"calls nobody reported usage for: {collector.total(UNKNOWN_USAGE)}")  # noqa: T201
    print(f"replayed steps not counted twice: {meter.stats.duplicates}")  # noqa: T201

    computed = [d for _, _, d in collector.counted if d.get("cost_confidence") == "estimated"]
    print(f"cost figures the kit computed rather than read: {len(computed)}")  # noqa: T201

    class Down:
        """A store whose collector is unreachable."""

        def count(self, name: str, value: float, **dimensions: str) -> None:  # noqa: ARG002
            """Refuse every counter."""
            message = f"collector unreachable for {name}"
            raise RuntimeError(message)

    outage = SpendMeter(Down(), pricing=table)
    outage.record(_record("acme", "gpt-5", billed))
    print(f"\ncollector down, run unaffected, dropped={outage.stats.dropped}")  # noqa: T201


if __name__ == "__main__":
    main()
