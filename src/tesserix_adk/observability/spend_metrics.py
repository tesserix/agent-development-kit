"""Spend and performance as alertable series, with the gaps left visible as gaps.

Discovering a runaway retry loop from the provider's monthly invoice is discovering it a
month late. These counters come off the same records the spans are built from, so metrics
and traces cannot disagree about a run, and they carry the pricing version that produced
each figure so a mid-window price change does not retroactively rewrite what was recorded.

Nothing is estimated and presented as measured. A call whose usage the provider never
reported lands in an explicit unknown series rather than as a zero, because a zero is
summed by whoever reads it and understates the bill.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import Field

from tesserix_adk.core import AdkModel, Cost, CostConfidence, CountSource
from tesserix_adk.observability.metrics import COST, TOKENS, Dimensions

if TYPE_CHECKING:
    from tesserix_adk.observability.attribution import SpendRecord
    from tesserix_adk.observability.metrics import Meter

__all__ = [
    "BUDGET_BREACHES",
    "DROPPED",
    "ITERATIONS",
    "LATENCY",
    "TOOL_CALLS",
    "UNKNOWN_USAGE",
    "UNPRICED",
    "MeterStats",
    "ModelRate",
    "PricingTable",
    "SpendMeter",
]

LATENCY = "adk.latency_seconds"
ITERATIONS = "adk.iterations"
TOOL_CALLS = "adk.tool_calls"
BUDGET_BREACHES = "adk.budget_breaches"
UNKNOWN_USAGE = "adk.unknown_usage"
"""Calls the provider reported no usage for. A gap somebody can see, not a zero."""

UNPRICED = "adk.unpriced"
"""Calls nobody can price. The tokens are still counted; only the money is missing."""

DROPPED = "adk.metrics_dropped"
"""Counted locally, because a metric lost on the way out cannot report its own loss."""

_MILLION = Decimal(1_000_000)


class ModelRate(AdkModel):
    """What a model costs, per million tokens.

    Args:
        input_per_million: Price of a million prompt tokens.
        output_per_million: Price of a million generated tokens.
        cached_per_million: Price of a million tokens served from the provider's cache.
        currency: ISO 4217 code the rate is quoted in.
        source: Where the rate came from — a published list, a contract, or the figure a
            deployment put on its own hardware. A self-hosted model priced at nothing
            reads as free capacity, which is how a GPU bill surprises somebody.
    """

    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    cached_per_million: Decimal | None = None
    currency: str = "USD"
    source: str = "published"

    def of(self, *, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> Cost:
        """What this rate makes of a call's tokens.

        Returns:
            A cost marked estimated: it is computed here rather than billed by anyone, and
            a computed figure presented as measured is the one nobody re-checks.
        """
        cached_rate = self.cached_per_million
        fresh = input_tokens - cached_tokens if cached_rate is not None else input_tokens
        return Cost(
            input=self.input_per_million * Decimal(fresh) / _MILLION,
            output=self.output_per_million * Decimal(output_tokens) / _MILLION,
            cache_read=(cached_rate or Decimal(0)) * Decimal(cached_tokens) / _MILLION,
            currency=self.currency,
            confidence=CostConfidence.ESTIMATED,
        )


class PricingTable(AdkModel):
    """Model prices as of a stated version.

    Args:
        version: Which revision of the prices these are. Recorded on every cost series, so
            a price change starts a new series rather than rewriting an old one.
        rates: The rate for each model.
    """

    version: str = Field(min_length=1)
    rates: Mapping[str, ModelRate] = {}

    def rate_of(self, model: str) -> ModelRate | None:
        """The rate for `model`, or None where the table does not price it."""
        return self.rates.get(model)


@dataclass(slots=True)
class MeterStats:
    """What the meter did, for the counters that should sit near zero.

    Args:
        recorded: Records counted.
        duplicates: Records seen again under a key already counted.
        unknown: Calls whose usage the provider never reported.
        unpriced: Calls nobody could price.
        dropped: Counter emissions the store refused.
    """

    recorded: int = 0
    duplicates: int = 0
    unknown: int = 0
    unpriced: int = 0
    dropped: int = 0
    seen: set[str] = field(default_factory=set)


class SpendMeter:
    """Emits the spend and performance series for the records a run produced.

    Args:
        meter: Where the counters go.
        pricing: The table used where a record arrived without a cost of its own.
        dimensions: Which values this deployment wants its series split by.

    Example:
        >>> table = PricingTable(version="2026-08-01")
        >>> SpendMeter(meter, pricing=table).stats.recorded   # doctest: +SKIP
        0
    """

    def __init__(
        self,
        meter: Meter,
        *,
        pricing: PricingTable,
        dimensions: Dimensions | None = None,
    ) -> None:
        self._meter = meter
        self._pricing = pricing
        self._dimensions = dimensions or Dimensions()
        self.stats = MeterStats()

    def record(
        self,
        record: SpendRecord,
        *,
        seconds: float | None = None,
        iterations: int = 0,
        tool_calls: int = 0,
        breached: bool = False,
        key: str | None = None,
    ) -> None:
        """Count one metered step.

        Args:
            record: The step, as the spans were built from it.
            seconds: How long it took, where it was timed.
            iterations: Loop iterations this step covered.
            tool_calls: Tool calls it fanned out to.
            breached: Whether a budget ceiling was crossed. Enforcement is elsewhere; this
                only reports what enforcement decided.
            key: Identifies the step, so a replay after a process restart is counted once.
                Absent, every call counts — a caller that cannot identify a step is better
                served by a double count than by a silent drop.

        Returns:
            Nothing. A store that refuses a counter costs the counter, never the run.
        """
        if key is not None and key in self.stats.seen:
            self.stats.duplicates += 1
            return
        if key is not None:
            self.stats.seen.add(key)
        self.stats.recorded += 1
        usage = record.usage
        base = self._dimensions.of(record) | {
            "agent_version": record.attribution.agent_version,
            "pricing_version": self._pricing.version,
        }
        unknown = usage.source is not CountSource.PROVIDER and usage.cost is None
        if unknown:
            self.stats.unknown += 1
            self._count(UNKNOWN_USAGE, 1, base)
        else:
            self._count(TOKENS, usage.input_tokens + usage.output_tokens, base)
            self._money(record, base)
        if seconds is not None:
            self._count(LATENCY, seconds, base)
        if iterations:
            self._count(ITERATIONS, iterations, base)
        if tool_calls:
            self._count(TOOL_CALLS, tool_calls, base)
        if breached:
            self._count(BUDGET_BREACHES, 1, base)

    def _money(self, record: SpendRecord, base: dict[str, str]) -> None:
        """Count what the step cost, or count that nobody could say."""
        cost = record.usage.cost or self._priced(record)
        if cost is None:
            self.stats.unpriced += 1
            self._count(UNPRICED, 1, base)
            return
        self._count(
            COST,
            float(cost.total),
            base | {"currency": cost.currency, "cost_confidence": str(cost.confidence)},
        )

    def _priced(self, record: SpendRecord) -> Cost | None:
        """What the table makes of the call, where the provider priced nothing."""
        rate = self._pricing.rate_of(record.attribution.model)
        if rate is None:
            return None
        usage = record.usage
        return rate.of(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
        )

    def _count(self, name: str, value: float, dimensions: Mapping[str, str]) -> None:
        """Emit one counter, absorbing a store that refuses it."""
        try:
            self._meter.count(name, value, **dict(dimensions))
        except Exception:
            self.stats.dropped += 1
