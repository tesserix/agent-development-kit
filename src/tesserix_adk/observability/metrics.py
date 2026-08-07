"""Spend as metrics, so it survives a sampled-away trace.

Traces are sampled; bills are not. A cost question answered from spans alone is answered
from whatever fraction the sampler kept, which is a number that looks precise and is wrong.
The counters here are emitted for every run whatever the trace does with it.

The dimension set is deliberately small and deliberately not the span's. A tenant id per
series is a metric store that falls over at the worst possible moment, so a deployment
states which tenants it wants separated and the rest land in one bucket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core import AdkModel

if TYPE_CHECKING:
    from tesserix_adk.observability.attribution import SpendRecord

__all__ = ["CALLS", "COST", "TOKENS", "Dimensions", "Meter"]

COST = "adk.cost"
TOKENS = "adk.tokens"
CALLS = "adk.calls"


@runtime_checkable
class Meter(Protocol):
    """Sideband emission of counters.

    Like `Tracer`, this fails open: a collector outage degrades observability and must
    never stop a run, so implementations swallow their own transport failures.
    """

    def count(self, name: str, value: float, **dimensions: str) -> None:
        """Add `value` to the counter `name` under `dimensions`. Never raises."""
        ...


class Dimensions(AdkModel):
    """Which values a deployment wants its metrics split by.

    An allow-list rather than a cap: bucketing whatever arrived after the store filled up
    gives a different answer every week. Anything not listed lands in `other` and is still
    counted — the money is never dropped, only the ability to break it out, and the span
    still carries the full identity for an investigation that needs it.

    Args:
        tenants: Tenants kept as their own series. None keeps every tenant, which is right
            for a single-tenant deployment and wrong for a marketplace.
        agents: Agents kept as their own series. None keeps every agent.
        models: Models kept as their own series. None keeps every model.
        other: The bucket everything else lands in.

    Example:
        >>> kept = Dimensions(tenants=frozenset({"acme"}))
        >>> kept.bucket("globex", kept.tenants)
        'other'
    """

    tenants: frozenset[str] | None = None
    agents: frozenset[str] | None = None
    models: frozenset[str] | None = None
    other: str = "other"

    def bucket(self, value: str, allowed: frozenset[str] | None) -> str:
        """Return `value` if it is kept as its own series, else the shared bucket."""
        return value if allowed is None or value in allowed else self.other

    def of(self, record: SpendRecord) -> dict[str, str]:
        """The dimensions to count `record` under.

        Args:
            record: The metered step being counted.

        Returns:
            A low-cardinality dimension set: tenant, agent, model, task class, outcome,
            whether it was estimated, and the currency.
        """
        attribution = record.attribution
        return {
            "tenant": self.bucket(attribution.tenant, self.tenants),
            "agent": self.bucket(attribution.agent, self.agents),
            "model": self.bucket(attribution.model, self.models),
            "task_class": attribution.task_class,
            "outcome": str(record.outcome),
            "estimated": "true" if record.estimated else "false",
            "currency": record.cost.currency,
        }
