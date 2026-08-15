"""Two products, one set of attribute names, and numbers nobody guessed.

Run it with `uv run python examples/telemetry_convention.py`.
"""

from __future__ import annotations

from decimal import Decimal

from tesserix_adk.core import AttributionError, Cost, CostConfidence, tenant_scope
from tesserix_adk.observability import (
    RESERVED_PREFIX,
    AttributeSet,
    CacheStatus,
    Measured,
    Outcome,
    Unavailability,
    conforms,
)


def refunds(**overrides: object) -> AttributeSet:
    """One agent's telemetry, with the tenant and user taken from the bound scope."""
    fields: dict[str, object] = {
        "agent": "refunds",
        "agent_version": "3.1.0",
        "model": "gpt-5",
        "provider": "openai",
        "run_id": "run-1",
        "outcome": Outcome.ANSWERED,
        "cache": CacheStatus.MISS,
        "input_tokens": Measured.of(1200),
        "output_tokens": Measured.of(340),
        "latency_seconds": Measured.of(2.5),
        "cost": Cost(input=Decimal("0.0012"), output=Decimal("0.0044")),
        "pricing_version": "2026-08-01",
    }
    return AttributeSet.here(**(fields | overrides))


def main() -> None:
    """The same run in two products, an unpriced model, and a squatted name."""
    with tenant_scope("acme", user="ada"):
        checkout = refunds(extra={"checkout.step": "review"})
        support = refunds(extra={"support.queue": "tier-2"})
        unpriced = refunds(
            cost=Cost(confidence=CostConfidence.UNKNOWN),
            input_tokens=Measured.missing(Unavailability.NOT_REPORTED),
        )

    rendered = checkout.rendered()
    print(f"tenant and user came from the scope: {rendered['adk.tenant']}/{rendered['adk.user']}")  # noqa: T201
    confidence = rendered["adk.cost.confidence"]
    print(f"cost={rendered['adk.cost']} {rendered['adk.currency']} ({confidence})")  # noqa: T201

    mine = {name for name in rendered if name.startswith(RESERVED_PREFIX)}
    theirs = {name for name in support.rendered() if name.startswith(RESERVED_PREFIX)}
    print(f"\ntwo products, same {len(mine)} names: identical={mine == theirs}")  # noqa: T201

    missing = unpriced.rendered()
    tokens, cost = missing["adk.input_tokens.unavailable"], missing["adk.cost.unavailable"]
    print(f"\nnothing guessed: {tokens!r}, {cost!r}")  # noqa: T201

    dimensions = checkout.metric_dimensions()
    print(f"\nsafe as metric dimensions: {len(dimensions)} of {len(rendered)}, no run_id or user")  # noqa: T201

    try:
        conforms(rendered | {"adk.my_own_field": "value"})
    except AttributionError as error:
        print(f"\nsquatting refused: {error}")  # noqa: T201


if __name__ == "__main__":
    main()
