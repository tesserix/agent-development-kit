"""Persisted state that survives an upgrade: envelopes, canonical bytes, migrations on read.

Run it with `uv run python examples/state_versioning.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from tesserix_adk.core import (
    AdkModel,
    Envelope,
    StateKind,
    StateMigration,
    StateRegistry,
    UnsupportedStateVersionError,
    canonical_json,
    packed,
    unpacked,
)

KIND = "orders"


class Order(AdkModel):
    """An order as this build declares it, after the currency migration."""

    id: str
    placed_at: datetime
    total: Decimal
    currency: str = ""


def registry() -> StateRegistry:
    """A registry for one consumer-owned kind, at version 2, knowing the step to it."""
    registered = StateRegistry({KIND: 2})
    registered.register(
        StateMigration(
            kind=KIND,
            from_version=1,
            to_version=2,
            migrate=lambda payload: {**payload, "currency": "EUR"},
            note="orders were EUR-only before this",
        )
    )
    return registered


def written_by_the_previous_release() -> str:
    """What the last minor stored: version 1, with no currency field at all."""
    return Envelope(
        kind=KIND,
        schema_version=1,
        payload={
            "id": "o-1",
            "placed_at": {"$datetime": "2026-08-01T09:00:00+00:00"},
            "total": {"$decimal": "412.35"},
        },
    ).to_json()


def main() -> None:
    """Read an old record, hash a new one, and watch the two refusals."""
    known = registry()

    order = unpacked(written_by_the_previous_release(), Order, kind=KIND, registry=known)
    print("migrated on read:", order.id, order.total, order.currency)  # noqa: T201
    print("exact, not a float:", order.total * 3 == Decimal("1237.05"))  # noqa: T201

    blob = packed(order, kind=KIND, registry=known)
    print("stored at version:", Envelope.from_json(blob).schema_version)  # noqa: T201
    print("same bytes twice:", blob == packed(order, kind=KIND, registry=known))  # noqa: T201
    print("digest:", Envelope.from_json(blob).digest()[:12])  # noqa: T201

    ahead = Envelope(kind=KIND, schema_version=9, payload={"id": "o-2"})
    try:
        known.upgraded(ahead)
    except UnsupportedStateVersionError as refused:
        print("left for a newer worker:", refused.found, "vs", refused.supported)  # noqa: T201

    newer = Envelope(
        kind=KIND,
        schema_version=2,
        payload={
            "id": "o-3",
            "placed_at": {"$datetime": "2026-08-20T09:00:00+00:00"},
            "total": {"$decimal": "10.00"},
            "currency": "GBP",
            "gift_note": "for Ada",
        },
    )
    kept: dict[str, Any] = newer.preserved(Order)
    print("this build has no place for:", sorted(kept))  # noqa: T201
    rewritten = packed(newer.opened(Order), kind=KIND, preserved=kept, registry=known)
    print("and wrote it back anyway:", "gift_note" in rewritten)  # noqa: T201

    print("what the kit itself versions:", sorted(k.value for k in StateKind))  # noqa: T201
    print("canonical:", canonical_json({"b": 1, "a": datetime(2026, 1, 1, tzinfo=UTC)}))  # noqa: T201


if __name__ == "__main__":
    main()
