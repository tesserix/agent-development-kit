"""A run that rolls back publishes nothing, and one that commits is delivered once.

Runs offline against an in-process stand-in for PostgreSQL.

    uv run python examples/outbox.py
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from pydantic import SecretStr

from tesserix_adk.adapters import OutboxRelay, PostgresOutbox, PostgresOutboxSettings
from tesserix_adk.core import Delivery, EventEnvelope, Eventing, RunCompleted
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.testing import FakeClock

SETTINGS = PostgresOutboxSettings(dsn=SecretStr("postgresql://adk@localhost:5432/adk"))


class Database:
    """Just enough PostgreSQL: rows, one transaction at a time, and a rollback that discards."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._staged: list[dict[str, Any]] = []

    async def fetch(self, statement: str, *args: Any) -> list[list[Any]]:  # noqa: ANN401 — bound parameters are whatever the column holds
        """Answer the handful of statements this example sends."""
        if "INSERT INTO" in statement:
            self._staged.append({"id": len(self.rows) + len(self._staged), "payload": args[6]})
            return []
        if "UPDATE" in statement and "claimed_by" in statement:
            return [[row["id"], _run_of(row), row["payload"]] for row in self.rows]
        if "published_at = $2" in statement:
            self.rows = [row for row in self.rows if row["id"] not in set(args[0])]
        return []

    @asynccontextmanager
    async def transaction(self) -> Any:  # noqa: ANN401 — an async context manager over itself
        """Commit what was staged, or discard it if the body raised."""
        try:
            yield self
        except Exception:
            self._staged.clear()
            raise
        self.rows.extend(self._staged)
        self._staged.clear()


class Transport:
    """The real publisher, on the far side of the relay."""

    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    async def publish(self, event: EventEnvelope) -> None:
        """Take one event."""
        self.events.append(event)

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        """Take a batch."""
        self.events.extend(events)


def _run_of(row: dict[str, Any]) -> str:
    return str(json.loads(row["payload"])["run_id"])


async def main() -> None:
    """Roll one run back, commit the next, and see what the transport was told."""
    database, transport, clock = Database(), Transport(), FakeClock()
    outbox = PostgresOutbox(database, clock=clock, settings=SETTINGS)
    eventing = Eventing(clock=clock, delivery=Delivery.GUARANTEED)
    relay = OutboxRelay(database, transport, clock=clock, worker="relay-1")

    with tenant_scope("acme", user="ada"):
        try:
            async with database.transaction() as tx:
                await outbox.bound(tx).publish(await _event(eventing, "run_abandoned"))
                raise RuntimeError("the rest of the unit of work failed")
        except RuntimeError as failure:
            print(f"rolled back: {failure}")  # noqa: T201

        async with database.transaction() as tx:
            await outbox.bound(tx).publish(await _event(eventing, "run_finished"))

    print(f"rows waiting: {len(database.rows)}")  # noqa: T201
    print(f"delivered: {await relay.deliver()}")  # noqa: T201
    print(f"transport saw: {[event.run_id for event in transport.events]}")  # noqa: T201
    print(f"nothing left: {await relay.deliver()}")  # noqa: T201


async def _event(eventing: Eventing, run_id: str) -> EventEnvelope:
    event = await eventing.emit(RunCompleted(run_id=run_id, iterations=2))
    if event is None:
        raise SystemExit("the envelope was not built")
    return event


if __name__ == "__main__":
    asyncio.run(main())
