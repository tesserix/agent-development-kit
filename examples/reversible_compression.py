"""Compression the model can undo, and the boundaries the handle does not cross.

Four scenarios: a compressed admission carrying a handle, the original pulled back whole,
an expansion windowed to the budget left, and a handle offered by the wrong run.

Run it with `python examples/reversible_compression.py`.
"""

from __future__ import annotations

import asyncio
import json

from tesserix_adk.core import ClaimUnavailableError
from tesserix_adk.memory import ContentRouter, ReversibleRouter
from tesserix_adk.runtime import MemoryAuditSink, MemoryClaimCheckStore
from tesserix_adk.testing import FakeClock

ROWS = json.dumps(
    [
        {"id": index, "region": "apac", "status": "active", "host": f"node-{index:03d}"}
        for index in range(300)
    ]
)


async def main() -> None:
    """Admit a large result reversibly, then read it back three different ways."""
    clock = FakeClock()
    audit = MemoryAuditSink()
    router = ReversibleRouter(
        ContentRouter(threshold_tokens=64),
        MemoryClaimCheckStore(clock),
        ttl_seconds=60.0,
        audit=audit,
        clock=clock,
    )

    admitted = await router.admit(
        ROWS, budget_tokens=4_000, tenant="acme", run_id="run-1", untrusted=True
    )
    print(  # noqa: T201
        f"admitted: {admitted.original_tokens}->{admitted.compressed_tokens} tokens "
        f"(ratio {admitted.ratio:.2f}), handle {admitted.handle[:14]}…"
    )

    whole = await router.expand(admitted.handle, tenant="acme", run_id="run-1", user="u-1")
    print(f"expanded whole: {whole.chars} characters, truncated={whole.truncated}")  # noqa: T201

    windowed = await router.expand(
        admitted.handle, tenant="acme", run_id="run-1", budget_tokens=100
    )
    print(  # noqa: T201
        f"expanded within budget: {len(windowed.content)} of {windowed.chars} characters, "
        f"truncated={windowed.truncated}"
    )

    try:
        await router.expand(admitted.handle, tenant="acme", run_id="run-2")
    except ClaimUnavailableError as gone:
        print(f"another run: {gone}")  # noqa: T201

    clock.advance(61.0)
    try:
        await router.expand(admitted.handle, tenant="acme", run_id="run-1")
    except ClaimUnavailableError:
        print("past its retention window: the handle expired rather than dangled")  # noqa: T201

    for event in await audit.records(tenant="acme"):
        print(f"audited: {event.decision} {event.tool} — {event.reason}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
