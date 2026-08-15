"""Taking back the hotel when the flight leg fails.

Run it with `uv run python examples/compensation.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import Applied, CompensationIncomplete, IrreversibleActionError
from tesserix_adk.runtime import MemoryCompensationLedger, MemoryIdempotencyStore
from tesserix_adk.workflows import Compensations, Saga, compensating_activity

TENANT = "acme"

BOOKED: list[str] = []


@compensating_activity("book_hotel")
async def cancel_hotel(action: Applied) -> str:
    """Give the room back."""
    BOOKED.remove(action.idempotency_key)
    return f"cancelled {action.idempotency_key}"


@compensating_activity("charge_card", irreversible=True)
async def unwind_charge(action: Applied) -> str:
    """Never called automatically. Money does not come back on its own."""
    return f"reconcile {action.idempotency_key} by hand"


COMPENSATIONS = Compensations(cancel_hotel, unwind_charge)


def scope(run_id: str, **extra: object) -> Saga:
    """A saga over a fresh in-memory ledger."""
    return Saga(
        run_id,
        tenant=TENANT,
        ledger=MemoryCompensationLedger(),
        compensations=COMPENSATIONS,
        **extra,  # type: ignore[arg-type]
    )


def booking(key: str, reference: str) -> object:
    """A forward call that really books something."""

    async def call() -> str:
        BOOKED.append(key)
        return reference

    return call


def failing() -> object:
    """A forward call whose result never comes back."""

    async def call() -> str:
        raise RuntimeError("the supplier connection went away")

    return call


async def the_hotel_comes_back_off() -> None:
    """The ordinary case: one leg applied, the next fails, the first is taken back."""
    saga = scope("run_1")
    try:
        async with saga:
            await saga.apply("book_hotel", key="run_1:hotel", call=booking("run_1:hotel", "7QK2ZP"))
            raise RuntimeError("no seats on BA117")
    except RuntimeError as failed:
        print(f"the run still failed: {failed}")  # noqa: T201

    print(f"outcome: {type(saga.outcome).__name__}")  # noqa: T201
    print(f"still booked: {BOOKED}")  # noqa: T201


async def what_cannot_come_back() -> None:
    """A step nothing can take back is refused before the money moves."""
    saga = scope("run_2")
    try:
        async with saga:
            await saga.apply("charge_card", key="run_2:card", call=booking("run_2:card", "txn-9"))
    except IrreversibleActionError as refused:
        print(f"refused: {refused}")  # noqa: T201
        print(f"worth retrying: {refused.retryable}")  # noqa: T201

    print(f"nothing was charged: {'run_2:card' not in BOOKED}")  # noqa: T201


async def approved_and_still_never_auto_compensated() -> None:
    """Approval lets it run. It does not make it reversible."""
    saga = scope("run_3")
    try:
        async with saga:
            await saga.apply(
                "charge_card",
                key="run_3:card",
                call=booking("run_3:card", "txn-9"),
                approved=True,
            )
            raise RuntimeError("the leg after it failed")
    except RuntimeError:
        pass

    outcome = saga.outcome
    if not isinstance(outcome, CompensationIncomplete):
        raise RuntimeError("an irreversible charge cannot end as a clean unwind")
    print(f"outstanding for a person: {[one.tool for one in outcome.outstanding]}")  # noqa: T201
    print(f"why: {outcome.reversals[0].detail}")  # noqa: T201


async def a_step_nobody_can_decide() -> None:
    """The call never returned. The idempotency store is asked before anything is assumed."""
    idempotency = MemoryIdempotencyStore()
    await idempotency.record("run_4:hotel", tenant=TENANT, outcome="7QK2ZP", ttl_seconds=60)
    BOOKED.append("run_4:hotel")
    saga = scope("run_4", idempotency=idempotency)
    try:
        async with saga:
            await saga.apply("book_hotel", key="run_4:hotel", call=failing())
    except RuntimeError:
        pass

    print(f"outcome: {type(saga.outcome).__name__}")  # noqa: T201
    print(f"still booked: {BOOKED}")  # noqa: T201


async def one_branch_at_a_time() -> None:
    """Fan-out: the flight leg unwinds and the hotel leg stands."""
    saga = scope("run_5")
    async with saga:
        await saga.apply(
            "book_hotel", key="run_5:hotel", call=booking("run_5:hotel", "7QK2ZP"), branch="hotel"
        )
        await saga.apply(
            "book_hotel", key="run_5:room2", call=booking("run_5:room2", "8LM3PQ"), branch="flight"
        )
    await saga.unwind("the flight leg failed", branch="flight")

    print(f"still booked: {BOOKED}")  # noqa: T201


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await the_hotel_comes_back_off()
    await what_cannot_come_back()
    await approved_and_still_never_auto_compensated()
    await a_step_nobody_can_decide()
    await one_branch_at_a_time()


if __name__ == "__main__":
    asyncio.run(main())
