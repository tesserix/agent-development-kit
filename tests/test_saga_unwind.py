"""The flight fails after the hotel is booked, and nobody is left holding a room."""

from __future__ import annotations

import asyncio

import pytest

from tesserix_adk.core import (
    Applied,
    CompensatedFailure,
    CompensationIncomplete,
    ReversalOutcome,
    arguments_digest,
)
from tesserix_adk.runtime import (
    Compensator,
    MemoryCompensationLedger,
    MemoryIdempotencyStore,
)

pytestmark = pytest.mark.anyio


def applied(
    tool: str = "book_hotel",
    *,
    step: int = 1,
    branch: str = "",
    irreversible: bool = False,
    result_ref: str = "res-1",
    key: str = "",
) -> Applied:
    """One forward action that really happened."""
    return Applied(
        run_id="r1",
        tenant="acme",
        branch=branch,
        step=step,
        tool=tool,
        idempotency_key=key or f"{tool}:1",
        arguments_digest=arguments_digest({"night": "2026-09-01"}),
        result_ref=result_ref,
        irreversible=irreversible,
    )


class Cancels:
    """A reversal that records what it was asked to take back."""

    def __init__(self, *, irreversible: bool = False, fails: int = 0) -> None:
        self.irreversible = irreversible
        self.fails = fails
        self.asked: list[Applied] = []

    async def compensate(self, action: Applied) -> str:
        self.asked.append(action)
        if self.fails:
            self.fails -= 1
            raise RuntimeError("supplier is down")
        return f"cancelled {action.idempotency_key}"


async def ledger_of(*actions: Applied) -> MemoryCompensationLedger:
    """A ledger holding exactly what was applied."""
    ledger = MemoryCompensationLedger()
    for action in actions:
        await ledger.record(action)
    return ledger


class TestTheHappyUnwind:
    async def test_what_was_applied_is_taken_back(self) -> None:
        cancels = Cancels()
        compensator = Compensator(await ledger_of(applied()), reversals={"book_hotel": cancels})

        ended = await compensator.unwind("r1", tenant="acme", cause="no seats on BA117")

        assert isinstance(ended, CompensatedFailure)
        assert [one.idempotency_key for one in cancels.asked] == ["book_hotel:1"]

    async def test_it_is_taken_back_under_the_key_the_forward_call_used(self) -> None:
        """A fresh key cancels a different booking, or nothing at all."""
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(applied(key="book:BA117:ada")), reversals={"book_hotel": cancels}
        )

        await compensator.unwind("r1", tenant="acme", cause="x")

        assert cancels.asked[0].idempotency_key == "book:BA117:ada"

    async def test_the_original_error_is_carried_with_the_reversal(self) -> None:
        compensator = Compensator(await ledger_of(applied()), reversals={"book_hotel": Cancels()})

        ended = await compensator.unwind("r1", tenant="acme", cause="no seats on BA117")

        assert ended.cause == "no seats on BA117"
        assert ended.reversals[0].outcome is ReversalOutcome.REVERSED

    async def test_the_newest_action_is_reversed_first(self) -> None:
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(applied("book_hotel", step=1), applied("hold_seat", step=2)),
            reversals={"book_hotel": cancels, "hold_seat": cancels},
        )

        await compensator.unwind("r1", tenant="acme", cause="x")

        assert [one.tool for one in cancels.asked] == ["hold_seat", "book_hotel"]

    async def test_a_compensated_run_never_reports_success(self) -> None:
        compensator = Compensator(await ledger_of(applied()), reversals={"book_hotel": Cancels()})

        assert (await compensator.unwind("r1", tenant="acme", cause="x")).succeeded is False

    async def test_a_reversal_that_fails_once_is_tried_again(self) -> None:
        cancels = Cancels(fails=1)
        compensator = Compensator(await ledger_of(applied()), reversals={"book_hotel": cancels})

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensatedFailure)
        assert ended.reversals[0].attempts == 2

    async def test_a_run_that_applied_nothing_unwinds_to_a_clean_failure(self) -> None:
        compensator = Compensator(MemoryCompensationLedger(), reversals={})

        ended = await compensator.unwind("r1", tenant="acme", cause="model refused")

        assert isinstance(ended, CompensatedFailure)
        assert ended.reversals == ()


class TestWhenItCannotAllComeBack:
    async def test_an_exhausted_reversal_halts_in_incomplete(self) -> None:
        cancels = Cancels(fails=99)
        compensator = Compensator(await ledger_of(applied()), reversals={"book_hotel": cancels})

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensationIncomplete)
        assert ended.reversals[0].attempts == 3

    async def test_it_names_every_action_still_applied_and_its_key(self) -> None:
        compensator = Compensator(
            await ledger_of(applied()), reversals={"book_hotel": Cancels(fails=99)}
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensationIncomplete)
        assert [one.idempotency_key for one in ended.outstanding] == ["book_hotel:1"]

    async def test_the_supplier_s_own_refusal_is_what_a_person_reads(self) -> None:
        compensator = Compensator(
            await ledger_of(applied()), reversals={"book_hotel": Cancels(fails=99)}
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert "supplier is down" in ended.reversals[0].detail

    async def test_one_supplier_being_down_does_not_strand_the_rest(self) -> None:
        working = Cancels()
        compensator = Compensator(
            await ledger_of(applied("book_hotel", step=1), applied("hold_seat", step=2)),
            reversals={"book_hotel": working, "hold_seat": Cancels(fails=99)},
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensationIncomplete)
        assert [one.tool for one in ended.outstanding] == ["hold_seat"]
        assert working.asked

    async def test_an_action_with_no_paired_reversal_is_reported_not_assumed_away(self) -> None:
        compensator = Compensator(await ledger_of(applied()), reversals={})

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensationIncomplete)
        assert "no reversal is paired" in ended.reversals[0].detail

    async def test_an_incomplete_compensation_never_reports_success(self) -> None:
        compensator = Compensator(await ledger_of(applied()), reversals={})

        assert (await compensator.unwind("r1", tenant="acme", cause="x")).succeeded is False


class TestWhatIsNeverAutoCompensated:
    async def test_money_movement_is_refused_rather_than_reversed(self) -> None:
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(applied("refund", irreversible=True)), reversals={"refund": cancels}
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert cancels.asked == []
        assert isinstance(ended, CompensationIncomplete)
        assert ended.reversals[0].outcome is ReversalOutcome.REFUSED_IRREVERSIBLE

    async def test_the_refusal_says_it_needs_a_person(self) -> None:
        compensator = Compensator(
            await ledger_of(applied("refund", irreversible=True)), reversals={"refund": Cancels()}
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert "by hand" in ended.reversals[0].detail

    async def test_the_reversible_work_around_it_still_comes_back(self) -> None:
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(
                applied("book_hotel", step=1), applied("refund", step=2, irreversible=True)
            ),
            reversals={"book_hotel": cancels, "refund": Cancels()},
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert [one.tool for one in cancels.asked] == ["book_hotel"]
        assert isinstance(ended, CompensationIncomplete)
        assert [one.tool for one in ended.outstanding] == ["refund"]


class TestAnActionNobodyCanDecide:
    async def test_it_is_queried_before_anything_is_reversed(self) -> None:
        idempotency = MemoryIdempotencyStore()
        await idempotency.record(
            "book_hotel:1", tenant="acme", outcome="PNR 7QK2ZP", ttl_seconds=60
        )
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(applied(result_ref="")),
            reversals={"book_hotel": cancels},
            idempotency=idempotency,
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensatedFailure)
        assert cancels.asked[0].result_ref == "PNR 7QK2ZP"

    async def test_an_answer_nobody_has_is_never_blindly_reversed(self) -> None:
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(applied(result_ref="")),
            reversals={"book_hotel": cancels},
            idempotency=MemoryIdempotencyStore(),
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert cancels.asked == []
        assert isinstance(ended, CompensationIncomplete)
        assert ended.reversals[0].outcome is ReversalOutcome.UNKNOWN

    async def test_it_is_never_blindly_retried_either(self) -> None:
        """The forward call is not made again to find out; that is the second booking."""
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(applied(result_ref="")), reversals={"book_hotel": cancels}
        )

        await compensator.unwind("r1", tenant="acme", cause="x")

        assert cancels.asked == []

    async def test_a_deployment_with_nowhere_to_ask_leaves_it_outstanding(self) -> None:
        compensator = Compensator(
            await ledger_of(applied(result_ref="")), reversals={"book_hotel": Cancels()}
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensationIncomplete)
        assert "nothing can say" in ended.reversals[0].detail


class TestFanOut:
    async def test_a_branch_unwinds_without_touching_its_sibling(self) -> None:
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(
                applied("book_hotel", branch="hotel"), applied("book_flight", branch="flight")
            ),
            reversals={"book_hotel": cancels, "book_flight": cancels},
        )

        await compensator.unwind("r1", tenant="acme", cause="x", branch="flight")

        assert [one.tool for one in cancels.asked] == ["book_flight"]

    async def test_the_sibling_is_still_outstanding_afterwards(self) -> None:
        ledger = await ledger_of(
            applied("book_hotel", branch="hotel"), applied("book_flight", branch="flight")
        )
        compensator = Compensator(ledger, reversals={"book_flight": Cancels()})

        await compensator.unwind("r1", tenant="acme", cause="x", branch="flight")

        left = await ledger.outstanding("r1", tenant="acme")
        assert [one.tool for one in left] == ["book_hotel"]

    async def test_every_branch_unwinds_where_none_is_named(self) -> None:
        cancels = Cancels()
        compensator = Compensator(
            await ledger_of(
                applied("book_hotel", branch="hotel"), applied("book_flight", branch="flight")
            ),
            reversals={"book_hotel": cancels, "book_flight": cancels},
        )

        ended = await compensator.unwind("r1", tenant="acme", cause="x")

        assert isinstance(ended, CompensatedFailure)
        assert len(cancels.asked) == 2


class TestCancellationArrivingMidUnwind:
    async def test_the_unwind_finishes_rather_than_stopping_halfway(self) -> None:
        slow = Cancels()
        original = slow.compensate

        async def unhurried(action: Applied) -> str:
            await asyncio.sleep(0.05)
            return await original(action)

        slow.compensate = unhurried  # type: ignore[method-assign]
        compensator = Compensator(
            await ledger_of(applied("book_hotel", step=1), applied("hold_seat", step=2)),
            reversals={"book_hotel": slow, "hold_seat": slow},
        )

        unwinding = asyncio.ensure_future(compensator.unwind("r1", tenant="acme", cause="x"))
        await asyncio.sleep(0.01)
        unwinding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await unwinding

        await asyncio.sleep(0.2)
        assert [one.tool for one in slow.asked] == ["hold_seat", "book_hotel"]

    async def test_nothing_is_left_applied_after_it_finishes(self) -> None:
        ledger = await ledger_of(applied())
        compensator = Compensator(ledger, reversals={"book_hotel": Cancels()})

        unwinding = asyncio.ensure_future(compensator.unwind("r1", tenant="acme", cause="x"))
        await asyncio.sleep(0)
        unwinding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await unwinding

        await asyncio.sleep(0.05)
        assert await ledger.outstanding("r1", tenant="acme") == ()
