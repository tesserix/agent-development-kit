"""The hotel is booked and the flight fails. What the run does about the hotel."""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    Applied,
    CompensatedFailure,
    CompensationIncomplete,
    CompensationLedger,
    IrreversibleActionError,
    Reversal,
    ReversalOutcome,
    arguments_digest,
    verify_conformance,
)
from tesserix_adk.runtime import MemoryCompensationLedger

pytestmark = pytest.mark.anyio


def applied(
    tool: str = "book_hotel",
    *,
    step: int = 1,
    branch: str = "",
    irreversible: bool = False,
    result_ref: str = "res-1",
) -> Applied:
    """One forward action that really happened."""
    return Applied(
        run_id="r1",
        tenant="acme",
        branch=branch,
        step=step,
        tool=tool,
        idempotency_key=f"{tool}:1",
        arguments_digest=arguments_digest({"night": "2026-09-01"}),
        result_ref=result_ref,
        irreversible=irreversible,
    )


class TestWhatIsWrittenDownWhenSomethingHappens:
    def test_an_applied_action_carries_the_key_its_reversal_needs(self) -> None:
        assert applied().idempotency_key == "book_hotel:1"

    def test_the_arguments_are_carried_as_a_digest_not_as_arguments(self) -> None:
        """A ledger that keeps the arguments keeps whatever was in them."""
        first = arguments_digest({"card": "4242424242424242", "amount": 40})
        again = arguments_digest({"amount": 40, "card": "4242424242424242"})

        assert first == again
        assert "4242" not in first

    def test_an_action_whose_result_never_came_back_says_so(self) -> None:
        assert applied(result_ref="").outcome_known is False

    def test_an_action_that_returned_is_decided(self) -> None:
        assert applied().outcome_known is True


class TestTheLedger:
    async def test_it_satisfies_the_protocol(self) -> None:
        verify_conformance(MemoryCompensationLedger(), CompensationLedger)

    async def test_what_was_applied_is_outstanding_until_it_is_reversed(self) -> None:
        ledger = MemoryCompensationLedger()
        await ledger.record(applied())

        assert [one.tool for one in await ledger.outstanding("r1", tenant="acme")] == ["book_hotel"]

    async def test_a_reversed_action_is_no_longer_outstanding(self) -> None:
        ledger = MemoryCompensationLedger()
        await ledger.record(applied())

        await ledger.mark(Reversal(applied=applied(), outcome=ReversalOutcome.REVERSED))

        assert await ledger.outstanding("r1", tenant="acme") == ()

    async def test_an_action_that_could_not_be_reversed_stays_outstanding(self) -> None:
        """It is the list a person reconciles by hand; losing it loses the money."""
        ledger = MemoryCompensationLedger()
        await ledger.record(applied())

        await ledger.mark(
            Reversal(applied=applied(), outcome=ReversalOutcome.FAILED, detail="supplier down")
        )

        assert len(await ledger.outstanding("r1", tenant="acme")) == 1

    async def test_the_newest_action_comes_back_first(self) -> None:
        """Unwinding is reverse order, so the ledger hands them back that way."""
        ledger = MemoryCompensationLedger()
        await ledger.record(applied("book_hotel", step=1))
        await ledger.record(applied("book_flight", step=2))

        order = [one.tool for one in await ledger.outstanding("r1", tenant="acme")]

        assert order == ["book_flight", "book_hotel"]

    async def test_one_tenant_never_sees_another_s_applied_work(self) -> None:
        ledger = MemoryCompensationLedger()
        await ledger.record(applied())

        assert await ledger.outstanding("r1", tenant="other") == ()

    async def test_a_branch_is_unwound_without_touching_its_sibling(self) -> None:
        ledger = MemoryCompensationLedger()
        await ledger.record(applied("book_hotel", branch="hotel"))
        await ledger.record(applied("book_flight", branch="flight"))

        found = await ledger.outstanding("r1", tenant="acme", branch="flight")

        assert [one.tool for one in found] == ["book_flight"]

    async def test_recording_the_same_action_twice_records_it_once(self) -> None:
        """A retried record after a crash is the same applied action, not a second one."""
        ledger = MemoryCompensationLedger()
        await ledger.record(applied())
        await ledger.record(applied())

        assert len(await ledger.outstanding("r1", tenant="acme")) == 1

    async def test_erasure_reaches_the_ledger(self) -> None:
        ledger = MemoryCompensationLedger()
        await ledger.record(applied())

        assert await ledger.forget(tenant="acme") == 1
        assert await ledger.outstanding("r1", tenant="acme") == ()

    async def test_every_attempt_is_kept_including_the_ones_that_settled_nothing(self) -> None:
        """A reversal that failed twice is what an operator sees, not just the last state."""
        ledger = MemoryCompensationLedger()
        await ledger.record(applied())
        await ledger.mark(Reversal(applied=applied(), outcome=ReversalOutcome.FAILED))
        await ledger.mark(Reversal(applied=applied(), outcome=ReversalOutcome.REVERSED))

        assert [one.outcome for one in ledger.attempts] == [
            ReversalOutcome.FAILED,
            ReversalOutcome.REVERSED,
        ]


class TestTheTerminalStatesAConsumerReads:
    def test_a_compensated_failure_carries_the_original_error(self) -> None:
        ended = CompensatedFailure(
            run_id="r1",
            tenant="acme",
            cause="no seats on BA117",
            reversals=(Reversal(applied=applied(), outcome=ReversalOutcome.REVERSED),),
        )

        assert ended.cause == "no seats on BA117"
        assert ended.succeeded is False

    def test_an_incomplete_compensation_names_every_action_left_applied(self) -> None:
        ended = CompensationIncomplete(
            run_id="r1",
            tenant="acme",
            cause="no seats on BA117",
            outstanding=(applied(),),
            reversals=(Reversal(applied=applied(), outcome=ReversalOutcome.FAILED),),
        )

        assert [one.idempotency_key for one in ended.outstanding] == ["book_hotel:1"]
        assert ended.succeeded is False

    def test_an_incomplete_compensation_is_not_a_compensated_failure(self) -> None:
        """A consumer that cannot tell them apart reports a clean rollback that never was."""
        ended = CompensationIncomplete(
            run_id="r1", tenant="acme", cause="x", outstanding=(applied(),)
        )

        assert isinstance(ended, CompensationIncomplete)
        assert not isinstance(ended, CompensatedFailure)

    def test_an_incomplete_compensation_with_nothing_outstanding_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outstanding"):
            CompensationIncomplete(run_id="r1", tenant="acme", cause="x", outstanding=())


class TestRefusingToUndoWhatCannotBeUndone:
    def test_the_refusal_names_the_action_and_the_key(self) -> None:
        refused = IrreversibleActionError(
            "money already moved",
            run_id="r1",
            tool="refund",
            idempotency_key="refund:1",
        )

        assert refused.tool == "refund"
        assert refused.idempotency_key == "refund:1"

    def test_it_is_not_retryable_because_retrying_moves_more_money(self) -> None:
        refused = IrreversibleActionError("x", run_id="r1", tool="refund", idempotency_key="k")

        assert refused.retryable is False
