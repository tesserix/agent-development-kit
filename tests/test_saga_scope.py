"""Declaring the reversal next to the forward action, and the scope that drives it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    Applied,
    CompensatedFailure,
    CompensationIncomplete,
    IrreversibleActionError,
    ReversalOutcome,
)
from tesserix_adk.runtime import MemoryCompensationLedger, MemoryIdempotencyStore
from tesserix_adk.workflows import Compensations, Saga, compensating_activity

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

pytestmark = pytest.mark.anyio

CANCELLED: list[str] = []


@compensating_activity("book_hotel")
async def cancel_hotel(action: Applied) -> str:
    """Give the room back."""
    CANCELLED.append(action.idempotency_key)
    return f"cancelled {action.idempotency_key}"


@compensating_activity("refund", irreversible=True)
async def unrefund(action: Applied) -> str:
    """Never called. Money does not come back on its own."""
    raise AssertionError(f"{action.tool} was auto-compensated")


@pytest.fixture(autouse=True)
def _nothing_cancelled_yet() -> None:
    CANCELLED.clear()


def saga(**overrides: object) -> Saga:
    """A scope over an in-memory ledger."""
    fields: dict[str, object] = {
        "run_id": "r1",
        "tenant": "acme",
        "ledger": MemoryCompensationLedger(),
        "compensations": Compensations(cancel_hotel, unrefund),
    }
    return Saga(**{**fields, **overrides})  # type: ignore[arg-type]


class TestPairingAReversalWithItsForwardAction:
    def test_the_reversal_says_what_it_reverses(self) -> None:
        assert cancel_hotel.forward == "book_hotel"

    def test_a_reversal_of_something_irreversible_says_so_before_it_is_needed(self) -> None:
        assert unrefund.irreversible is True
        assert cancel_hotel.irreversible is False

    def test_the_registry_finds_the_reversal_for_a_tool(self) -> None:
        assert Compensations(cancel_hotel).of("book_hotel") is cancel_hotel

    def test_a_tool_nothing_reverses_is_not_pretended_to_be_covered(self) -> None:
        assert Compensations(cancel_hotel).of("charge_card") is None

    def test_two_reversals_for_one_tool_are_refused(self) -> None:
        """Which of them runs would otherwise depend on registration order."""
        with pytest.raises(ValueError, match="book_hotel"):
            Compensations(cancel_hotel, cancel_hotel)

    def test_the_registry_names_what_it_covers(self) -> None:
        """So a consumer can check its coverage against its tool set before a run."""
        assert set(Compensations(cancel_hotel, unrefund).tools) == {"book_hotel", "refund"}

    def test_the_docstring_survives_the_decorator(self) -> None:
        assert cancel_hotel.__doc__ is not None
        assert "room" in cancel_hotel.__doc__


class TestApplyingWorkInsideTheScope:
    async def test_a_step_that_succeeds_is_written_down_with_its_result(self) -> None:
        scope = saga()

        async with scope:
            await scope.apply("book_hotel", key="hotel:1", call=_returning("PNR 7QK2ZP"))

        applied = await scope.ledger.outstanding("r1", tenant="acme")
        assert [one.result_ref for one in applied] == ["PNR 7QK2ZP"]

    async def test_the_step_returns_what_the_call_returned(self) -> None:
        scope = saga()

        async with scope:
            got = await scope.apply("book_hotel", key="hotel:1", call=_returning("PNR 7QK2ZP"))

        assert got == "PNR 7QK2ZP"

    async def test_a_scope_that_finished_has_no_outcome_to_report(self) -> None:
        scope = saga()

        async with scope:
            await scope.apply("book_hotel", key="hotel:1", call=_returning("ok"))

        assert scope.outcome is None

    async def test_a_call_that_never_returned_is_left_undecided_not_unrecorded(self) -> None:
        """The record is written before the call, or a crash mid-call leaves no trace of it."""
        scope = saga()

        with pytest.raises(RuntimeError):
            async with scope:
                await scope.apply("book_hotel", key="hotel:1", call=_raising())

        assert isinstance(scope.outcome, CompensationIncomplete)
        assert scope.outcome.outstanding[0].outcome_known is False


class TestWhenAStepFails:
    async def test_the_work_already_applied_is_taken_back(self) -> None:
        scope = saga()

        with pytest.raises(RuntimeError, match="no seats"):
            await _hotel_then_failure(scope)

        assert CANCELLED == ["hotel:1"]

    async def test_the_original_error_still_reaches_the_consumer(self) -> None:
        """Compensating is not handling. The caller's own error handling still runs."""
        scope = saga()

        with pytest.raises(RuntimeError, match="no seats"):
            await _hotel_then_failure(scope)

    async def test_the_outcome_is_a_compensated_failure_carrying_both_halves(self) -> None:
        scope = saga()

        with pytest.raises(RuntimeError):
            await _hotel_then_failure(scope)

        assert isinstance(scope.outcome, CompensatedFailure)
        assert scope.outcome.cause == "no seats on BA117"
        assert scope.outcome.reversals[0].outcome is ReversalOutcome.REVERSED

    async def test_a_reversal_that_cannot_run_ends_incomplete(self) -> None:
        scope = saga(compensations=Compensations())

        with pytest.raises(RuntimeError):
            await _hotel_then_failure(scope)

        assert isinstance(scope.outcome, CompensationIncomplete)

    async def test_nothing_applied_means_nothing_to_unwind(self) -> None:
        scope = saga()

        with pytest.raises(RuntimeError):
            await _nothing_then_failure(scope)

        assert isinstance(scope.outcome, CompensatedFailure)
        assert scope.outcome.reversals == ()


class TestRefusingWhatCannotBeUndone:
    async def test_an_irreversible_step_is_refused_before_the_money_moves(self) -> None:
        scope = saga()

        with pytest.raises(IrreversibleActionError, match="refund"):
            async with scope:
                await scope.apply("refund", key="refund:1", call=_raising())

    async def test_the_refusal_names_the_key_the_call_would_have_used(self) -> None:
        scope = saga()

        with pytest.raises(IrreversibleActionError) as refused:
            async with scope:
                await scope.apply("refund", key="refund:1", call=_returning("ok"))

        assert refused.value.idempotency_key == "refund:1"

    async def test_an_approved_irreversible_step_runs(self) -> None:
        scope = saga()

        async with scope:
            got = await scope.apply(
                "refund", key="refund:1", call=_returning("txn-9"), approved=True
            )

        assert got == "txn-9"

    async def test_an_approved_irreversible_step_is_still_never_auto_compensated(self) -> None:
        scope = saga()

        with pytest.raises(RuntimeError):
            await _approved_refund_then_failure(scope)

        assert isinstance(scope.outcome, CompensationIncomplete)
        assert scope.outcome.reversals[0].outcome is ReversalOutcome.REFUSED_IRREVERSIBLE


class TestFanOutBranches:
    async def test_each_branch_keeps_its_own_sequence(self) -> None:
        scope = saga()

        async with scope:
            await scope.apply("book_hotel", key="hotel:1", call=_returning("ok"), branch="hotel")
            await scope.apply("book_hotel", key="hotel:2", call=_returning("ok"), branch="flight")

        one = await scope.ledger.outstanding("r1", tenant="acme", branch="hotel")
        assert [action.idempotency_key for action in one] == ["hotel:1"]

    async def test_one_branch_unwinding_leaves_the_other_alone(self) -> None:
        scope = saga()

        async with scope:
            await scope.apply("book_hotel", key="hotel:1", call=_returning("ok"), branch="hotel")
            await scope.apply("book_hotel", key="hotel:2", call=_returning("ok"), branch="flight")
        await scope.unwind("the flight leg failed", branch="flight")

        assert CANCELLED == ["hotel:2"]


class TestTheUnknownOutcome:
    async def test_it_is_queried_before_the_unwind_decides_anything(self) -> None:
        idempotency = MemoryIdempotencyStore()
        await idempotency.record("hotel:1", tenant="acme", outcome="PNR 7QK2ZP", ttl_seconds=60)
        scope = saga(idempotency=idempotency)

        with pytest.raises(RuntimeError):
            async with scope:
                await scope.apply("book_hotel", key="hotel:1", call=_raising())

        assert isinstance(scope.outcome, CompensatedFailure)
        assert CANCELLED == ["hotel:1"]


async def _hotel_then_failure(scope: Saga) -> None:
    """One leg applied, the next one fails."""
    async with scope:
        await scope.apply("book_hotel", key="hotel:1", call=_returning("PNR 7QK2ZP"))
        raise RuntimeError("no seats on BA117")


async def _nothing_then_failure(scope: Saga) -> None:
    """It failed before anything was applied at all."""
    async with scope:
        raise RuntimeError("the model refused")


async def _approved_refund_then_failure(scope: Saga) -> None:
    """A refund somebody approved, and a leg after it that failed."""
    async with scope:
        await scope.apply("refund", key="refund:1", call=_returning("txn-9"), approved=True)
        raise RuntimeError("the leg after it failed")


def _returning(value: str) -> Callable[[], Awaitable[str]]:
    """A forward call that succeeds."""

    async def call() -> str:
        return value

    return call


def _raising() -> Callable[[], Awaitable[str]]:
    """A forward call whose result never comes back."""

    async def call() -> str:
        raise RuntimeError("the connection went away")

    return call
