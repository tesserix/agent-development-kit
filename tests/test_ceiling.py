"""A ceiling that holds under concurrency, splitting and retries, or it is not a ceiling.

The three ways a limit leaks are two actions both fitting under the same headroom, one
action arriving as ten small ones, and a timed-out action being retried onto fresh
headroom. Each has its own class here, and the arithmetic is `Decimal` throughout: a
ceiling a hundredth of a rupee out is a ceiling nobody can reconcile.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from tesserix_adk.core import (
    ActionClass,
    ActionRegistry,
    ActionRequest,
    Agent,
    ApprovalDecision,
    ApprovalRecord,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    AutonomyOutcome,
    Ceiling,
    CeilingExceededError,
    CommitmentLedger,
    HoldState,
    InexactAmountError,
    InMemoryCeilingLedger,
    InMemoryGrants,
    ModelCapabilities,
    ToolCall,
    Usage,
    WindowKind,
    exact,
)
from tesserix_adk.runtime import AgentRunner, AutonomyGate, ModelResponse, RevocationWatch
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

NOW = 1_000.0
DAY = 86_400.0
LIMIT = Ceiling(amount=Decimal("10000"), currency="INR", window_seconds=DAY)
REGISTRY = ActionRegistry(
    {
        "change_booking": ActionClass(
            name="booking.change", amount_field="amount", currency_field="currency"
        )
    }
)


def ledger(clock: FakeClock | None = None) -> InMemoryCeilingLedger:
    """A ledger on a clock a test can move."""
    return InMemoryCeilingLedger(clock=clock or FakeClock(start=NOW))


async def reserving(
    held: InMemoryCeilingLedger, amount: str, key: str, ceiling: Ceiling = LIMIT
) -> Any:
    """One reservation for the usual tenant and class."""
    return await held.reserve(
        tenant="acme",
        action_class="booking.change",
        ceiling=ceiling,
        amount=Decimal(amount),
        idempotency_key=key,
    )


class TestReservingUnderACeiling:
    """Headroom is what is left after everything held and everything committed."""

    async def test_a_reservation_under_the_ceiling_is_held(self) -> None:
        held = ledger()
        reserved = await reserving(held, "3000", "call-1")
        assert reserved.state is HoldState.HELD
        assert reserved.amount == Decimal("3000")

    async def test_holding_counts_against_the_headroom_before_anything_commits(self) -> None:
        held = ledger()
        await reserving(held, "9000", "call-1")
        with pytest.raises(CeilingExceededError):
            await reserving(held, "2000", "call-2")

    async def test_committing_keeps_it_counted(self) -> None:
        held = ledger()
        await reserving(held, "9000", "call-1")
        await held.commit("call-1")
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("9000")

    async def test_releasing_gives_the_headroom_back(self) -> None:
        held = ledger()
        await reserving(held, "9000", "call-1")
        await held.release("call-1")
        assert (await reserving(held, "9000", "call-2")).state is HoldState.HELD

    async def test_a_refusal_says_what_was_asked_and_what_was_left(self) -> None:
        held = ledger()
        await reserving(held, "9000", "call-1")
        with pytest.raises(CeilingExceededError) as raised:
            await reserving(held, "2000", "call-2")
        assert raised.value.details["headroom"] == "1000"
        assert raised.value.details["requested"] == "2000"

    async def test_a_ledger_is_what_the_ladder_reads_headroom_from(self) -> None:
        assert isinstance(ledger(), CommitmentLedger)


class TestTwoActionsUnderOneHeadroom:
    """The primary scenario: four concurrent 3000s against 10000 commit exactly three."""

    async def test_only_what_fits_is_reserved(self) -> None:
        held = ledger()
        outcomes = await asyncio.gather(
            *(reserving(held, "3000", f"call-{n}") for n in range(4)),
            return_exceptions=True,
        )
        refused = [one for one in outcomes if isinstance(one, CeilingExceededError)]
        assert len(refused) == 1

    async def test_the_committed_total_is_exact(self) -> None:
        held = ledger()
        await asyncio.gather(
            *(reserving(held, "3000", f"call-{n}") for n in range(4)),
            return_exceptions=True,
        )
        for n in range(4):
            with suppressing():
                await held.commit(f"call-{n}")
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("9000")

    async def test_the_last_hundredth_is_not_rounded_away(self) -> None:
        held = ledger()
        await reserving(held, "9999.99", "call-1")
        with pytest.raises(CeilingExceededError):
            await reserving(held, "0.02", "call-2")


class TestSplittingOneActionIntoMany:
    """Ten actions under the limit are one action over it, and aggregate the same way."""

    async def test_small_actions_aggregate_against_the_same_ceiling(self) -> None:
        held = ledger()
        for n in range(10):
            await reserving(held, "1000", f"call-{n}")
            await held.commit(f"call-{n}")
        with pytest.raises(CeilingExceededError):
            await reserving(held, "1", "call-11")

    async def test_a_different_class_has_its_own_ceiling(self) -> None:
        held = ledger()
        await reserving(held, "10000", "call-1")
        other = await held.reserve(
            tenant="acme",
            action_class="payment.refund",
            ceiling=LIMIT,
            amount=Decimal("10000"),
            idempotency_key="call-2",
        )
        assert other.state is HoldState.HELD

    async def test_a_different_tenant_has_its_own_ceiling(self) -> None:
        held = ledger()
        await reserving(held, "10000", "call-1")
        other = await held.reserve(
            tenant="other",
            action_class="booking.change",
            ceiling=LIMIT,
            amount=Decimal("10000"),
            idempotency_key="call-2",
        )
        assert other.state is HoldState.HELD

    async def test_a_currency_is_part_of_what_a_ceiling_counts(self) -> None:
        held = ledger()
        await reserving(held, "10000", "call-1")
        other = await held.reserve(
            tenant="acme",
            action_class="booking.change",
            ceiling=Ceiling(amount=Decimal("10000"), currency="USD", window_seconds=DAY),
            amount=Decimal("10000"),
            idempotency_key="call-2",
        )
        assert other.state is HoldState.HELD


class TestRetryingWhatMayAlreadyHaveHappened:
    """A retry asks about the action it already reserved, never for more headroom."""

    async def test_the_same_key_gets_the_same_reservation(self) -> None:
        held = ledger()
        first = await reserving(held, "3000", "call-1")
        again = await reserving(held, "3000", "call-1")
        assert again.id == first.id

    async def test_a_retry_does_not_take_fresh_headroom(self) -> None:
        held = ledger()
        for _ in range(4):
            await reserving(held, "3000", "call-1")
        assert (await reserving(held, "7000", "call-2")).state is HoldState.HELD

    async def test_committing_twice_is_committing_once(self) -> None:
        held = ledger()
        await reserving(held, "3000", "call-1")
        await held.commit("call-1")
        await held.commit("call-1")
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("3000")

    async def test_a_committed_action_is_never_released_by_a_late_retry(self) -> None:
        held = ledger()
        await reserving(held, "3000", "call-1")
        await held.commit("call-1")
        await held.release("call-1")
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("3000")

    async def test_a_key_nobody_reserved_settles_to_nothing(self) -> None:
        held = ledger()
        assert await held.commit("call-nothing") is None
        await held.release("call-nothing")

    async def test_a_reservation_reused_after_release_is_a_new_one(self) -> None:
        held = ledger()
        first = await reserving(held, "3000", "call-1")
        await held.release("call-1")
        again = await reserving(held, "3000", "call-1")
        assert again.id != first.id


class TestReservationsNobodyCameBackFor:
    """A crashed process must not hold headroom until somebody notices."""

    async def test_a_held_reservation_expires_after_its_ttl(self) -> None:
        clock = FakeClock(start=NOW)
        held = InMemoryCeilingLedger(clock=clock, hold_seconds=60.0)
        await reserving(held, "10000", "call-1")
        await clock.sleep(61.0)
        assert (await reserving(held, "10000", "call-2")).state is HoldState.HELD

    async def test_an_expired_hold_is_reaped_rather_than_left_to_be_found(self) -> None:
        clock = FakeClock(start=NOW)
        held = InMemoryCeilingLedger(clock=clock, hold_seconds=60.0)
        await reserving(held, "10000", "call-1")
        await clock.sleep(61.0)
        assert await held.reap() == 1
        assert await held.reap() == 0

    async def test_a_reaped_reservation_cannot_then_be_committed(self) -> None:
        clock = FakeClock(start=NOW)
        held = InMemoryCeilingLedger(clock=clock, hold_seconds=60.0)
        await reserving(held, "10000", "call-1")
        await clock.sleep(61.0)
        await held.reap()
        assert await held.commit("call-1") is None

    async def test_a_committed_action_is_never_reaped(self) -> None:
        clock = FakeClock(start=NOW)
        held = InMemoryCeilingLedger(clock=clock, hold_seconds=60.0)
        await reserving(held, "10000", "call-1")
        await held.commit("call-1")
        await clock.sleep(61.0)
        assert await held.reap() == 0


class TestWhenAWindowEnds:
    """Which window an action lands in is decided when it is reserved, not when it settles."""

    async def test_a_rolling_window_forgets_what_fell_out_of_it(self) -> None:
        clock = FakeClock(start=NOW)
        held = ledger(clock)
        await reserving(held, "10000", "call-1")
        await held.commit("call-1")
        await clock.sleep(DAY + 1)
        assert (await reserving(held, "10000", "call-2")).state is HoldState.HELD

    async def test_a_rolling_window_still_counts_what_is_inside_it(self) -> None:
        clock = FakeClock(start=NOW)
        held = ledger(clock)
        await reserving(held, "10000", "call-1")
        await held.commit("call-1")
        await clock.sleep(DAY - 1)
        with pytest.raises(CeilingExceededError):
            await reserving(held, "1", "call-2")

    async def test_a_calendar_window_resets_on_its_boundary(self) -> None:
        clock = FakeClock(start=NOW)
        held = InMemoryCeilingLedger(clock=clock, windows=WindowKind.CALENDAR)
        await reserving(held, "10000", "call-1")
        await held.commit("call-1")
        await clock.sleep(DAY)
        assert (await reserving(held, "10000", "call-2")).state is HoldState.HELD

    async def test_a_calendar_window_holds_until_its_boundary(self) -> None:
        clock = FakeClock(start=NOW)
        held = InMemoryCeilingLedger(clock=clock, windows=WindowKind.CALENDAR)
        await reserving(held, "10000", "call-1")
        await held.commit("call-1")
        await clock.sleep(DAY - NOW - 1)
        with pytest.raises(CeilingExceededError):
            await reserving(held, "1", "call-2")

    async def test_a_rollover_between_reserve_and_commit_settles_where_it_was_reserved(
        self,
    ) -> None:
        clock = FakeClock(start=NOW)
        held = InMemoryCeilingLedger(clock=clock, windows=WindowKind.CALENDAR)
        await reserving(held, "10000", "call-1")
        await clock.sleep(DAY)
        await held.commit("call-1")
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("0")


class TestArithmeticThatCannotDrift:
    """A float amount is not an amount, and neither is a string somebody typed."""

    def test_a_decimal_is_taken_as_it_is(self) -> None:
        assert exact(Decimal("10.05")) == Decimal("10.05")

    def test_an_integer_is_exact_and_is_taken(self) -> None:
        assert exact(7) == Decimal("7")

    def test_a_float_is_refused_rather_than_rounded(self) -> None:
        with pytest.raises(InexactAmountError, match="float"):
            exact(10.05)

    def test_free_text_is_refused(self) -> None:
        with pytest.raises(InexactAmountError, match="amount"):
            exact("about ten thousand")

    def test_a_numeric_string_is_taken_exactly(self) -> None:
        assert exact("10.05") == Decimal("10.05")

    def test_a_negative_amount_is_refused(self) -> None:
        with pytest.raises(InexactAmountError, match="negative"):
            exact("-1")

    async def test_the_ledger_refuses_an_amount_it_cannot_add_up(self) -> None:
        held = ledger()
        with pytest.raises(InexactAmountError):
            await held.reserve(
                tenant="acme",
                action_class="booking.change",
                ceiling=LIMIT,
                amount=10.05,  # type: ignore[arg-type]
                idempotency_key="call-1",
            )


class TestMoneyComingBack:
    """A credit is recorded and audited; it never quietly becomes headroom to spend."""

    async def test_a_credit_is_recorded_against_the_class(self) -> None:
        held = ledger()
        await reserving(held, "10000", "call-1")
        await held.commit("call-1")
        credit = await held.credit(
            tenant="acme",
            action_class="booking.change",
            currency="INR",
            amount=Decimal("5000"),
            authorised_by="ops@acme.example",
            reason="the hotel refunded it",
        )
        assert credit.authorised_by == "ops@acme.example"

    async def test_a_credit_does_not_make_fresh_headroom(self) -> None:
        held = ledger()
        await reserving(held, "10000", "call-1")
        await held.commit("call-1")
        await held.credit(
            tenant="acme",
            action_class="booking.change",
            currency="INR",
            amount=Decimal("5000"),
            authorised_by="ops@acme.example",
            reason="the hotel refunded it",
        )
        with pytest.raises(CeilingExceededError):
            await reserving(held, "1", "call-2")

    async def test_a_credit_nobody_signed_is_refused(self) -> None:
        held = ledger()
        with pytest.raises(ValueError, match="authorised"):
            await held.credit(
                tenant="acme",
                action_class="booking.change",
                currency="INR",
                amount=Decimal("5000"),
                authorised_by="",
                reason="",
            )

    async def test_what_was_credited_is_readable_for_reconciliation(self) -> None:
        held = ledger()
        await held.credit(
            tenant="acme",
            action_class="booking.change",
            currency="INR",
            amount=Decimal("5000"),
            authorised_by="ops@acme.example",
            reason="the hotel refunded it",
        )
        assert [one.amount for one in held.credits()] == [Decimal("5000")]


class TestTheLadderOverALedger:
    """The ladder reads headroom from the same ledger the reservations are in."""

    def _ladder(self, held: InMemoryCeilingLedger) -> AutonomyLadder:
        grant = AutonomyGrant(
            id="g1",
            tenant="acme",
            action_class="booking.change",
            level=AutonomyLevel.ACT_WITHIN_LIMITS,
            granted_by="ops@acme.example",
            issued_at=NOW,
            expires_at=NOW + DAY,
            ceiling=LIMIT,
        )
        return AutonomyLadder(
            REGISTRY,
            grants=InMemoryGrants([grant]),
            commitments=held,
            clock=FakeClock(start=NOW),
        )

    async def _asking(self, ladder: AutonomyLadder, amount: int) -> Any:
        return await ladder.decide(
            ActionRequest(
                tool="change_booking",
                tenant="acme",
                arguments={"amount": amount, "currency": "INR"},
            )
        )

    async def test_what_is_held_is_already_gone_from_the_headroom(self) -> None:
        held = ledger()
        ladder = self._ladder(held)
        await reserving(held, "9000", "call-1")
        decided = await self._asking(ladder, 2000)
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.headroom == Decimal("1000")

    async def test_a_ceiling_already_spent_leaves_no_headroom_for_a_rounding(self) -> None:
        held = ledger()
        await reserving(held, "10000", "call-1")
        decided = await self._asking(self._ladder(held), 1)
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.headroom == Decimal("0")


class TestTheGateTakingHeadroom:
    """The gate takes the headroom, because a read and a take are not the same check."""

    def _gate(self, held: InMemoryCeilingLedger) -> AutonomyGate:
        return AutonomyGate(_ladder(held), commitments=held)

    async def _deciding(self, gate: AutonomyGate, amount: int, key: str | None) -> Any:
        return await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": amount, "currency": "INR"},
            run_id="run_1",
            key=key,
        )

    async def test_acting_takes_the_headroom_it_is_about_to_use(self) -> None:
        held = ledger()
        decided = await self._deciding(self._gate(held), 4000, "run_1:call-1")
        assert decided.outcome is AutonomyOutcome.ACT
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("4000")

    async def test_a_pending_escalation_holds_the_money_it_is_about(self) -> None:
        held = ledger()
        gate = self._gate(held)
        await self._deciding(gate, 9000, "run_1:call-1")
        assert (
            await self._deciding(gate, 9000, "run_1:call-2")
        ).outcome is AutonomyOutcome.ESCALATE

    async def test_the_same_call_retried_asks_about_the_same_action(self) -> None:
        held = ledger()
        gate = self._gate(held)
        for _ in range(3):
            await self._deciding(gate, 4000, "run_1:call-1")
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("4000")

    async def test_a_ledger_that_refuses_sends_the_call_to_a_human(self) -> None:
        held = ledger()
        gate = self._gate(held)
        await reserving(held, "9000", "somebody-else")
        decided = await self._deciding(gate, 2000, "run_1:call-1")
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert "over the" in decided.reason

    async def test_a_call_nobody_gave_a_key_for_takes_nothing(self) -> None:
        held = ledger()
        await self._deciding(self._gate(held), 4000, None)
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("0")

    async def test_a_gate_with_no_ledger_decides_as_before(self) -> None:
        gate = AutonomyGate(_ladder(ledger()))
        assert (await self._deciding(gate, 4000, "run_1:call-1")).outcome is AutonomyOutcome.ACT

    async def test_a_refused_class_never_reaches_the_ledger(self) -> None:
        held = ledger()
        gate = AutonomyGate(
            AutonomyLadder(
                ActionRegistry({"change_booking": ActionClass(name="autonomy.grant")}),
                grants=InMemoryGrants([_grant()]),
                clock=FakeClock(start=NOW),
            ),
            commitments=held,
        )
        assert (await self._deciding(gate, 10, "run_1:call-1")).outcome is AutonomyOutcome.REFUSE

    async def test_a_refusal_takes_nothing_even_where_a_ceiling_covered_it(self) -> None:
        held = ledger()
        clock = FakeClock(start=NOW)
        watch = RevocationWatch(clock=clock, stale_after_seconds=30.0)
        gate = AutonomyGate(_ladder(held), revocations=watch, commitments=held)
        await clock.sleep(31.0)
        decided = await self._deciding(gate, 4000, "run_1:call-1")
        assert decided.outcome is AutonomyOutcome.REFUSE
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("0")

    async def test_settling_a_key_with_no_ledger_behind_it_does_nothing(self) -> None:
        await AutonomyGate(_ladder(ledger())).settle("run_1:call-1", happened=True)


class TestALoopThatSettlesWhatItHeld:
    """Headroom taken at dispatch is committed by what happened, not by what was asked."""

    async def test_a_call_that_went_out_commits_what_it_held(self) -> None:
        held = ledger()
        await _settling(held, 900)
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("900")

    async def test_a_call_that_errored_keeps_it_because_it_may_have_happened(self) -> None:
        held = ledger()
        await _settling(held, 900, failing=True)
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("900")

    async def test_a_call_a_human_declined_gives_the_headroom_back(self) -> None:
        held = ledger()
        await _settling(held, 900, approvals=Desk(), declared=True)
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("0")

    async def test_a_runner_with_no_gate_settles_nothing(self) -> None:
        held = ledger()
        await _settling(held, 900, gated=False)
        assert await held.committed(
            tenant="acme", action_class="booking.change", window_seconds=DAY
        ) == Decimal("0")


def _grant() -> AutonomyGrant:
    """The one grant every ladder here answers from."""
    return AutonomyGrant(
        id="g1",
        tenant="acme",
        action_class="booking.change",
        level=AutonomyLevel.ACT_WITHIN_LIMITS,
        granted_by="ops@acme.example",
        issued_at=NOW,
        expires_at=NOW + DAY,
        ceiling=LIMIT,
    )


def _ladder(held: InMemoryCeilingLedger) -> AutonomyLadder:
    """A ladder reading the same ledger the reservations go into."""
    return AutonomyLadder(
        REGISTRY, grants=InMemoryGrants([_grant()]), commitments=held, clock=FakeClock(start=NOW)
    )


class Desk:
    """An approval backend that declines, which is what puts headroom back."""

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Decline, naming a reason a human would have given."""
        return ApprovalDecision(
            record_id=record.id,
            granted=False,
            decided_by="ada",
            decided_at=NOW,
            reason="above the desk limit",
        )


async def _settling(
    held: InMemoryCeilingLedger,
    amount: int,
    *,
    failing: bool = False,
    gated: bool = True,
    approvals: Desk | None = None,
    declared: bool = False,
) -> Any:
    """One run over a booking change, with the ledger wired into the gate."""

    @tool(requires_approval=declared)
    async def change_booking(amount: int, currency: str) -> str:
        """Change a booking.

        Args:
            amount: How much it moves.
            currency: In what.
        """
        if failing:
            raise RuntimeError("the booking system said no")
        return f"changed by {amount} {currency}"

    registry = ToolRegistry((change_booking,), clock=FakeClock())
    runner = AgentRunner(
        provider=ScriptedProvider(
            _asking(amount),
            ModelResponse(content="Done.", usage=Usage(input_tokens=1, output_tokens=1)),
            capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
        ),
        clock=FakeClock(),
        tools=registry.view(allow=("change_booking",), agent="planner"),
        approvals=approvals,
        autonomy=AutonomyGate(_ladder(held), commitments=held) if gated else None,
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Settle the booking.",
        free_text=True,
        model="scripted-1",
        tools=("change_booking",),
    )
    try:
        return await runner.run(agent, "settle it", tenant="acme", run_id="run_1")
    finally:
        change_booking.release()


def _asking(amount: int) -> ModelResponse:
    """The one response that asks for the booking change."""
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-1", name="change_booking", arguments={"amount": amount, "currency": "INR"}
            ),
        ),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


class suppressing:  # noqa: N801 — a context manager reads as a verb here
    """Swallow the one refusal the concurrency scenario expects."""

    def __enter__(self) -> None:
        """Nothing to hand over."""

    def __exit__(self, kind: object, value: object, traceback: object) -> bool:
        """Swallow a ceiling refusal and nothing else."""
        return isinstance(value, CeilingExceededError)
