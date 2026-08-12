"""Withdrawing autonomy, and having it stop work that was already under way.

Granting autonomy is only acceptable if it can be taken back at once. A grant read at run
start is not withdrawable — a multi-day run would keep acting for days — so the check is
per action, and a run suspended on a human decision is re-checked when that decision
arrives rather than acting on what was true when it went to sleep.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    ApprovalDecision,
    ApprovalRecord,
    ModelCapabilities,
    Run,
    RunEvent,
    RunEventKind,
    RunState,
    ToolCall,
    Usage,
)
from tesserix_adk.core.autonomy import (
    ActionClass,
    ActionRegistry,
    ActionRequest,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    AutonomyOutcome,
    Ceiling,
    InFlightPolicy,
    InMemoryGrants,
    Revocation,
)
from tesserix_adk.core.errors import ConfigurationError, GrantRevokedError
from tesserix_adk.runtime import AgentRunner, AutonomyGate, ModelResponse, RevocationWatch
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

NOW = 1_000.0
DAY = 86_400.0
BOOKING = ActionClass(name="booking.change", amount_field="amount", currency_field="currency")
REFUND = ActionClass(name="payment.refund", amount_field="amount", currency_field="currency")
REGISTRY = ActionRegistry({"change_booking": BOOKING, "refund_payment": REFUND})


def grant(**fields: object) -> AutonomyGrant:
    """One grant, filled in enough to be issued."""
    defaults: dict[str, object] = {
        "id": "g1",
        "tenant": "acme",
        "action_class": "booking.change",
        "level": AutonomyLevel.ACT_WITHIN_LIMITS,
        "granted_by": "ops@acme.example",
        "issued_at": NOW,
        "expires_at": NOW + DAY,
        "ceiling": Ceiling(amount=Decimal("5000"), currency="INR", window_seconds=DAY),
    }
    return AutonomyGrant.model_validate(defaults | fields)


def withdrawn(**fields: object) -> Revocation:
    """One revocation, as an operator would record it."""
    defaults: dict[str, object] = {
        "revoked_by": "ops@acme.example",
        "revoked_at": NOW + 1,
        "reason": "the desk withdrew it",
    }
    return Revocation.model_validate(defaults | fields)


def asking(tool_name: str = "change_booking", **fields: object) -> ActionRequest:
    """One action an agent wants to take."""
    defaults: dict[str, object] = {
        "tool": tool_name,
        "tenant": "acme",
        "arguments": {"amount": 10, "currency": "INR"},
    }
    return ActionRequest.model_validate(defaults | fields)


class TestWhatARevocationCovers:
    """A revocation names one grant, or everything of a class, or everything for a tenant."""

    def test_one_grant_by_id(self) -> None:
        assert withdrawn(grant_id="g1").covers(grant()) is True
        assert withdrawn(grant_id="g2").covers(grant()) is False

    def test_every_grant_of_one_class_for_one_tenant(self) -> None:
        wide = withdrawn(tenant="acme", action_class="booking.change")
        assert wide.covers(grant()) is True
        assert wide.covers(grant(action_class="payment.refund")) is False

    def test_every_class_for_one_tenant(self) -> None:
        wide = withdrawn(tenant="acme")
        assert wide.covers(grant()) is True
        assert wide.covers(grant(action_class="payment.refund")) is True
        assert wide.covers(grant(tenant="other")) is False

    def test_a_revocation_that_names_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names a grant"):
            withdrawn()

    def test_it_records_who_withdrew_it_and_when(self) -> None:
        assert withdrawn(grant_id="g1").revoked_by == "ops@acme.example"
        assert withdrawn(grant_id="g1").revoked_at == NOW + 1


class TestTheLadderAfterARevocation:
    """The grant is read per action, so withdrawing it lands on the very next one."""

    async def test_an_action_before_the_revocation_is_covered(self) -> None:
        grants = InMemoryGrants([grant()])
        held = AutonomyLadder(REGISTRY, grants=grants, clock=FakeClock(start=NOW))
        assert (await held.decide(asking())).outcome is AutonomyOutcome.ACT

    async def test_the_next_action_after_it_is_not(self) -> None:
        grants = InMemoryGrants([grant()])
        held = AutonomyLadder(REGISTRY, grants=grants, clock=FakeClock(start=NOW))
        await grants.revoke(withdrawn(grant_id="g1"))
        decided = await held.decide(asking())
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.grant_id is None

    async def test_revoking_one_class_leaves_the_others_alone(self) -> None:
        grants = InMemoryGrants([grant(), grant(id="g2", action_class="payment.refund")])
        held = AutonomyLadder(REGISTRY, grants=grants, clock=FakeClock(start=NOW))
        await grants.revoke(withdrawn(tenant="acme", action_class="booking.change"))
        assert (await held.decide(asking())).outcome is AutonomyOutcome.ESCALATE
        assert (await held.decide(asking("refund_payment"))).outcome is AutonomyOutcome.ACT

    async def test_a_revoked_grant_is_never_reactivated(self) -> None:
        grants = InMemoryGrants([grant()])
        await grants.revoke(withdrawn(grant_id="g1"))
        with pytest.raises(ConfigurationError, match="already exists"):
            await grants.issue(grant())

    async def test_re_granting_mints_a_new_id_and_that_one_works(self) -> None:
        grants = InMemoryGrants([grant()])
        held = AutonomyLadder(REGISTRY, grants=grants, clock=FakeClock(start=NOW))
        await grants.revoke(withdrawn(grant_id="g1"))
        await grants.issue(grant(id="g1-again", issued_at=NOW + 2))
        decided = await held.decide(asking())
        assert decided.outcome is AutonomyOutcome.ACT
        assert decided.grant_id == "g1-again"

    async def test_what_was_withdrawn_stays_readable(self) -> None:
        grants = InMemoryGrants([grant()])
        await grants.revoke(withdrawn(grant_id="g1"))
        assert [held.id for held in grants.all_grants()] == ["g1"]


class TestTheWatchOnTheBus:
    """A broadcast makes revocation prompt; the store is what makes it true."""

    async def test_a_broadcast_revocation_stops_the_next_action(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        await watch.heard(withdrawn(grant_id="g1"))
        assert watch.revoked(grant()) is True

    async def test_a_grant_nobody_withdrew_is_untouched(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        await watch.heard(withdrawn(grant_id="g2"))
        assert watch.revoked(grant()) is False

    async def test_a_view_nobody_has_confirmed_lately_is_not_trusted(self) -> None:
        clock = FakeClock(start=NOW)
        watch = RevocationWatch(clock=clock, stale_after_seconds=30.0)
        await clock.sleep(31.0)
        assert watch.fresh() is False

    async def test_hearing_from_the_bus_is_a_confirmation(self) -> None:
        clock = FakeClock(start=NOW)
        watch = RevocationWatch(clock=clock, stale_after_seconds=30.0)
        await clock.sleep(31.0)
        await watch.heard(withdrawn(grant_id="g2"))
        assert watch.fresh() is True

    async def test_a_gate_over_a_stale_watch_refuses_rather_than_acting(self) -> None:
        clock = FakeClock(start=NOW)
        watch = RevocationWatch(clock=clock, stale_after_seconds=30.0)
        gate = AutonomyGate(_ladder(grant()), revocations=watch)
        await clock.sleep(31.0)
        decided = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 10, "currency": "INR"},
            run_id="run_1",
        )
        assert decided.outcome is AutonomyOutcome.REFUSE
        assert "stale" in decided.reason

    async def test_it_follows_the_bus_until_the_bus_ends(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        await watch.follow(Bus(withdrawn(grant_id="g1")))
        assert watch.revoked(grant()) is True

    async def test_a_caller_that_just_read_the_store_confirms_the_view(self) -> None:
        clock = FakeClock(start=NOW)
        watch = RevocationWatch(clock=clock, stale_after_seconds=30.0)
        await clock.sleep(31.0)
        watch.confirm()
        assert watch.fresh() is True

    async def test_a_live_grant_a_fresh_watch_knows_nothing_about_still_acts(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        decided = await _gate(watch, grant()).decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 10, "currency": "INR"},
            run_id="run_1",
        )
        assert decided.outcome is AutonomyOutcome.ACT

    async def test_a_stale_watch_does_not_manufacture_authority(self) -> None:
        clock = FakeClock(start=NOW)
        watch = RevocationWatch(clock=clock, stale_after_seconds=30.0)
        gate = AutonomyGate(_ladder(), revocations=watch)
        await clock.sleep(31.0)
        decided = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 10, "currency": "INR"},
            run_id="run_1",
        )
        assert decided.outcome is AutonomyOutcome.ESCALATE


class TestRunsAlreadyUnderWay:
    """The decision a run was suspended on is re-checked when it wakes, not trusted."""

    async def test_a_revocation_while_a_human_decides_stops_the_call(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        called, run = await _settling(
            _gate(watch, grant(ceiling=_tight())),
            approvals=Desk(revoking=watch),
            policy=InFlightPolicy.CANCEL,
        )
        assert called == []
        assert run.state is RunState.FAILED
        assert _events(run, RunEventKind.GRANT_REVOKED)

    async def test_the_same_run_can_be_told_to_ask_instead_of_stopping(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        called, run = await _settling(
            _gate(watch, grant(ceiling=_tight())),
            approvals=Desk(revoking=watch),
            policy=InFlightPolicy.ASK_ALWAYS,
        )
        assert called == [{"amount": 9000, "currency": "INR"}]
        assert run.state is RunState.COMPLETED
        assert _events(run, RunEventKind.GRANT_REVOKED)

    async def test_a_call_nobody_revoked_goes_out_after_the_approval(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        called, run = await _settling(
            _gate(watch, grant(ceiling=_tight())), approvals=Desk(), policy=InFlightPolicy.CANCEL
        )
        assert called == [{"amount": 9000, "currency": "INR"}]
        assert not _events(run, RunEventKind.GRANT_REVOKED)

    async def test_the_refusal_names_the_grant_that_was_withdrawn(self) -> None:
        watch = RevocationWatch(clock=FakeClock(start=NOW))
        await watch.heard(withdrawn(grant_id="g1"))
        gate = _gate(watch, grant())
        decided = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 10, "currency": "INR"},
            run_id="run_1",
        )
        assert decided.outcome is AutonomyOutcome.REFUSE
        assert decided.grant_id == "g1"

    def test_the_typed_outcome_says_which_grant_and_who_withdrew_it(self) -> None:
        failure = GrantRevokedError("withdrawn", grant_id="g1", revoked_by="ops@acme.example")
        assert failure.grant_id == "g1"
        assert failure.details["revoked_by"] == "ops@acme.example"


class Bus:
    """A broadcast that carries what the test put on it and then ends."""

    def __init__(self, *revocations: Revocation) -> None:
        self._revocations = revocations

    async def publish(self, revocation: Revocation) -> None:
        """Unused here; a watch only ever listens."""

    async def listen(self) -> AsyncIterator[Revocation]:
        """Hand over each revocation once."""
        for revocation in self._revocations:
            yield revocation


class Desk:
    """An approval desk that can withdraw the grant while the human is deciding."""

    def __init__(self, *, revoking: RevocationWatch | None = None) -> None:
        self._revoking = revoking
        self.requested: list[ApprovalRecord] = []

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Grant it, having first withdrawn the authority behind it where asked to."""
        self.requested.append(record)
        if self._revoking is not None:
            await self._revoking.heard(withdrawn(grant_id="g1"))
        return ApprovalDecision(
            record_id=record.id, granted=True, decided_by="ada", decided_at=NOW, reason=""
        )


def _tight() -> Ceiling:
    """A ceiling low enough that the one call in these tests has to be approved."""
    return Ceiling(amount=Decimal("100"), currency="INR", window_seconds=DAY)


def _ladder(*grants: AutonomyGrant) -> AutonomyLadder:
    """A ladder over the grants a test issued."""
    return AutonomyLadder(REGISTRY, grants=InMemoryGrants(grants), clock=FakeClock(start=NOW))


def _gate(watch: RevocationWatch, *grants: AutonomyGrant) -> AutonomyGate:
    """A gate that consults the watch before it lets anything through."""
    return AutonomyGate(_ladder(*grants), revocations=watch)


def _events(run: Run[Any], kind: RunEventKind) -> list[RunEvent]:
    """Every event of one kind, in order."""
    return [event for event in run.events if event.kind is kind]


async def _settling(
    autonomy: AutonomyGate,
    *,
    approvals: Desk,
    policy: InFlightPolicy,
) -> tuple[list[dict[str, Any]], Run[Any]]:
    """A run over one booking change that has to be approved, returning what executed."""
    called: list[dict[str, Any]] = []

    @tool
    async def change_booking(amount: int, currency: str) -> str:
        """Change a booking.

        Args:
            amount: How much it moves.
            currency: In what.
        """
        called.append({"amount": amount, "currency": currency})
        return f"changed by {amount} {currency}"

    registry = ToolRegistry((change_booking,), clock=FakeClock())
    runner = AgentRunner(
        provider=ScriptedProvider(
            _calling(),
            ModelResponse(content="Done.", usage=Usage(input_tokens=1, output_tokens=1)),
            capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
        ),
        clock=FakeClock(),
        tools=registry.view(allow=("change_booking",), agent="planner"),
        approvals=approvals,
        autonomy=autonomy,
        revoked_runs=policy,
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Settle the booking.",
        free_text=True,
        model="scripted-1",
        tools=("change_booking",),
    )
    try:
        return called, await runner.run(agent, "settle it", tenant="acme", run_id="run_1")
    finally:
        change_booking.release()


def _calling() -> ModelResponse:
    """A model turn that calls the booking tool for more than the ceiling covers."""
    return ModelResponse(
        content="",
        tool_calls=(
            ToolCall(
                id="call_1",
                name="change_booking",
                arguments={"amount": 9000, "currency": "INR"},
            ),
        ),
        usage=Usage(input_tokens=1, output_tokens=1),
    )
