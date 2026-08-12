"""What an agent may do unattended, and what it has to come back and ask about.

The ladder is the only thing that answers that question. Every test here is one way an
action can fail to be covered by a grant, and every one of them lands on asking a human
rather than on acting: an unmatched action has never been permitted by anybody.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

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
    RESERVED_ACTION_CLASS,
    ActionClass,
    ActionRegistry,
    ActionRequest,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    AutonomyOutcome,
    Ceiling,
    InMemoryGrants,
)
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.runtime.autonomy import AutonomyGate, InMemoryReports, ReportLog
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

NOW = 1_000.0
DAY = 86_400.0

BOOKING = ActionClass(name="booking.change", amount_field="amount", currency_field="currency")
REFUND = ActionClass(
    name="payment.refund", irreversible=True, amount_field="amount", currency_field="currency"
)
NOTIFY = ActionClass(name="comms.notify")
REGISTRY = ActionRegistry(
    {
        "change_booking": BOOKING,
        "refund_payment": REFUND,
        "send_email": NOTIFY,
        "grant_autonomy": ActionClass(name=RESERVED_ACTION_CLASS),
    }
)


class Spent:
    """A ledger that answers with one number and records what it was asked."""

    def __init__(self, total: Decimal) -> None:
        self.total = total
        self.asked: list[tuple[str, str, float]] = []

    async def committed(self, *, tenant: str, action_class: str, window_seconds: float) -> Decimal:
        """The total, and a note of the window it was asked over."""
        self.asked.append((tenant, action_class, window_seconds))
        return self.total


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


def ladder(*grants: AutonomyGrant, registry: ActionRegistry = REGISTRY) -> AutonomyLadder:
    """A ladder over the grants a test issued, on a clock the test controls."""
    return AutonomyLadder(registry, grants=InMemoryGrants(grants), clock=FakeClock(start=NOW))


def asking(tool: str = "change_booking", **fields: object) -> ActionRequest:
    """One action an agent wants to take."""
    defaults: dict[str, object] = {
        "tool": tool,
        "tenant": "acme",
        "arguments": {"amount": 900, "currency": "INR"},
    }
    return ActionRequest.model_validate(defaults | {"tool": tool} | fields)


class TestWhatAGrantIs:
    """A grant is issued by someone, for something, until a moment."""

    def test_a_ceilinged_level_without_a_ceiling_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="ceiling"):
            grant(ceiling=None)

    def test_a_grant_that_never_expires_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expires"):
            grant(expires_at=NOW - 1)

    def test_a_ceiling_in_something_that_is_not_a_currency_is_refused(self) -> None:
        with pytest.raises(ValueError, match="currency"):
            Ceiling(amount=Decimal("1"), currency="rupees", window_seconds=DAY)

    def test_a_ceiling_of_nothing_is_refused_rather_than_read_as_unlimited(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            Ceiling(amount=Decimal("0"), currency="INR", window_seconds=DAY)

    def test_an_irreversible_class_cannot_be_left_uncapped(self) -> None:
        with pytest.raises(ConfigurationError, match="irreversible"):
            AutonomyLadder(
                REGISTRY,
                grants=InMemoryGrants(
                    [
                        grant(
                            action_class="payment.refund",
                            level=AutonomyLevel.ACT_AND_REPORT,
                            ceiling=None,
                        )
                    ]
                ),
                clock=FakeClock(start=NOW),
            ).validate_grants()


class TestIssuingOne:
    """Issuance is a second protocol, and an id is never reused."""

    async def test_an_issued_grant_answers_the_next_action(self) -> None:
        grants = InMemoryGrants()
        await grants.issue(grant())
        held = AutonomyLadder(REGISTRY, grants=grants, clock=FakeClock(start=NOW))
        assert (await held.decide(asking())).outcome is AutonomyOutcome.ACT

    async def test_an_id_that_is_already_in_use_is_refused(self) -> None:
        grants = InMemoryGrants([grant()])
        with pytest.raises(ConfigurationError, match="already exists"):
            await grants.issue(grant(level=AutonomyLevel.ACT_AND_REPORT))

    def test_a_class_says_whether_it_carries_money(self) -> None:
        assert BOOKING.priced is True
        assert NOTIFY.priced is False

    def test_a_reader_that_cannot_enumerate_is_checked_as_it_answers(self) -> None:
        class Opaque:
            async def grants_for(
                self,
                *,
                tenant: str,  # noqa: ARG002 — the protocol's shape, and this one answers nothing
                action_class: str,  # noqa: ARG002 — same
            ) -> list[AutonomyGrant]:
                return []

        AutonomyLadder(REGISTRY, grants=Opaque(), clock=FakeClock(start=NOW)).validate_grants()

    def test_a_commitment_below_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            asking(committed=Decimal("-1"))


class TestActingInsideTheCeiling:
    """Inside the headroom the agent acts; at the edge it asks."""

    async def test_an_action_inside_the_headroom_is_taken_unattended(self) -> None:
        decided = await ladder(grant()).decide(asking())
        assert decided.outcome is AutonomyOutcome.ACT
        assert decided.grant_id == "g1"

    async def test_the_headroom_is_never_rounded_up_to_fit(self) -> None:
        decided = await ladder(grant()).decide(asking(committed=Decimal("4200")))
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.headroom == Decimal("800")
        assert decided.grant_id == "g1"

    async def test_an_action_exactly_at_the_headroom_is_still_inside_it(self) -> None:
        decided = await ladder(grant()).decide(asking(committed=Decimal("4100")))
        assert decided.outcome is AutonomyOutcome.ACT

    async def test_a_hundredth_over_is_over(self) -> None:
        decided = await ladder(grant()).decide(
            asking(arguments={"amount": "900.01", "currency": "INR"}, committed=Decimal("4100"))
        )
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_an_amount_the_arguments_do_not_carry_is_asked_about(self) -> None:
        decided = await ladder(grant()).decide(asking(arguments={"currency": "INR"}))
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert "amount" in decided.reason

    async def test_a_ceiling_on_a_class_that_carries_no_amount_asks(self) -> None:
        decided = await ladder(grant(action_class="comms.notify")).decide(
            asking(tool="send_email", arguments={})
        )
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_decision_to_act_says_so_without_being_unpacked(self) -> None:
        assert (await ladder(grant()).decide(asking())).unattended is True
        assert (await ladder().decide(asking())).unattended is False

    async def test_an_amount_that_is_not_a_number_is_asked_about(self) -> None:
        decided = await ladder(grant()).decide(
            asking(arguments={"amount": "lots", "currency": "INR"})
        )
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_another_currency_is_asked_about_rather_than_converted(self) -> None:
        decided = await ladder(grant()).decide(asking(arguments={"amount": 10, "currency": "USD"}))
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert "currency" in decided.reason


class TestFailingClosed:
    """Every way an action can fail to match a grant ends at a human."""

    async def test_an_action_with_no_grant_at_all_asks(self) -> None:
        decided = await ladder().decide(asking())
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.level is AutonomyLevel.ASK_ALWAYS
        assert decided.grant_id is None

    async def test_a_tool_in_no_action_class_asks(self) -> None:
        decided = await ladder(grant()).decide(asking(tool="delete_everything"))
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert "unregistered" in decided.reason

    async def test_a_class_registered_after_the_grant_was_issued_asks(self) -> None:
        later = ActionRegistry(
            dict(REGISTRY.tools) | {"cancel_trip": ActionClass(name="trip.cancel")}
        )
        decided = await ladder(grant(), registry=later).decide(asking(tool="cancel_trip"))
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_an_expired_grant_asks(self) -> None:
        late = AutonomyLadder(
            REGISTRY,
            grants=InMemoryGrants([grant(expires_at=NOW + 1)]),
            clock=FakeClock(start=NOW + 2),
        )
        decided = await late.decide(asking())
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.grant_id is None

    async def test_another_tenant_s_grant_is_not_this_tenant_s(self) -> None:
        decided = await ladder(grant(tenant="other")).decide(asking())
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_parent_tenant_s_grant_does_not_reach_a_child_by_default(self) -> None:
        decided = await ladder(grant(tenant="acme")).decide(asking(tenant="acme/eu"))
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_parent_tenant_s_grant_reaches_a_child_when_it_says_so(self) -> None:
        decided = await ladder(grant(includes_subtenants=True)).decide(asking(tenant="acme/eu"))
        assert decided.outcome is AutonomyOutcome.ACT

    async def test_a_tenant_that_merely_starts_the_same_is_a_different_tenant(self) -> None:
        decided = await ladder(grant(includes_subtenants=True)).decide(asking(tenant="acmecorp"))
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_grant_to_one_user_does_not_cover_another(self) -> None:
        decided = await ladder(grant(user="ada")).decide(asking(user="bob"))
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_grant_to_one_user_covers_that_user(self) -> None:
        decided = await ladder(grant(user="ada")).decide(asking(user="ada"))
        assert decided.outcome is AutonomyOutcome.ACT


class TestTheLevels:
    """Three levels, and what each of them buys."""

    async def test_ask_always_asks_even_for_nothing(self) -> None:
        decided = await ladder(
            grant(level=AutonomyLevel.ASK_ALWAYS, ceiling=None, action_class="comms.notify")
        ).decide(asking(tool="send_email", arguments={}))
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.grant_id == "g1"

    async def test_act_and_report_on_a_class_with_no_money_acts(self) -> None:
        decided = await ladder(
            grant(level=AutonomyLevel.ACT_AND_REPORT, ceiling=None, action_class="comms.notify")
        ).decide(asking(tool="send_email", arguments={}))
        assert decided.outcome is AutonomyOutcome.ACT
        assert decided.reports is True

    async def test_act_and_report_is_still_bounded_where_it_carries_a_ceiling(self) -> None:
        decided = await ladder(grant(level=AutonomyLevel.ACT_AND_REPORT)).decide(
            asking(committed=Decimal("4200"))
        )
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_report_nobody_received_degrades_the_next_action_to_asking(self) -> None:
        decided = await ladder(grant(level=AutonomyLevel.ACT_AND_REPORT)).decide(
            asking(reports_outstanding=True)
        )
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert "report" in decided.reason

    async def test_an_outstanding_report_does_not_hold_up_a_different_level(self) -> None:
        decided = await ladder(grant()).decide(asking(reports_outstanding=True))
        assert decided.outcome is AutonomyOutcome.ACT
        assert decided.reports is False

    async def test_act_within_limits_does_not_oblige_a_report(self) -> None:
        assert (await ladder(grant()).decide(asking())).reports is False


class TestNobodyGrantsThemselves:
    """The one thing an agent may never do is widen what it may do."""

    async def test_an_agent_reaching_for_the_grant_tool_is_refused_outright(self) -> None:
        decided = await ladder(grant()).decide(asking(tool="grant_autonomy", arguments={}))
        assert decided.outcome is AutonomyOutcome.REFUSE
        assert "escalation" in decided.reason

    async def test_it_is_refused_even_where_a_grant_says_otherwise(self) -> None:
        held = grant(action_class=RESERVED_ACTION_CLASS, level=AutonomyLevel.ACT_AND_REPORT)
        decided = await ladder(held).decide(asking(tool="grant_autonomy", arguments={}))
        assert decided.outcome is AutonomyOutcome.REFUSE

    def test_the_runtime_is_given_no_way_to_issue_one(self) -> None:
        assert not hasattr(ladder(grant()), "issue")


class TestWhichGrantAnswers:
    """Two grants can cover one action, and which one answered has to be recorded."""

    async def test_the_most_recently_issued_grant_decides(self) -> None:
        old = grant(id="old", issued_at=NOW - DAY)
        tight = Ceiling(amount=Decimal("1"), currency="INR", window_seconds=DAY)
        new = grant(id="new", ceiling=tight)
        decided = await ladder(old, new).decide(asking())
        assert decided.grant_id == "new"
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_grant_for_another_class_does_not_answer_this_one(self) -> None:
        decided = await ladder(grant(action_class="payment.refund")).decide(asking())
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.grant_id is None

    async def test_the_decision_names_the_class_it_resolved(self) -> None:
        assert (await ladder(grant()).decide(asking())).action_class == "booking.change"


class TestAskingTheLedgerWhatIsSpent:
    """A caller that does not know what is committed lets the ladder go and find out."""

    async def test_the_ledger_is_asked_over_the_grant_s_own_window(self) -> None:
        spent = Spent(Decimal("4200"))
        held = AutonomyLadder(
            REGISTRY,
            grants=InMemoryGrants([grant()]),
            clock=FakeClock(start=NOW),
            commitments=spent,
        )
        decided = await held.decide(asking())
        assert decided.outcome is AutonomyOutcome.ESCALATE
        assert decided.headroom == Decimal("800")
        assert spent.asked == [("acme", "booking.change", DAY)]

    async def test_what_the_caller_states_is_taken_over_what_the_ledger_holds(self) -> None:
        spent = Spent(Decimal("4900"))
        held = AutonomyLadder(
            REGISTRY,
            grants=InMemoryGrants([grant()]),
            clock=FakeClock(start=NOW),
            commitments=spent,
        )
        assert (await held.decide(asking(committed=Decimal("0")))).outcome is AutonomyOutcome.ACT
        assert spent.asked == []

    async def test_a_ladder_with_no_ledger_sees_nothing_committed(self) -> None:
        assert (await ladder(grant()).decide(asking())).outcome is AutonomyOutcome.ACT


class TestNamingTheClass:
    """The runtime needs the class before it has a decision, to look up owed reports."""

    def test_a_registered_tool_names_its_class(self) -> None:
        assert ladder().classify("refund_payment") == "payment.refund"

    def test_an_unregistered_tool_names_nothing(self) -> None:
        assert ladder().classify("launch_rocket") is None


class TestTheGateTheLoopHolds:
    """What the loop asks, and what acting unattended leaves owed behind it."""

    async def test_acting_under_act_and_report_leaves_a_report_owed(self) -> None:
        reports = InMemoryReports()
        gate = AutonomyGate(ladder(grant(level=AutonomyLevel.ACT_AND_REPORT)), reports=reports)
        decided = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-1",
        )
        assert decided.outcome is AutonomyOutcome.ACT
        assert await reports.outstanding(tenant="acme", action_class="booking.change") is True

    async def test_the_next_action_asks_while_that_report_is_undelivered(self) -> None:
        reports = InMemoryReports()
        gate = AutonomyGate(ladder(grant(level=AutonomyLevel.ACT_AND_REPORT)), reports=reports)
        await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-1",
        )
        second = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-2",
        )
        assert second.outcome is AutonomyOutcome.ESCALATE
        assert "undelivered" in second.reason

    async def test_delivering_the_report_lets_the_next_action_go_again(self) -> None:
        reports = InMemoryReports()
        gate = AutonomyGate(ladder(grant(level=AutonomyLevel.ACT_AND_REPORT)), reports=reports)
        await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-1",
        )
        await reports.delivered(tenant="acme", action_class="booking.change", run_id="run-1")
        second = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-2",
        )
        assert second.outcome is AutonomyOutcome.ACT

    async def test_act_within_limits_leaves_nothing_owed(self) -> None:
        reports = InMemoryReports()
        gate = AutonomyGate(ladder(grant()), reports=reports)
        await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-1",
        )
        assert await reports.outstanding(tenant="acme", action_class="booking.change") is False

    async def test_a_gate_with_no_report_log_still_decides(self) -> None:
        gate = AutonomyGate(ladder(grant(level=AutonomyLevel.ACT_AND_REPORT)))
        decided = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-1",
        )
        assert decided.outcome is AutonomyOutcome.ACT

    async def test_an_unregistered_tool_owes_no_report_lookup(self) -> None:
        reports = InMemoryReports()
        gate = AutonomyGate(ladder(grant()), reports=reports)
        decided = await gate.decide(
            tool="launch_rocket", tenant="acme", arguments={}, run_id="run-1"
        )
        assert decided.outcome is AutonomyOutcome.ESCALATE

    async def test_a_user_scoped_grant_answers_only_that_user(self) -> None:
        gate = AutonomyGate(ladder(grant(user="ana@acme.example")))
        mine = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-1",
            user="ana@acme.example",
        )
        theirs = await gate.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run-2",
            user="bo@acme.example",
        )
        assert mine.outcome is AutonomyOutcome.ACT
        assert theirs.outcome is AutonomyOutcome.ESCALATE

    def test_the_log_is_recognised_by_shape(self) -> None:
        assert isinstance(InMemoryReports(), ReportLog)


class TestTheLoopHoldingToTheGrant:
    """What the runner does with a decision, which is the only place it matters."""

    async def test_a_call_inside_the_ceiling_goes_out_unattended(self) -> None:
        called, run = await _settling(gate(grant()), _calling(amount=900))
        assert called == [{"amount": 900, "currency": "INR"}]
        assert run.state is RunState.COMPLETED
        assert not _events(run, RunEventKind.AUTONOMY_ESCALATED)

    async def test_a_call_over_the_ceiling_waits_for_a_human(self) -> None:
        approvals = Desk()
        called, run = await _settling(gate(grant()), _calling(amount=9000), approvals=approvals)
        assert called == [{"amount": 9000, "currency": "INR"}]
        assert "over the" in str(_events(run, RunEventKind.AUTONOMY_ESCALATED)[0].detail)
        assert approvals.requested[0].reason.startswith("beyond this agent's autonomy")

    async def test_a_human_declining_the_escalation_stops_the_call(self) -> None:
        called, run = await _settling(
            gate(grant()), _calling(amount=9000), approvals=Desk(granted=False)
        )
        assert called == []
        assert run.state is RunState.COMPLETED
        assert _events(run, RunEventKind.APPROVAL_DENIED)

    async def test_an_escalation_with_nowhere_to_escalate_to_does_not_dispatch(self) -> None:
        with pytest.raises(ConfigurationError, match="approval gate"):
            await _settling(gate(grant()), _calling(amount=9000))

    async def test_a_refused_class_fails_the_run_rather_than_asking_anyone(self) -> None:
        called, run = await _settling(
            gate(grant(), tool_class=ActionClass(name=RESERVED_ACTION_CLASS)),
            _calling(amount=1),
            approvals=Desk(),
        )
        assert called == []
        assert run.state is RunState.FAILED
        assert _events(run, RunEventKind.AUTONOMY_REFUSED)

    async def test_autonomy_never_waives_an_approval_the_tool_declared(self) -> None:
        approvals = Desk()
        called, _ = await _settling(
            gate(grant()), _calling(amount=900), approvals=approvals, declared=True
        )
        assert called == [{"amount": 900, "currency": "INR"}]
        assert approvals.requested[0].reason.endswith("is declared to require approval")

    async def test_a_runner_with_no_gate_is_unchanged(self) -> None:
        called, run = await _settling(None, _calling(amount=9000))
        assert called == [{"amount": 9000, "currency": "INR"}]
        assert run.state is RunState.COMPLETED


class Desk:
    """An approval backend that answers as scripted and remembers what it was asked."""

    def __init__(self, *, granted: bool = True) -> None:
        self._granted = granted
        self.requested: list[ApprovalRecord] = []

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Answer, and keep the record so a test can read what the human was told."""
        self.requested.append(record)
        return ApprovalDecision(
            record_id=record.id,
            granted=self._granted,
            decided_by="ada",
            decided_at=NOW,
            reason="" if self._granted else "above the desk limit",
        )


def gate(*grants: AutonomyGrant, tool_class: ActionClass = BOOKING) -> AutonomyGate:
    """A gate over the one tool the loop tests call."""
    registry = ActionRegistry({"change_booking": tool_class})
    return AutonomyGate(
        AutonomyLadder(registry, grants=InMemoryGrants(grants), clock=FakeClock(start=NOW))
    )


def _events(run: Run[Any], kind: RunEventKind) -> list[RunEvent]:
    """Every event of one kind, in order."""
    return [event for event in run.events if event.kind is kind]


def _calling(amount: int, currency: str = "INR") -> ModelResponse:
    """A model turn that calls the booking tool."""
    return ModelResponse(
        content="",
        tool_calls=(
            ToolCall(
                id="call_1",
                name="change_booking",
                arguments={"amount": amount, "currency": currency},
            ),
        ),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


async def _settling(
    autonomy: AutonomyGate | None,
    *responses: ModelResponse,
    approvals: Desk | None = None,
    declared: bool = False,
) -> tuple[list[dict[str, Any]], Run[Any]]:
    """A run over a booking change, returning what actually executed."""
    called: list[dict[str, Any]] = []

    @tool(requires_approval=declared)
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
            *responses,
            ModelResponse(content="Done.", usage=Usage(input_tokens=1, output_tokens=1)),
            capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
        ),
        clock=FakeClock(),
        tools=registry.view(allow=("change_booking",), agent="planner"),
        approvals=approvals,
        autonomy=autonomy,
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
