"""What an agent did unattended, and what it declined to do.

The question after an autonomy incident is not answerable from telemetry: spans are
sampled, dropped under load and stripped of the context that made the decision. So every
test here is one half of the same claim — that a decision reached the audit store exactly
once, and that the payload behind it did not.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from tesserix_adk.core import (
    Agent,
    ModelCapabilities,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    Usage,
)
from tesserix_adk.core.audit import (
    AuditDecision,
    AuditEvent,
    AuditSink,
    digest_of_arguments,
    pseudonym,
)
from tesserix_adk.core.autonomy import (
    RESERVED_ACTION_CLASS,
    ActionClass,
    ActionRegistry,
    AutonomyGrant,
    AutonomyLadder,
    AutonomyLevel,
    Ceiling,
    InMemoryGrants,
)
from tesserix_adk.core.errors import AuditUnavailableError
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.runtime.audit import AuditTrail, MemoryAuditSink
from tesserix_adk.runtime.autonomy import AutonomyGate
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

NOW = 1_000.0
DAY = 86_400.0
CARD = "4111 1111 1111 1111"

BOOKING = ActionClass(name="booking.change", amount_field="amount", currency_field="currency")
REGISTRY = ActionRegistry({"change_booking": BOOKING, "send_email": ActionClass(name="comms")})


class Unreachable:
    """A sink that is down, which is the case audit has to fail closed on."""

    async def append(self, event: AuditEvent) -> AuditEvent:  # noqa: ARG002 — the protocol's shape
        """Refuse, as a store nobody can reach does."""
        raise TimeoutError("no route to the audit store")

    async def records(self, **asked: object) -> tuple[AuditEvent, ...]:  # noqa: ARG002 — the protocol's shape
        """Refuse, as a store nobody can reach does."""
        raise TimeoutError("no route to the audit store")

    async def pseudonymise(self, **asked: object) -> int:  # noqa: ARG002 — the protocol's shape
        """Refuse, as a store nobody can reach does."""
        raise TimeoutError("no route to the audit store")


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


def trail(sink: AuditSink | None = None) -> AuditTrail:
    """A trail over whatever sink the test wants, on a clock it controls."""
    return AuditTrail(sink if sink is not None else MemoryAuditSink(), clock=FakeClock(start=NOW))


def gate(
    *grants: AutonomyGrant,
    audit: AuditTrail | None = None,
    tool_class: ActionClass = BOOKING,
) -> AutonomyGate:
    """The gate the loop holds, wired to an audit trail."""
    registry = ActionRegistry(
        {"change_booking": tool_class, "send_email": ActionClass(name="comms")}
    )
    ladder = AutonomyLadder(registry, grants=InMemoryGrants(grants), clock=FakeClock(start=NOW))
    return AutonomyGate(ladder, audit=audit if audit is not None else trail())


def event(**fields: object) -> AuditEvent:
    """One decision, filled in enough to be appended."""
    defaults: dict[str, object] = {
        "run_id": "run_1",
        "sequence": 0,
        "tenant": "acme",
        "user": "ada",
        "agent_name": "planner",
        "agent_version": "1.0.0",
        "tool": "change_booking",
        "action_class": "booking.change",
        "level": AutonomyLevel.ACT_WITHIN_LIMITS,
        "decision": AuditDecision.EXECUTED,
        "reason": "within the grant",
        "grant_id": "g1",
        "arguments_digest": digest_of_arguments({"amount": 900}),
        "idempotency_key": "run_1:change_booking:1",
        "recorded_at": NOW,
    }
    return AuditEvent.model_validate(defaults | fields)


class TestWhatARecordSays:
    async def test_an_executed_action_names_the_grant_that_permitted_it(self) -> None:
        sink = MemoryAuditSink()
        decided = await gate(grant()).decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run_1",
        )
        await trail(sink).record(
            decided,
            AuditDecision.EXECUTED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 900, "currency": "INR"},
        )

        assert (await sink.records(tenant="acme"))[0].grant_id == "g1"

    async def test_it_carries_the_headroom_either_side_of_the_action(self) -> None:
        """ "Was there room for this?" is unanswerable from the ceiling alone, later."""
        sink = MemoryAuditSink()
        decided = await gate(grant()).decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run_1",
        )
        await trail(sink).record(
            decided,
            AuditDecision.EXECUTED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 900, "currency": "INR"},
            amount=Decimal("900"),
        )

        recorded = (await sink.records(tenant="acme"))[0]
        assert recorded.headroom_before == Decimal("5000")
        assert recorded.headroom_after == Decimal("4100")

    async def test_an_action_with_no_ceiling_records_no_headroom_rather_than_zero(self) -> None:
        sink = MemoryAuditSink()
        unlimited = grant(action_class="comms", ceiling=None, level=AutonomyLevel.ACT_AND_REPORT)
        decided = await gate(unlimited).decide(
            tool="send_email", tenant="acme", arguments={}, run_id="run_1"
        )
        await trail(sink).record(
            decided,
            AuditDecision.EXECUTED,
            run_id="run_1",
            tenant="acme",
            tool="send_email",
            arguments={},
        )

        assert (await sink.records(tenant="acme"))[0].headroom_after is None

    async def test_an_approved_execution_names_who_approved_it(self) -> None:
        sink = MemoryAuditSink()
        await trail(sink).record(
            None,
            AuditDecision.EXECUTED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 9000},
            approver="ada@acme.example",
        )

        assert (await sink.records(tenant="acme"))[0].approver == "ada@acme.example"

    async def test_an_unattended_action_names_nobody_as_approver(self) -> None:
        sink = MemoryAuditSink()
        await trail(sink).record(
            None,
            AuditDecision.EXECUTED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 900},
        )

        recorded = (await sink.records(tenant="acme"))[0]
        assert recorded.approver is None
        assert recorded.unattended


class TestARefusalIsAsVisibleAsAnAction:
    async def test_a_refusal_is_recorded_with_the_same_weight_as_an_execution(self) -> None:
        """Nobody can show that a ceiling held if only the actions were written down."""
        sink = MemoryAuditSink()
        await trail(sink).record(
            None,
            AuditDecision.REFUSED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 90_000},
            reason="beyond the ceiling",
        )

        recorded = (await sink.records(tenant="acme"))[0]
        assert recorded.decision is AuditDecision.REFUSED
        assert recorded.reason == "beyond the ceiling"

    async def test_the_record_of_a_refusal_does_not_carry_the_refused_payload(self) -> None:
        sink = MemoryAuditSink()
        await trail(sink).record(
            None,
            AuditDecision.REFUSED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"card": CARD},
            reason="beyond the ceiling",
        )

        assert CARD not in (await sink.records(tenant="acme"))[0].model_dump_json()

    async def test_an_escalation_is_recorded_too(self) -> None:
        sink = MemoryAuditSink()
        await trail(sink).record(
            None,
            AuditDecision.ESCALATED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 9_000},
            reason="over the ceiling, so a human is being asked",
        )

        assert (await sink.records(tenant="acme"))[0].decision is AuditDecision.ESCALATED


class TestWhatIsNeverStored:
    def test_a_card_number_is_not_in_the_digest_of_it(self) -> None:
        assert CARD not in digest_of_arguments({"card": CARD})

    def test_an_email_address_is_not_in_the_digest_of_it(self) -> None:
        assert "ada@acme.example" not in digest_of_arguments({"who": "ada@acme.example"})

    def test_a_bearer_token_is_not_in_the_digest_of_it(self) -> None:
        bearer = "sk-live-0123456789"  # gitleaks:allow — credential to nothing
        assert bearer not in digest_of_arguments({"key": bearer})

    def test_the_same_arguments_digest_the_same_whatever_the_key_order(self) -> None:
        """Two records of one action must be recognisable as one action."""
        assert digest_of_arguments({"a": 1, "b": 2}) == digest_of_arguments({"b": 2, "a": 1})

    def test_a_changed_argument_digests_differently(self) -> None:
        assert digest_of_arguments({"amount": 900}) != digest_of_arguments({"amount": 901})

    def test_two_different_card_numbers_still_digest_differently(self) -> None:
        """Scrubbing before the digest must not collapse every payment into one record."""
        assert digest_of_arguments({"card": CARD, "amount": 1}) != digest_of_arguments(
            {"card": CARD, "amount": 2}
        )

    def test_a_nested_secret_is_scrubbed_too(self) -> None:
        assert CARD not in digest_of_arguments({"payment": {"card": CARD}})

    def test_a_secret_in_a_list_is_scrubbed_too(self) -> None:
        assert CARD not in digest_of_arguments({"cards": [CARD]})

    def test_a_deployment_can_add_a_shape_of_its_own(self) -> None:
        digested = digest_of_arguments({"ref": "CASE-99887766"}, extra_patterns=(r"CASE-\d+",))
        assert digested != digest_of_arguments({"ref": "CASE-99887766"})


class TestReconstructingTheOrder:
    async def test_a_run_numbers_its_decisions_from_zero(self) -> None:
        sink = MemoryAuditSink()
        held = trail(sink)
        for name in ("change_booking", "send_email"):
            await held.record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool=name,
                arguments={},
            )

        assert [one.sequence for one in await sink.records(tenant="acme")] == [0, 1]

    async def test_concurrent_branches_do_not_share_a_sequence_number(self) -> None:
        """Fan-out is the case where ordering is reconstructable or it is nothing."""
        sink = MemoryAuditSink()
        held = trail(sink)

        async def acted(which: int) -> None:
            await held.record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={"branch": which},
            )

        await asyncio.gather(*(acted(which) for which in range(8)))

        sequences = [one.sequence for one in await sink.records(tenant="acme")]
        assert sorted(sequences) == list(range(8))

    async def test_another_run_numbers_its_own_decisions(self) -> None:
        sink = MemoryAuditSink()
        held = trail(sink)
        for run_id in ("run_1", "run_2"):
            await held.record(
                None,
                AuditDecision.EXECUTED,
                run_id=run_id,
                tenant="acme",
                tool="change_booking",
                arguments={},
            )

        assert [one.sequence for one in await sink.records(tenant="acme")] == [0, 0]


class TestTheSameDecisionTwice:
    async def test_a_retried_activity_does_not_yield_two_records(self) -> None:
        sink = MemoryAuditSink()
        held = trail(sink)
        for _ in range(2):
            await held.record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={"amount": 900},
                key="call_1",
            )

        assert len(await sink.records(tenant="acme")) == 1

    async def test_the_second_write_returns_what_was_already_recorded(self) -> None:
        held = trail()
        first = await held.record(
            None,
            AuditDecision.EXECUTED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 900},
            key="call_1",
        )
        again = await held.record(
            None,
            AuditDecision.EXECUTED,
            run_id="run_1",
            tenant="acme",
            tool="change_booking",
            arguments={"amount": 900},
            key="call_1",
        )

        assert again == first

    async def test_an_escalation_and_the_execution_after_it_are_two_records(self) -> None:
        """One call, two decisions: a human was asked, and then the call went out."""
        sink = MemoryAuditSink()
        held = trail(sink)
        for decision in (AuditDecision.ESCALATED, AuditDecision.EXECUTED):
            await held.record(
                None,
                decision,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={"amount": 9_000},
                key="call_1",
            )

        assert len(await sink.records(tenant="acme")) == 2

    async def test_two_calls_in_one_run_are_two_records(self) -> None:
        sink = MemoryAuditSink()
        held = trail(sink)
        for which in ("call_1", "call_2"):
            await held.record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={"amount": 900},
                key=which,
            )

        assert len(await sink.records(tenant="acme")) == 2

    async def test_a_call_with_no_key_is_still_one_record_per_payload(self) -> None:
        sink = MemoryAuditSink()
        held = trail(sink)
        for _ in range(2):
            await held.record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={"amount": 900},
            )

        assert len(await sink.records(tenant="acme")) == 1


class TestWhenTheStoreCannotBeReached:
    async def test_recording_fails_closed_rather_than_returning(self) -> None:
        with pytest.raises(AuditUnavailableError):
            await trail(Unreachable()).record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={},
            )

    async def test_the_failure_names_the_run_and_the_tool(self) -> None:
        with pytest.raises(AuditUnavailableError, match="change_booking"):
            await trail(Unreachable()).record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={},
            )

    async def test_the_failure_does_not_quote_the_payload(self) -> None:
        with pytest.raises(AuditUnavailableError) as refused:
            await trail(Unreachable()).record(
                None,
                AuditDecision.EXECUTED,
                run_id="run_1",
                tenant="acme",
                tool="change_booking",
                arguments={"card": CARD},
            )

        assert CARD not in str(refused.value)


class TestAskingWhatAnAgentDid:
    async def test_a_tenant_sees_its_own_decisions_and_no_others(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(tenant="acme"))
        await sink.append(event(tenant="globex", run_id="run_2"))

        assert [one.tenant for one in await sink.records(tenant="acme")] == ["acme"]

    async def test_a_period_excludes_what_happened_outside_it(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(recorded_at=NOW, idempotency_key="k1"))
        await sink.append(event(recorded_at=NOW + DAY, idempotency_key="k2"))

        held = await sink.records(tenant="acme", since=NOW + 1, until=NOW + DAY + 1)
        assert [one.idempotency_key for one in held] == ["k2"]

    async def test_the_declines_come_back_with_the_actions(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(decision=AuditDecision.EXECUTED, idempotency_key="k1"))
        await sink.append(event(decision=AuditDecision.REFUSED, idempotency_key="k2"))

        assert len(await sink.records(tenant="acme")) == 2

    async def test_one_kind_of_decision_can_be_asked_for_on_its_own(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(decision=AuditDecision.EXECUTED, idempotency_key="k1"))
        await sink.append(event(decision=AuditDecision.REFUSED, idempotency_key="k2"))

        held = await sink.records(tenant="acme", decision=AuditDecision.REFUSED)
        assert [one.idempotency_key for one in held] == ["k2"]

    async def test_records_come_back_in_the_order_they_happened(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(sequence=1, idempotency_key="k2"))
        await sink.append(event(sequence=0, idempotency_key="k1", run_id="run_0"))

        assert [one.sequence for one in await sink.records(tenant="acme")] == [1, 0]


class TestAnErasureRequest:
    async def test_the_person_is_pseudonymised_and_the_decision_is_kept(self) -> None:
        """The decision is what makes unattended operation defensible; the name is not."""
        sink = MemoryAuditSink()
        await sink.append(event(user="ada@acme.example"))

        assert await sink.pseudonymise(tenant="acme", subject="ada@acme.example") == 1
        kept = (await sink.records(tenant="acme"))[0]
        assert kept.user == pseudonym("ada@acme.example")
        assert kept.decision is AuditDecision.EXECUTED

    async def test_an_approver_is_pseudonymised_too(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(user=None, approver="ada@acme.example"))

        await sink.pseudonymise(tenant="acme", subject="ada@acme.example")
        assert (await sink.records(tenant="acme"))[0].approver == pseudonym("ada@acme.example")

    async def test_nobody_else_is_touched(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(user="ada@acme.example"))
        await sink.append(event(user="grace@acme.example", idempotency_key="k2"))

        assert await sink.pseudonymise(tenant="acme", subject="ada@acme.example") == 1
        assert (await sink.records(tenant="acme"))[1].user == "grace@acme.example"

    async def test_another_tenant_s_records_are_not_touched(self) -> None:
        sink = MemoryAuditSink()
        await sink.append(event(tenant="globex", user="ada@acme.example"))

        assert await sink.pseudonymise(tenant="acme", subject="ada@acme.example") == 0

    def test_the_same_person_pseudonymises_to_the_same_name(self) -> None:
        """Otherwise a series of decisions by one person stops being one series."""
        assert pseudonym("ada@acme.example") == pseudonym("ada@acme.example")

    def test_a_different_person_pseudonymises_differently(self) -> None:
        assert pseudonym("ada@acme.example") != pseudonym("grace@acme.example")

    def test_a_pseudonym_does_not_contain_the_person(self) -> None:
        assert "ada@acme.example" not in pseudonym("ada@acme.example")

    def test_a_deployment_salt_makes_the_pseudonym_unlinkable_across_stores(self) -> None:
        assert pseudonym("ada@acme.example", salt="acme") != pseudonym("ada@acme.example")


class TestTheRecordItself:
    def test_a_record_is_frozen(self) -> None:
        """An audit record that can be edited in place is not an audit record."""
        with pytest.raises(ValueError, match="frozen"):
            event().user = "mallory"

    def test_a_record_round_trips(self) -> None:
        assert AuditEvent.model_validate_json(event().model_dump_json()) == event()

    def test_a_record_without_a_tenant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tenant"):
            event(tenant="")

    def test_an_approved_action_is_not_unattended(self) -> None:
        assert not event(approver="ada").unattended

    def test_a_refusal_is_not_an_unattended_action(self) -> None:
        assert not event(decision=AuditDecision.REFUSED).unattended


class TestWhatTheGateRecords:
    async def test_a_refusal_reaches_the_store_before_the_run_hears_about_it(self) -> None:
        sink = MemoryAuditSink()
        held = gate(grant(), audit=trail(sink), tool_class=ActionClass(name=RESERVED_ACTION_CLASS))

        await held.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run_1",
        )

        assert (await sink.records(tenant="acme"))[0].decision is AuditDecision.REFUSED

    async def test_an_escalation_reaches_the_store(self) -> None:
        sink = MemoryAuditSink()
        held = gate(grant(), audit=trail(sink))

        await held.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 9_000, "currency": "INR"},
            run_id="run_1",
        )

        assert (await sink.records(tenant="acme"))[0].decision is AuditDecision.ESCALATED

    async def test_a_permitted_action_is_not_recorded_as_done_before_it_is(self) -> None:
        """The gate says it may; only the loop knows the call was cleared to go out."""
        sink = MemoryAuditSink()
        held = gate(grant(), audit=trail(sink))

        await held.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run_1",
        )

        assert await sink.records(tenant="acme") == ()

    async def test_a_gate_with_no_trail_still_decides(self) -> None:
        """Audit is wiring a deployment adds; a kit that breaks without it is not usable."""
        held = AutonomyGate(
            AutonomyLadder(REGISTRY, grants=InMemoryGrants((grant(),)), clock=FakeClock(start=NOW))
        )

        decided = await held.decide(
            tool="change_booking",
            tenant="acme",
            arguments={"amount": 900, "currency": "INR"},
            run_id="run_1",
        )

        assert decided.unattended


class TestNothingGoesOutUnaudited:
    async def test_an_unattended_call_is_recorded_as_executed(self) -> None:
        sink = MemoryAuditSink()

        called, _ = await _settling(gate(grant(), audit=trail(sink)), _calling(900))

        assert called == [900]
        recorded = await sink.records(tenant="acme")
        assert [one.decision for one in recorded] == [AuditDecision.EXECUTED]
        assert recorded[0].agent_name == "planner"

    async def test_the_record_names_the_run_and_the_tool(self) -> None:
        sink = MemoryAuditSink()

        await _settling(gate(grant(), audit=trail(sink)), _calling(900))

        recorded = (await sink.records(tenant="acme"))[0]
        assert recorded.run_id == "run_1"
        assert recorded.tool == "change_booking"

    async def test_a_call_nobody_could_audit_does_not_go_out(self) -> None:
        """Autonomy without a durable record of it is not autonomy anybody may have."""
        called, run = await _settling(gate(grant(), audit=trail(Unreachable())), _calling(900))

        assert called == []
        assert run.state is RunState.FAILED
        assert "AuditUnavailableError" in _terminated(run)

    async def test_a_refused_call_is_in_the_store_even_though_the_run_failed(self) -> None:
        sink = MemoryAuditSink()

        called, run = await _settling(
            gate(grant(), audit=trail(sink), tool_class=ActionClass(name=RESERVED_ACTION_CLASS)),
            _calling(900),
        )

        assert called == []
        assert run.state is RunState.FAILED
        assert (await sink.records(tenant="acme"))[0].decision is AuditDecision.REFUSED

    async def test_a_run_with_no_autonomy_gate_is_unaffected(self) -> None:
        called, run = await _settling(None, _calling(900))

        assert called == [900]
        assert run.state is RunState.COMPLETED


def _terminated(run: Run[Any]) -> str:
    """What the run says ended it."""
    return next(
        event.detail or ""
        for event in reversed(run.events)
        if event.kind is RunEventKind.TERMINATED
    )


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
    autonomy: AutonomyGate | None, *responses: ModelResponse
) -> tuple[list[int], Run[Any]]:
    """A run over a booking change, returning what actually executed."""
    called: list[int] = []

    @tool()
    async def change_booking(amount: int, currency: str) -> str:
        """Change a booking.

        Args:
            amount: How much it moves.
            currency: In what.
        """
        called.append(amount)
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
