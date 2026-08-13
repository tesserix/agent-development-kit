"""A conversation changing hands, carrying a declared payload rather than a transcript.

The shortcut every staged flow reaches for is forwarding the whole conversation to the
next agent. That leaks context the target has no business seeing, pays for it by the
token, and lets the target infer permissions nobody granted it. This file is the
counter-argument: what crosses is a typed contract, the target holds no more than the
agent that handed to it, and the tenant is not a parameter anybody could get wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, Field

from tesserix_adk.core import (
    Agent,
    ConfigurationError,
    GuardrailError,
    HandoffContractError,
    RunEventKind,
    RunState,
    Usage,
)
from tesserix_adk.core.guards import GuardrailPipeline
from tesserix_adk.core.run import Run as CoreRun
from tesserix_adk.runtime import (
    AgentRunner,
    Handoff,
    HandoffContract,
    HandoffDesk,
    ModelResponse,
    Receiver,
)
from tesserix_adk.runtime.delegation import Delegation, DelegationScope
from tesserix_adk.testing import FakeClock, FakeGuardrail, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from tesserix_adk.core.run import Run

HELD = frozenset({"read_account", "issue_credit"})


class Ticket(BaseModel):
    """What the billing agent accepts, and nothing else."""

    account: str = Field(min_length=1)
    complaint: str = Field(min_length=1)


class Escalation(BaseModel):
    """What the human desk accepts."""

    account: str = Field(min_length=1)
    urgency: int = Field(ge=1, le=5)


class Desk:
    """A queue that keeps what it was handed, standing in for a person."""

    def __init__(self) -> None:
        self.taken: list[Handoff] = []

    async def receive(self, handoff: Handoff) -> None:
        self.taken.append(handoff)


def agent(name: str, *tools: str, **overrides: object) -> Agent[Any]:
    fields: dict[str, object] = {
        "name": name,
        "instructions": f"You are {name}.",
        "free_text": True,
        "model": "claude-sonnet-5",
        "tools": tools,
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answer(text: str = "Credited, and the account is clear.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def billing() -> Receiver:
    return Receiver(
        agent=agent("billing", "read_account", "issue_credit", "close_account"),
        contract=HandoffContract(accepts=Ticket),
    )


def receivers() -> tuple[Receiver, ...]:
    return (
        billing(),
        Receiver(
            agent=agent("shipping", "read_account"),
            contract=HandoffContract(accepts=Ticket),
        ),
    )


def ticket() -> Ticket:
    return Ticket(account="ac_9", complaint="charged twice in March")


def desk(
    *responses: ModelResponse | BaseException,
    targets: tuple[Receiver, ...] | None = None,
    tools: frozenset[str] = HELD,
    guardrails: GuardrailPipeline | None = None,
    user: str | None = "ada",
    **overrides: object,
) -> HandoffDesk:
    runner = AgentRunner(
        provider=ScriptedProvider(*responses),
        clock=FakeClock(),
        tools=FakeToolRegistry(
            dict.fromkeys(("read_account", "issue_credit", "close_account"), str)
        ),
    )
    return HandoffDesk(
        runner,
        targets if targets is not None else receivers(),
        agent=agent("triage", *sorted(tools), **overrides),
        delegation=Delegation.root(
            run_id="run_1",
            tenant="acme",
            agent="triage",
            user=user,
            scope=DelegationScope(tools=tools),
        ),
        guardrails=guardrails,
    )


class TestDeclaringWhoMayBeHandedTo:
    def test_a_desk_with_nowhere_to_hand_to_is_refused_where_it_is_written(self) -> None:
        with pytest.raises(ConfigurationError, match="no receiver"):
            desk(targets=())

    def test_two_receivers_answering_to_one_name_are_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="one name"):
            desk(targets=(billing(), billing()))

    def test_a_receiver_that_is_neither_an_agent_nor_a_queue_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="neither"):
            Receiver(contract=HandoffContract(accepts=Ticket))

    def test_a_receiver_that_is_both_an_agent_and_a_queue_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="both"):
            Receiver(
                agent=agent("billing", "read_account"),
                queue=Desk(),
                name="billing",
                contract=HandoffContract(accepts=Ticket),
            )

    def test_a_queue_with_no_name_is_refused_because_nothing_names_it(self) -> None:
        with pytest.raises(ConfigurationError, match="name"):
            Receiver(queue=Desk(), contract=HandoffContract(accepts=Ticket))

    def test_a_receiver_is_named_by_its_agent(self) -> None:
        assert billing().name == "billing"

    def test_the_desk_says_who_it_can_hand_to(self) -> None:
        assert desk().receivers == ("billing", "shipping")


class TestWhatCrossesWithTheConversation:
    async def test_the_payload_is_validated_against_the_targets_contract(self) -> None:
        handing = desk(answer())
        result = await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort the double charge"
        )
        assert result.handoff.state == {"account": "ac_9", "complaint": "charged twice in March"}

    async def test_a_payload_the_contract_does_not_accept_fails_before_the_target_runs(
        self,
    ) -> None:
        handing = desk(answer())
        with pytest.raises(HandoffContractError) as raised:
            await handing.hand_off(
                "billing", reason="a billing question", state={"account": "ac_9"}, task="sort it"
            )
        assert raised.value.reason == "contract"
        assert raised.value.violations == ("complaint",)

    async def test_nothing_reaches_the_target_when_the_contract_is_violated(self) -> None:
        handing = desk(answer())
        with pytest.raises(HandoffContractError):
            await handing.hand_off(
                "billing", reason="a billing question", state={"account": "ac_9"}, task="sort it"
            )
        assert handing.handoffs == ()

    async def test_a_refusal_names_the_agent_that_made_it_and_the_run(self) -> None:
        handing = desk(answer())
        with pytest.raises(HandoffContractError) as raised:
            await handing.hand_off(
                "billing", reason="a billing question", state={"account": "ac_9"}, task="sort it"
            )
        assert (raised.value.source, raised.value.target) == ("triage", "billing")
        assert raised.value.run_id == "run_1"

    async def test_a_payload_of_the_declared_model_is_taken_as_it_stands(self) -> None:
        handing = desk(answer())
        result = await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert result.run is not None
        assert result.run.state is RunState.COMPLETED

    async def test_handing_to_nobody_on_the_desk_is_refused_typed(self) -> None:
        handing = desk(answer())
        with pytest.raises(HandoffContractError) as raised:
            await handing.hand_off(
                "litigation", reason="an appeal", state=ticket(), task="argue it"
            )
        assert raised.value.reason == "unknown_target"

    async def test_a_reason_is_required_because_a_handoff_nobody_explained_is_a_bug(self) -> None:
        handing = desk(answer())
        with pytest.raises(ConfigurationError, match="reason"):
            await handing.hand_off("billing", reason="  ", state=ticket(), task="sort it")

    async def test_the_transcript_is_not_what_crosses(self) -> None:
        """Only the declared payload reaches the target, whatever the source said."""
        handing = desk(answer())
        result = await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort the double charge"
        )
        assert result.run is not None
        prompt = " ".join(
            part.text
            for message in result.run.messages
            for part in message.content
            if hasattr(part, "text")
        )
        assert "charged twice in March" in prompt
        assert "You are triage" not in prompt


class TestWhatTheTargetIsAllowedToHold:
    async def test_the_target_holds_the_intersection_rather_than_its_own_allowlist(self) -> None:
        """`close_account` is billing's own and not triage's, so it does not cross."""
        result = await desk(answer()).hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert result.run is not None
        assert result.run.grant is not None
        assert result.run.grant.tools == ("read_account", "issue_credit")

    async def test_a_target_sharing_no_tool_with_the_source_never_runs(self) -> None:
        stranger = (
            Receiver(agent=agent("stranger", "wire"), contract=HandoffContract(accepts=Ticket)),
        )
        handing = desk(targets=stranger)
        with pytest.raises(HandoffContractError) as raised:
            await handing.hand_off("stranger", reason="a question", state=ticket(), task="sort it")
        assert raised.value.reason == "no_tools"

    async def test_the_target_runs_one_level_below_the_agent_that_handed_to_it(self) -> None:
        result = await desk(answer()).hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert result.run is not None
        assert (result.run.depth, result.run.path) == (1, ("triage", "billing"))

    async def test_the_tenant_and_the_user_cross_unchanged(self) -> None:
        result = await desk(answer()).hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert result.run is not None
        assert (result.run.tenant, result.run.user) == ("acme", "ada")

    def test_there_is_no_way_to_hand_off_into_another_tenant(self) -> None:
        """The widening path is unrepresentable: `hand_off` takes no tenant at all."""
        import inspect

        taken = inspect.signature(HandoffDesk.hand_off).parameters
        assert "tenant" not in taken


class TestSessionContinuity:
    async def test_recalled_memory_passes_through_to_the_target(self) -> None:
        handing = desk(answer())
        result = await handing.hand_off(
            "billing",
            reason="a billing question",
            state=ticket(),
            task="sort it",
            memory=("the customer called about this in February",),
        )
        assert result.run is not None
        prompt = " ".join(
            part.text
            for message in result.run.messages
            for part in message.content
            if hasattr(part, "text")
        )
        assert "called about this in February" in prompt

    async def test_the_conversation_so_far_crosses_when_it_is_handed_over(self) -> None:
        first = await desk(answer("What is the account?")).hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert first.run is not None
        handing = desk(answer())
        result = await handing.hand_off(
            "shipping",
            reason="it turned out to be a delivery",
            state=ticket(),
            task="where is it",
            history=first.run.messages,
        )
        assert result.run is not None
        assert len(result.run.messages) > len(first.run.messages)


def _review_desk(queue: Desk) -> Receiver:
    return Receiver(queue=queue, name="review_desk", contract=HandoffContract(accepts=Escalation))


class TestHandingToAPerson:
    async def test_a_queue_is_a_target_like_any_other(self) -> None:
        queue = Desk()
        handing = desk(targets=(_review_desk(queue),))
        result = await handing.hand_off(
            "review_desk",
            reason="the customer asked for a person",
            state=Escalation(account="ac_9", urgency=4),
            task="review the credit",
        )
        assert result.queued
        assert result.run is None
        assert queue.taken[0].to_agent == "review_desk"

    async def test_a_queue_is_held_to_the_same_contract(self) -> None:
        queue = Desk()
        handing = desk(targets=(_review_desk(queue),))
        with pytest.raises(HandoffContractError):
            await handing.hand_off(
                "review_desk",
                reason="the customer asked for a person",
                state={"account": "ac_9", "urgency": 9},
                task="review the credit",
            )
        assert queue.taken == []


class TestWhatIsWrittenDown:
    async def test_the_handoff_is_recorded_with_who_why_and_what_crossed(self) -> None:
        handing = desk(answer())
        await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        recorded = handing.handoffs[0]
        assert (recorded.from_agent, recorded.to_agent) == ("triage", "billing")
        assert recorded.reason == "a billing question"
        assert recorded.scope == ("read_account", "issue_credit")

    async def test_the_chain_reads_as_one_trace(self) -> None:
        handing = desk(answer(), answer())
        await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        await handing.hand_off(
            "shipping", reason="a delivery question", state=ticket(), task="find it"
        )
        assert [one.trace for one in handing.handoffs] == [
            ("triage", "billing"),
            ("triage", "shipping"),
        ]
        assert {one.run_id for one in handing.handoffs} == {"run_1"}

    async def test_a_handoff_and_a_refusal_are_both_events(self) -> None:
        handing = desk(answer())
        await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        with pytest.raises(HandoffContractError):
            await handing.hand_off(
                "billing", reason="a billing question", state={"account": "ac_9"}, task="sort it"
            )
        assert [one.kind for one in handing.events] == [
            RunEventKind.HANDED_OFF,
            RunEventKind.HANDOFF_REFUSED,
        ]

    async def test_what_the_target_spent_is_attributed_to_the_target(self) -> None:
        handing = desk(answer())
        await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert handing.events[0].usage is not None
        assert handing.events[0].usage.input_tokens == 10


class TestWhatMustNotReachTheTarget:
    async def test_the_payload_passes_the_guardrail_chain_before_it_crosses(self) -> None:
        guard = FakeGuardrail("no_pii")
        handing = desk(answer(), guardrails=GuardrailPipeline([guard]))
        await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert guard.checked
        assert "charged twice in March" in guard.checked[0]

    async def test_a_guardrail_that_blocks_it_stops_the_handoff(self) -> None:
        blocking = GuardrailPipeline([FakeGuardrail("no_pii", allow=False)])
        handing = desk(answer(), guardrails=blocking)
        with pytest.raises(GuardrailError):
            await handing.hand_off(
                "billing", reason="a billing question", state=ticket(), task="sort it"
            )
        assert handing.handoffs == ()

    async def test_what_is_recorded_is_what_the_guardrail_left(self) -> None:
        """Redaction happens before telemetry sees the payload, not after."""
        redacting = GuardrailPipeline([FakeGuardrail("no_pii", redacts="[redacted]")])
        handing = desk(answer(), guardrails=redacting)
        result = await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        assert "March" not in result.handoff.rendered


class TestHandingOffFromARunThatIsStillGoing:
    async def test_a_source_run_that_has_not_settled_is_refused(self) -> None:
        """A call still in flight has to finish or be abandoned before the conversation moves."""
        handing = desk(answer())
        unsettled: Run[Any] = _running()
        with pytest.raises(HandoffContractError) as raised:
            await handing.hand_off(
                "billing",
                reason="a billing question",
                state=ticket(),
                task="sort it",
                after=unsettled,
            )
        assert raised.value.reason == "in_flight"

    async def test_a_settled_source_run_hands_over(self) -> None:
        handing = desk(answer(), answer())
        first = await handing.hand_off(
            "billing", reason="a billing question", state=ticket(), task="sort it"
        )
        result = await handing.hand_off(
            "shipping",
            reason="a delivery question",
            state=ticket(),
            task="find it",
            after=first.run,
        )
        assert result.run is not None
        assert result.run.state is RunState.COMPLETED


def _running() -> Run[Any]:
    """A run that has not reached a terminal state, as one mid-tool-call would be."""
    return CoreRun(
        id="run_2",
        tenant="acme",
        user="ada",
        agent_name="triage",
        agent_version="1.0.0",
        model="claude-sonnet-5",
        prompt_version="p1",
        state=RunState.RUNNING,
    )
