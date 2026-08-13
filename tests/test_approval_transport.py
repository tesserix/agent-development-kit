"""Who is asked, over what, and what happens when nobody answers.

The gate itself is settled elsewhere: this file is about the wire between a held call and
the person who decides about it. The failures it exists to prevent are a run that waits for
ever because the queue is down, an answer replayed into a second execution, and a service
account approving its own agent's payment.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.approvals import (
    ConsoleApprovals,
    NatsApprovals,
    WebhookApprovals,
)
from tesserix_adk.core import (
    Agent,
    ApprovalDecision,
    ApprovalDeliveryError,
    ApprovalGate,
    ApprovalRecord,
    ApprovalTransport,
    ConfigurationError,
    ModelCapabilities,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse, TransportGate, self_granted
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

if TYPE_CHECKING:
    from collections.abc import Mapping

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)
NOW = 1_000.0

type Posted = tuple[str, bytes, "Mapping[str, str]"]


class Answering:
    """A transport that carries the answer back in its own reply."""

    def __init__(self, *, granted: bool = True, by: str = "ada") -> None:
        self.granted = granted
        self.by = by
        self.delivered: list[ApprovalRecord] = []

    async def deliver(self, record: ApprovalRecord) -> ApprovalDecision:
        self.delivered.append(record)
        return ApprovalDecision(
            record_id=record.id, granted=self.granted, decided_by=self.by, decided_at=NOW
        )


class Posting:
    """A transport that only puts the question somewhere; the answer arrives elsewhere."""

    def __init__(self) -> None:
        self.delivered: list[ApprovalRecord] = []

    async def deliver(self, record: ApprovalRecord) -> None:
        self.delivered.append(record)


class Down:
    """A transport that cannot reach anybody."""

    async def deliver(self, record: ApprovalRecord) -> None:  # noqa: ARG002 — never arrives
        raise RuntimeError("approval queue unreachable")


def record(tool_name: str = "wire_funds", agent: str = "planner") -> ApprovalRecord:
    """One held call, as it goes onto the wire."""
    return ApprovalRecord.for_call(
        run_id="run_1",
        tenant="acme",
        agent_name=agent,
        tool_name=tool_name,
        arguments={"amount": 500, "iban": "GB33BUKB20201555555555"},
        reason=f"{tool_name} is declared to require approval",
        requested_at=NOW,
    )


class TestAskingOverATransport:
    """The gate holds the call; the transport only decides where the question goes."""

    async def test_a_transport_that_answers_in_its_reply_is_the_decision(self) -> None:
        transport = Answering()

        decision = await TransportGate(transport).request(record())

        assert decision.granted
        assert [held.tool_name for held in transport.delivered] == ["wire_funds"]

    async def test_the_question_carries_a_digest_rather_than_the_arguments(self) -> None:
        """A queue outlives the run and is read by people who are not party to it."""
        transport = Answering()

        await TransportGate(transport).request(record())

        assert "GB33BUKB20201555555555" not in transport.delivered[0].summary

    async def test_a_transport_that_answers_out_of_band_holds_the_call(self) -> None:
        gate = TransportGate(Posting())
        held = record()
        asked = asyncio.create_task(gate.request(held))
        await _until(lambda: gate.waiting == (held.id,))

        assert gate.decide(_grant(held))

        assert (await asked).granted

    async def test_a_transport_that_cannot_deliver_never_becomes_a_grant(self) -> None:
        with pytest.raises(RuntimeError, match="unreachable"):
            await TransportGate(Down()).request(record())

    def test_the_gate_is_an_approval_gate_by_shape(self) -> None:
        assert isinstance(TransportGate(Posting()), ApprovalGate)
        assert isinstance(Posting(), ApprovalTransport)


class TestWhenNobodyAnswers:
    """Silence is a refusal. A gate that opens on a timeout is not a gate."""

    async def test_a_wait_that_runs_out_is_a_denial(self) -> None:
        clock = FakeClock(start=NOW, auto_advance=False)
        gate = TransportGate(Posting(), clock=clock, wait_seconds=60.0)
        asked = asyncio.create_task(gate.request(record()))
        await _until(lambda: bool(clock.slept))

        clock.advance(60.0)
        decision = await asked

        assert not decision.granted
        assert "60" in decision.reason

    async def test_the_timeout_denial_names_the_gate_rather_than_a_person(self) -> None:
        """Nobody decided this, and the audit trail must not read as though somebody did."""
        clock = FakeClock(start=NOW, auto_advance=False)
        gate = TransportGate(Posting(), clock=clock, wait_seconds=30.0)
        asked = asyncio.create_task(gate.request(record()))
        await _until(lambda: bool(clock.slept))

        clock.advance(30.0)
        decision = await asked

        assert decision.decided_by == "system:timeout"
        assert decision.decided_at == NOW + 30.0

    async def test_a_gate_nobody_gave_a_clock_gives_up_on_the_real_one(self) -> None:
        """The default wall clock is the one production runs on, so it has to time out too."""
        gate = TransportGate(Posting(), wait_seconds=0.0)

        decision = await gate.request(record())

        assert not decision.granted
        assert decision.decided_by == "system:timeout"

    async def test_a_call_nobody_answered_is_no_longer_waiting(self) -> None:
        clock = FakeClock(start=NOW, auto_advance=False)
        gate = TransportGate(Posting(), clock=clock, wait_seconds=30.0)
        asked = asyncio.create_task(gate.request(record()))
        await _until(lambda: bool(clock.slept))

        clock.advance(30.0)
        await asked

        assert gate.waiting == ()


class TestAnAnswerNobodyIsWaitingFor:
    """One decision is one execution, and a decision for a finished run is nothing."""

    async def test_a_second_answer_to_the_same_request_changes_nothing(self) -> None:
        gate = TransportGate(Posting())
        held = record()
        asked = asyncio.create_task(gate.request(held))
        await _until(lambda: bool(gate.waiting))
        gate.decide(_grant(held))
        await asked

        assert not gate.decide(_grant(held, by="mallory"))

    async def test_an_answer_to_a_request_nobody_raised_is_stale(self) -> None:
        assert not TransportGate(Posting()).decide(_grant(record()))

    async def test_an_answer_after_the_run_was_cancelled_is_stale(self) -> None:
        gate = TransportGate(Posting())
        held = record()
        asked = asyncio.create_task(gate.request(held))
        await _until(lambda: bool(gate.waiting))

        gate.abandon(held.id, why="the run was cancelled")
        with pytest.raises(asyncio.CancelledError):
            await asked

        assert not gate.decide(_grant(held))

    async def test_abandoning_a_call_nobody_holds_is_not_an_error(self) -> None:
        TransportGate(Posting()).abandon("nothing", why="the run was cancelled")


class TestAnApproverWhoIsNotTheAgent:
    """A service account approving its own agent's payment is not a second pair of eyes."""

    def test_a_decision_naming_the_asking_agent_is_self_granted(self) -> None:
        held = record()

        assert self_granted(held, _grant(held, by="planner"))
        assert self_granted(held, _grant(held, by="agent:planner"))
        assert self_granted(held, _grant(held, by="service/Planner "))

    def test_a_decision_naming_a_person_is_not(self) -> None:
        held = record()

        assert not self_granted(held, _grant(held, by="ada"))
        assert not self_granted(held, _grant(held, by="planner@acme.example"))

    async def test_a_grant_from_the_agents_own_identity_is_refused(self) -> None:
        called, run = await _run(
            TransportGate(Answering(by="agent:planner")),
            _calling("wire_funds", amount=500),
            _answering(),
        )

        assert called == []
        assert run.state is RunState.COMPLETED
        assert "approval_self_granted" in (_first(run, RunEventKind.TOOL_REFUSED).detail or "")

    async def test_a_denial_from_the_gate_itself_is_still_a_denial(self) -> None:
        """Only a grant needs a second pair of eyes; a refusal by the gate is one."""
        called, run = await _run(
            TransportGate(Answering(granted=False, by="system:timeout")),
            _calling("wire_funds", amount=500),
            _answering(),
        )

        assert called == []
        assert "approval_denied" in (_first(run, RunEventKind.TOOL_REFUSED).detail or "")


class TestAToolNobodyMayCall:
    """Refuse before troubling a human: an unreachable tool is not an approval question."""

    async def test_a_tool_outside_the_allowlist_never_reaches_the_gate(self) -> None:
        transport = Answering()
        called, run = await _run(
            TransportGate(transport), _calling("wire_gold", amount=500), _answering()
        )

        assert transport.delivered == []
        assert called == []
        assert run.state is RunState.FAILED


class TestOverNats:
    """The question is published per tenant, so a subscriber can be scoped to its own."""

    async def test_the_record_is_published_on_the_tenants_subject(self) -> None:
        published: list[tuple[str, bytes]] = []

        async def publish(subject: str, payload: bytes) -> None:
            published.append((subject, payload))

        held = record()
        await NatsApprovals(_publisher(publish)).deliver(held)

        subject, payload = published[0]
        assert subject == "adk.approvals.acme"
        assert json.loads(payload)["id"] == held.id

    async def test_a_tenant_that_could_widen_the_subject_is_refused(self) -> None:
        held = record().model_copy(update={"tenant": "acme.>"})

        with pytest.raises(ConfigurationError, match="subject token"):
            await NatsApprovals(_publisher(_nothing)).deliver(held)

    async def test_the_subject_root_is_checked_when_it_is_configured(self) -> None:
        with pytest.raises(ConfigurationError, match="subject token"):
            NatsApprovals(_publisher(_nothing), subject="adk approvals")


class TestOverAWebhook:
    """A signed POST, and an answer only where the receiver actually gave one."""

    async def test_the_body_is_signed_so_the_receiver_can_tell_who_asked(self) -> None:
        posts: list[Posted] = []

        assert await _webhook(202, b"", posts).deliver(record()) is None

        url, content, headers = posts[0]
        assert url == "https://desk.example/approvals"
        assert headers["X-Adk-Signature"].startswith("sha256=")
        assert json.loads(content)["tenant"] == "acme"

    async def test_the_signature_covers_the_body_that_was_sent(self) -> None:
        posts: list[Posted] = []

        await _webhook(202, b"", posts).deliver(record())

        _, content, headers = posts[0]
        expected = hmac.new(b"s3cret", content, hashlib.sha256).hexdigest()
        assert headers["X-Adk-Signature"] == f"sha256={expected}"

    async def test_a_receiver_that_answers_in_the_response_is_the_decision(self) -> None:
        answer = _grant(record()).model_dump_json().encode()

        decision = await _webhook(200, answer).deliver(record())

        assert decision is not None
        assert decision.granted

    async def test_an_answer_about_another_request_is_refused(self) -> None:
        answer = _grant(record(tool_name="issue_refund")).model_dump_json().encode()

        with pytest.raises(ApprovalDeliveryError, match="a different request"):
            await _webhook(200, answer).deliver(record())

    async def test_a_receiver_that_refused_the_post_is_not_silence(self) -> None:
        with pytest.raises(ApprovalDeliveryError, match="500"):
            await _webhook(500, b"nope").deliver(record())

    async def test_a_body_that_is_not_a_decision_is_not_read_as_one(self) -> None:
        with pytest.raises(ApprovalDeliveryError, match="not a decision"):
            await _webhook(200, b"{}").deliver(record())

    def test_a_url_that_is_not_https_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="https"):
            WebhookApprovals(
                _poster(_nothing), url="http://desk.example/approvals", secret=SecretStr("s3cret")
            )


class TestFromTheCommandLine:
    """The desk at a terminal, for a single operator and for a local run."""

    async def test_an_operator_who_types_yes_grants_it(self) -> None:
        shown: list[str] = []
        console = ConsoleApprovals(
            approver="ada", ask=lambda: "y\n", show=shown.append, clock=FakeClock(start=NOW)
        )

        decision = await console.deliver(record())

        assert decision is not None
        assert decision.granted
        assert decision.decided_by == "ada"
        assert any("wire_funds" in line for line in shown)

    async def test_anything_that_is_not_a_yes_is_a_no(self) -> None:
        console = ConsoleApprovals(approver="ada", ask=lambda: "", show=_nothing_shown)

        decision = await console.deliver(record())

        assert decision is not None
        assert not decision.granted

    async def test_what_is_shown_carries_no_argument_values(self) -> None:
        shown: list[str] = []
        console = ConsoleApprovals(approver="ada", ask=lambda: "n", show=shown.append)

        await console.deliver(record())

        assert not any("GB33BUKB20201555555555" in line for line in shown)

    async def test_the_question_goes_to_stdout_when_nowhere_else_is_named(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        console = ConsoleApprovals(approver="ada", ask=lambda: "n")

        await console.deliver(record())

        assert "wire_funds" in capsys.readouterr().out

    def test_an_approver_nobody_can_name_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="approver"):
            ConsoleApprovals(approver="  ", ask=lambda: "y", show=_nothing_shown)


def _grant(held: ApprovalRecord, *, by: str = "ada") -> ApprovalDecision:
    return ApprovalDecision(record_id=held.id, granted=True, decided_by=by, decided_at=NOW)


def _nothing_shown(line: str) -> None:
    """Swallow what a console approval would print."""


async def _nothing(*args: Any, **kwargs: Any) -> Any:
    """Stand in for a client no test in this class reaches."""


def _publisher(publish: Any) -> Any:
    """A NATS-shaped client with only the method the adapter uses."""
    return type("Publisher", (), {"publish": staticmethod(publish)})()


def _webhook(status: int, answer: bytes, posts: list[Posted] | None = None) -> WebhookApprovals:
    """A webhook over a receiver that answers as the test says, recording what it was sent."""

    async def post(url: str, *, content: bytes, headers: Mapping[str, str]) -> tuple[int, bytes]:
        if posts is not None:
            posts.append((url, content, headers))
        return status, answer

    return WebhookApprovals(
        _poster(post), url="https://desk.example/approvals", secret=SecretStr("s3cret")
    )


def _poster(post: Any) -> Any:
    """An HTTP-shaped client with only the method the adapter uses."""
    return type("Poster", (), {"post": staticmethod(post)})()


async def _until(ready: Any, *, tries: int = 100) -> None:
    """Yield to the loop until the gate has actually parked the call."""
    for _ in range(tries):
        if ready():
            return
        await asyncio.sleep(0)
    raise AssertionError("the call was never held")


def _first(run: Run[Any], kind: RunEventKind) -> Any:
    return next(event for event in run.events if event.kind is kind)


async def _run(approvals: Any, *responses: ModelResponse) -> tuple[list[dict[str, Any]], Run[Any]]:
    """A run over a money-moving tool, returning what actually executed."""
    called: list[dict[str, Any]] = []

    @tool(requires_approval=True)
    async def wire_funds(amount: int) -> str:
        """Move money.

        Args:
            amount: How much.
        """
        called.append({"amount": amount})
        return f"sent {amount}"

    registry = ToolRegistry((wire_funds,), clock=FakeClock())
    runner = AgentRunner(
        provider=ScriptedProvider(*responses, capabilities=CAPABLE),
        clock=FakeClock(),
        tools=registry.view(allow=("wire_funds",), agent="planner"),
        approvals=approvals,
    )
    agent: Agent[Any] = Agent(
        name="planner",
        instructions="Settle the booking.",
        free_text=True,
        model="scripted-1",
        tools=("wire_funds",),
        approval_required_tools=("wire_funds",),
    )
    try:
        return called, await runner.run(agent, "settle it", tenant="acme", run_id="run_1")
    finally:
        wire_funds.release()


def _calling(name: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id=f"call_{name}", name=name, arguments=arguments),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _answering() -> ModelResponse:
    return ModelResponse(content="Done.", usage=Usage(input_tokens=1, output_tokens=1))
