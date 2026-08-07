"""Putting a run on the wire: framing, resume, authorisation and control messages.

Every product that has written this bridge by hand has written it slightly differently, and
the disagreements are not cosmetic: one reconnect semantics loses events silently, another
leaks a raw provider payload because redaction lived in the UI. The transports here frame
the same typed events either way, fail closed on a client-supplied run id, and say
explicitly when something was missed rather than closing the gap quietly.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.adapters import (
    SSE_HEADERS,
    ApprovalInbox,
    PayloadElided,
    RunBroker,
    StreamGap,
    TransportAuthorizationError,
    WebSocketBridge,
    sse_events,
)
from tesserix_adk.core import (
    Agent,
    ApprovalDecision,
    ApprovalRecord,
    NoOutput,
    RunState,
    TextPart,
    Usage,
)
from tesserix_adk.runtime import (
    AgentRunner,
    AnswerDelta,
    ModelResponse,
    ProgressEvent,
    RunStream,
    ToolCallStarted,
)
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider, StallingProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

AGENT = Agent(name="chatter", instructions="Chat.", model="claude-sonnet-5", free_text=True)


def answer(text: str = "Kyoto, four nights.") -> ModelResponse:
    return ModelResponse(content=text, usage=Usage(input_tokens=10, output_tokens=5))


def stream_of(
    *responses: ModelResponse, run_id: str = "run_1", **overrides: object
) -> RunStream[NoOutput]:
    fields: dict[str, object] = {
        "provider": ScriptedProvider(*responses, capabilities=CAPABLE),
        "clock": FakeClock(),
    }
    runner = AgentRunner(**{**fields, **overrides})  # type: ignore[arg-type]
    return runner.stream(AGENT, "plan a trip", tenant="acme", run_id=run_id)


async def collected(events: AsyncIterator[ProgressEvent]) -> list[ProgressEvent]:
    return [event async for event in events]


async def texted(frames: AsyncIterator[str]) -> list[str]:
    return [frame async for frame in frames]


class Socket:
    """A websocket peer that says what it was told and answers from a script."""

    def __init__(self, *inbound: str, vanishes: bool = False) -> None:
        self._inbound = list(inbound)
        self._vanishes = vanishes
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        """Record what the bridge framed."""
        self.sent.append(data)

    async def receive_text(self) -> str:
        """The next scripted message, or a peer that never says anything again."""
        if self._inbound:
            return self._inbound.pop(0)
        if self._vanishes:
            raise ConnectionResetError("the peer went away without a close frame")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def close(self, code: int = 1000) -> None:
        """Note that the bridge closed the connection."""
        del code
        self.closed = True


class TestFraming:
    async def test_every_event_is_framed_with_its_kind_and_sequence(self) -> None:
        """`event:` and `id:` are what let a browser dispatch and resume without parsing."""
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        frames = await texted(sse_events(broker.subscribe("run_1", tenant="acme")))
        first = [line for line in frames if line.startswith("event: run_started")]
        assert first
        assert "id: 0\n" in "".join(frames)

    async def test_a_frame_carries_the_event_as_json_on_one_data_line(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        frames = await texted(sse_events(broker.subscribe("run_1", tenant="acme")))
        payloads = [
            json.loads(line.removeprefix("data: "))
            for frame in frames
            for line in frame.splitlines()
            if line.startswith("data: ")
        ]
        assert payloads[0]["kind"] == "run_started"
        assert payloads[-1]["kind"] == "run_completed"

    async def test_the_stream_opens_with_a_reconnection_hint(self) -> None:
        """Without `retry`, every browser picks its own reconnect delay."""
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        frames = await texted(
            sse_events(broker.subscribe("run_1", tenant="acme"), retry_milliseconds=2500)
        )
        assert frames[0] == "retry: 2500\n\n"

    async def test_an_idle_stream_is_kept_open_by_heartbeats(self) -> None:
        """A proxy closes a connection that goes quiet, mid-run, with no error anywhere."""
        provider = StallingProvider(capabilities=CAPABLE)
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(provider=provider), tenant="acme")
        async with aclosing(
            sse_events(broker.subscribe("run_1", tenant="acme"), heartbeat_seconds=0.01)
        ) as frames:
            seen = [await anext(frames) for _ in range(4)]
        await broker.cancel("run_1", tenant="acme")
        assert ": heartbeat\n\n" in seen

    def test_the_documented_headers_defeat_proxy_buffering(self) -> None:
        assert SSE_HEADERS["Content-Type"] == "text/event-stream"
        assert SSE_HEADERS["X-Accel-Buffering"] == "no"
        assert "no-transform" in SSE_HEADERS["Cache-Control"]

    async def test_a_stream_that_speaks_again_after_a_quiet_spell_is_still_framed(self) -> None:
        """A heartbeat is a pause in the stream, not the end of it."""

        async def slow() -> AsyncIterator[ProgressEvent]:
            await asyncio.sleep(0.03)
            yield AnswerDelta(run_id="run_1", sequence=1, text="late")

        frames = await texted(sse_events(slow(), heartbeat_seconds=0.01))
        assert ": heartbeat\n\n" in frames
        assert frames[-1].startswith("event: answer_delta\nid: 1\n")


class TestBothTransportsAgree:
    async def test_the_payloads_are_identical_over_sse_and_websocket(self) -> None:
        """A client that switches transport must not see a different run."""
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        over_sse = [
            json.loads(line.removeprefix("data: "))
            for frame in await texted(sse_events(broker.subscribe("run_1", tenant="acme")))
            for line in frame.splitlines()
            if line.startswith("data: ")
        ]

        socket = Socket()
        broker.register(stream_of(answer(), run_id="run_2"), tenant="acme")
        await WebSocketBridge(broker).serve(socket, run_id="run_2", tenant="acme")
        over_ws = [json.loads(text) for text in socket.sent]

        assert [event["kind"] for event in over_sse] == [event["kind"] for event in over_ws]


class TestResuming:
    async def test_a_client_that_reconnects_receives_what_it_missed(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        whole = await collected(broker.subscribe("run_1", tenant="acme"))
        resumed = await collected(broker.subscribe("run_1", tenant="acme", after=1))
        assert [event.sequence for event in resumed] == [
            event.sequence for event in whole if event.sequence > 1
        ]

    async def test_a_client_that_missed_more_than_was_kept_is_told_so(self) -> None:
        """Silently closing the gap is how a UI ends up showing a run that never happened."""
        broker = RunBroker[NoOutput](history=2)
        broker.register(stream_of(answer()), tenant="acme")
        await collected(broker.subscribe("run_1", tenant="acme"))
        resumed = await collected(broker.subscribe("run_1", tenant="acme", after=0))
        assert isinstance(resumed[0], StreamGap)
        assert resumed[0].missing > 0

    async def test_reconnecting_after_the_run_ended_gives_the_terminal_event(self) -> None:
        """An empty stream reads as a run still going, which it is not."""
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        whole = await collected(broker.subscribe("run_1", tenant="acme"))
        again = await collected(broker.subscribe("run_1", tenant="acme", after=whole[-2].sequence))
        assert [event.kind for event in again] == ["run_completed"]


class TestTheBoundaryFailsClosed:
    async def test_another_tenant_cannot_subscribe_to_a_run(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        with pytest.raises(TransportAuthorizationError):
            await collected(broker.subscribe("run_1", tenant="rival"))

    async def test_a_run_id_nobody_registered_is_refused_rather_than_reported(self) -> None:
        """Telling a client which run ids exist is telling it about another tenant's runs."""
        with pytest.raises(TransportAuthorizationError):
            await collected(RunBroker[NoOutput]().subscribe("run_9", tenant="acme"))

    async def test_another_tenant_cannot_cancel_a_run(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(provider=StallingProvider(capabilities=CAPABLE)), tenant="acme")
        with pytest.raises(TransportAuthorizationError):
            await broker.cancel("run_1", tenant="rival")
        await broker.cancel("run_1", tenant="acme")

    async def test_a_cancel_message_for_another_tenant_s_run_is_refused(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(provider=StallingProvider(capabilities=CAPABLE)), tenant="acme")
        socket = Socket(json.dumps({"type": "cancel", "run_id": "run_1"}))
        with pytest.raises(TransportAuthorizationError):
            await WebSocketBridge(broker).serve(socket, run_id="run_1", tenant="rival")
        assert not socket.sent
        await broker.cancel("run_1", tenant="acme")


class TestControlMessages:
    async def test_a_cancel_message_ends_the_run(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(provider=StallingProvider(capabilities=CAPABLE)), tenant="acme")
        socket = Socket(json.dumps({"type": "cancel", "run_id": "run_1"}))
        await WebSocketBridge(broker).serve(socket, run_id="run_1", tenant="acme")
        assert broker.run("run_1", tenant="acme").state is RunState.CANCELLED

    async def test_an_approval_decision_reaches_the_waiting_run(self) -> None:
        inbox = ApprovalInbox()
        record = ApprovalRecord.for_call(
            run_id="run_1",
            tenant="acme",
            agent_name="chatter",
            tool_name="refund",
            arguments={"amount": 10},
            reason="spends money",
        )
        held = asyncio.ensure_future(inbox.request(record))
        await inbox.wait_for(record.id)
        inbox.decide(
            ApprovalDecision(record_id=record.id, granted=True, decided_by="ops"), tenant="acme"
        )
        assert (await held).granted

    async def test_an_approval_for_another_tenant_is_refused(self) -> None:
        inbox = ApprovalInbox()
        record = ApprovalRecord.for_call(
            run_id="run_1",
            tenant="acme",
            agent_name="chatter",
            tool_name="refund",
            arguments={"amount": 10},
            reason="spends money",
        )
        held = asyncio.ensure_future(inbox.request(record))
        await inbox.wait_for(record.id)
        with pytest.raises(TransportAuthorizationError):
            inbox.decide(
                ApprovalDecision(record_id=record.id, granted=True, decided_by="thief"),
                tenant="rival",
            )
        inbox.decide(
            ApprovalDecision(record_id=record.id, granted=False, decided_by="ops"), tenant="acme"
        )
        assert not (await held).granted

    async def test_an_unknown_control_message_is_ignored_rather_than_fatal(self) -> None:
        """A newer client sending a message this version has never heard of stays connected."""
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(provider=StallingProvider(capabilities=CAPABLE)), tenant="acme")
        socket = Socket(
            json.dumps({"type": "telemetry", "fps": 60}),
            "not json at all",
            json.dumps(["cancel"]),
            json.dumps({"type": "cancel"}),
        )
        await WebSocketBridge(broker).serve(socket, run_id="run_1", tenant="acme")
        assert socket.closed
        assert broker.run("run_1", tenant="acme").state is RunState.CANCELLED

    async def test_an_approval_decision_arrives_over_the_connection(self) -> None:
        """The point of the inbox: the person deciding is on the other end of the socket."""
        inbox = ApprovalInbox()
        record = ApprovalRecord.for_call(
            run_id="run_1",
            tenant="acme",
            agent_name="chatter",
            tool_name="refund",
            arguments={"amount": 10},
            reason="spends money",
        )
        held = asyncio.ensure_future(inbox.request(record))
        await inbox.wait_for(record.id)
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(provider=StallingProvider(capabilities=CAPABLE)), tenant="acme")
        decision = ApprovalDecision(record_id=record.id, granted=True, decided_by="ops")
        socket = Socket(
            json.dumps({"type": "approval", "decision": decision.model_dump(mode="json")}),
            json.dumps({"type": "cancel"}),
        )
        bridge = WebSocketBridge(broker, approvals=inbox)
        await bridge.serve(socket, run_id="run_1", tenant="acme")
        assert (await held).granted

    async def test_deciding_twice_does_not_disturb_the_run(self) -> None:
        """Two reviewers clicking approve is a race, not an error."""
        inbox = ApprovalInbox()
        record = ApprovalRecord.for_call(
            run_id="run_1",
            tenant="acme",
            agent_name="chatter",
            tool_name="refund",
            arguments={"amount": 10},
            reason="spends money",
        )
        held = asyncio.ensure_future(inbox.request(record))
        await inbox.wait_for(record.id)
        first = ApprovalDecision(record_id=record.id, granted=True, decided_by="ops")
        inbox.decide(first, tenant="acme")
        inbox.decide(
            ApprovalDecision(record_id=record.id, granted=False, decided_by="ops2"), tenant="acme"
        )
        assert (await held).decided_by == "ops"

    async def test_a_peer_that_vanishes_cancels_the_run(self) -> None:
        """A run nobody is connected to still calls providers and still bills."""
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(provider=StallingProvider(capabilities=CAPABLE)), tenant="acme")
        await WebSocketBridge(broker).serve(Socket(vanishes=True), run_id="run_1", tenant="acme")
        assert broker.run("run_1", tenant="acme").state is RunState.CANCELLED


class TestWhatNeverReachesTheWire:
    async def test_an_oversized_event_is_referenced_rather_than_truncated(self) -> None:
        """Half a JSON document is not a smaller JSON document."""
        huge = ToolCallStarted(
            run_id="run_1",
            sequence=4,
            call_id="c1",
            tool="ledger",
            arguments=json.dumps({"rows": ["entry " * 8] * 40}),
        )
        framed = "".join(await texted(sse_events(_replayed([huge]), payload_limit_bytes=512)))
        payload = json.loads(
            next(line for line in framed.splitlines() if line.startswith("data: ")).removeprefix(
                "data: "
            )
        )
        assert payload["kind"] == "payload_elided"
        assert payload["elided"] == "tool_call_started"
        assert payload["reference"] == "run_1:4"
        assert payload["size_bytes"] > 512

    async def test_a_credential_shaped_value_is_masked_at_the_boundary(self) -> None:
        """The runtime scrubs too. A boundary that trusts its input is not a boundary."""
        leaky = ToolCallStarted(
            run_id="run_1",
            sequence=3,
            call_id="c1",
            tool="lookup",
            arguments='{"token": "sk-live-01234567890abcdef"}',  # gitleaks:allow
        )
        framed = "".join(await texted(sse_events(_replayed([leaky]))))
        assert "sk-live-01234567890abcdef" not in framed
        assert "[redacted]" in framed


class TestPayloadElision:
    def test_an_elision_says_what_it_replaced_and_how_big_it_was(self) -> None:
        elided = PayloadElided(
            run_id="run_1", sequence=4, elided="answer_delta", size_bytes=9001, reference="run_1:4"
        )
        assert elided.kind == "payload_elided"
        assert elided.size_bytes == 9001


async def _replayed(events: list[ProgressEvent] | list[Any]) -> AsyncIterator[ProgressEvent]:
    for event in events:
        yield event


class TestOneRunDrivenOnce:
    async def test_two_subscribers_see_the_same_run(self) -> None:
        broker = RunBroker[NoOutput]()
        provider = ScriptedProvider(answer(), capabilities=CAPABLE)
        broker.register(stream_of(provider=provider), tenant="acme")
        both = await asyncio.gather(
            collected(broker.subscribe("run_1", tenant="acme")),
            collected(broker.subscribe("run_1", tenant="acme")),
        )
        assert [event.sequence for event in both[0]] == [event.sequence for event in both[1]]
        assert len(provider.requests) == 1

    async def test_the_finished_run_is_readable_through_the_broker(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        await collected(broker.subscribe("run_1", tenant="acme"))
        run = broker.run("run_1", tenant="acme")
        assert run.state is RunState.COMPLETED
        part = run.messages[-1].content[0]
        assert isinstance(part, TextPart)
        assert part.text == "Kyoto, four nights."

    async def test_the_deltas_still_reassemble_to_the_answer(self) -> None:
        broker = RunBroker[NoOutput]()
        broker.register(stream_of(answer()), tenant="acme")
        events = await collected(broker.subscribe("run_1", tenant="acme"))
        deltas = [event.text for event in events if isinstance(event, AnswerDelta)]
        assert "".join(deltas) == "Kyoto, four nights."
