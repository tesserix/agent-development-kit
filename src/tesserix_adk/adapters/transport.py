"""Putting a run on the wire, over SSE or a websocket, without each product inventing it.

The bridge from an agent stream to a browser has been written by hand many times, and the
hand-written versions disagree about the things that matter: what a reconnect means, how a
gap is reported, where redaction happens. Here they agree, because both transports frame
the same typed events and both go through the same boundary.

The boundary fails closed. A run id arrives from a client, so it is checked against the
tenant that registered the run before anything is framed or cancelled — an unknown id is
refused exactly as another tenant's is, because saying which ids exist is saying something
about another tenant's runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, Field

from tesserix_adk.core import AdkError, ApprovalDecision, scrub
from tesserix_adk.runtime import ProgressEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping

    from tesserix_adk.core import ApprovalRecord, Run
    from tesserix_adk.runtime import RunStream

__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "DEFAULT_PAYLOAD_LIMIT_BYTES",
    "DEFAULT_RETRY_MILLISECONDS",
    "SSE_HEADERS",
    "ApprovalInbox",
    "PayloadElided",
    "RunBroker",
    "StreamGap",
    "TransportAuthorizationError",
    "WebSocketBridge",
    "WebSocketLike",
    "sse_events",
]

DEFAULT_HEARTBEAT_SECONDS = 15.0
DEFAULT_RETRY_MILLISECONDS = 3000
DEFAULT_PAYLOAD_LIMIT_BYTES = 64 * 1024
DEFAULT_HISTORY = 512

SSE_HEADERS: Mapping[str, str] = {
    "Content-Type": "text/event-stream",
    # `no-transform` stops a compressing proxy from buffering the whole body first.
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_HEARTBEAT = ": heartbeat\n\n"


class TransportAuthorizationError(AdkError):
    """Raised when a client asked about a run that is not its tenant's to ask about."""


class StreamGap(ProgressEvent):
    """Events a reconnecting client will never receive, said out loud.

    A gap closed quietly is a UI rendering a run that did not happen. This says how much is
    missing and where the stream picks up, so a client can refetch the run rather than
    believe what it has.

    Args:
        missing: How many events fell out of the buffer before the client came back.
        resumed_from: The first sequence number it will now receive.
    """

    kind: Literal["stream_gap"] = "stream_gap"
    missing: int = Field(ge=1)
    resumed_from: int = Field(ge=0)


class PayloadElided(ProgressEvent):
    """An event too large for the transport, referenced rather than cut in half.

    Truncating JSON produces a document that does not parse; truncating a tool result
    produces one that does parse and is wrong. Neither goes on the wire.

    Args:
        elided: The `kind` of the event this replaces.
        size_bytes: How large the framed payload would have been.
        reference: `run_id:sequence` — what to fetch the full payload by.
    """

    kind: Literal["payload_elided"] = "payload_elided"
    elided: str
    size_bytes: int = Field(ge=0)
    reference: str


class _Room[OutputT: BaseModel]:
    """One registered run: who owns it, what it has said, and who is listening."""

    def __init__(self, stream: RunStream[OutputT], tenant: str, history: int) -> None:
        self.stream = stream
        self.tenant = tenant
        self.events: deque[ProgressEvent] = deque(maxlen=history)
        self.evicted = 0
        self.done = False
        self.listeners: set[asyncio.Queue[ProgressEvent | None]] = set()
        self.task: asyncio.Task[None] | None = None

    def post(self, event: ProgressEvent) -> None:
        """Keep the event for a client that is not here yet, and give it to those who are."""
        if len(self.events) == self.events.maxlen:
            self.evicted += 1
        self.events.append(event)
        for listener in self.listeners:
            listener.put_nowait(event)

    def finish(self) -> None:
        """Tell every listener the run is over."""
        self.done = True
        for listener in self.listeners:
            listener.put_nowait(None)


class RunBroker[OutputT: BaseModel]:
    """One run, driven once, readable by however many transports attach to it.

    A reconnect is a second reader of a run already in flight, so the run cannot be tied to
    the connection that started it. The broker owns the driving; a transport only reads.

    Args:
        history: How many events to keep per run for a client that reconnects. Beyond this
            a resuming client is told what it missed rather than quietly given less.
    """

    def __init__(self, *, history: int = DEFAULT_HISTORY) -> None:
        self._history = history
        self._rooms: dict[str, _Room[OutputT]] = {}

    def register(self, stream: RunStream[OutputT], *, tenant: str) -> str:
        """Take ownership of `stream` on behalf of `tenant`.

        The run starts when the first transport attaches, not here, so a registration
        nobody ever connects to costs nothing and misses nothing.

        Returns:
            The run id a client subscribes and sends control messages by.
        """
        self._rooms[stream.run_id] = _Room(stream, tenant, self._history)
        return stream.run_id

    def _driving(self, room: _Room[OutputT]) -> None:
        """Make sure the run is being driven, exactly once, however many are listening."""
        if room.task is None:
            room.task = asyncio.create_task(self._drain(room))

    async def _drain(self, room: _Room[OutputT]) -> None:
        async for event in room.stream:
            room.post(event)
        room.finish()

    async def subscribe(
        self, run_id: str, *, tenant: str, after: int | None = None
    ) -> AsyncGenerator[ProgressEvent]:
        """Every event after `after`, missed ones first, then live ones as they arrive.

        Yields:
            The events, in sequence order, preceded by a `StreamGap` where the client was
            away longer than the buffer is deep.

        Raises:
            TransportAuthorizationError: If the run is not this tenant's.
        """
        room = self._authorised(run_id, tenant)
        listener: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
        room.listeners.add(listener)
        self._driving(room)
        try:
            for missed in self._backlog(room, -1 if after is None else after):
                yield missed
            if room.done:
                return
            while (live := await listener.get()) is not None:
                yield live
        finally:
            room.listeners.discard(listener)

    def _backlog(self, room: _Room[OutputT], seen: int) -> Iterable[ProgressEvent]:
        """What the client missed, with a gap notice where some of it is already gone."""
        kept = [event for event in room.events if event.sequence > seen]
        first_kept = kept[0].sequence if kept else seen + 1
        if first_kept > seen + 1:
            yield StreamGap(
                run_id=room.stream.run_id,
                sequence=seen + 1,
                missing=first_kept - seen - 1,
                resumed_from=first_kept,
            )
        yield from kept

    async def cancel(self, run_id: str, *, tenant: str) -> None:
        """Stop a run this tenant owns, through the run's own cancellation path.

        A run nobody ever attached to is cancelled before it starts, so it never calls a
        provider at all.

        Raises:
            TransportAuthorizationError: If the run is not this tenant's.
        """
        room = self._authorised(run_id, tenant)
        await room.stream.aclose()
        if room.task is not None:
            await room.task

    def authorise(self, run_id: str, *, tenant: str) -> None:
        """Check this tenant may speak about this run, before anything is framed.

        Raises:
            TransportAuthorizationError: If it may not.
        """
        self._authorised(run_id, tenant)

    def run(self, run_id: str, *, tenant: str) -> Run[OutputT]:
        """The finished record for a run this tenant owns.

        Raises:
            TransportAuthorizationError: If the run is not this tenant's.
        """
        return self._authorised(run_id, tenant).stream.run

    def _authorised(self, run_id: str, tenant: str) -> _Room[OutputT]:
        """The room, if this tenant may have it.

        Raises:
            TransportAuthorizationError: If it may not, or if there is no such run. The two
                are one answer on purpose: which ids exist is itself tenant information.
        """
        room = self._rooms.get(run_id)
        if room is None or room.tenant != tenant:
            raise TransportAuthorizationError(
                "no such run for this tenant", run_id=run_id, tenant=tenant
            )
        return room


async def sse_events(
    events: AsyncIterator[ProgressEvent],
    *,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    retry_milliseconds: int = DEFAULT_RETRY_MILLISECONDS,
    payload_limit_bytes: int = DEFAULT_PAYLOAD_LIMIT_BYTES,
) -> AsyncGenerator[str]:
    """Frame `events` as server-sent events, heartbeating while the run is quiet.

    Yields:
        Whole frames, ready to write to the response body. `SSE_HEADERS` are the headers
        that stop an intermediary buffering them into one lump at the end.
    """
    yield f"retry: {retry_milliseconds}\n\n"
    iterator = events.__aiter__()
    # The pending read is shielded, and outlives the timeout: cancelling it would throw
    # CancelledError into the source generator, so heartbeating would end the run.
    pending: asyncio.Task[ProgressEvent] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            try:
                event = await asyncio.wait_for(asyncio.shield(pending), heartbeat_seconds)
            except StopAsyncIteration:
                pending = None
                return
            except TimeoutError:
                yield _HEARTBEAT
                continue
            pending = None
            framed = wire_payload(event, limit=payload_limit_bytes)
            yield f"event: {framed.kind}\nid: {framed.sequence}\ndata: {_dumped(framed)}\n\n"
    finally:
        if pending is not None:
            pending.cancel()


def wire_payload(
    event: ProgressEvent, *, limit: int = DEFAULT_PAYLOAD_LIMIT_BYTES
) -> ProgressEvent:
    """The event as it may leave the process: redacted, and within the transport's limit.

    Redaction happens here as well as in the runtime because a boundary that trusts its
    input is not a boundary — an event that arrived from a queue or another process was
    scrubbed by whatever put it there, which is to say by nothing this code can see.
    """
    scrubbed = _scrubbed(event)
    encoded = _dumped(scrubbed)
    if len(encoded.encode()) <= limit:
        return scrubbed
    return PayloadElided(
        run_id=event.run_id,
        sequence=event.sequence,
        at=event.at,
        elided=event.kind,
        size_bytes=len(encoded.encode()),
        reference=f"{event.run_id}:{event.sequence}",
    )


def _scrubbed(event: ProgressEvent) -> ProgressEvent:
    # `type is str` rather than `isinstance`: a StrEnum is a str, and masking one would
    # turn a discriminator into prose the far end cannot switch on.
    fields = {
        name: scrub(value)
        for name, value in event
        if type(value) is str and name not in {"kind", "run_id"}
    }
    return event.model_copy(update=fields)


def _dumped(event: ProgressEvent) -> str:
    return json.dumps(event.model_dump(mode="json"), separators=(",", ":"))


class WebSocketLike(Protocol):
    """The three methods a websocket needs to have. Any ASGI framework's will do.

    A protocol rather than a dependency: the kit has no business deciding which web
    framework a product runs.
    """

    async def send_text(self, data: str) -> None:
        """Send one text frame."""
        ...

    async def receive_text(self) -> str:
        """Wait for one text frame, raising when the peer is gone."""
        ...

    async def close(self, code: int = 1000) -> None:
        """Close the connection."""
        ...


class ApprovalInbox:
    """Where a held tool call waits for a decision that arrives over the control channel.

    An `ApprovalGate` the runner can be given directly, so a websocket client answering a
    held call needs no glue of its own.
    """

    def __init__(self) -> None:
        self._waiting: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._records: dict[str, ApprovalRecord] = {}
        self._arrived: dict[str, asyncio.Event] = {}

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Hold this call until somebody decides about it."""
        pending: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self._waiting[record.id] = pending
        self._records[record.id] = record
        self._arrived.setdefault(record.id, asyncio.Event()).set()
        return await pending

    async def wait_for(self, record_id: str) -> None:
        """Wait until a call with this id is actually held. For tests and for a UI poll."""
        await self._arrived.setdefault(record_id, asyncio.Event()).wait()

    def decide(self, decision: ApprovalDecision, *, tenant: str) -> None:
        """Answer a held call.

        Raises:
            TransportAuthorizationError: If no such call is held for this tenant. A decision
                is permission, so it is checked against the run that asked rather than
                against the client's claim about it.
        """
        record = self._records.get(decision.record_id)
        if record is None or record.tenant != tenant:
            raise TransportAuthorizationError(
                "no call awaiting this tenant's decision", tenant=tenant
            )
        pending = self._waiting.pop(decision.record_id, None)
        if pending is not None and not pending.done():
            pending.set_result(decision)


class WebSocketBridge[OutputT: BaseModel]:
    """A run over a websocket: events out, control messages in.

    Args:
        broker: Where the run is registered.
        approvals: Where an approval decision goes, when the run is gated by one.
        payload_limit_bytes: Above this an event is referenced rather than framed.
    """

    def __init__(
        self,
        broker: RunBroker[OutputT],
        *,
        approvals: ApprovalInbox | None = None,
        payload_limit_bytes: int = DEFAULT_PAYLOAD_LIMIT_BYTES,
    ) -> None:
        self._broker = broker
        self._approvals = approvals
        self._limit = payload_limit_bytes

    async def serve(
        self, connection: WebSocketLike, *, run_id: str, tenant: str, after: int | None = None
    ) -> None:
        """Pump `run_id` to `connection` until the run ends or the peer goes away.

        Raises:
            TransportAuthorizationError: If the run is not this tenant's. Checked before a
                single event is framed.
        """
        self._broker.authorise(run_id, tenant=tenant)
        control = asyncio.create_task(self._controlled(connection, run_id=run_id, tenant=tenant))
        try:
            async for event in self._broker.subscribe(run_id, tenant=tenant, after=after):
                await connection.send_text(_dumped(wire_payload(event, limit=self._limit)))
        finally:
            control.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await control
            await connection.close()

    async def _controlled(self, connection: WebSocketLike, *, run_id: str, tenant: str) -> None:
        """Read control messages until the peer stops talking, then end the run.

        A peer that vanished without a close frame is a consumer that has gone; the run it
        was watching keeps calling providers and keeps billing until somebody says stop.
        """
        with contextlib.suppress(ConnectionError, OSError):
            while True:
                message = await connection.receive_text()
                await self._acted(message, run_id=run_id, tenant=tenant)
        await self._broker.cancel(run_id, tenant=tenant)

    async def _acted(self, message: str, *, run_id: str, tenant: str) -> None:
        """Do what one control message asks, ignoring one this version does not know.

        An unknown message type is a newer client talking to an older server. Tearing the
        connection down for it turns a forward-compatible change into an outage.
        """
        try:
            parsed: object = json.loads(message)
        except ValueError:
            return
        if not isinstance(parsed, dict):
            return
        match parsed.get("type"):
            case "cancel":
                await self._broker.cancel(run_id, tenant=tenant)
            case "approval" if self._approvals is not None:
                self._approvals.decide(
                    ApprovalDecision.model_validate(parsed.get("decision")), tenant=tenant
                )
            case _:
                return
