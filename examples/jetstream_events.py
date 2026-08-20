"""Publishing events onto JetStream, and reading them back durably. Runs offline.

uv run python examples/jetstream_events.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from tesserix_adk.adapters import DurableConsumer, JetStreamEventPublisher, StreamRequirement
from tesserix_adk.core import Delivery, EventEnvelope, Eventing, RunCompleted, RunStarted
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.testing import FakeClock


@dataclass(slots=True)
class _Ack:
    stream: str = "ADK_EVENTS"
    seq: int = 1
    duplicate: bool = False


@dataclass(slots=True)
class _Config:
    subjects: tuple[str, ...] = ("adk.events.>",)
    retention: str = "limits"
    max_age: float = 604_800.0
    max_msg_size: int = -1


@dataclass(slots=True)
class _Broker:
    """A stand-in for JetStream that keeps what it is told, once per message id."""

    stored: list[tuple[str, bytes]] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)

    async def stream_info(self, name: str) -> _Config:
        del name
        return _Config()

    async def publish(
        self,
        subject: str,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 — the client's own signature
    ) -> _Ack:
        del timeout
        identifier = (headers or {})["Nats-Msg-Id"]
        if identifier in self.seen:
            return _Ack(duplicate=True)
        self.seen.add(identifier)
        self.stored.append((subject, payload))
        return _Ack()


@dataclass(slots=True)
class _Message:
    data: bytes
    settled: str = ""

    async def ack(self) -> None:
        self.settled = "acked"

    async def nak(self) -> None:
        self.settled = "naked"

    async def term(self) -> None:
        self.settled = "terminated"


@dataclass(slots=True)
class _Subscription:
    messages: list[_Message]

    async def fetch(
        self,
        batch: int,
        timeout: float | None = None,  # noqa: ASYNC109 — the client's own signature
    ) -> list[_Message]:
        del timeout
        taken, self.messages = self.messages[:batch], self.messages[batch:]
        return taken


async def main() -> None:
    """Publish two events onto a stand-in stream, then read them back."""
    broker = _Broker()
    publisher = await JetStreamEventPublisher.open(
        broker,
        clock=FakeClock(auto_advance=True),
        requirement=StreamRequirement(),
        delivery=Delivery.GUARANTEED,
    )
    eventing = Eventing(publisher, clock=FakeClock(), delivery=Delivery.GUARANTEED)

    with tenant_scope("acme", user="ada"):
        started = await eventing.emit(RunStarted(run_id="run_1", agent="desk"))
        await eventing.emit(RunCompleted(run_id="run_1", iterations=2), caused_by=started)

    if started is None:  # pragma: no cover — guaranteed delivery returns the envelope
        raise SystemExit("nothing was published")
    print("published on:", broker.stored[0][0])  # noqa: T201

    await publisher.publish(started)
    print("a retried publish is still one message:", len(broker.stored), "stored")  # noqa: T201
    print("the broker recognised it:", publisher.duplicates)  # noqa: T201

    read: list[EventEnvelope] = []
    messages = [_Message(data=payload) for _, payload in broker.stored]
    consumer = DurableConsumer(_Subscription(messages), handler=read.append)
    await consumer.consume()
    print("consumed:", [event.type.value for event in read])  # noqa: T201
    print("settled:", {message.settled for message in messages})  # noqa: T201

    def explode(event: EventEnvelope) -> None:
        del event
        raise RuntimeError("downstream is down")

    failing = _Message(data=broker.stored[0][1])
    await DurableConsumer(_Subscription([failing]), handler=explode).consume()
    print("a failed handler leaves it for redelivery:", failing.settled)  # noqa: T201

    unreadable = _Message(data=b"{not json")
    await DurableConsumer(_Subscription([unreadable]), handler=read.append).consume()
    print("nothing that is not an envelope is replayed for ever:", unreadable.settled)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
