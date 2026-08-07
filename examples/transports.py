"""Putting a run on the wire: SSE frames, a websocket, a reconnect and a refusal.

A broker drives the run once; SSE and a websocket are two readings of it. A scripted
provider stands in for a vendor, so nothing here reaches the network and no key is needed.

Run it with `python examples/transports.py`.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from tesserix_adk.adapters import (
    SSE_HEADERS,
    RunBroker,
    StreamGap,
    TransportAuthorizationError,
    WebSocketBridge,
    sse_events,
)
from tesserix_adk.core import Agent, NoOutput, Usage
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider

if TYPE_CHECKING:
    from tesserix_adk.runtime import RunStream

AGENT = Agent(name="concierge", instructions="Plan trips.", model="claude-sonnet-5", free_text=True)


class PrintingSocket:
    """A websocket peer that prints what it was sent and then asks to stop."""

    def __init__(self, *inbound: str) -> None:
        self._inbound = list(inbound)
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        """Record a frame the bridge pushed."""
        self.sent.append(data)

    async def receive_text(self) -> str:
        """The next scripted control message, then silence for the rest of the run."""
        if self._inbound:
            return self._inbound.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def close(self, code: int = 1000) -> None:
        """Note the close the bridge performs on its way out."""
        del code


def a_run() -> RunStream[NoOutput]:
    """A scripted run, ready to register."""
    runner = AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(
                content="Four nights near Kyoto, in the eastern hills.",
                usage=Usage(input_tokens=12, output_tokens=9),
            ),
            capabilities=CAPABLE,
        ),
        clock=FakeClock(),
    )
    return runner.stream(AGENT, "Plan four nights near Kyoto.", tenant="acme", run_id="run_1")


async def over_sse() -> None:
    """Frame a run as server-sent events, headers and all."""
    broker = RunBroker[NoOutput]()
    broker.register(a_run(), tenant="acme")
    print("headers:", json.dumps(SSE_HEADERS))  # noqa: T201
    async for frame in sse_events(broker.subscribe("run_1", tenant="acme")):
        print(frame.splitlines()[0])  # noqa: T201


async def over_a_websocket() -> None:
    """The same events, same payloads, over a socket that also talks back."""
    broker = RunBroker[NoOutput]()
    broker.register(a_run(), tenant="acme")
    socket = PrintingSocket(json.dumps({"type": "telemetry", "fps": 60}))
    await WebSocketBridge(broker).serve(socket, run_id="run_1", tenant="acme")
    kinds = [json.loads(frame)["kind"] for frame in socket.sent]
    print("websocket kinds:", " ".join(kinds))  # noqa: T201


async def reconnecting() -> None:
    """A client that was away asks from its last sequence and is told what it missed."""
    broker = RunBroker[NoOutput](history=2)
    broker.register(a_run(), tenant="acme")
    async for _ in broker.subscribe("run_1", tenant="acme"):
        pass
    resumed = [event async for event in broker.subscribe("run_1", tenant="acme", after=0)]
    first = resumed[0]
    if isinstance(first, StreamGap):
        print(f"gap: {first.missing} event(s) missed, resuming at {first.resumed_from}")  # noqa: T201


async def a_run_that_is_not_yours() -> None:
    """The boundary fails closed: a run id from a client is a claim, not a fact."""
    broker = RunBroker[NoOutput]()
    broker.register(a_run(), tenant="acme")
    try:
        await broker.cancel("run_1", tenant="rival")
    except TransportAuthorizationError as refusal:
        print("refused:", refusal)  # noqa: T201
    await broker.cancel("run_1", tenant="acme")


async def main() -> None:
    """Run every transport."""
    await over_sse()
    await over_a_websocket()
    await reconnecting()
    await a_run_that_is_not_yours()


if __name__ == "__main__":
    asyncio.run(main())
