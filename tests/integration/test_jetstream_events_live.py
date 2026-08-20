"""The event publisher and durable consumer against a real NATS JetStream.

Opted into with `-m integration` and an environment variable, because the default lane
reaches no network. The unit tests check what the adapter sends; this checks that a real
stream deduplicates on the header it sends and redelivers what the consumer declines,
which no fake can tell you.

    ADK_TEST_NATS_URL=nats://localhost:4222 uv run pytest tests/integration -m integration
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.adapters.jetstream import (
    DurableConsumer,
    JetStreamEventPublisher,
    StreamRequirement,
)
from tesserix_adk.core import Delivery, Eventing, RunStarted, tenant_scope
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from tesserix_adk.core import EventEnvelope

pytestmark = [pytest.mark.integration, pytest.mark.allow_network]

NATS_URL = os.environ.get("ADK_TEST_NATS_URL", "")
TENANT = "acme"


async def _stream(root: str) -> tuple[Any, Any, str]:
    """A connection, a JetStream context, and a stream of our own to publish onto."""
    nats = pytest.importorskip("nats")
    connection = await nats.connect(NATS_URL)
    js = connection.jetstream()
    name = f"ADK_TEST_{uuid.uuid4().hex[:8].upper()}"
    await js.add_stream(name=name, subjects=[f"{root}.>"], max_age=86_400.0)
    return connection, js, name


@pytest.mark.skipif(not NATS_URL, reason="no ephemeral NATS configured")
class TestAgainstARealStream:
    async def test_a_retried_publish_is_one_message_on_the_stream(self) -> None:
        root = f"adk-test-{uuid.uuid4().hex[:8]}"
        connection, js, name = await _stream(root)
        try:
            publisher = await JetStreamEventPublisher.open(
                js,
                clock=FakeClock(auto_advance=True),
                subject_root=root,
                requirement=StreamRequirement(name=name),
                delivery=Delivery.GUARANTEED,
            )
            eventing = Eventing(publisher, clock=FakeClock(), delivery=Delivery.GUARANTEED)
            with tenant_scope(TENANT):
                event = await eventing.emit(RunStarted(run_id="run_1", agent="desk"))
            assert event is not None
            await publisher.publish(event)
            info = await js.stream_info(name)
            assert info.state.messages == 1
            assert publisher.duplicates == 1
        finally:
            await js.delete_stream(name)
            await connection.close()

    async def test_a_declined_message_comes_back_and_an_acked_one_does_not(self) -> None:
        root = f"adk-test-{uuid.uuid4().hex[:8]}"
        connection, js, name = await _stream(root)
        try:
            publisher = await JetStreamEventPublisher.open(
                js,
                clock=FakeClock(auto_advance=True),
                subject_root=root,
                requirement=StreamRequirement(name=name),
                delivery=Delivery.GUARANTEED,
            )
            eventing = Eventing(publisher, clock=FakeClock(), delivery=Delivery.GUARANTEED)
            with tenant_scope(TENANT):
                await eventing.emit(RunStarted(run_id="run_1", agent="desk"))
            subscription = await js.pull_subscribe(f"{root}.{TENANT}.>", durable="adk_test")

            def explode(event: EventEnvelope) -> None:
                del event
                raise RuntimeError("downstream is down")

            await DurableConsumer(subscription, handler=explode, fetch_timeout=2.0).consume()

            read: list[EventEnvelope] = []
            again = DurableConsumer(subscription, handler=read.append, fetch_timeout=2.0)
            await again.consume()
            assert [event.run_id for event in read] == ["run_1"]
            assert await again.consume() == 0
        finally:
            await js.delete_stream(name)
            await connection.close()

    async def test_a_stream_that_forgets_too_soon_is_refused(self) -> None:
        root = f"adk-test-{uuid.uuid4().hex[:8]}"
        connection, js, name = await _stream(root)
        try:
            with pytest.raises(Exception, match="retains"):
                await JetStreamEventPublisher.open(
                    js,
                    clock=FakeClock(auto_advance=True),
                    subject_root=root,
                    requirement=StreamRequirement(name=name, min_age_seconds=999_999_999.0),
                )
        finally:
            await js.delete_stream(name)
            await connection.close()
