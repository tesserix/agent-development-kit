"""The state and queue conformance suites, unchanged, against a real Redis.

Opted into with `-m integration` and an environment variable, because the default lane
reaches no network. The unit tests check what the adapters send; this checks that what
they send is valid Lua that does what the protocol says, which no fake can tell you.

    ADK_TEST_REDIS_URL=redis://localhost:6379/9 uv run pytest tests/integration -m integration
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr

from tesserix_adk.adapters.state import RedisStateStore, RedisStoreSettings, RedisWorkQueue
from tesserix_adk.core.queue import QueuePolicy
from tesserix_adk.testing import FakeClock, StateStoreConformance, WorkQueueConformance

if TYPE_CHECKING:
    from tesserix_adk.core.queue import WorkQueue
    from tesserix_adk.core.state import StateStore

pytestmark = [pytest.mark.integration, pytest.mark.allow_network]

REDIS_URL = os.environ.get("ADK_TEST_REDIS_URL", "")
SETTINGS = RedisStoreSettings(dsn=SecretStr(REDIS_URL or "redis://localhost:6379/9"), durable=False)


def client() -> Any:
    """An async Redis client, or a skip where the extra is not installed."""
    redis = pytest.importorskip("redis.asyncio")
    return redis.from_url(REDIS_URL)


@pytest.mark.skipif(not REDIS_URL, reason="no ephemeral Redis configured")
class TestRunStateInRedis(StateStoreConformance):
    """Identical assertions to the in-process store's, against a real server."""

    def make_store(self) -> StateStore:
        return RedisStateStore(
            client(),
            clock=FakeClock(),
            settings=SETTINGS,
            namespace=f"adk:test:{uuid.uuid4().hex}",
        )


@pytest.mark.skipif(not REDIS_URL, reason="no ephemeral Redis configured")
class TestWorkQueuedInRedis(WorkQueueConformance):
    """Identical assertions to the in-process queue's, against a real server."""

    def make_queue(self) -> WorkQueue:
        return RedisWorkQueue(
            client(),
            clock=FakeClock(),
            settings=SETTINGS,
            policy=QueuePolicy(max_attempts=3),
            namespace=f"adk:test:{uuid.uuid4().hex}",
        )

    async def advance(self, queue: WorkQueue, seconds: float) -> None:
        assert isinstance(queue, RedisWorkQueue)
        clock = queue._clock
        assert isinstance(clock, FakeClock)
        clock.advance(seconds)
