"""Durable runs and durable work, with a stand-in where the server would be.

The fake below answers the way Redis would; the interesting part is what the adapters
refuse, and what they send. Run it with `python examples/state_adapters.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import SecretStr

from tesserix_adk.adapters import RedisStateStore, RedisStoreSettings, RedisWorkQueue
from tesserix_adk.core import ConfigurationError, StateConflictError
from tesserix_adk.core.primitives import Usage
from tesserix_adk.core.queue import QueuePolicy, WorkItem
from tesserix_adk.core.state import RunRecord, StateDelta, StateKey
from tesserix_adk.testing import FakeClock

NOW = 1_000.0
SETTINGS = RedisStoreSettings(dsn=SecretStr("redis://:s3cret@cache:6379/0"))
KEY = StateKey(tenant="acme", id="run_1")
HEALTHY = {"maxmemory-policy": "noeviction", "appendonly": "yes"}


class ShoutingRedis:
    """Prints the first key of every script, and answers with what it was told to."""

    def __init__(self, *replies: Any, config: dict[str, str] | None = None) -> None:  # noqa: ANN401 — whatever the server would say
        self.replies = list(replies)
        self.config = config or dict(HEALTHY)

    async def eval(
        self,
        script: str,  # noqa: ARG002 — a real server would run it
        numkeys: int,  # noqa: ARG002 — a real server would split the keys off here
        *args: str,
    ) -> Any:  # noqa: ANN401 — the reply shape is the script's, not the client's
        """Answer one call."""
        print("  key:", args[0])  # noqa: T201
        return self.replies.pop(0) if self.replies else None

    async def config_get(self, pattern: str) -> dict[str, str]:  # noqa: ARG002 — the fake holds one config
        """What the server says it is configured as."""
        return self.config


def run(**fields: Any) -> RunRecord:  # noqa: ANN401 — the record's own field types
    """One run, filled in enough to be stored."""
    return RunRecord(run_id="run_1", tenant="acme", agent_name="planner", **fields)


def store(client: ShoutingRedis) -> RedisStateStore:
    """A state store on a clock this file controls."""
    return RedisStateStore(client, clock=FakeClock(start=NOW), settings=SETTINGS)


async def a_cache_is_refused_before_it_loses_a_run() -> None:
    """An instance that may evict answers a lost run exactly like a run that never was."""
    print("preflight:")  # noqa: T201
    cache = ShoutingRedis(config={"maxmemory-policy": "allkeys-lru", "appendonly": "yes"})
    try:
        await RedisStateStore.open(cache, clock=FakeClock(start=NOW), settings=SETTINGS)
    except ConfigurationError as refused:
        print("  refused:", refused)  # noqa: T201


async def a_write_that_lost_the_race_is_refused() -> None:
    """Two workers read version 2; the second one to write does not overwrite the first."""
    print("versions:")  # noqa: T201
    try:
        await store(ShoutingRedis(["conflict", "5"])).put_run(run(version=2))
    except StateConflictError as lost:
        print("  held:", lost.expected_version, "| live:", lost.actual_version)  # noqa: T201


async def patches_add_up_wherever_they_arrive() -> None:
    """Token counts go to the server as amounts, so two patches at once both land."""
    print("patches:")  # noqa: T201
    counters = [run().model_dump_json(), ["3", "40", "9", "0", "2", "0"]]
    patched = await store(ShoutingRedis(counters)).patch_run(
        KEY, StateDelta(usage=Usage(input_tokens=40, output_tokens=9), iterations=1)
    )
    print("  version:", patched.version, "| input tokens:", patched.usage.input_tokens)  # noqa: T201


async def work_outlives_the_worker_holding_it() -> None:
    """A claim, and the item a lapsed lease gives back one attempt worse off."""
    print("queue:")  # noqa: T201
    item = WorkItem(id="i1", tenant="acme")
    queue = RedisWorkQueue(
        ShoutingRedis(None, [[item.model_dump_json(), "1030"]], "ok"),
        clock=FakeClock(start=NOW),
        settings=SETTINGS,
        policy=QueuePolicy(max_attempts=3, backoff_seconds=2.0),
    )
    await queue.enqueue(item)
    for given in await queue.reap():
        print("  back:", given.state.value, "| attempt", given.attempts, "|", given.failures[-1])  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await a_cache_is_refused_before_it_loses_a_run()
    await a_write_that_lost_the_race_is_refused()
    await patches_add_up_wherever_they_arrive()
    await work_outlives_the_worker_holding_it()


if __name__ == "__main__":
    asyncio.run(main())
