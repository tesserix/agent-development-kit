"""Runs and work in one PostgreSQL, with a stand-in where the server would be.

The fake below answers the way PostgreSQL would; the interesting part is what the adapters
send, what they refuse, and that a state change and the work it queues can be made to commit
together. Run it with `python examples/sql_state_adapters.py`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import SecretStr

from tesserix_adk.adapters import (
    EXPECTED_SCHEMA,
    PostgresStateStore,
    PostgresStoreSettings,
    PostgresWorkQueue,
)
from tesserix_adk.core import ConfigurationError
from tesserix_adk.core.primitives import Usage
from tesserix_adk.core.queue import QueuePolicy, WorkItem
from tesserix_adk.core.state import RunRecord, StateDelta, StateKey
from tesserix_adk.testing import FakeClock

NOW = 1_000.0
SETTINGS = PostgresStoreSettings(dsn=SecretStr("postgresql://adk:s3cret@db:5432/adk"))
KEY = StateKey(tenant="acme", id="run_1")


class ShoutingSql:
    """Prints the first line of every statement, and answers with what it was told to."""

    def __init__(self, *replies: Any) -> None:  # noqa: ANN401 — whatever the server would say
        self.replies = list(replies)

    async def fetch(self, statement: str, *args: Any) -> Any:  # noqa: ANN401 — the row shape is the statement's
        """Answer one statement."""
        print("  sql:", statement.strip().splitlines()[0], "|", args[:2])  # noqa: T201
        return self.replies.pop(0) if self.replies else []


def run(**fields: Any) -> RunRecord:  # noqa: ANN401 — the record's own field types
    """One run, filled in enough to be stored."""
    return RunRecord(run_id="run_1", tenant="acme", agent_name="planner", **fields)


def stored() -> list[Any]:
    """A run row: the sequence, the blob, then the counters the columns own."""
    return [7, run().model_dump_json(), 3, 40, 9, 500, 2, 6]


async def the_schema_is_the_deployment_s_to_apply() -> None:
    """The kit reads the version it was written for and refuses anything else."""
    print("schema:")  # noqa: T201
    print("  tables:", EXPECTED_SCHEMA.count("CREATE TABLE"))  # noqa: T201
    try:
        await PostgresStateStore.open(
            ShoutingSql([[99]], [["5s"]]), clock=FakeClock(start=NOW), settings=SETTINGS
        )
    except ConfigurationError as refused:
        print("  refused:", refused)  # noqa: T201


async def patches_add_up_in_the_database() -> None:
    """Token counts go to the server as amounts, so two patches at once both land."""
    print("patches:")  # noqa: T201
    sql = ShoutingSql([stored()])
    store = PostgresStateStore(sql, clock=FakeClock(start=NOW), settings=SETTINGS)
    patched = await store.patch_run(
        KEY, StateDelta(usage=Usage(input_tokens=40, output_tokens=9), iterations=1)
    )
    print("  version:", patched.version, "| input tokens:", patched.usage.input_tokens)  # noqa: T201


async def a_claim_steps_over_what_another_worker_holds() -> None:
    """`FOR UPDATE ... SKIP LOCKED`: no worker waits, and no item goes out twice."""
    print("queue:")  # noqa: T201
    item = WorkItem(id="i1", tenant="acme")
    queue = PostgresWorkQueue(
        ShoutingSql([[item.model_dump_json()]], [[1]]),
        clock=FakeClock(start=NOW),
        settings=SETTINGS,
        policy=QueuePolicy(max_attempts=3, lease_seconds=45),
    )
    claimed = await queue.claim(worker="w1")
    print("  claimed:", claimed and claimed.id, "| lease:", claimed and claimed.lease_expires_at)  # noqa: T201


async def a_run_and_the_work_it_queued_commit_together() -> None:
    """One session, one transaction: the outbox problem stops existing."""
    print("together:")  # noqa: T201
    session = ShoutingSql([[1]], [])
    store = PostgresStateStore(ShoutingSql(), clock=FakeClock(start=NOW), settings=SETTINGS)
    queue = PostgresWorkQueue(ShoutingSql(), clock=FakeClock(start=NOW), settings=SETTINGS)
    await store.bound(session).put_run(run())
    await queue.bound(session).enqueue(WorkItem(id="i1", tenant="acme"))
    print("  statements in the caller's transaction:", len(session.replies) == 0)  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await the_schema_is_the_deployment_s_to_apply()
    await patches_add_up_in_the_database()
    await a_claim_steps_over_what_another_worker_holds()
    await a_run_and_the_work_it_queued_commit_together()


if __name__ == "__main__":
    asyncio.run(main())
