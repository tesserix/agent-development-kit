"""A worker that dies holding an item, and the four things that follow from it.

Four scenarios: a claim that is exclusive; a lease that lapses and comes back; an item that
keeps failing until the dead letter catches it; and one tenant's backlog failing to starve
another's. Run it with `python examples/work_queue.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import QueuePolicy, WorkItem, WorkPriority
from tesserix_adk.runtime import MemoryWorkQueue
from tesserix_adk.testing import FakeClock

TENANT = "acme"


def a_queue(clock: FakeClock, **policy: float | int) -> MemoryWorkQueue:
    """A queue whose leases and attempts are short enough to watch."""
    return MemoryWorkQueue(QueuePolicy(**policy), clock)


def a_run(item_id: str = "run_1", *, tenant: str = TENANT) -> WorkItem:
    """One agent run, waiting for whichever worker gets to it."""
    return WorkItem(id=item_id, tenant=tenant, payload={"agent": "planner"})


async def one_worker_at_a_time() -> None:
    """A claim is exclusive for as long as the lease lasts."""
    queue = a_queue(FakeClock(), lease_seconds=30.0)
    await queue.enqueue(a_run())
    first = await queue.claim(worker="worker-7")
    second = await queue.claim(worker="worker-8")

    print("\n=== one worker at a time ===")  # noqa: T201
    print(f"worker-7 got: {first.id if first else None}")  # noqa: T201
    print(f"worker-8 got: {second.id if second else None}")  # noqa: T201


async def a_worker_that_died() -> None:
    """Nobody renews the lease, so the reaper gives the work to somebody who will."""
    clock = FakeClock()
    queue = a_queue(clock, lease_seconds=10.0, backoff_seconds=0.0)
    await queue.enqueue(a_run())
    await queue.claim(worker="worker-7")
    clock.advance(11.0)
    reaped = await queue.reap()
    carried_on = await queue.claim(worker="worker-8")

    print("\n=== a worker that died ===")  # noqa: T201
    print(f"reaped: {[(item.id, item.attempts) for item in reaped]}")  # noqa: T201
    print(f"why: {reaped[0].failures[0]}")  # noqa: T201
    print(f"now held by: {carried_on.worker if carried_on else None}")  # noqa: T201


async def an_item_that_never_succeeds() -> None:
    """Three attempts, then the dead letter — not a loop that spends budget on failing."""
    queue = a_queue(FakeClock(), max_attempts=3, backoff_seconds=0.0)
    await queue.enqueue(a_run())
    for attempt in range(3):
        await queue.claim(worker="worker-7")
        await queue.fail("run_1", tenant=TENANT, worker="worker-7", error=f"boom {attempt}")
    dead = await queue.dead_letters(tenant=TENANT)

    print("\n=== an item that never succeeds ===")  # noqa: T201
    print(f"dead-lettered: {[item.id for item in dead]}")  # noqa: T201
    print(f"after: {dead[0].attempts} attempts")  # noqa: T201
    print(f"carrying: {list(dead[0].failures)}")  # noqa: T201
    print(f"claimable again: {await queue.claim(worker='worker-8') is not None}")  # noqa: T201


async def a_backlog_that_starves_nobody() -> None:
    """Tenants are served in turn, so the loud one does not own the queue."""
    queue = a_queue(FakeClock())
    for index in range(5):
        await queue.enqueue(a_run(f"loud_{index}", tenant="loud"))
    quiet = a_run("quiet_1", tenant="quiet").model_copy(update={"priority": WorkPriority.LOW})
    await queue.enqueue(quiet)
    served = [await queue.claim(worker=f"worker-{index}") for index in range(3)]

    print("\n=== a backlog that starves nobody ===")  # noqa: T201
    print(f"served: {[item.tenant for item in served if item]}")  # noqa: T201
    print(f"depth left: {(await queue.stats()).depth}")  # noqa: T201


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await one_worker_at_a_time()
    await a_worker_that_died()
    await an_item_that_never_succeeds()
    await a_backlog_that_starves_nobody()


if __name__ == "__main__":
    asyncio.run(main())
