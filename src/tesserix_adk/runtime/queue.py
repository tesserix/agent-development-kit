"""A work queue in a dict, for a single replica and for tests.

Nothing here outlives the process, which is what surviving a rolled pod is not. It exists
so the semantics a real adapter has to keep — leases evaluated by the store, attempts that
survive redelivery, per-tenant fairness, a dead letter nothing falls out of — can be
exercised without Redis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.errors import LeaseLostError, WorkItemNotFoundError
from tesserix_adk.core.queue import QueuePolicy, QueueStats, WorkItem, WorkState

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import Clock

__all__ = ["MemoryWorkQueue"]

LEASE_LAPSED = "lease lapsed with the item unfinished"
WORKER_RESTARTED = "the worker holding it restarted"


class MemoryWorkQueue:
    """A `WorkQueue` in a dict, keyed by tenant and item.

    Items take a store-wide sequence number when first enqueued and keep it, so redelivery
    puts an item back where it was rather than at the end: an item that has already failed
    four times is the one most worth finishing or dead-lettering, not the one to starve.

    Tenants are served in rotation. Priority orders a tenant's own work and nothing else,
    because a tenant that enqueues everything at `URGENT` would otherwise be right to.
    """

    def __init__(self, policy: QueuePolicy | None = None, clock: Clock | None = None) -> None:
        self._policy = policy or QueuePolicy()
        self._clock = clock
        self._items: dict[tuple[str, str], tuple[int, WorkItem]] = {}
        self._deduped: dict[tuple[str, str], str] = {}
        self._sequence = 0
        self._turn = 0
        self._reaped = 0
        self._duplicates = 0

    @property
    def policy(self) -> QueuePolicy:
        """How long a claim lasts here, and how many attempts an item gets."""
        return self._policy

    async def enqueue(self, item: WorkItem) -> WorkItem:
        """Add `item`, or return the live item it already is.

        Re-enqueueing an id that is still in the queue returns what is stored rather than
        replacing it: an item whose attempts were reset by a well-meaning retry is an item
        the cap can never catch.
        """
        existing = self._live_duplicate(item)
        if existing is not None:
            self._duplicates += 1
            return existing
        held = self._items.get((item.tenant, item.id))
        if held is not None and not held[1].terminal:
            self._duplicates += 1
            return held[1]
        now = self._now()
        placed = item.model_copy(
            update={
                "enqueued_at": item.enqueued_at or now,
                "available_at": item.available_at or now,
                "state": WorkState.QUEUED,
                "worker": None,
                "lease_expires_at": None,
            }
        )
        self._sequence += 1
        self._items[placed.tenant, placed.id] = (self._sequence, placed)
        if placed.dedupe_key is not None:
            self._deduped[placed.tenant, placed.dedupe_key] = placed.id
        return placed

    async def claim(
        self, *, worker: str, queue: str = "default", lease_seconds: float | None = None
    ) -> WorkItem | None:
        """Take the next due item for `worker`, taking each tenant's turn in rotation."""
        now = self._now()
        due = [
            (sequence, item)
            for sequence, item in self._items.values()
            if item.queue == queue and item.state is WorkState.QUEUED and item.available_at <= now
        ]
        if not due:
            return None
        chosen = self._next_in_turn(due)
        lease = lease_seconds or self._policy.lease_seconds
        held = chosen.model_copy(
            update={
                "state": WorkState.CLAIMED,
                "worker": worker,
                "lease_expires_at": now + lease,
                "first_claimed_at": now,
            }
        )
        return self._keep(held)

    async def heartbeat(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Extend the claim, up to the total the policy allows a single claim to run for."""
        item = self._holding(item_id, tenant=tenant, worker=worker)
        now = self._now()
        if now - item.held_since >= self._policy.max_lease_seconds:
            raise LeaseLostError(
                f"{item_id!r} has been held for longer than {self._policy.max_lease_seconds}s",
                item_id=item_id,
                worker=worker,
                holder=worker,
                reason="capped",
            )
        return self._keep(
            item.model_copy(update={"lease_expires_at": now + self._policy.lease_seconds})
        )

    async def complete(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Finish the item. Its dedupe key is free again; the same job may be enqueued anew."""
        item = self._holding(item_id, tenant=tenant, worker=worker)
        self._release_dedupe(item)
        return self._keep(
            item.model_copy(
                update={
                    "state": WorkState.COMPLETED,
                    "worker": None,
                    "lease_expires_at": None,
                }
            )
        )

    async def fail(
        self, item_id: str, *, tenant: str, worker: str, error: str, retryable: bool = True
    ) -> WorkItem:
        """Give the item back for a retry after a backoff, or to the dead letter."""
        item = self._holding(item_id, tenant=tenant, worker=worker)
        return self._keep(self._returned(item, error, retryable=retryable))

    async def reap(self) -> tuple[WorkItem, ...]:
        """Requeue everything whose lease has lapsed, by this store's clock."""
        now = self._now()
        lapsed = [item for _, item in self._items.values() if item.lapsed_at(now)]
        moved = tuple(self._keep(self._returned(item, LEASE_LAPSED)) for item in lapsed)
        self._reaped += len(moved)
        return moved

    async def adopt(self, *, worker: str) -> tuple[WorkItem, ...]:
        """Give back what `worker` held before it restarted, without waiting out the lease."""
        owned = [item for _, item in self._items.values() if item.held and item.worker == worker]
        return tuple(
            self._keep(self._returned(item, WORKER_RESTARTED, backoff=False)) for item in owned
        )

    async def dead_letters(self, *, tenant: str, limit: int = 50) -> tuple[WorkItem, ...]:
        """Return items that ran out of attempts, in the order they were enqueued."""
        dead = sorted(
            (
                (sequence, item)
                for sequence, item in self._items.values()
                if item.tenant == tenant and item.state is WorkState.DEAD_LETTERED
            ),
            key=lambda held: held[0],
        )
        return tuple(item for _, item in dead[:limit])

    async def stats(self, *, queue: str = "default") -> QueueStats:
        """Return the depth, ages and counters a deployment alerts on."""
        now = self._now()
        items = [item for _, item in self._items.values() if item.queue == queue]
        waiting = [item for item in items if item.state is WorkState.QUEUED]
        return QueueStats(
            depth=len(waiting),
            claimed=sum(1 for item in items if item.held),
            dead_lettered=sum(1 for item in items if item.state is WorkState.DEAD_LETTERED),
            oldest_age_seconds=max((now - item.enqueued_at for item in waiting), default=0.0),
            reaped=self._reaped,
            duplicates_suppressed=self._duplicates,
        )

    def _returned(
        self, item: WorkItem, error: str, *, retryable: bool = True, backoff: bool = True
    ) -> WorkItem:
        """The item as it looks once a worker has given it back, one attempt worse off."""
        attempts = item.attempts + 1
        failures = (*item.failures, error)
        if not retryable or self._policy.exhausted(attempts):
            return item.model_copy(
                update={
                    "state": WorkState.DEAD_LETTERED,
                    "worker": None,
                    "lease_expires_at": None,
                    "attempts": attempts,
                    "failures": failures,
                }
            )
        wait = self._policy.backoff_for(attempts) if backoff else 0.0
        return item.model_copy(
            update={
                "state": WorkState.QUEUED,
                "worker": None,
                "lease_expires_at": None,
                "first_claimed_at": None,
                "available_at": self._now() + wait,
                "attempts": attempts,
                "failures": failures,
            }
        )

    def _holding(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """The item, if this worker still holds it — and a refusal if it does not."""
        held = self._items.get((tenant, item_id))
        if held is None:
            raise WorkItemNotFoundError(
                f"no item {item_id!r} in {tenant!r}", item_id=item_id, tenant=tenant
            )
        item = held[1]
        if not item.held or item.worker != worker:
            raise LeaseLostError(
                f"{item_id!r} is no longer held by {worker!r}",
                item_id=item_id,
                worker=worker,
                holder=item.worker,
                reason="taken" if item.held else "expired",
            )
        if item.lapsed_at(self._now()):
            raise LeaseLostError(
                f"the claim {worker!r} holds on {item_id!r} has expired",
                item_id=item_id,
                worker=worker,
                holder=item.worker,
                reason="expired",
            )
        return item

    def _live_duplicate(self, item: WorkItem) -> WorkItem | None:
        """The item this one would duplicate, where one is still worth waiting for."""
        if item.dedupe_key is None:
            return None
        named = self._deduped.get((item.tenant, item.dedupe_key))
        if named is None:
            return None
        held = self._items.get((item.tenant, named))
        return None if held is None or held[1].terminal else held[1]

    def _next_in_turn(self, due: list[tuple[int, WorkItem]]) -> WorkItem:
        """The best item of whichever tenant's turn it is, so no tenant is starved."""
        tenants = sorted({item.tenant for _, item in due})
        self._turn = (self._turn + 1) % len(tenants)
        serving = tenants[self._turn]
        best = sorted(
            (held for held in due if held[1].tenant == serving),
            key=lambda held: (-held[1].priority, held[0]),
        )
        return best[0][1]

    def _keep(self, item: WorkItem) -> WorkItem:
        """Store the item at the position it has held since it was enqueued."""
        sequence = self._items[item.tenant, item.id][0]
        self._items[item.tenant, item.id] = (sequence, item)
        return item

    def _release_dedupe(self, item: WorkItem) -> None:
        """A finished job is no longer the one a second enqueue would collapse into."""
        if item.dedupe_key is not None:
            self._deduped.pop((item.tenant, item.dedupe_key), None)

    def _now(self) -> float:
        return 0.0 if self._clock is None else self._clock.now()
