"""Work handed to another process, and what happens when that process dies holding it.

A run dispatched to a background worker has no owner once the pod is rolled: nothing times
it out, nothing retries it, and nothing can say afterwards whether it finished. So work is
claimed under a lease rather than taken, the lease is renewed by a living worker and lapses
with a dead one, and whatever lapses is requeued with its attempt counted. Delivery is
at-least-once — a slow worker's item is redelivered while it is still working — so handlers
must be idempotent, which is what `docs/tool-idempotency.md` is for.

The decisions behind these types are in `docs/work-queue.md`.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.models import AdkModel

__all__ = [
    "QueuePolicy",
    "QueueStats",
    "WorkItem",
    "WorkPriority",
    "WorkQueue",
    "WorkState",
]


class WorkPriority(IntEnum):
    """How soon an item is claimed relative to its tenant's other work.

    Ordered rather than named so a store can sort by it. Priority orders within a tenant
    and never across them: a tenant that enqueues everything at `URGENT` would otherwise
    starve every other tenant, and it would be right to.
    """

    LOW = 10
    NORMAL = 20
    HIGH = 30
    URGENT = 40


class WorkState(StrEnum):
    """Where an item is.

    `CLAIMED` says a worker held it when the store last heard; it does not say the worker
    is alive. That is what the lease is for.
    """

    QUEUED = "queued"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    DEAD_LETTERED = "dead_lettered"


class WorkItem(AdkModel):
    """One unit of work, as the queue holds it.

    Args:
        id: What this item is called, unique within the tenant.
        tenant: The isolation boundary. Claiming never crosses it.
        queue: Which named queue it sits in, so one store can back several.
        payload: What the handler needs to do the work. Strings only: a queue is a
            queryable store, and an object graph in it is an object graph an operator can
            read. Anything large goes behind a claim check.
        priority: How soon, relative to this tenant's other work.
        dedupe_key: What makes this the same logical job as another. A second enqueue
            under a live key returns the first item rather than adding a second.
        attempts: How many times a worker has held it. Incremented when a lease lapses or
            a handler fails, never when it succeeds.
        state: See `WorkState`.
        worker: Who holds it, where anyone does.
        enqueued_at: Unix seconds it was first enqueued, for the oldest-item metric.
        available_at: Unix seconds before which it will not be claimed. Backoff is this.
        lease_expires_at: Unix seconds the claim lapses, or `None` where it is not held.
        first_claimed_at: When it was first claimed, so renewals can be bounded.
        failures: What went wrong on each attempt, newest last. Carried to the dead letter
            so the item explains itself without a log search.

    Example:
        >>> WorkItem(id="w1", tenant="acme").state
        <WorkState.QUEUED: 'queued'>
    """

    id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    queue: str = Field(default="default", min_length=1)
    payload: dict[str, str] = Field(default_factory=dict)
    priority: WorkPriority = WorkPriority.NORMAL
    dedupe_key: str | None = None
    attempts: int = Field(default=0, ge=0)
    state: WorkState = WorkState.QUEUED
    worker: str | None = None
    enqueued_at: float = 0.0
    available_at: float = 0.0
    lease_expires_at: float | None = None
    first_claimed_at: float | None = None
    failures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _refuse_a_claim_with_nobody_holding_it(self) -> WorkItem:
        """A claimed item with no worker is an item no reaper can attribute or free."""
        if self.state is WorkState.CLAIMED and not self.worker:
            raise ValueError("a claimed item names the worker holding it")
        return self

    @property
    def held(self) -> bool:
        """Whether a worker last claimed it and has not given it back."""
        return self.state is WorkState.CLAIMED

    @property
    def held_since(self) -> float:
        """When the current claim started, or zero where nothing has ever claimed it."""
        return self.first_claimed_at or 0.0

    @property
    def terminal(self) -> bool:
        """Whether the queue is done with it, either way."""
        return self.state in {WorkState.COMPLETED, WorkState.DEAD_LETTERED}

    def lapsed_at(self, now: float) -> bool:
        """Whether the claim on it has expired, by the store's clock rather than a worker's."""
        return self.held and self.lease_expires_at is not None and self.lease_expires_at <= now


class QueuePolicy(AdkModel):
    """How long a claim lasts, how often it may be retried, and how long it waits.

    Args:
        lease_seconds: How long a claim holds before it lapses. Long enough to outlast a
            garbage-collection pause, short enough that a dead worker's item is picked up
            while somebody still cares.
        max_attempts: How many attempts an item gets before the dead letter. A poisonous
            item that loops forever spends model budget on failing to finish.
        backoff_seconds: The first wait after a failure, doubled per attempt.
        backoff_cap_seconds: The longest that doubling reaches.
        max_lease_seconds: The total a single claim may be renewed for. A worker that
            heartbeats forever holds work nothing else can pick up, and a stuck run is
            indistinguishable from a busy one to everything except this bound.

    Example:
        >>> QueuePolicy(backoff_seconds=1.0).backoff_for(3)
        4.0
    """

    lease_seconds: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=5, ge=1)
    backoff_seconds: float = Field(default=1.0, ge=0)
    backoff_cap_seconds: float = Field(default=300.0, ge=0)
    max_lease_seconds: float = Field(default=3600.0, gt=0)

    def backoff_for(self, attempts: int) -> float:
        """How long the item waits after its `attempts`-th failure.

        Doubling, capped. Deterministic rather than jittered: a queue redelivers one item
        to one worker, so there is no thundering herd to spread, and a test that has to
        allow for jitter stops asserting the interval at all.
        """
        if attempts <= 0:
            return 0.0
        doubled = self.backoff_seconds * float(2 ** (attempts - 1))
        return min(doubled, self.backoff_cap_seconds)

    def exhausted(self, attempts: int) -> bool:
        """Whether an item on `attempts` has run out of them."""
        return attempts >= self.max_attempts

    def overheld(self, item: WorkItem, now: float) -> bool:
        """Whether a single claim has been renewed for longer than the bound allows."""
        return now - item.held_since >= self.max_lease_seconds

    def claimed(
        self, item: WorkItem, *, worker: str, now: float, lease_seconds: float | None = None
    ) -> WorkItem:
        """The item as it looks once `worker` has taken it under a lease."""
        return item.model_copy(
            update={
                "state": WorkState.CLAIMED,
                "worker": worker,
                "lease_expires_at": now + (lease_seconds or self.lease_seconds),
                "first_claimed_at": now,
            }
        )

    def renewed(self, item: WorkItem, *, now: float) -> WorkItem:
        """The item with its lease extended from `now`."""
        return item.model_copy(update={"lease_expires_at": now + self.lease_seconds})

    def completed(self, item: WorkItem) -> WorkItem:
        """The item as it looks once the work is done and nobody holds it."""
        return item.model_copy(
            update={"state": WorkState.COMPLETED, "worker": None, "lease_expires_at": None}
        )

    def returned(
        self,
        item: WorkItem,
        *,
        error: str,
        now: float,
        retryable: bool = True,
        backoff: bool = True,
    ) -> WorkItem:
        """The item as it looks once a worker has given it back, one attempt worse off.

        Every store decides retry against dead letter here rather than in its own driver
        code: two implementations that disagree about when an item is poisonous are two
        deployments with different retry semantics and one set of tests.
        """
        attempts = item.attempts + 1
        failures = (*item.failures, error)
        if not retryable or self.exhausted(attempts):
            return item.model_copy(
                update={
                    "state": WorkState.DEAD_LETTERED,
                    "worker": None,
                    "lease_expires_at": None,
                    "attempts": attempts,
                    "failures": failures,
                }
            )
        return item.model_copy(
            update={
                "state": WorkState.QUEUED,
                "worker": None,
                "lease_expires_at": None,
                "first_claimed_at": None,
                "available_at": now + (self.backoff_for(attempts) if backoff else 0.0),
                "attempts": attempts,
                "failures": failures,
            }
        )


class QueueStats(AdkModel):
    """What the queue looks like right now, for the metrics a deployment alerts on.

    Gauges are of the moment; counters accumulate for the life of the store. Depth alone
    says nothing — a queue of ten items ten seconds old is healthy and a queue of one item
    an hour old is an incident — so age is here beside it.

    Args:
        depth: Items waiting to be claimed, including ones still backing off.
        claimed: Items a worker is holding.
        dead_lettered: Items that ran out of attempts and are still in the dead letter.
        oldest_age_seconds: How long the oldest waiting item has been waiting.
        reaped: How many claims have lapsed and been requeued, cumulative.
        duplicates_suppressed: How many enqueues a dedupe key collapsed, cumulative.
    """

    depth: int = Field(default=0, ge=0)
    claimed: int = Field(default=0, ge=0)
    dead_lettered: int = Field(default=0, ge=0)
    oldest_age_seconds: float = Field(default=0.0, ge=0)
    reaped: int = Field(default=0, ge=0)
    duplicates_suppressed: int = Field(default=0, ge=0)


@runtime_checkable
class WorkQueue(Protocol):
    """Where work waits for a worker, and what becomes of it when one dies holding it.

    Delivery is at-least-once. A worker that is merely slow has its item redelivered while
    it is still working on it, so handlers are idempotent or they are wrong. Lease expiry
    is evaluated by the store, never by a worker: two workers' clocks disagree, and a
    queue that trusted theirs would free work that is being done and keep work that is not.

    Implementations raise `QueueUnavailableError` rather than losing an enqueue, and
    `LeaseLostError` rather than letting a worker act on a claim it no longer holds.
    """

    async def enqueue(self, item: WorkItem) -> WorkItem:
        """Add `item`, or return the live item its dedupe key already names.

        Raises:
            QueueUnavailableError: If the store could not take it. Never a silent drop.
        """
        ...

    async def claim(
        self, *, worker: str, queue: str = "default", lease_seconds: float | None = None
    ) -> WorkItem | None:
        """Take the next item for `worker`, or `None` where there is nothing due.

        Tenants are served in turn, so one tenant's backlog cannot starve another's.
        Priority and age order the choice within the tenant whose turn it is.
        """
        ...

    async def heartbeat(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Extend the claim, saying the worker is still alive and still working.

        Raises:
            LeaseLostError: If the claim lapsed, was taken by another worker, or has been
                renewed for longer than the policy allows.
            WorkItemNotFoundError: If there is no such item.
        """
        ...

    async def complete(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Finish the item. It is not claimable again.

        Raises:
            LeaseLostError: If the worker no longer holds it — in which case another
                worker has it, and this one's result is a duplicate.
            WorkItemNotFoundError: If there is no such item.
        """
        ...

    async def fail(
        self, item_id: str, *, tenant: str, worker: str, error: str, retryable: bool = True
    ) -> WorkItem:
        """Give the item back as failed, for retry after a backoff or for the dead letter.

        `retryable=False` skips the remaining attempts: an item that cannot succeed does
        not become able to by waiting.

        Raises:
            LeaseLostError: If the worker no longer holds it.
            WorkItemNotFoundError: If there is no such item.
        """
        ...

    async def reap(self) -> tuple[WorkItem, ...]:
        """Requeue every item whose lease has lapsed, and return what moved.

        An item at the attempt cap goes to the dead letter instead. Sweeping is explicit
        rather than a background task the kit owns: the caller decides how often, and a
        process that has stopped sweeping is visible in its own scheduler.
        """
        ...

    async def adopt(self, *, worker: str) -> tuple[WorkItem, ...]:
        """Release what `worker` held before it restarted, and return what moved.

        A worker that comes back under its own name cannot know what it was doing, so it
        gives the work back immediately rather than waiting out a lease nobody is renewing.
        The attempt still counts: the work was tried, and the cap exists to bound tries.
        """
        ...

    async def dead_letters(self, *, tenant: str, limit: int = 50) -> tuple[WorkItem, ...]:
        """Return items that ran out of attempts, oldest first, with their failures."""
        ...

    async def stats(self, *, queue: str = "default") -> QueueStats:
        """Return the depth, ages and counters a deployment alerts on."""
        ...
