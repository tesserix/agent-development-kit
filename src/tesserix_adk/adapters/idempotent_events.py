"""Handling an at-least-once event exactly once.

The broker will deliver the same event twice — after an ack timeout, a reaper requeue, or
a rolling deploy — and there is no configuration that makes it stop. What decides whether
that is harmless is whether the handler's effect is recorded somewhere the second delivery
can see. That record is the whole module.

The guarantee is narrower than it looks, so it is stated rather than implied: this
suppresses a second *execution* within the retention window. It makes the effect single
only where the handler's own state change shares the transaction the marker is written in.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.errors import (
    ConfigurationError,
    DuplicateInFlightError,
    EventIdReuseError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from tesserix_adk.adapters.jetstream import DeadLetter
    from tesserix_adk.core.events import EventEnvelope
    from tesserix_adk.core.idempotency import IdempotencyStore

__all__ = [
    "DEFAULT_DEDUPE_TTL_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "IdempotentConsumer",
    "dedupe_key",
]

DEFAULT_DEDUPE_TTL_SECONDS = 86_400.0
"""How long a processed event is remembered. Longer than any redelivery horizon."""

DEFAULT_MAX_ATTEMPTS = 5

_HANDLED = "handled"
_POISON = "poison"


def dedupe_key(*, group: str, event: EventEnvelope) -> str:
    """What one consumer group's handling of one event is recorded under.

    The tenant is not in the key because the store is already scoped by it, which is what
    makes an id clash between two tenants impossible rather than merely unlikely.
    """
    return f"{group}:{event.event_id}"


class IdempotentConsumer:
    """Runs a handler at most once per event, and stops retrying what never works.

    Args:
        handler: What to do with the event. Sync or async.
        store: Where the processed marker lives. Redis and PostgreSQL implementations
            ship with the kit.
        group: The consumer group. Two groups each handle every event once.
        ttl_seconds: How long a processed event is remembered.
        redelivery_horizon_seconds: The longest the broker could still redeliver an
            event, which is `max_deliver` times `ack_wait`. A marker that expires first
            is not a marker.
        max_attempts: How many failed attempts before the event is buried.
        dead_letter: Where a poison event goes with its failure history.
        transaction: Opens the transaction the handler's state change and the processed
            marker share. Without one the marker is written after the handler returns,
            which suppresses a second execution but cannot make the effect atomic.

    Raises:
        ConfigurationError: If the retention window is shorter than the redelivery
            horizon, which would let a redelivery arrive after its own marker expired.
    """

    __slots__ = (
        "_dead_letter",
        "_group",
        "_handler",
        "_max_attempts",
        "_store",
        "_transaction",
        "_ttl",
        "buried",
        "handled",
        "suppressed",
    )

    def __init__(
        self,
        handler: Callable[[EventEnvelope], Any],
        *,
        store: IdempotencyStore,
        group: str,
        ttl_seconds: float = DEFAULT_DEDUPE_TTL_SECONDS,
        redelivery_horizon_seconds: float = 0.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        dead_letter: DeadLetter | None = None,
        transaction: Callable[[], AbstractAsyncContextManager[object]] | None = None,
    ) -> None:
        if ttl_seconds < redelivery_horizon_seconds:
            raise ConfigurationError(
                f"a processed event is remembered for {ttl_seconds}s and the broker may "
                f"still redeliver it {redelivery_horizon_seconds}s later, so the marker "
                f"the redelivery would look for is already gone"
            )
        self._handler = handler
        self._store = store
        self._group = group
        self._ttl = ttl_seconds
        self._max_attempts = max(1, max_attempts)
        self._dead_letter = dead_letter
        self._transaction = transaction
        self.handled = 0
        self.suppressed = 0
        self.buried = 0

    async def handle(self, event: EventEnvelope) -> bool:
        """Handle `event` unless it has already been handled by this group.

        Returns:
            Whether the handler ran. `False` where this delivery was recognised as one
            this group has already applied, which is what an operator replay reports as
            suppressed rather than redelivered.

        Raises:
            DuplicateInFlightError: Another worker is handling this event right now, so
                this delivery is left for redelivery rather than acknowledged behind it.
            EventIdReuseError: A different event arrived under an id already recorded,
                which would otherwise suppress an unrelated effect.
            Exception: Whatever the handler raised, so the broker redelivers it. Once
                `max_attempts` is reached the event is buried and nothing is raised.
        """
        key = dedupe_key(group=self._group, event=event)
        identity = _identity(event)
        claim = await self._store.begin(key, tenant=event.tenant, ttl_seconds=self._ttl)
        if claim.outcome is not None:
            self._recognised(event, claim.outcome, identity)
            if not _reconsidered(event, claim.outcome):
                self.suppressed += 1
                return False
        if claim.in_flight:
            self.suppressed += 1
            raise DuplicateInFlightError(
                f"{event.event_id} is being handled by another worker of {self._group}",
                run_id=event.run_id,
                tenant=event.tenant,
            )
        try:
            await self._ran(event, key, identity)
        except Exception as failure:
            await self._failed(event, key, identity, failure)
            return False
        self.handled += 1
        return True

    async def _ran(self, event: EventEnvelope, key: str, identity: str) -> None:
        """The handler and its marker, inside one transaction where there is one."""
        if self._transaction is None:
            await self._handled(event, key, identity)
            return
        async with self._transaction():
            await self._handled(event, key, identity)

    async def _handled(self, event: EventEnvelope, key: str, identity: str) -> None:
        """Run the handler, then record that this group has handled this event."""
        outcome = self._handler(event)
        if hasattr(outcome, "__await__"):
            await outcome
        await self._store.record(
            key, tenant=event.tenant, outcome=f"{_HANDLED}:{identity}", ttl_seconds=self._ttl
        )

    async def _failed(
        self, event: EventEnvelope, key: str, identity: str, failure: Exception
    ) -> None:
        """Record the attempt, and bury the event once there have been enough of them."""
        history = await self._recorded_failures(event, key)
        history = (*history, type(failure).__name__)
        if len(history) >= self._max_attempts:
            await self._buried(event, key, identity, history)
            return
        await self._store.record(
            _failures_key(key),
            tenant=event.tenant,
            outcome=json.dumps(list(history)),
            ttl_seconds=self._ttl,
        )
        await self._store.abandon(key, tenant=event.tenant)
        raise failure

    async def _buried(
        self, event: EventEnvelope, key: str, identity: str, history: tuple[str, ...]
    ) -> None:
        """Keep it where somebody can look at it, and stop redelivering it."""
        if self._dead_letter is not None:
            await self._dead_letter.bury(event.to_json().encode(), reason=_POISON, history=history)
        await self._store.record(
            key, tenant=event.tenant, outcome=f"{_POISON}:{identity}", ttl_seconds=self._ttl
        )
        self.buried += 1

    async def _recorded_failures(self, event: EventEnvelope, key: str) -> tuple[str, ...]:
        """What went wrong on the earlier attempts, as their exception types."""
        claim = await self._store.begin(
            _failures_key(key), tenant=event.tenant, ttl_seconds=self._ttl
        )
        if claim.outcome is None:
            return ()
        return tuple(json.loads(claim.outcome))

    def _recognised(self, event: EventEnvelope, outcome: str, identity: str) -> None:
        """Refuse an id that has been reused for something else.

        Deduplicating on an id a publisher reissued would suppress an effect that never
        happened, which is the one failure a dedupe record is supposed to prevent.
        """
        if outcome.rsplit(":", 1)[-1] != identity:
            raise EventIdReuseError(
                f"{event.event_id} was already recorded for a different event",
                run_id=event.run_id,
                tenant=event.tenant,
            )


def _reconsidered(event: EventEnvelope, outcome: str) -> bool:
    """Whether an operator has decided this buried event should be handled after all.

    A marker that says the handler already ran still suppresses, replay or not — that is the
    double-charge it exists to prevent. A marker that says the event was given up on is a
    decision, and the replay is an operator overturning it.
    """
    return bool(event.replay_id) and outcome.startswith(f"{_POISON}:")


def _failures_key(key: str) -> str:
    return f"{key}:failures"


def _identity(event: EventEnvelope) -> str:
    """A digest of everything about the event except the id it claims to be.

    The replay marker is dropped with it: a replay is the same event coming round again,
    and an identity that changed with the marker would read as an id reused for something
    else, which is exactly the suppression a replay must not lose.
    """
    body = event.model_dump(mode="json")
    body.pop("event_id", None)
    body.pop("replay_id", None)
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
