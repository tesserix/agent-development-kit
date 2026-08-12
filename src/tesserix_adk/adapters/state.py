"""Run state and work queues in Redis, on an instance that has been asked to keep them.

Products already keep session state in Redis, each with its own key layout and no version
discipline, on an instance nobody checked was configured to hold anything. A cache with an
eviction policy will drop a run mid-flight and answer the next read with "no such run",
which is indistinguishable from a run that never started. So these stores refuse to open
against an instance that may evict what they write, and say what to change.

Everything that decides — the version compare, the retry-or-dead-letter choice, the
backoff — lives in `core`, and the Lua here exists only to make the decision land
atomically. The scripts read fields and move index entries; they never do the arithmetic.

The trade-offs, the key layout and the settings a deployment must set are in
`docs/state-adapters.md`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import Field

from tesserix_adk.adapters.memory import MemoryStoreSettings
from tesserix_adk.core.errors import (
    ConfigurationError,
    LeaseLostError,
    PoolExhaustedError,
    QueueUnavailableError,
    StateConflictError,
    StateInUseError,
    StateNotFoundError,
    StatePersistenceError,
    WorkItemNotFoundError,
)
from tesserix_adk.core.primitives import Usage
from tesserix_adk.core.queue import QueuePolicy, QueueStats, WorkItem, WorkState
from tesserix_adk.core.run import RunState
from tesserix_adk.core.state import RunRecord, SessionRecord, StateDelta, StateKey, StatePage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.core.state import StateQuery

__all__ = [
    "COUNTERS",
    "DEFAULT_QUEUE_NAMESPACE",
    "DEFAULT_STATE_NAMESPACE",
    "InspectableRedis",
    "RedisStateStore",
    "RedisStoreSettings",
    "RedisWorkQueue",
]

DEFAULT_STATE_NAMESPACE = "adk:state"
DEFAULT_QUEUE_NAMESPACE = "adk:queue"

COUNTERS = ("version", "input_tokens", "output_tokens", "cost_micros", "iterations", "cursor")
"""The run fields that accumulate, kept in a hash so two workers' patches both land.

They are not in the JSON blob: `HINCRBY` commutes and a read-modify-write of a blob does
not. The version is here with them, so a patch bumps it and a stale compare-and-set loses.
"""

T = TypeVar("T")


class RedisStoreSettings(MemoryStoreSettings):
    """How to reach Redis, and what this deployment needs it to promise.

    The connection half is the same for every adapter; only the promises differ.

    Args:
        durable: Whether what is written must still be there later. Set it false only for
            a queue whose work can be recreated from somewhere else.
        max_value_bytes: The largest record this store will write. Redis holds far more
            than this, but a value big enough to matter is a value that blocks the single
            thread every other worker is waiting on.
    """

    durable: bool = True
    max_value_bytes: int = Field(default=262_144, ge=1)


class RedisStateStore:
    """Session and run state in Redis, with the version compare inside the write.

    Each run is a JSON blob and a small hash. The blob holds what a worker sets; the hash
    holds the numbers that accumulate and the version. A write states the version it read
    and the script commits at that version plus one or refuses, so two workers holding the
    same run cannot both think they wrote it.

    Args:
        client: A Redis client that can `eval` and be asked for its configuration.
        clock: Where `updated_at` comes from.
        settings: Connection, retry and size policy.
        namespace: Key prefix, so this can share an instance with everything else.
        session_ttl_seconds: How long a session lives without being written, or None to
            keep it until something deletes it. Runs never expire: a run that vanishes
            mid-flight is a run nothing can resume and nothing can reap.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        client: InspectableRedis,
        *,
        clock: Clock,
        settings: RedisStoreSettings,
        namespace: str = DEFAULT_STATE_NAMESPACE,
        session_ttl_seconds: float | None = None,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._client = client
        self._clock = clock
        self._settings = settings
        self._namespace = namespace
        self._ttl = session_ttl_seconds
        self._retry = _Retrying(clock, settings, entropy, _state_failure(type(self).__name__))

    @classmethod
    async def open(cls, client: InspectableRedis, **kwargs: Any) -> RedisStateStore:  # noqa: ANN401 — the constructor's own keywords
        """Construct the store and refuse an instance that would evict what it writes.

        The check is here rather than in a runbook because an instance configured as a
        cache loses a run quietly, weeks later, under load.

        Raises:
            ConfigurationError: If the server may evict keys, or keeps nothing across a
                restart, and this deployment asked for durable state.
        """
        store = cls(client, **kwargs)
        await store.preflight()
        return store

    async def preflight(self) -> None:
        """Check the server is configured to hold state. Called at startup, never per request."""
        if not self._settings.durable:
            return
        config = await self._retry(lambda: self._client.config_get("*"))
        _refuse_a_cache(config, type(self).__name__)

    async def get_session(self, key: StateKey) -> SessionRecord | None:
        """Return the session, or None where there is not one."""
        blob = await self._retry(lambda: self._client.eval(GET_BLOB, 1, self._session(key)))
        return None if blob is None else SessionRecord.model_validate_json(_text(blob))

    async def put_session(self, record: SessionRecord) -> SessionRecord:
        """Store the session at its read version plus one, or refuse."""
        written = record.model_copy(
            update={"version": record.version + 1, "updated_at": self._clock.now()}
        )
        blob = self._sized(written.model_dump_json(), written.key)
        reply = await self._retry(
            lambda: self._client.eval(
                PUT_SESSION,
                1,
                self._session(record.key),
                str(record.version),
                blob,
                str(int((self._ttl or 0) * 1_000)),
            )
        )
        _refuse_a_stale_write(reply, record.key, record.version)
        return written

    async def delete_session(self, key: StateKey, *, cascade: bool = False) -> None:
        """Remove the session, and its runs where `cascade` says so."""
        reply = await self._retry(
            lambda: self._client.eval(
                DELETE_SESSION,
                3,
                self._session(key),
                f"{self._session(key)}:runs",
                f"{self._scope(key.tenant)}:runs",
                "1" if cascade else "0",
                self._scope(key.tenant),
                _TERMINAL_STATES,
            )
        )
        answer = [_text(part) for part in reply]
        if answer[0] == "live":
            raise StateInUseError(
                f"{_named(key)} still has runs that have not finished",
                key=_named(key),
                live_runs=tuple(answer[1:]),
            )

    async def get_run(self, key: StateKey) -> RunRecord | None:
        """Return the run, with everything patched into it since it was written."""
        reply = await self._retry(
            lambda: self._client.eval(GET_RUN, 2, self._run(key), f"{self._run(key)}:n")
        )
        return None if reply is None else _merged(reply)

    async def put_run(self, record: RunRecord) -> RunRecord:
        """Store the run at its read version plus one, or refuse."""
        written = record.scrubbed().model_copy(update={"updated_at": self._clock.now()})
        key = record.key
        session = record.session_id or ""
        blob = self._sized(_zeroed(written).model_dump_json(), key)
        reply = await self._retry(
            lambda: self._client.eval(
                PUT_RUN,
                5,
                self._run(key),
                f"{self._run(key)}:n",
                f"{self._scope(key.tenant)}:runs",
                f"{self._scope(key.tenant)}:seq",
                # Unused where the run belongs to no session, but a script takes keys, not maybes.
                self._session(StateKey(tenant=key.tenant, id=session or key.id)) + ":runs",
                str(record.version),
                blob,
                str(record.usage.input_tokens),
                str(record.usage.output_tokens),
                str(record.cost_micros),
                str(record.iterations),
                str(record.message_cursor),
                key.id,
                session,
            )
        )
        _refuse_a_stale_write(reply, key, record.version)
        return written.model_copy(update={"version": record.version + 1})

    async def patch_run(self, key: StateKey, delta: StateDelta) -> RunRecord:
        """Add `delta` to the run, server-side, without reading it first."""
        usage = delta.usage or Usage(input_tokens=0, output_tokens=0)
        reply = await self._retry(
            lambda: self._client.eval(
                PATCH_RUN,
                2,
                self._run(key),
                f"{self._run(key)}:n",
                str(usage.input_tokens),
                str(usage.output_tokens),
                str(delta.cost_micros),
                str(delta.iterations),
                str(delta.messages_read),
            )
        )
        if reply is None:
            raise StateNotFoundError(f"no run {_named(key)}", key=_named(key), kind="run")
        return _merged(reply)

    async def delete_run(self, key: StateKey) -> None:
        """Remove the run, its counters, and its place in the listing."""
        await self._retry(
            lambda: self._client.eval(
                DELETE_RUN,
                3,
                self._run(key),
                f"{self._run(key)}:n",
                f"{self._scope(key.tenant)}:runs",
                key.id,
                self._scope(key.tenant),
            )
        )

    async def list_runs(self, query: StateQuery) -> StatePage:
        """Return one page of the tenant's runs, in the order the store first saw them."""
        reply = await self._retry(
            lambda: self._client.eval(
                LIST_RUNS,
                1,
                f"{self._scope(query.tenant)}:runs",
                self._scope(query.tenant),
                f"({query.cursor}" if query.cursor else "-inf",
                str(query.limit),
                query.state.value if query.state else "",
                "" if query.updated_before is None else str(query.updated_before),
            )
        )
        found = [(_text(entry[0]), _merged(entry[1])) for entry in reply or ()]
        # One more than the page was asked for, because a cursor that leads nowhere loops a reaper.
        more = len(found) > query.limit
        page = found[: query.limit]
        return StatePage(
            records=tuple(record for _, record in page), cursor=page[-1][0] if more else None
        )

    def _sized(self, blob: str, key: StateKey) -> str:
        """The blob, if the store will take it — and a refusal naming what will."""
        size = len(blob.encode())
        if size > self._settings.max_value_bytes:
            raise StatePersistenceError(
                f"{_named(key)} is {size} bytes, over the {self._settings.max_value_bytes}"
                " this store writes; hold a record this size in the PostgreSQL state store",
                store=type(self).__name__,
                reason="too_large",
            )
        return blob

    def _scope(self, tenant: str) -> str:
        return f"{self._namespace}:{{{_segment(tenant)}}}"

    def _run(self, key: StateKey) -> str:
        return f"{self._scope(key.tenant)}:run:{_segment(key.id)}"

    def _session(self, key: StateKey) -> str:
        return f"{self._scope(key.tenant)}:session:{_segment(key.id)}"


class RedisWorkQueue:
    """A work queue in Redis, where the lease is a sorted set and the reaper is a scan.

    A claim moves the item out of its tenant's ready set and into the queue's lease set in
    one script, so two workers cannot hold it whatever they believe about the time. The
    item's own record is written immediately afterwards; a crash between the two leaves an
    item that is leased and says it is queued, which the next `reap` corrects — a lease
    nobody renews lapses whatever the record says.

    Args:
        client: A Redis client that can `eval` and be asked for its configuration.
        clock: Where the store's idea of now comes from. One clock decides every lease:
            each worker passes its own, so a deployment runs them off NTP or accepts
            leases that lapse early on a drifting node.
        settings: Connection, retry and size policy.
        policy: How long a claim lasts and how many attempts an item gets.
        namespace: Key prefix. It is also the cluster hash tag, so one queue lives in one
            slot — sharding is by namespace, not by key.
        entropy: Jitter source, injectable so a test can read the backoff back.
    """

    def __init__(
        self,
        client: InspectableRedis,
        *,
        clock: Clock,
        settings: RedisStoreSettings,
        policy: QueuePolicy | None = None,
        namespace: str = DEFAULT_QUEUE_NAMESPACE,
        entropy: Callable[[], float] = random.random,
    ) -> None:
        self._client = client
        self._clock = clock
        self._settings = settings
        self._policy = policy or QueuePolicy()
        self._base = f"{{{namespace}}}"
        self._retry = _Retrying(clock, settings, entropy, _queue_failure(type(self).__name__))

    @classmethod
    async def open(cls, client: InspectableRedis, **kwargs: Any) -> RedisWorkQueue:  # noqa: ANN401 — the constructor's own keywords
        """Construct the queue and refuse an instance that would evict work off it."""
        queue = cls(client, **kwargs)
        await queue.preflight()
        return queue

    async def preflight(self) -> None:
        """Check the server will hold the work it is given. Called at startup."""
        if not self._settings.durable:
            return
        config = await self._retry(lambda: self._client.config_get("*"))
        _refuse_a_cache(config, type(self).__name__)

    @property
    def policy(self) -> QueuePolicy:
        """How long a claim lasts here, and how many attempts an item gets."""
        return self._policy

    async def enqueue(self, item: WorkItem) -> WorkItem:
        """Add `item`, or return the live item its dedupe key already names."""
        now = self._clock.now()
        placed = item.model_copy(
            update={
                "enqueued_at": item.enqueued_at or now,
                "available_at": item.available_at or now,
                "state": WorkState.QUEUED,
                "worker": None,
                "lease_expires_at": None,
            }
        )
        ref = _ref(placed.tenant, placed.id)
        blob = self._sized(placed)  # Outside the retry: an oversized item is not a bad connection.
        held = await self._retry(
            lambda: self._client.eval(
                ENQUEUE,
                7,
                self._item(ref),
                self._ready(placed.queue, placed.tenant),
                self._ages(placed.queue),
                self._tenants(placed.queue),
                f"{self._base}:queues",
                self._dedupe(placed.tenant, placed.dedupe_key or ref),
                f"{self._base}:counters",
                ref,
                placed.tenant,
                placed.queue,
                blob,
                str(placed.available_at),
                str(placed.enqueued_at),
                "1" if placed.dedupe_key else "0",
                f"{self._base}:item:",
            )
        )
        return placed if held is None else WorkItem.model_validate_json(_text(held))

    async def claim(
        self, *, worker: str, queue: str = "default", lease_seconds: float | None = None
    ) -> WorkItem | None:
        """Take the next due item, taking each tenant's turn in rotation."""
        now = self._clock.now()
        lease = lease_seconds or self._policy.lease_seconds
        reply = await self._retry(
            lambda: self._client.eval(
                CLAIM,
                4,
                self._tenants(queue),
                self._ages(queue),
                self._leases(queue),
                f"{self._base}:holders",
                self._base,
                _segment(queue),
                worker,
                str(now),
                str(now + lease),
                str(_SCAN),
            )
        )
        if reply is None:
            return None
        item = WorkItem.model_validate_json(_text(reply[1]))
        claimed = self._policy.claimed(item, worker=worker, now=now, lease_seconds=lease)
        await self._settle(claimed, worker=worker)
        return claimed

    async def heartbeat(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Extend the claim, up to the total the policy allows a single claim to run for."""
        item = await self._holding(item_id, tenant=tenant, worker=worker)
        now = self._clock.now()
        if self._policy.overheld(item, now):
            raise LeaseLostError(
                f"{item_id!r} has been held for longer than {self._policy.max_lease_seconds}s",
                item_id=item_id,
                worker=worker,
                holder=worker,
                reason="capped",
            )
        renewed = self._policy.renewed(item, now=now)
        await self._settle(renewed, worker=worker)
        return renewed

    async def complete(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """Finish the item. Its dedupe key is free again; the same job may be enqueued anew."""
        item = await self._holding(item_id, tenant=tenant, worker=worker)
        done = self._policy.completed(item)
        await self._settle(done, worker=worker)
        return done

    async def fail(
        self, item_id: str, *, tenant: str, worker: str, error: str, retryable: bool = True
    ) -> WorkItem:
        """Give the item back for a retry after a backoff, or to the dead letter."""
        item = await self._holding(item_id, tenant=tenant, worker=worker)
        given = self._policy.returned(item, error=error, now=self._clock.now(), retryable=retryable)
        await self._settle(given, worker=worker)
        return given

    async def reap(self) -> tuple[WorkItem, ...]:
        """Requeue everything whose lease has lapsed, by this store's clock."""
        now = self._clock.now()
        lapsed = await self._retry(
            lambda: self._client.eval(
                LAPSED, 1, f"{self._base}:queues", self._base, str(now), str(_SCAN)
            )
        )
        return await self._give_back(lapsed, LEASE_LAPSED, backoff=True, reaped=True)

    async def adopt(self, *, worker: str) -> tuple[WorkItem, ...]:
        """Give back what `worker` held before it restarted, without waiting out the lease."""
        owned = await self._retry(
            lambda: self._client.eval(OWNED, 1, self._worker(worker), self._base, str(_SCAN))
        )
        return await self._give_back(owned, WORKER_RESTARTED, backoff=False, reaped=False)

    async def dead_letters(self, *, tenant: str, limit: int = 50) -> tuple[WorkItem, ...]:
        """Return items that ran out of attempts, in the order they were enqueued."""
        blobs = await self._retry(
            lambda: self._client.eval(
                DEAD_LETTERS, 1, self._dead(tenant), self._base, str(limit - 1)
            )
        )
        return tuple(WorkItem.model_validate_json(_text(blob)) for blob in blobs or () if blob)

    async def stats(self, *, queue: str = "default") -> QueueStats:
        """Return the depth, ages and counters a deployment alerts on."""
        reply = await self._retry(
            lambda: self._client.eval(
                STATS,
                3,
                self._ages(queue),
                self._leases(queue),
                f"{self._base}:counters",
                self._dead_count(queue),
            )
        )
        depth, claimed, oldest, dead, reaped, duplicates = (float(_text(v) or 0) for v in reply)
        return QueueStats(
            depth=int(depth),
            claimed=int(claimed),
            dead_lettered=int(dead),
            oldest_age_seconds=max(self._clock.now() - oldest, 0.0) if depth else 0.0,
            reaped=int(reaped),
            duplicates_suppressed=int(duplicates),
        )

    async def _holding(self, item_id: str, *, tenant: str, worker: str) -> WorkItem:
        """The item, if this worker still holds it — and a refusal if it does not."""
        ref = _ref(tenant, item_id)
        reply = await self._retry(
            lambda: self._client.eval(
                HELD,
                2,
                self._item(ref),
                f"{self._base}:holders",
                self._base,
                ref,
                worker,
                str(self._clock.now()),
            )
        )
        answer = [None if part is None else _text(part) for part in reply]
        if answer[0] == "missing":
            raise WorkItemNotFoundError(
                f"no item {item_id!r} in {tenant!r}", item_id=item_id, tenant=tenant
            )
        if answer[0] == "lost":
            raise LeaseLostError(
                f"{item_id!r} is no longer held by {worker!r}",
                item_id=item_id,
                worker=worker,
                holder=answer[1] or None,
                reason=answer[2] or "expired",
            )
        return WorkItem.model_validate_json(answer[1] or "")

    async def _give_back(
        self,
        refs: Any,  # noqa: ANN401 — whatever the scan script returned
        error: str,
        *,
        backoff: bool,
        reaped: bool,
    ) -> tuple[WorkItem, ...]:
        """Return every item in `refs` to the queue, where its lease has not moved since."""
        now = self._clock.now()
        moved = []
        for entry in refs or ():
            item = WorkItem.model_validate_json(_text(entry[0]))
            given = self._policy.returned(item, error=error, now=now, backoff=backoff)
            settled = await self._settle(
                given, worker=item.worker or "", fence=_text(entry[1]), reaped=reaped
            )
            if settled:
                moved.append(given)
        return tuple(moved)

    async def _settle(
        self, item: WorkItem, *, worker: str, fence: str = "", reaped: bool = False
    ) -> bool:
        """Write the item where its new state says it belongs, if the claim still stands."""
        ref = _ref(item.tenant, item.id)
        blob = self._sized(item)  # Outside the retry: an oversized item is not a bad connection.
        reply = await self._retry(
            lambda: self._client.eval(
                SETTLE,
                7,
                self._item(ref),
                f"{self._base}:holders",
                self._leases(item.queue),
                self._ready(item.queue, item.tenant),
                self._ages(item.queue),
                self._dead(item.tenant),
                f"{self._base}:counters",
                self._base,
                ref,
                worker,
                blob,
                str(item.available_at),
                str(item.enqueued_at),
                str(item.lease_expires_at or 0.0),
                fence,
                "1" if reaped else "0",
                self._dead_count(item.queue),
                self._worker(worker),
            )
        )
        return _text(reply) == "ok"

    def _sized(self, item: WorkItem) -> str:
        """The item as it will be written, if the store will take it."""
        blob = item.model_dump_json()
        size = len(blob.encode())
        if size > self._settings.max_value_bytes:
            raise StatePersistenceError(
                f"{item.id!r} is {size} bytes, over the {self._settings.max_value_bytes} this"
                " queue writes; put a payload this size behind a claim check",
                store=type(self).__name__,
                reason="too_large",
            )
        return blob

    def _item(self, ref: str) -> str:
        return f"{self._base}:item:{ref}"

    def _ready(self, queue: str, tenant: str) -> str:
        return f"{self._base}:ready:{_segment(queue)}:{_segment(tenant)}"

    def _ages(self, queue: str) -> str:
        return f"{self._base}:ages:{_segment(queue)}"

    def _tenants(self, queue: str) -> str:
        return f"{self._base}:tenants:{_segment(queue)}"

    def _leases(self, queue: str) -> str:
        return f"{self._base}:leases:{_segment(queue)}"

    def _dead(self, tenant: str) -> str:
        return f"{self._base}:dead:{_segment(tenant)}"

    def _dead_count(self, queue: str) -> str:
        return f"dead:{_segment(queue)}"

    def _worker(self, worker: str) -> str:
        return f"{self._base}:worker:{_segment(worker)}"

    def _dedupe(self, tenant: str, key: str) -> str:
        return f"{self._base}:dedupe:{_segment(tenant)}:{key}"


LEASE_LAPSED = "lease lapsed with the item unfinished"
WORKER_RESTARTED = "the worker holding it restarted"

_SCAN = 200
"""How many entries one script looks at. A script that scanned everything would block."""

_TERMINAL_STATES = "," + ",".join(state.value for state in RunState if state.is_terminal) + ","

_EVICTING = ("allkeys-lru", "allkeys-lfu", "allkeys-random", "volatile-ttl")


class InspectableRedis(Protocol):
    """The two calls these stores make: run a script, and say how the server is configured.

    Narrow because the kit does not want an opinion about which Redis client a deployment
    runs. `CONFIG` cannot be called from Lua, which is why it is a second method rather
    than another script.
    """

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:  # noqa: ANN401 — the reply shape is the script's, not the client's
        """Run a Lua script server-side, which is where atomicity comes from."""
        ...

    async def config_get(self, pattern: str) -> Mapping[str, str]:
        """Return the server's configuration, for the parameters matching `pattern`."""
        ...


@dataclass(frozen=True, slots=True)
class _Retrying:
    """Bounded, jittered retries around one driver call."""

    clock: Clock
    settings: RedisStoreSettings
    entropy: Callable[[], float]
    failed: Callable[[int], Exception]

    async def __call__(self, work: Callable[[], Awaitable[T]]) -> T:
        """Run `work`, retrying transport failures until the attempt budget is spent."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await work()
            except Exception as failure:
                self._reraise_if_terminal(failure, attempt)
                await self.clock.sleep(self._backoff(attempt))

    def _reraise_if_terminal(self, failure: Exception, attempt: int) -> None:
        if "pool" in type(failure).__name__.lower():
            raise PoolExhaustedError("redis connection pool exhausted") from failure
        if attempt >= self.settings.max_attempts:
            raise self.failed(attempt) from failure

    def _backoff(self, attempt: int) -> float:
        """Jittered, because every worker that started together retries together."""
        return self.settings.backoff_seconds * float(2 ** (attempt - 1)) * (0.5 + self.entropy())


def _state_failure(store: str) -> Callable[[int], Exception]:
    """What an unreachable store raises. Never silence: a write that failed is not a write."""

    def failed(attempt: int) -> Exception:
        return StatePersistenceError(
            f"{store} unreachable after {attempt} attempts", store=store, reason="unavailable"
        )

    return failed


def _queue_failure(store: str) -> Callable[[int], Exception]:
    """What an unreachable queue raises, so an enqueue is never a silent drop."""

    def failed(attempt: int) -> Exception:
        return QueueUnavailableError(
            f"{store} unreachable after {attempt} attempts", queue=store, operation="eval"
        )

    return failed


def _refuse_a_cache(config: Mapping[str, str], store: str) -> None:
    """Refuse a server that would evict what was written, or forget it on restart."""
    policy = str(config.get("maxmemory-policy", "noeviction"))
    if policy in _EVICTING:
        raise ConfigurationError(
            f"{store} needs maxmemory-policy noeviction; this server is {policy}, which will"
            " drop a run mid-flight and answer the next read as though it never existed"
        )
    if str(config.get("appendonly", "no")) == "no" and not str(config.get("save", "")).strip():
        raise ConfigurationError(
            f"{store} needs persistence; this server has appendonly off and no save points,"
            " so a restart loses every run in flight"
        )


def _refuse_a_stale_write(reply: Any, key: StateKey, expected: int) -> None:  # noqa: ANN401 — whatever the driver decoded
    """A write against a version that moved is refused, with both numbers."""
    answer = [_text(part) for part in reply]
    if answer[0] == "conflict":
        raise StateConflictError(
            f"{_named(key)} moved to version {answer[1]} while it was held at {expected}",
            key=_named(key),
            expected_version=expected,
            actual_version=int(answer[1]),
        )


def _merged(reply: Sequence[Any]) -> RunRecord:
    """A run as the blob it was written as, plus everything patched into it since."""
    record = RunRecord.model_validate_json(_text(reply[0]))
    version, inputs, outputs, cost, iterations, cursor = (int(_text(n) or 0) for n in reply[1])
    delta = StateDelta(
        usage=Usage(input_tokens=inputs, output_tokens=outputs),
        cost_micros=cost,
        iterations=iterations,
        messages_read=cursor,
    )
    return delta.applied_to(record).model_copy(update={"version": version})


def _zeroed(record: RunRecord) -> RunRecord:
    """The record without the numbers the hash owns, so nothing is counted twice."""
    return record.model_copy(
        update={
            "usage": Usage(input_tokens=0, output_tokens=0),
            "cost_micros": 0,
            "iterations": 0,
            "message_cursor": 0,
            "version": 0,
        }
    )


def _segment(value: str) -> str:
    """A key segment nothing else can imitate, whatever separators are in the value.

    Length-prefixed rather than escaped: a tenant called `a:b` and a tenant called `a`
    holding an id that starts `b:` build the same key under any convention that only joins.
    """
    return f"{len(value.encode())}:{value}"


def _ref(tenant: str, item_id: str) -> str:
    """What an index entry calls an item: its tenant, unambiguously, and then its id."""
    return f"{_segment(tenant)}:{item_id}"


def _named(key: StateKey) -> str:
    return f"{key.tenant}/{key.id}"


def _text(value: Any) -> str:  # noqa: ANN401 — bytes or str, depending on the client's decoding
    """What the server said, as text. Clients differ on whether that is bytes."""
    if value is None:
        return ""
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)


GET_BLOB = """
return redis.call('GET', KEYS[1])
"""

PUT_SESSION = """
local blob = redis.call('GET', KEYS[1])
local current = 0
if blob then current = tonumber(cjson.decode(blob).version) or 0 end
if current ~= tonumber(ARGV[1]) then return {'conflict', tostring(current)} end
local ttl = tonumber(ARGV[3])
if ttl > 0 then redis.call('SET', KEYS[1], ARGV[2], 'PX', ttl)
else redis.call('SET', KEYS[1], ARGV[2]) end
return {'ok', tostring(current + 1)}
"""

DELETE_SESSION = """
local runs = redis.call('SMEMBERS', KEYS[2])
local live = {}
for _, id in ipairs(runs) do
  local blob = redis.call('GET', ARGV[2] .. ':run:' .. #id .. ':' .. id)
  if blob then
    local state = cjson.decode(blob).state
    if not string.find(ARGV[3], ',' .. state .. ',', 1, true) then live[#live + 1] = id end
  end
end
if #live > 0 and ARGV[1] ~= '1' then
  local refusal = {'live'}
  for _, id in ipairs(live) do refusal[#refusal + 1] = id end
  return refusal
end
if ARGV[1] == '1' then
  for _, id in ipairs(runs) do
    local run = ARGV[2] .. ':run:' .. #id .. ':' .. id
    redis.call('DEL', run, run .. ':n')
    redis.call('ZREM', KEYS[3], id)
  end
end
redis.call('DEL', KEYS[1], KEYS[2])
return {'ok'}
"""

COUNTED = """
local function counted(key)
  local n = redis.call('HMGET', key, 'version', 'input_tokens', 'output_tokens',
                       'cost_micros', 'iterations', 'cursor')
  for i = 1, 6 do if not n[i] then n[i] = '0' end end
  return n
end
"""
"""Read the six accumulating fields, absent ones as zero.

Prepended to every script that reads them: a nil inside a Lua table truncates the reply,
so a run whose hash has not been written yet would come back as a record with no version.
"""

GET_RUN = (
    COUNTED
    + """
local blob = redis.call('GET', KEYS[1])
if not blob then return false end
return {blob, counted(KEYS[2])}
"""
)

PUT_RUN = """
local current = tonumber(redis.call('HGET', KEYS[2], 'version')) or 0
if current ~= tonumber(ARGV[1]) then return {'conflict', tostring(current)} end
redis.call('SET', KEYS[1], ARGV[2])
redis.call('HSET', KEYS[2], 'version', current + 1, 'input_tokens', ARGV[3],
           'output_tokens', ARGV[4], 'cost_micros', ARGV[5], 'iterations', ARGV[6],
           'cursor', ARGV[7])
if current == 0 then redis.call('ZADD', KEYS[3], redis.call('INCR', KEYS[4]), ARGV[8]) end
if ARGV[9] ~= '' then redis.call('SADD', KEYS[5], ARGV[8]) end
return {'ok', tostring(current + 1)}
"""

PATCH_RUN = (
    COUNTED
    + """
if redis.call('EXISTS', KEYS[1]) == 0 then return false end
redis.call('HINCRBY', KEYS[2], 'version', 1)
redis.call('HINCRBY', KEYS[2], 'input_tokens', ARGV[1])
redis.call('HINCRBY', KEYS[2], 'output_tokens', ARGV[2])
redis.call('HINCRBY', KEYS[2], 'cost_micros', ARGV[3])
redis.call('HINCRBY', KEYS[2], 'iterations', ARGV[4])
redis.call('HINCRBY', KEYS[2], 'cursor', ARGV[5])
return {redis.call('GET', KEYS[1]), counted(KEYS[2])}
"""
)

DELETE_RUN = """
local blob = redis.call('GET', KEYS[1])
if blob then
  local held = cjson.decode(blob)
  if held.session_id and held.session_id ~= cjson.null then
    local session = ARGV[2] .. ':session:' .. #held.session_id .. ':' .. held.session_id
    redis.call('SREM', session .. ':runs', ARGV[1])
  end
end
redis.call('DEL', KEYS[1], KEYS[2])
redis.call('ZREM', KEYS[3], ARGV[1])
return 1
"""

LIST_RUNS = (
    COUNTED
    + """
local wanted = tonumber(ARGV[3]) + 1
local found = {}
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], ARGV[2], '+inf', 'WITHSCORES',
                       'LIMIT', 0, wanted * 4)
for i = 1, #ids, 2 do
  local run = ARGV[1] .. ':run:' .. #ids[i] .. ':' .. ids[i]
  local blob = redis.call('GET', run)
  if blob then
    local held = cjson.decode(blob)
    local keep = true
    if ARGV[4] ~= '' and held.state ~= ARGV[4] then keep = false end
    if ARGV[5] ~= '' and tonumber(held.updated_at) >= tonumber(ARGV[5]) then keep = false end
    if keep and #found < wanted then
      found[#found + 1] = {ids[i + 1], {blob, counted(run .. ':n')}}
    end
  end
end
return found
"""
)

ENQUEUE = """
local function live(blob)
  if not blob then return nil end
  local held = cjson.decode(blob)
  if held.state == 'completed' or held.state == 'dead_lettered' then return nil end
  return blob
end
if ARGV[7] == '1' then
  local named = redis.call('GET', KEYS[6])
  if named then
    local held = live(redis.call('GET', ARGV[8] .. named))
    if held then redis.call('HINCRBY', KEYS[7], 'duplicates', 1) return held end
  end
end
local held = live(redis.call('GET', KEYS[1]))
if held then redis.call('HINCRBY', KEYS[7], 'duplicates', 1) return held end
redis.call('SET', KEYS[1], ARGV[4])
redis.call('ZADD', KEYS[2], ARGV[5], ARGV[1])
redis.call('ZADD', KEYS[3], ARGV[6], ARGV[1])
redis.call('ZADD', KEYS[4], 'NX', 0, ARGV[2])
redis.call('SADD', KEYS[5], ARGV[3])
if ARGV[7] == '1' then redis.call('SET', KEYS[6], ARGV[1]) end
return false
"""

CLAIM = """
local tenants = redis.call('ZRANGE', KEYS[1], 0, -1)
for _, tenant in ipairs(tenants) do
  local ready = ARGV[1] .. ':ready:' .. ARGV[2] .. ':' .. #tenant .. ':' .. tenant
  local due = redis.call('ZRANGEBYSCORE', ready, '-inf', ARGV[4], 'LIMIT', 0, tonumber(ARGV[6]))
  local best, blob, priority = nil, nil, -1
  for _, ref in ipairs(due) do
    local held = redis.call('GET', ARGV[1] .. ':item:' .. ref)
    if held and tonumber(cjson.decode(held).priority) > priority then
      best, blob, priority = ref, held, tonumber(cjson.decode(held).priority)
    end
  end
  if best then
    redis.call('ZREM', ready, best)
    redis.call('ZREM', KEYS[2], best)
    redis.call('ZADD', KEYS[3], ARGV[5], best)
    redis.call('HSET', KEYS[4], best, ARGV[3])
    redis.call('SADD', ARGV[1] .. ':worker:' .. #ARGV[3] .. ':' .. ARGV[3], best)
    redis.call('ZADD', KEYS[1], redis.call('INCR', ARGV[1] .. ':turn'), tenant)
    return {best, blob}
  end
end
return false
"""

HELD = """
local blob = redis.call('GET', KEYS[1])
if not blob then return {'missing'} end
local holder = redis.call('HGET', KEYS[2], ARGV[2]) or ''
if holder ~= ARGV[3] then
  return {'lost', holder, (holder ~= '') and 'taken' or 'expired'}
end
local queue = cjson.decode(blob).queue
local lease = redis.call('ZSCORE', ARGV[1] .. ':leases:' .. #queue .. ':' .. queue, ARGV[2])
if not lease or tonumber(lease) <= tonumber(ARGV[4]) then
  return {'lost', holder, 'expired'}
end
return {'ok', blob}
"""

SETTLE = """
if ARGV[8] ~= '' and redis.call('ZSCORE', KEYS[3], ARGV[2]) ~= ARGV[8] then return 'moved' end
if redis.call('HGET', KEYS[2], ARGV[2]) ~= ARGV[3] then return 'moved' end
local held = cjson.decode(ARGV[4])
redis.call('SET', KEYS[1], ARGV[4])
if held.state == 'claimed' then
  redis.call('ZADD', KEYS[3], ARGV[7], ARGV[2])
  return 'ok'
end
redis.call('ZREM', KEYS[3], ARGV[2])
redis.call('HDEL', KEYS[2], ARGV[2])
redis.call('SREM', ARGV[11], ARGV[2])
if held.state == 'queued' then
  redis.call('ZADD', KEYS[4], ARGV[5], ARGV[2])
  redis.call('ZADD', KEYS[5], ARGV[6], ARGV[2])
else
  if held.dedupe_key and held.dedupe_key ~= cjson.null then
    redis.call('DEL', ARGV[1] .. ':dedupe:' .. #held.tenant .. ':' .. held.tenant ..
               ':' .. held.dedupe_key)
  end
  if held.state == 'dead_lettered' then
    redis.call('ZADD', KEYS[6], held.enqueued_at, ARGV[2])
    redis.call('HINCRBY', KEYS[7], ARGV[10], 1)
  end
end
if ARGV[9] == '1' then redis.call('HINCRBY', KEYS[7], 'reaped', 1) end
return 'ok'
"""

LAPSED = """
local found = {}
for _, queue in ipairs(redis.call('SMEMBERS', KEYS[1])) do
  local leases = ARGV[1] .. ':leases:' .. #queue .. ':' .. queue
  local refs = redis.call('ZRANGEBYSCORE', leases, '-inf', ARGV[2], 'WITHSCORES',
                          'LIMIT', 0, tonumber(ARGV[3]))
  for i = 1, #refs, 2 do
    local blob = redis.call('GET', ARGV[1] .. ':item:' .. refs[i])
    if blob then found[#found + 1] = {blob, refs[i + 1]} end
  end
end
return found
"""

OWNED = """
local found = {}
for _, ref in ipairs(redis.call('SMEMBERS', KEYS[1])) do
  local blob = redis.call('GET', ARGV[1] .. ':item:' .. ref)
  if blob then
    local queue = cjson.decode(blob).queue
    local lease = redis.call('ZSCORE', ARGV[1] .. ':leases:' .. #queue .. ':' .. queue, ref)
    if lease then found[#found + 1] = {blob, lease} end
  end
end
return found
"""

DEAD_LETTERS = """
local found = {}
for _, ref in ipairs(redis.call('ZRANGE', KEYS[1], 0, tonumber(ARGV[2]))) do
  found[#found + 1] = redis.call('GET', ARGV[1] .. ':item:' .. ref)
end
return found
"""

STATS = """
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
return {
  redis.call('ZCARD', KEYS[1]),
  redis.call('ZCARD', KEYS[2]),
  oldest[2] or '0',
  redis.call('HGET', KEYS[3], ARGV[1]) or '0',
  redis.call('HGET', KEYS[3], 'reaped') or '0',
  redis.call('HGET', KEYS[3], 'duplicates') or '0',
}
"""
