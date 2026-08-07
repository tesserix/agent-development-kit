"""Spend ledgers: in-process, Redis, PostgreSQL, and a coalescing wrapper.

The in-memory one is the reference implementation and the one local development runs
against. The other two put the same semantics in a store every replica can see, which is
the only place a per-tenant ceiling can actually live.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from tesserix_adk.core.errors import BudgetExceededError, BudgetUnavailableError
from tesserix_adk.core.ledger import SEPARATOR, LedgerKey, Reservation, WindowSpend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.core.protocols import Clock

__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "CoalescingLedger",
    "InMemoryLedger",
    "PostgresLedger",
    "RedisClient",
    "RedisLedger",
    "SqlExecutor",
]

DEFAULT_LEASE_SECONDS = 300.0


class _Held:
    """A reservation as the ledger keeps it, with whatever the run has admitted spending."""

    def __init__(self, reservation: Reservation, tenant: str, key: LedgerKey) -> None:
        self.reservation = reservation
        self.tenant = tenant
        self.key = key
        self.progress = Decimal(0)


class _Entry:
    """One settled amount, with the moment it happened, which is what a window filters on."""

    def __init__(self, at: float, amount: Decimal) -> None:
        self.at = at
        self.amount = amount


class InMemoryLedger:
    """A ledger in this process. The reference semantics, and enough for one replica.

    A deployment that runs more than one replica needs a shared store; this one is honest
    about holding the window in memory rather than pretending to coordinate.

    Args:
        clock: Where time comes from. Injected so a window can be tested without waiting
            an hour for it.
        shards: How many counters a busy tenant's writes spread over. Reads sum all of
            them, so the ceiling is unaffected by the choice.
        degraded_allowed: Whether an unreachable store may be carried on through. Off, and
            turning it on is a configuration decision that gets recorded on every hold it
            waves through.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        shards: int = 1,
        degraded_allowed: bool = False,
    ) -> None:
        self._clock = clock
        self._shards = max(shards, 1)
        self._degraded_allowed = degraded_allowed
        self._entries: dict[str, list[list[_Entry]]] = {}
        self._held: dict[str, _Held] = {}
        self._lock = asyncio.Lock()
        self._high_water = clock.now()
        self._broken: BudgetUnavailableError | None = None
        self.reservations = 0
        self.degradations = 0

    def break_with(self, failure: BudgetUnavailableError) -> None:
        """Make the store unreachable, which is what a fail-closed test needs."""
        self._broken = failure

    def _now(self) -> float:
        """Time that never goes backwards.

        A clock stepped back would reopen a calendar bucket that has already been spent,
        which is a second allowance for anybody who can nudge NTP.
        """
        self._high_water = max(self._high_water, self._clock.now())
        return self._high_water

    def _shard(self, key: str) -> int:
        return hash(key) % self._shards if self._shards > 1 else 0

    def _unreachable(self) -> None:
        if self._broken is not None and not self._degraded_allowed:
            raise self._broken

    async def reserve(
        self,
        key: LedgerKey,
        amount: Decimal,
        *,
        ceiling: Decimal,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Reservation:
        """Hold `amount`, atomically with the ceiling check."""
        self._unreachable()
        async with self._lock:
            now = self._now()
            self._sweep(now)
            if self._broken is not None:
                return self._degraded(key, amount, now, lease_seconds)
            spend = self._window(key, now)
            if spend.committed + amount > ceiling:
                raise BudgetExceededError(
                    f"{key.name} has {spend.remaining(ceiling)} left of {ceiling} "
                    f"{spend.currency} and this run asked for {amount}",
                    tenant=key.tenant,
                )
            held = Reservation(
                id=uuid.uuid4().hex,
                key=key.at(now),
                amount=amount,
                taken_at=now,
                expires_at=now + lease_seconds,
            )
            self._held[held.id] = _Held(held, key.tenant, key)
            self.reservations += 1
            return held

    def _degraded(
        self, key: LedgerKey, amount: Decimal, now: float, lease_seconds: float
    ) -> Reservation:
        """A hold nobody checked, marked as one."""
        self.degradations += 1
        return Reservation(
            id=uuid.uuid4().hex,
            key=key.at(now),
            amount=amount,
            taken_at=now,
            expires_at=now + lease_seconds,
            degraded=True,
        )

    async def settle(self, reservation: Reservation, actual: Decimal) -> None:
        """Record `actual` and give back whatever was held over it."""
        self._unreachable()
        async with self._lock:
            now = self._now()
            held = self._claim(reservation, now)
            self._record(held.key, now, actual)

    async def release(self, reservation: Reservation) -> None:
        """Give a hold back unspent."""
        self._unreachable()
        async with self._lock:
            self._claim(reservation, self._now())

    async def record_progress(self, reservation: Reservation, spent: Decimal) -> None:
        """Tell the ledger what a run has spent so far, so a lapsed lease can settle.

        Without it, reconciliation can only release: nothing recorded what the replica
        that died had already been charged for.
        """
        self._unreachable()
        async with self._lock:
            held = self._held.get(reservation.id)
            if held is None:
                raise BudgetUnavailableError(
                    f"reservation {reservation.id} is not open: it was settled, released or "
                    f"expired",
                    tenant=reservation.key.split(":")[0],
                )
            held.progress = spent

    async def read_window(self, key: LedgerKey) -> WindowSpend:
        """What the window holds now."""
        self._unreachable()
        async with self._lock:
            now = self._now()
            self._sweep(now)
            return self._window(key, now)

    async def reconcile(self) -> int:
        """Close out lapsed leases. Settles what a run admitted, releases the rest."""
        self._unreachable()
        async with self._lock:
            return self._sweep(self._now())

    async def forget(self, tenant: str) -> WindowSpend:
        """Drop a tenant's records, returning the aggregate that went."""
        self._unreachable()
        async with self._lock:
            now = self._now()
            prefix = f"{tenant}:"
            settled = Decimal(0)
            for name in [n for n in self._entries if n.startswith(prefix)]:
                settled += sum(
                    (entry.amount for shard in self._entries.pop(name) for entry in shard),
                    Decimal(0),
                )
            reserved = Decimal(0)
            for held_id in [i for i, h in self._held.items() if h.tenant == tenant]:
                reserved += self._held.pop(held_id).reservation.amount
            return WindowSpend(settled=settled, reserved=reserved, resets_at=now)

    def _claim(self, reservation: Reservation, now: float) -> _Held:
        """Take a hold out of the book, or say why it was not there to take."""
        self._sweep(now)
        held = self._held.pop(reservation.id, None)
        if held is None:
            reason = "expired" if reservation.expires_at <= now else "already closed"
            raise BudgetUnavailableError(
                f"reservation {reservation.id} is not open: it {reason}",
                tenant=reservation.key.split(":")[0],
            )
        return held

    def _sweep(self, now: float) -> int:
        """Close every lapsed lease, settling what its run admitted spending."""
        lapsed = [held for held in self._held.values() if held.reservation.expires_at <= now]
        for held in lapsed:
            del self._held[held.reservation.id]
            if held.progress > 0:
                self._record(held.key, now, held.progress)
        return len(lapsed)

    def _record(self, key: LedgerKey, now: float, amount: Decimal) -> None:
        name = key.name
        shards = self._entries.setdefault(name, [[] for _ in range(self._shards)])
        shards[self._shard(f"{name}:{now}:{amount}")].append(_Entry(now, amount))

    def _window(self, key: LedgerKey, now: float) -> WindowSpend:
        opened = key.window.opened_at(now)
        settled = sum(
            (
                entry.amount
                for shard in self._entries.get(key.name, [])
                for entry in shard
                if entry.at >= opened
            ),
            Decimal(0),
        )
        reserved = sum(
            (held.reservation.amount for held in self._held.values() if held.key.name == key.name),
            Decimal(0),
        )
        return WindowSpend(settled=settled, reserved=reserved, resets_at=key.window.resets_at(now))


class _Block:
    """An allowance taken in one round trip and drawn down locally."""

    def __init__(self, reservation: Reservation) -> None:
        self.reservation = reservation
        self.left = reservation.amount
        self.spent = Decimal(0)
        self.outstanding: dict[str, Decimal] = {}


class CoalescingLedger:
    """A ledger that buys allowance in blocks, so not every call is a round trip.

    A shared ledger consulted on every model call adds its latency to every model call.
    Reserving a block and drawing it down locally removes most of those trips without
    weakening the ceiling: the block is *held* in the ledger the whole time, so no other
    replica can spend it, and what the block does not use goes back on `flush`.

    Args:
        inner: The ledger the blocks are taken from.
        block: How much to take at a time. Larger means fewer trips and more allowance
            parked on one replica.
    """

    def __init__(self, inner: InMemoryLedger, *, block: Decimal) -> None:
        self._inner = inner
        self._block = block
        self._blocks: dict[str, _Block] = {}
        self._keys: dict[str, LedgerKey] = {}
        self._lock = asyncio.Lock()

    async def reserve(
        self,
        key: LedgerKey,
        amount: Decimal,
        *,
        ceiling: Decimal,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Reservation:
        """Draw `amount` from the held block, taking another where it does not fit."""
        if amount > self._block:
            return await self._inner.reserve(
                key, amount, ceiling=ceiling, lease_seconds=lease_seconds
            )
        async with self._lock:
            name = key.name
            self._keys[name] = key
            block = self._blocks.get(name)
            if block is None or block.left < amount:
                if block is not None:
                    await self._give_back(block)
                block = _Block(
                    await self._inner.reserve(
                        key, self._block, ceiling=ceiling, lease_seconds=lease_seconds
                    )
                )
                self._blocks[name] = block
            block.left -= amount
            drawn = block.reservation.model_copy(update={"id": uuid.uuid4().hex, "amount": amount})
            block.outstanding[drawn.id] = amount
            return drawn

    async def settle(self, reservation: Reservation, actual: Decimal) -> None:
        """Charge `actual` against the block this draw came from."""
        async with self._lock:
            block = self._holding(reservation)
            if block is None:
                await self._inner.settle(reservation, actual)
                return
            block.left += block.outstanding.pop(reservation.id) - actual
            block.spent += actual

    async def release(self, reservation: Reservation) -> None:
        """Give a draw back to the block."""
        async with self._lock:
            block = self._holding(reservation)
            if block is None:
                await self._inner.release(reservation)
                return
            block.left += block.outstanding.pop(reservation.id)

    async def record_progress(self, reservation: Reservation, spent: Decimal) -> None:
        """Record a draw's progress against the block, so a lapsed block settles for it."""
        async with self._lock:
            block = self._holding(reservation)
            if block is None:
                await self._inner.record_progress(reservation, spent)
                return
            await self._inner.record_progress(block.reservation, block.spent + spent)

    async def read_window(self, key: LedgerKey) -> WindowSpend:
        """What the ledger underneath holds, blocks included."""
        return await self._inner.read_window(key)

    async def reconcile(self) -> int:
        """Reconciliation belongs to the store, not to the cache in front of it."""
        return await self._inner.reconcile()

    async def forget(self, tenant: str) -> WindowSpend:
        """Drop this replica's blocks for `tenant`, then erase it underneath."""
        async with self._lock:
            for name in [n for n, k in self._keys.items() if k.tenant == tenant]:
                self._blocks.pop(name, None)
                del self._keys[name]
            return await self._inner.forget(tenant)

    async def flush(self) -> None:
        """Settle every held block for what it actually spent, giving back the rest."""
        async with self._lock:
            for name, block in list(self._blocks.items()):
                await self._give_back(block)
                del self._blocks[name]

    async def _give_back(self, block: _Block) -> None:
        await self._inner.settle(block.reservation, block.spent)

    def _holding(self, reservation: Reservation) -> _Block | None:
        return next(
            (b for b in self._blocks.values() if reservation.id in b.outstanding),
            None,
        )


class RedisClient(Protocol):
    """The one call the Redis ledger makes, and nothing else.

    Narrow because the kit does not want an opinion about which Redis client a deployment
    runs, nor a hard dependency on one in order to type-check.
    """

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        """Run a Lua script server-side, which is where atomicity comes from."""
        ...


class SqlExecutor(Protocol):
    """The one call the PostgreSQL ledger makes."""

    async def fetch(self, statement: str, *args: Any) -> Sequence[Sequence[Any]]:
        """Run a statement and return its rows."""
        ...


class _Shared:
    """What both shared stores do the same way: time, shards and failing closed."""

    def __init__(self, clock: Clock, shards: int, degraded_allowed: bool) -> None:
        self._clock = clock
        self._shards = max(shards, 1)
        self._degraded_allowed = degraded_allowed
        self._high_water = clock.now()
        self.degradations = 0

    def _now(self) -> float:
        """Time that never goes backwards, so a stepped clock cannot reopen a window."""
        self._high_water = max(self._high_water, self._clock.now())
        return self._high_water

    def _shard(self, seed: str) -> int:
        return hash(seed) % self._shards if self._shards > 1 else 0

    def _degrade(
        self, failure: Exception, key: LedgerKey, amount: Decimal, now: float, lease_seconds: float
    ) -> Reservation:
        """Carry on unchecked, if a deployment said in advance that it wanted to."""
        if not self._degraded_allowed:
            raise BudgetUnavailableError(str(failure), tenant=key.tenant) from failure
        self.degradations += 1
        return Reservation(
            id=uuid.uuid4().hex,
            key=key.at(now),
            amount=amount,
            taken_at=now,
            expires_at=now + lease_seconds,
            degraded=True,
        )

    def _unavailable(self, failure: Exception, tenant: str) -> BudgetUnavailableError:
        return BudgetUnavailableError(str(failure), tenant=tenant)


class RedisLedger(_Shared):
    """The window in Redis, with every operation a single server-side script.

    Atomicity is the whole point: the ceiling check and the hold have to happen without
    another replica slipping between them, which no sequence of round trips can promise.
    The scripts are exercised against a real server by `SpendLedgerConformance`.

    Args:
        client: Anything that can `eval` Lua. The `redis` extra installs one.
        clock: Where time comes from.
        namespace: Key prefix, so a ledger can share a Redis with everything else.
        shards: How many counters a busy tenant's writes spread over.
        degraded_allowed: Whether an unreachable server may be carried on through.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        clock: Clock,
        namespace: str = "adk:ledger",
        shards: int = 1,
        degraded_allowed: bool = False,
    ) -> None:
        super().__init__(clock, shards, degraded_allowed)
        self._client = client
        self._namespace = namespace

    def _base(self, key: LedgerKey, now: float) -> str:
        return f"{self._namespace}:{key.at(now)}"

    async def reserve(
        self,
        key: LedgerKey,
        amount: Decimal,
        *,
        ceiling: Decimal,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Reservation:
        """Hold `amount`, with the ceiling checked inside the script that holds it."""
        now = self._now()
        base = self._base(key, now)
        held_id = uuid.uuid4().hex
        try:
            granted, remaining = await self._client.eval(
                RESERVE,
                2,
                f"{base}:s{self._shard(held_id)}",
                f"{base}:held",
                str(now),
                str(float(key.window.seconds)),
                str(amount),
                str(ceiling),
                held_id,
                str(now + lease_seconds),
                str(self._shards),
            )
        except Exception as failure:  # any transport failure reads as unavailability
            return self._degrade(failure, key, amount, now, lease_seconds)
        if not int(granted):
            raise BudgetExceededError(
                f"{key.name} has {remaining} left of {ceiling} and this run asked for {amount}",
                tenant=key.tenant,
            )
        return Reservation(
            id=held_id,
            key=key.at(now),
            amount=amount,
            taken_at=now,
            expires_at=now + lease_seconds,
        )

    async def settle(self, reservation: Reservation, actual: Decimal) -> None:
        """Close the hold and record `actual` against the window it came from."""
        await self._close(reservation, actual)

    async def release(self, reservation: Reservation) -> None:
        """Close the hold, recording nothing."""
        await self._close(reservation, Decimal(0))

    async def _close(self, reservation: Reservation, actual: Decimal) -> None:
        if reservation.degraded:
            self.degradations += 1
            return
        tenant = reservation.key.split(SEPARATOR)[0]
        now = self._now()
        base = f"{self._namespace}:{reservation.key}"
        try:
            closed = await self._client.eval(
                SETTLE,
                2,
                f"{base}:s{self._shard(reservation.id)}",
                f"{base}:held",
                reservation.id,
                str(actual),
                str(now),
            )
        except Exception as failure:
            raise self._unavailable(failure, tenant) from failure
        if not int(closed):
            raise BudgetUnavailableError(
                f"reservation {reservation.id} is not open: it expired, settled or was released",
                tenant=tenant,
            )

    async def record_progress(self, reservation: Reservation, spent: Decimal) -> None:
        """Tell the server what this run has spent, so a lapsed lease settles for it."""
        tenant = reservation.key.split(SEPARATOR)[0]
        base = f"{self._namespace}:{reservation.key}"
        try:
            recorded = await self._client.eval(
                PROGRESS, 1, f"{base}:held", reservation.id, str(spent)
            )
        except Exception as failure:
            raise self._unavailable(failure, tenant) from failure
        if not int(recorded):
            raise BudgetUnavailableError(f"reservation {reservation.id} is not open", tenant=tenant)

    async def read_window(self, key: LedgerKey) -> WindowSpend:
        """Sum every shard and every live hold, which is what a ceiling is checked against."""
        now = self._now()
        base = self._base(key, now)
        try:
            settled, reserved = await self._client.eval(
                READ,
                2,
                f"{base}:s",
                f"{base}:held",
                str(now),
                str(float(key.window.seconds)),
                str(self._shards),
            )
        except Exception as failure:
            raise self._unavailable(failure, key.tenant) from failure
        return WindowSpend(
            settled=Decimal(settled),
            reserved=Decimal(reserved),
            resets_at=key.window.resets_at(now),
        )

    async def reconcile(self) -> int:
        """Sweep lapsed leases across the namespace, settling admitted progress."""
        try:
            return int(
                await self._client.eval(RECONCILE, 0, f"{self._namespace}:*:held", str(self._now()))
            )
        except Exception as failure:
            raise self._unavailable(failure, "") from failure

    async def forget(self, tenant: str) -> WindowSpend:
        """Delete every key under this tenant, returning the aggregate that went."""
        try:
            settled, reserved = await self._client.eval(FORGET, 0, f"{self._namespace}:{tenant}:*")
        except Exception as failure:
            raise self._unavailable(failure, tenant) from failure
        return WindowSpend(settled=Decimal(settled), reserved=Decimal(reserved))


class PostgresLedger(_Shared):
    """The window in PostgreSQL, with every operation a single statement.

    One statement rather than a transaction of several, because two statements are two
    chances for another replica to slip between them. The SQL is exercised against a real
    database by `SpendLedgerConformance`.

    Args:
        executor: Anything that can `fetch` rows. The `postgres` extra installs one.
        clock: Where time comes from.
        table: Where the ledger lives.
        shards: Recorded on each row, so a busy tenant's writes spread over them.
        degraded_allowed: Whether an unreachable database may be carried on through.
    """

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        clock: Clock,
        table: str = "adk_spend_ledger",
        shards: int = 1,
        degraded_allowed: bool = False,
    ) -> None:
        super().__init__(clock, shards, degraded_allowed)
        self._sql = executor
        self._table = table

    async def ensure_schema(self) -> None:
        """Create the table if it is not there. Called by a deployment, never on a request.

        A ledger that runs DDL while serving fails at the worst moment, and a migration is
        a deployment's decision, not a library's.
        """
        await self._sql.fetch(SCHEMA.format(table=self._table))

    async def reserve(
        self,
        key: LedgerKey,
        amount: Decimal,
        *,
        ceiling: Decimal,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Reservation:
        """Hold `amount`, with the ceiling checked inside the statement that holds it."""
        now = self._now()
        held_id = uuid.uuid4().hex
        try:
            rows = await self._sql.fetch(
                RESERVE_SQL.format(table=self._table),
                key.at(now),
                key.tenant,
                held_id,
                amount,
                ceiling,
                now,
                float(key.window.seconds),
                now + lease_seconds,
                self._shard(held_id),
            )
        except Exception as failure:  # any transport failure reads as unavailability
            return self._degrade(failure, key, amount, now, lease_seconds)
        granted, remaining = rows[0][0], rows[0][1]
        if not granted:
            raise BudgetExceededError(
                f"{key.name} has {remaining} left of {ceiling} and this run asked for {amount}",
                tenant=key.tenant,
            )
        return Reservation(
            id=held_id,
            key=key.at(now),
            amount=amount,
            taken_at=now,
            expires_at=now + lease_seconds,
        )

    async def settle(self, reservation: Reservation, actual: Decimal) -> None:
        """Close the hold and record `actual`."""
        await self._close(reservation, actual)

    async def release(self, reservation: Reservation) -> None:
        """Close the hold, recording nothing."""
        await self._close(reservation, Decimal(0))

    async def _close(self, reservation: Reservation, actual: Decimal) -> None:
        if reservation.degraded:
            self.degradations += 1
            return
        tenant = reservation.key.split(SEPARATOR)[0]
        rows = await self._rows(
            SETTLE_SQL.format(table=self._table), tenant, reservation.id, actual, self._now()
        )
        if not rows:
            raise BudgetUnavailableError(
                f"reservation {reservation.id} is not open: it expired, settled or was released",
                tenant=tenant,
            )

    async def record_progress(self, reservation: Reservation, spent: Decimal) -> None:
        """Record what this run has spent so far against its open hold."""
        tenant = reservation.key.split(SEPARATOR)[0]
        rows = await self._rows(
            PROGRESS_SQL.format(table=self._table), tenant, reservation.id, spent
        )
        if not rows:
            raise BudgetUnavailableError(f"reservation {reservation.id} is not open", tenant=tenant)

    async def read_window(self, key: LedgerKey) -> WindowSpend:
        """Sum the window, holds included."""
        now = self._now()
        rows = await self._rows(
            READ_SQL.format(table=self._table),
            key.tenant,
            key.at(now),
            now,
            float(key.window.seconds),
        )
        settled, reserved = rows[0][0], rows[0][1]
        return WindowSpend(
            settled=Decimal(settled or 0),
            reserved=Decimal(reserved or 0),
            resets_at=key.window.resets_at(now),
        )

    async def reconcile(self) -> int:
        """Close lapsed leases, settling whatever progress they admitted."""
        rows = await self._rows(RECONCILE_SQL.format(table=self._table), "", self._now())
        return int(rows[0][0])

    async def forget(self, tenant: str) -> WindowSpend:
        """Delete the tenant's rows, returning the aggregate that went."""
        rows = await self._rows(FORGET_SQL.format(table=self._table), tenant, tenant)
        settled, reserved = rows[0][0], rows[0][1]
        return WindowSpend(settled=Decimal(settled or 0), reserved=Decimal(reserved or 0))

    async def _rows(self, statement: str, tenant: str, *args: Any) -> Sequence[Sequence[Any]]:
        try:
            return await self._sql.fetch(statement, *args)
        except Exception as failure:
            raise self._unavailable(failure, tenant) from failure


# The scripts below run server-side. Every one is a single round trip on purpose: the
# ceiling check and the write it authorises cannot be two trips without a race between them.

RESERVE = """
local settled, held = KEYS[1], KEYS[2]
local now, length = tonumber(ARGV[1]), tonumber(ARGV[2])
local amount, ceiling = tonumber(ARGV[3]), tonumber(ARGV[4])
local id, expires, shards = ARGV[5], tonumber(ARGV[6]), tonumber(ARGV[7])
local opened = now - length
local base = string.sub(settled, 1, string.find(settled, ':s%d*$') - 1)
local total = 0
for shard = 0, shards - 1 do
  local zset = base .. ':s' .. shard
  redis.call('ZREMRANGEBYSCORE', zset, '-inf', '(' .. opened)
  for _, member in ipairs(redis.call('ZRANGEBYSCORE', zset, opened, '+inf')) do
    total = total + tonumber(string.match(member, '|([^|]+)$'))
  end
end
local reserved = 0
local open = redis.call('HGETALL', held)
for i = 1, #open, 2 do
  local a, e, p = string.match(open[i + 1], '([^|]*)|([^|]*)|([^|]*)')
  if tonumber(e) <= now then
    redis.call('HDEL', held, open[i])
    if tonumber(p) > 0 then
      redis.call('ZADD', settled, now, open[i] .. '|' .. p)
      total = total + tonumber(p)
    end
  else
    reserved = reserved + tonumber(a)
  end
end
local left = ceiling - total - reserved
if amount > left then return {0, tostring(left)} end
redis.call('HSET', held, id, amount .. '|' .. expires .. '|0')
redis.call('EXPIRE', held, math.ceil(length) * 2)
return {1, tostring(left - amount)}
"""

SETTLE = """
local settled, held = KEYS[1], KEYS[2]
local id, actual, now = ARGV[1], tonumber(ARGV[2]), tonumber(ARGV[3])
if redis.call('HDEL', held, id) == 0 then return 0 end
if actual > 0 then redis.call('ZADD', settled, now, id .. '|' .. actual) end
return 1
"""

PROGRESS = """
local held, id, spent = KEYS[1], ARGV[1], ARGV[2]
local current = redis.call('HGET', held, id)
if not current then return 0 end
local amount, expires = string.match(current, '([^|]*)|([^|]*)|')
redis.call('HSET', held, id, amount .. '|' .. expires .. '|' .. spent)
return 1
"""

READ = """
local base, held = KEYS[1], KEYS[2]
local now, length, shards = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local opened = now - length
local total = 0
for shard = 0, shards - 1 do
  for _, member in ipairs(redis.call('ZRANGEBYSCORE', base .. shard, opened, '+inf')) do
    total = total + tonumber(string.match(member, '|([^|]+)$'))
  end
end
local reserved = 0
local open = redis.call('HGETALL', held)
for i = 1, #open, 2 do
  local a, e = string.match(open[i + 1], '([^|]*)|([^|]*)|')
  if tonumber(e) > now then reserved = reserved + tonumber(a) end
end
return {tostring(total), tostring(reserved)}
"""

RECONCILE = """
local pattern, now = ARGV[1], tonumber(ARGV[2])
local closed, cursor = 0, '0'
repeat
  local scan = redis.call('SCAN', cursor, 'MATCH', pattern, 'COUNT', 200)
  cursor = scan[1]
  for _, held in ipairs(scan[2]) do
    local settled = string.gsub(held, ':held$', ':s0')
    local open = redis.call('HGETALL', held)
    for i = 1, #open, 2 do
      local _, e, p = string.match(open[i + 1], '([^|]*)|([^|]*)|([^|]*)')
      if tonumber(e) <= now then
        redis.call('HDEL', held, open[i])
        if tonumber(p) > 0 then
          redis.call('ZADD', settled, now, open[i] .. '|' .. p)
        end
        closed = closed + 1
      end
    end
  end
until cursor == '0'
return closed
"""

FORGET = """
local pattern = ARGV[1]
local settled, reserved, cursor = 0, 0, '0'
repeat
  local scan = redis.call('SCAN', cursor, 'MATCH', pattern, 'COUNT', 200)
  cursor = scan[1]
  for _, name in ipairs(scan[2]) do
    if redis.call('TYPE', name).ok == 'zset' then
      for _, member in ipairs(redis.call('ZRANGE', name, 0, -1)) do
        settled = settled + tonumber(string.match(member, '|([^|]+)$'))
      end
    else
      local open = redis.call('HGETALL', name)
      for i = 1, #open, 2 do
        reserved = reserved + tonumber(string.match(open[i + 1], '([^|]*)|'))
      end
    end
    redis.call('DEL', name)
  end
until cursor == '0'
return {tostring(settled), tostring(reserved)}
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    id          text PRIMARY KEY,
    window_key  text NOT NULL,
    tenant      text NOT NULL,
    amount      numeric NOT NULL,
    progress    numeric NOT NULL DEFAULT 0,
    settled     numeric,
    at          double precision NOT NULL,
    expires_at  double precision NOT NULL,
    shard       integer NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS {table}_window ON {table} (window_key, at);
CREATE INDEX IF NOT EXISTS {table}_tenant ON {table} (tenant);
"""

RESERVE_SQL = """
WITH swept AS (
    UPDATE {table} SET settled = progress
    WHERE window_key = $1 AND settled IS NULL AND expires_at <= $6
    RETURNING 1
), window_now AS (
    SELECT coalesce(sum(settled), 0) AS spent,
           coalesce(sum(amount) FILTER (WHERE settled IS NULL), 0) AS held
    FROM {table}
    WHERE window_key = $1 AND at >= $6 - $7
), taken AS (
    INSERT INTO {table} (id, window_key, tenant, amount, at, expires_at, shard)
    SELECT $3, $1, $2, $4, $6, $8, $9 FROM window_now
    WHERE spent + held + $4 <= $5
    RETURNING 1
)
SELECT EXISTS (SELECT 1 FROM taken) AS granted,
       (SELECT $5 - spent - held FROM window_now) AS remaining
"""

SETTLE_SQL = """
UPDATE {table} SET settled = $3
WHERE id = $2 AND tenant = $1 AND settled IS NULL AND expires_at > $4
RETURNING id
"""

PROGRESS_SQL = """
UPDATE {table} SET progress = $3
WHERE id = $2 AND tenant = $1 AND settled IS NULL
RETURNING id
"""

READ_SQL = """
SELECT coalesce(sum(settled), 0),
       coalesce(sum(amount) FILTER (WHERE settled IS NULL AND expires_at > $3), 0)
FROM {table}
WHERE tenant = $1 AND window_key = $2 AND at >= $3 - $4
"""

RECONCILE_SQL = """
WITH closed AS (
    UPDATE {table} SET settled = progress
    WHERE settled IS NULL AND expires_at <= $2
    RETURNING 1
)
SELECT count(*) FROM closed
"""

FORGET_SQL = """
WITH gone AS (
    DELETE FROM {table} WHERE tenant = $1 OR tenant = $2
    RETURNING settled, amount, settled IS NULL AS open
)
SELECT coalesce(sum(settled), 0), coalesce(sum(amount) FILTER (WHERE open), 0) FROM gone
"""
