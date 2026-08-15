"""Holding a run while advancing it, and giving it up when the turn is over.

`Leaseholder` is what a worker uses: it takes the lease before reading the frontier, renews
it while the turn runs, and releases it at the end so the next worker does not wait out the
whole TTL. `MemoryLeaseStore` is the same decision in a dict, for tests and single-process
deployments.

Nothing here consults the worker's own clock for expiry. The store owns the clock, because
it is the only one every worker shares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.errors import RunLeaseError
from tesserix_adk.core.lease import DEFAULT_LEASE, RunLease

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

    from tesserix_adk.core.lease import LeasePolicy, LeaseStore
    from tesserix_adk.core.protocols import Clock

__all__ = ["Leaseholder", "MemoryLeaseStore"]


class MemoryLeaseStore:
    """A `LeaseStore` in a dict, deciding expiry on one clock.

    Nothing here outlives the process, so it protects a run from the other workers in this
    one and from nobody else. It exists so resume can be exercised without Redis.

    Args:
        clock: The store's clock. Every expiry decision is taken on it, and never on a
            timestamp a caller passed in.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._held: dict[tuple[str, str], RunLease] = {}

    async def acquire(
        self, run_id: str, *, tenant: str, holder: str, ttl_seconds: float
    ) -> RunLease:
        """Take the lease on `run_id`, or refuse because a live one is held elsewhere.

        Raises:
            RunLeaseError: If another holder's lease has not expired.
        """
        now = self._clock.now()
        current = self._held.get((tenant, run_id))
        if current is not None and current.held_at(now) and current.holder != holder:
            raise RunLeaseError(
                f"{run_id} is held by {current.holder} until {current.expires_at:g}",
                run_id=run_id,
                holder=current.holder,
                requested_by=holder,
                fence=current.fence,
            )
        taken = RunLease(
            run_id=run_id,
            tenant=tenant,
            holder=holder,
            fence=1 if current is None else current.fence + 1,
            expires_at=now + ttl_seconds,
        )
        self._held[tenant, run_id] = taken
        return taken

    async def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        """Extend `lease`, or refuse because it is no longer the current one.

        Raises:
            RunLeaseError: If the fence has moved on, which is what a lease taken by
                somebody else looks like from here.
        """
        current = self._held.get((lease.tenant, lease.run_id))
        if current is None or current.fence != lease.fence or current.holder != lease.holder:
            holder = "" if current is None else current.holder
            raise RunLeaseError(
                f"{lease.run_id} is no longer held by {lease.holder}",
                run_id=lease.run_id,
                holder=holder,
                requested_by=lease.holder,
                fence=lease.fence,
            )
        renewed = lease.model_copy(update={"expires_at": self._clock.now() + ttl_seconds})
        self._held[lease.tenant, lease.run_id] = renewed
        return renewed

    async def release(self, lease: RunLease) -> None:
        """Give the lease up, unless it has already been taken by someone else."""
        current = self._held.get((lease.tenant, lease.run_id))
        if current is not None and current.fence == lease.fence:
            del self._held[lease.tenant, lease.run_id]

    async def held(self, run_id: str, *, tenant: str) -> RunLease | None:
        """Return the current lease on `run_id`, expired or not."""
        return self._held.get((tenant, run_id))


class Leaseholder:
    """One worker's hold on one run, for as long as the turn takes.

    Used as an async context manager, so the lease is released on the way out however the
    turn ended. A holder that dies without releasing costs the next worker the remaining
    TTL, which is the price of not having to trust a dead process to tidy up.

    Args:
        store: Where the one-holder-at-a-time decision is made.
        holder: This worker's name, as it should appear in the rejection another gets.
        policy: How long a hold lasts and when it is renewed.
    """

    def __init__(
        self, store: LeaseStore, *, holder: str, policy: LeasePolicy = DEFAULT_LEASE
    ) -> None:
        self._store = store
        self._holder = holder
        self._policy = policy
        self._lease: RunLease | None = None

    @property
    def lease(self) -> RunLease | None:
        """The lease currently held, or `None` before acquisition and after release."""
        return self._lease

    @property
    def fence(self) -> int:
        """The fencing token to stamp on writes. Zero while nothing is held."""
        return 0 if self._lease is None else self._lease.fence

    async def acquire(self, run_id: str, *, tenant: str) -> RunLease:
        """Take the run, or raise because someone else has it.

        Raises:
            RunLeaseError: If another worker holds a live lease on the run.
        """
        self._lease = await self._store.acquire(
            run_id, tenant=tenant, holder=self._holder, ttl_seconds=self._policy.ttl_seconds
        )
        return self._lease

    async def renew(self) -> RunLease:
        """Extend the hold, so a long turn does not lose the run halfway through.

        Raises:
            RunLeaseError: If nothing is held, or the lease has been taken by another worker.
        """
        if self._lease is None:
            raise RunLeaseError("nothing is held to renew", requested_by=self._holder)
        self._lease = await self._store.renew(self._lease, ttl_seconds=self._policy.ttl_seconds)
        return self._lease

    async def keep(self, *, now: float) -> RunLease | None:
        """Renew only if the policy says the lease is close enough to expiry to need it.

        Args:
            now: The store's clock, as the last acquisition or renewal reported it.

        Returns:
            The lease, renewed or not, or `None` while nothing is held.
        """
        if self._lease is None:
            return None
        if self._policy.due_to_renew(self._lease, now=now):
            return await self.renew()
        return self._lease

    async def release(self) -> None:
        """Give the run up. Releasing nothing, or a lease already taken, is not an error."""
        if self._lease is not None:
            await self._store.release(self._lease)
            self._lease = None

    async def __aenter__(self) -> Self:
        """Enter with nothing held; `acquire` is explicit because it can be refused."""
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the run however the turn ended."""
        await self.release()
