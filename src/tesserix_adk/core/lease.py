"""Which worker is allowed to advance a run, and how the others find out they are not.

A checkpoint says where a run got to. It does not say who may carry it on, and two workers
that both read the same frontier both dispatch the same outstanding call. The lease is that
missing half: one holder at a time, an expiry so a dead worker does not strand the run, and a
fencing token so a holder whose lease has since been taken cannot write as though it still
held one.

Expiry is decided by the store's clock, never by the holder's. A worker whose clock runs fast
would otherwise treat a live lease as expired and take it; a worker whose clock runs slow
would keep writing after its own had gone. The fence closes the same gap from the other side:
whatever either clock believes, a write carrying a fence below the current one is refused.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.models import AdkModel

__all__ = ["DEFAULT_LEASE", "LeasePolicy", "LeaseStore", "RunLease"]


class RunLease(AdkModel):
    """One worker's claim on one run, for a bounded time.

    Args:
        run_id: The run it covers.
        tenant: The isolation boundary. A lease never spans one.
        holder: Which worker holds it, as a deployment names its workers. It appears in the
            rejection the loser gets, because "someone else has it" is not a diagnosis.
        fence: Monotonic per run. Every acquisition increases it, so a write from a holder
            whose lease was taken is refused on the number rather than on a timestamp.
        expires_at: Unix seconds, on the store's clock, after which anyone may take it.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    holder: str = Field(min_length=1)
    fence: int = Field(default=1, ge=1)
    expires_at: float = 0.0

    def held_at(self, now: float) -> bool:
        """Whether this lease is still live at `now`, which must be the store's clock.

        Example:
            >>> RunLease(run_id="r1", tenant="acme", holder="w1", expires_at=100.0).held_at(99.0)
            True
        """
        return now < self.expires_at

    def superseded_by(self, other: RunLease) -> bool:
        """Whether `other` has taken this lease's place, on the fence rather than the clock."""
        return other.fence > self.fence


class LeasePolicy(AdkModel):
    """How long a lease lives and when its holder should renew it.

    Args:
        ttl_seconds: How long one acquisition lasts. Long enough to finish a turn, short
            enough that a worker lost to a pod roll does not hold a run for an hour.
        renew_within: How far ahead of expiry a holder renews. A holder that renews only at
            expiry renews after it, on any pause longer than the network round trip.

    Example:
        >>> DEFAULT_LEASE.due_to_renew(RunLease(
        ...     run_id="r1", tenant="acme", holder="w1", expires_at=100.0), now=85.0)
        True
    """

    ttl_seconds: float = Field(default=60.0, gt=0)
    renew_within: float = Field(default=20.0, ge=0)

    def due_to_renew(self, lease: RunLease, *, now: float) -> bool:
        """Whether `lease` should be renewed now rather than at the next boundary."""
        return now >= lease.expires_at - self.renew_within


DEFAULT_LEASE = LeasePolicy()
"""A minute of exclusivity, renewed with twenty seconds to spare."""


@runtime_checkable
class LeaseStore(Protocol):
    """Where the one-holder-at-a-time decision is made.

    The decision belongs to the store because the store has the only clock every worker
    shares. An implementation that compares the caller's `now` against its own rows has
    reintroduced exactly the skew the lease exists to survive.
    """

    async def acquire(
        self, run_id: str, *, tenant: str, holder: str, ttl_seconds: float
    ) -> RunLease:
        """Take the lease on `run_id`, or say who has it.

        An expired lease is takeable and the taker's fence is one higher, so the previous
        holder's writes are refused from that moment whatever its own clock says.

        Raises:
            RunLeaseError: If another holder's lease is still live.
            StatePersistenceError: If the store could not take the write.
        """
        ...

    async def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        """Extend `lease`, or refuse because it is no longer the current one.

        Raises:
            RunLeaseError: If the lease expired and was taken, or the fence has moved.
        """
        ...

    async def release(self, lease: RunLease) -> None:
        """Give the lease up early. Releasing one that has already moved on is not an error."""
        ...

    async def held(self, run_id: str, *, tenant: str) -> RunLease | None:
        """Return the current lease on `run_id`, expired or not, or `None` if there is none."""
        ...
