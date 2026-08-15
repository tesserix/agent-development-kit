"""Isolation the calling code cannot forget, because it never states the tenant.

A store scoped by an argument is scoped by whoever remembers the argument. Here the
partition is derived from the bound tenant context at the moment of the call, so a
forgotten prefix or a dropped `WHERE` is not something a caller can express — and a
row that comes back under the wrong tenant is refused rather than quietly dropped,
because silently filtering it leaves the corruption there for the next read.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from pydantic import Field, field_validator

from tesserix_adk.core import AdkModel, TenantIsolationError, current_tenant

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

__all__ = [
    "ADAPTER_GUARANTEES",
    "ErasureSweep",
    "IsolationGuarantee",
    "Isolator",
    "Partition",
    "PartitionMechanism",
    "partitioned",
]

_GLOB = set("*?[]^")


class PartitionMechanism(StrEnum):
    """How a store keeps one tenant's data away from another's.

    There is deliberately no post-filtered search: taking the top `k` and dropping what
    is not ours returns fewer than `k` of our own rows whenever a neighbour ranks higher,
    which reads as a thin answer rather than as a bug.
    """

    KEY_PREFIX = "key prefix"
    ROW_PREDICATE = "row predicate"
    ROW_LEVEL_SECURITY = "row-level security"
    SCHEMA_SEPARATION = "schema separation"
    PRE_FILTERED_ANN = "pre-filtered nearest neighbours"


@dataclass(frozen=True, slots=True, kw_only=True)
class Partition:
    """The tenant's slice of a store, derived rather than passed.

    Built through `current`, which reads the bound context. Nothing here accepts a tenant
    from a caller, so no call site can name one that is not the one it is running under.
    """

    tenant: str

    @classmethod
    def current(cls, *, where: str = "partition") -> Self:
        """The partition for the work in hand.

        Args:
            where: What was about to happen, so a refusal names the egress point.

        Raises:
            MissingTenantContextError: Where nothing is bound. Absence is never read as
                every tenant and never filled in with a default.
        """
        return cls(tenant=current_tenant(where=where).tenant)

    def key(self, namespace: str, *parts: str) -> str:
        """A store key inside this partition, with the tenant ahead of everything else."""
        return ":".join((namespace, self.tenant, *parts))

    def pattern(self, namespace: str) -> str:
        """A match pattern reaching this partition and no further.

        Raises:
            ValueError: Where the tenant name contains glob syntax, which would let one
                tenant's name match another tenant's keys.
        """
        if _GLOB & set(self.tenant):
            raise ValueError(
                f"tenant {self.tenant!r} contains pattern syntax and cannot be matched on "
                f"safely; a glob in a name is a glob over the neighbours"
            )
        return f"{namespace}:{self.tenant}:*"

    def predicate(self, *, column: str = "tenant", position: int = 1) -> tuple[str, str]:
        """The mandatory tenant clause and the value to bind to it.

        Args:
            column: The column holding the tenant.
            position: Where the value falls among the statement's arguments.

        Returns:
            The clause and its argument, bound rather than interpolated.
        """
        return f"{column} = ${position}", self.tenant

    def verify(self, found: str, *, where: str) -> None:
        """Refuse a record that is not this tenant's.

        Raises:
            TenantIsolationError: Where the record names another tenant, or names none —
                an unlabelled record predates the guarantee and is not assumed to be ours.
        """
        if found == self.tenant:
            return
        raise TenantIsolationError(
            f"{where} read a record belonging to {found or 'no tenant'} while bound to "
            f"{self.tenant}; it is withheld rather than filtered away",
            tenant=self.tenant,
            found=found,
            where=where,
        )

    def only[ItemT](
        self,
        items: Sequence[ItemT],
        *,
        tenant_of: Callable[[ItemT], str],
        where: str,
    ) -> Sequence[ItemT]:
        """The same items, once every one of them is known to be this tenant's.

        Raises:
            TenantIsolationError: On the first item that is not, before any of them is
                returned. Nothing partial comes back.
        """
        for item in items:
            self.verify(tenant_of(item), where=where)
        return items


def partitioned[ItemT](
    items: Iterable[ItemT], *, tenant_of: Callable[[ItemT], str]
) -> dict[str, list[ItemT]]:
    """Split work that spans tenants into one batch per tenant.

    A bulk operation over mixed tenants is either refused or run once per tenant; it is
    never run as one statement whose scope is the union of everybody in it.
    """
    batches: dict[str, list[ItemT]] = {}
    for item in items:
        batches.setdefault(tenant_of(item), []).append(item)
    return batches


class IsolationGuarantee(AdkModel):
    """What isolation an adapter actually provides, and where it stops.

    Args:
        adapter: The adapter this is about.
        mechanism: How the separation is enforced.
        partitions: What the partition key is, in the store's own terms.
        limits: What the mechanism does not protect against. Required: every mechanism
            here has a limit, and a guarantee claiming none is a guarantee nobody read.
    """

    adapter: str = Field(min_length=1)
    mechanism: PartitionMechanism
    partitions: str = Field(min_length=1)
    limits: tuple[str, ...]

    @field_validator("limits")
    @classmethod
    def _stated(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """A limitless guarantee is a claim, not a guarantee."""
        if not value:
            raise ValueError("limits must state what the mechanism does not protect against")
        return value

    def statement(self) -> str:
        """The guarantee as a consumer should read it, limits included."""
        limits = "; ".join(self.limits)
        return (
            f"{self.adapter} isolates by {self.mechanism}: {self.partitions}. "
            f"Not protected against: {limits}."
        )


class ErasureSweep(AdkModel):
    """What a tenant-scoped erasure found still standing afterwards.

    Args:
        tenant: Whose data was erased.
        remaining: What each verification query counted after the delete — rows,
            embeddings, cache keys, summaries. Zero everywhere is the only clean answer.
    """

    tenant: str = Field(min_length=1)
    remaining: Mapping[str, int]

    @field_validator("remaining")
    @classmethod
    def _counted(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        """A sweep that verified nothing is not evidence that nothing remains."""
        if not value:
            raise ValueError("remaining must record what each verification query counted")
        return value

    @property
    def complete(self) -> bool:
        """Whether the erasure reached every artefact it was asked about."""
        return not self.outstanding

    @property
    def outstanding(self) -> tuple[str, ...]:
        """The artefacts that still hold something."""
        return tuple(sorted(name for name, count in self.remaining.items() if count))


ADAPTER_GUARANTEES: Mapping[str, IsolationGuarantee] = {
    guarantee.adapter: guarantee
    for guarantee in (
        IsolationGuarantee(
            adapter="RedisCacheStore",
            mechanism=PartitionMechanism.KEY_PREFIX,
            partitions="every key is <namespace>:<tenant>:<prompt>:<model>:<digest>, so a "
            "hit for one tenant cannot be a hit for another, including a cached error",
            limits=(
                "anything holding the raw connection can read any key",
                "a tenant name containing glob syntax is refused rather than matched on",
            ),
        ),
        IsolationGuarantee(
            adapter="RedisMemoryStore",
            mechanism=PartitionMechanism.KEY_PREFIX,
            partitions="every key and index entry is prefixed with the scope's tenant",
            limits=(
                "anything holding the raw connection can read any key",
                "a KEYS or SCAN run outside the adapter is unscoped",
            ),
        ),
        IsolationGuarantee(
            adapter="PostgresMemoryStore",
            mechanism=PartitionMechanism.ROW_PREDICATE,
            partitions="tenant is a bound argument in the WHERE of every read and the "
            "value written on every insert",
            limits=(
                "a migration or backfill run outside the adapter bypasses the predicate",
                "row-level security is not assumed; the predicate is the boundary",
            ),
        ),
        IsolationGuarantee(
            adapter="PgvectorMemoryStore",
            mechanism=PartitionMechanism.PRE_FILTERED_ANN,
            partitions="tenant is in the WHERE of the similarity statement, so ranking "
            "sees only this tenant's vectors",
            limits=(
                "recall inside the tenant's own partition still depends on the ANN index",
                "a query issued outside the adapter can omit the filter",
            ),
        ),
        IsolationGuarantee(
            adapter="PgvectorIndex",
            mechanism=PartitionMechanism.PRE_FILTERED_ANN,
            partitions="tenant and collection are bound arguments in the WHERE, applied "
            "before ranking rather than to the top k",
            limits=(
                "recall inside the tenant's own partition still depends on the ANN index",
                "a query issued outside the adapter can omit the filter",
            ),
        ),
        IsolationGuarantee(
            adapter="GraphMemoryStore",
            mechanism=PartitionMechanism.ROW_PREDICATE,
            partitions="nodes and edges carry the tenant and every traversal is bounded "
            "by it, so a path cannot leave the tenant midway",
            limits=(
                "an entity resolved from text can still name someone outside the tenant",
                "a traversal run outside the adapter is unscoped",
            ),
        ),
        IsolationGuarantee(
            adapter="RoutedMemoryStore",
            mechanism=PartitionMechanism.ROW_PREDICATE,
            partitions="routing changes which store answers, never which tenant is asked "
            "for; each backing store keeps its own guarantee",
            limits=("the guarantee is the weakest of the stores routed to",),
        ),
    )
}
"""What each tenant-partitioned store adapter actually guarantees, and where it stops.

Lease, queue, ledger and idempotency adapters are keyed by work rather than by tenant
data and are not listed here.
"""


@dataclass(frozen=True, slots=True)
class Isolator[RecordT]:
    """The adapter-side wrapper that scopes a read and checks what comes back.

    Args:
        tenant_of: How to read a record's stored tenant.
        where: The adapter and operation, so a refusal names the egress point.
    """

    tenant_of: Callable[[RecordT], str]
    where: str

    async def read(
        self, operation: Callable[[], Awaitable[Sequence[RecordT]]]
    ) -> Sequence[RecordT]:
        """Run a scoped read, refusing anything in the result that is not this tenant's.

        The partition is resolved before `operation` is called, so a read with no context
        bound never reaches the store at all.
        """
        partition = Partition.current(where=self.where)
        return partition.only(await operation(), tenant_of=self.tenant_of, where=self.where)
