"""A store two tenants share, and cannot see each other through.

Run it with `uv run python examples/store_isolation.py`.
"""

from __future__ import annotations

from tesserix_adk.adapters import (
    ADAPTER_GUARANTEES,
    ErasureSweep,
    Isolator,
    Partition,
    partitioned,
)
from tesserix_adk.core import MissingTenantContextError, TenantIsolationError, tenant_scope

ROWS = [
    {"tenant": "acme", "body": "the Q3 renewal terms"},
    {"tenant": "rival", "body": "the Q3 renewal terms"},
]


class FakeVectorStore:
    """Stands in for pgvector, with the tenant filter inside the statement."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def search(self) -> list[dict[str, str]]:
        """Rank only the bound tenant's rows, because the filter is not a post-filter."""
        partition = Partition.current(where="fake.search")
        clause, value = partition.predicate(position=1)
        self.statements.append(f"SELECT body FROM chunks WHERE {clause} -- $1={value}")  # noqa: S608 — the clause is built by the adapter, the tenant is bound
        return [row for row in ROWS if row["tenant"] == value]


class LeakyStore(FakeVectorStore):
    """The same store after a botched backfill left a neighbour's row behind."""

    async def search(self) -> list[dict[str, str]]:
        """Return what the database actually holds, mislabelled row included."""
        return [*await super().search(), {"tenant": "rival", "body": "somebody else's"}]


async def main() -> None:
    """Two tenants, one collection, one erasure and one corrupted row."""
    store = FakeVectorStore()
    isolator: Isolator[dict[str, str]] = Isolator(
        tenant_of=lambda row: row["tenant"], where="fake.search"
    )

    for tenant in ("acme", "rival"):
        with tenant_scope(tenant):
            found = await isolator.read(store.search)
            key = Partition.current().key("adk:cache", "same-digest")
            print(f"{tenant}: {[row['body'] for row in found]} cached at {key}")  # noqa: T201

    issued = "\n  ".join(store.statements)
    print(f"\nstatements issued:\n  {issued}")  # noqa: T201

    try:
        await isolator.read(store.search)
    except MissingTenantContextError as refused:
        print(f"\nno context, no query: {refused}")  # noqa: T201

    with tenant_scope("acme"):
        try:
            await isolator.read(LeakyStore().search)
        except TenantIsolationError as leaked:
            print(f"\nwithheld rather than filtered: found {leaked.found!r}")  # noqa: T201

    batches = partitioned(ROWS, tenant_of=lambda row: row["tenant"])
    print(f"\na mixed batch becomes {len(batches)} scoped batches")  # noqa: T201

    sweep = ErasureSweep(tenant="acme", remaining={"rows": 0, "embeddings": 2, "keys": 0})
    print(f"erasure complete={sweep.complete}, outstanding={sweep.outstanding}")  # noqa: T201

    print(f"\n{ADAPTER_GUARANTEES['PgvectorIndex'].statement()}")  # noqa: T201


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
