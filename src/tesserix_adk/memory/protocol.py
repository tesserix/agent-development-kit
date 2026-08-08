"""One protocol across four kinds of memory, so memory is a dependency and not a rewrite.

Three products have three memory shapes and none of them is portable, so a bug fixed in
one recurs unchanged in the next. The operations below are kind-specific on purpose: what
working memory does (append, expire) and what semantic memory does (index, rank) do not
collapse into a shared get/put without lying about one of them.

Stability: this protocol is public API under semver. Within a minor release it is
additive only — a method may be added with a default-implementing mixin, never removed,
renamed, or given a required parameter. A removal is announced one minor ahead with a
shim that still works. See docs/memory.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue

    from tesserix_adk.memory.capabilities import MemoryCapabilities
    from tesserix_adk.memory.records import MemoryHit, MemoryQuery, MemoryRecord
    from tesserix_adk.memory.scope import MemoryScope

__all__ = ["MemoryStore"]


@runtime_checkable
class MemoryStore(Protocol):
    """Working, profile, episodic and semantic memory behind one substitutable dependency.

    Every operation takes a `MemoryScope`. There is no unscoped overload, so a call site
    cannot forget one, and a record whose own scope disagrees with the call is refused
    rather than filed under whichever of the two the adapter happened to read.
    """

    @property
    def capabilities(self) -> MemoryCapabilities:
        """What this adapter supports. Checked by `require_memory` when it is bound."""
        ...

    async def write(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Replace working memory at `record.key`.

        Raises:
            MemoryScopeError: If the record does not belong to `scope`, or is not working.
            MemoryLimitError: If the value is larger than this adapter holds.
        """
        ...

    async def read(self, scope: MemoryScope, key: str) -> MemoryRecord | None:
        """Return the working record at `key`, or None where there is none or it expired.

        Raises:
            MemoryCorruptionError: If what is stored no longer validates as a record.
        """
        ...

    async def append(self, scope: MemoryScope, key: str, value: JsonValue) -> int:
        """Add `value` to the sequence at `key` and return its position, counting from 1.

        Concurrent appends are ordered and none is dropped. Returning the position is
        what lets a caller detect a lost write instead of assuming one.
        """
        ...

    async def expire(self, scope: MemoryScope, key: str, *, ttl_seconds: float) -> None:
        """Have `key` read as absent once `ttl_seconds` have passed."""
        ...

    async def upsert(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Write or replace a profile record.

        Raises:
            MemoryScopeError: If the record does not belong to `scope`, or is not profile.
            MemoryLimitError: If the value is larger than this adapter holds.
        """
        ...

    async def profile(self, scope: MemoryScope, key: str) -> MemoryRecord | None:
        """Return the profile record at `key`, or None.

        Raises:
            MemoryCorruptionError: If what is stored no longer validates as a record.
        """
        ...

    async def log(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Record that something happened. Episodes accumulate; nothing is replaced.

        Raises:
            MemoryScopeError: If the record does not belong to `scope`, or is not episodic.
        """
        ...

    async def episodes(self, scope: MemoryScope, query: MemoryQuery) -> Sequence[MemoryHit]:
        """Return episodes matching `query`, newest first.

        Raises:
            CapabilityError: If the query asks for `as_of` and this adapter has none.
            MemoryCorruptionError: If a stored episode no longer validates.
        """
        ...

    async def index(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Add a semantic record to the vector collection.

        Raises:
            CapabilityError: If this adapter declares no semantic support.
            EmbeddingDimensionError: If the embedding is missing or the wrong width.
        """
        ...

    async def search(self, scope: MemoryScope, query: MemoryQuery) -> Sequence[MemoryHit]:
        """Return semantic records ranked by resemblance, closest first.

        Raises:
            CapabilityError: If this adapter declares no semantic support.
            EmbeddingDimensionError: If the query embedding is missing or the wrong width.
            MemoryCorruptionError: If a stored record no longer validates.
        """
        ...

    async def erase(self, scope: MemoryScope) -> int:
        """Delete every record under `scope`, across all four kinds, and return how many.

        A read racing an erasure sees all of the scope or none of it, never half.

        Raises:
            CapabilityError: If this adapter cannot erase. Reporting zero rows erased
                would be indistinguishable from a scope that held nothing.
        """
        ...
