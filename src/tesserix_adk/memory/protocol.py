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

    from tesserix_adk.memory.beliefs import Belief, Supersession
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

    async def profile(
        self, scope: MemoryScope, key: str, *, as_of: float | None = None
    ) -> MemoryRecord | None:
        """Return the profile record live at `as_of`, or now, or None.

        None also means "held but not recalled": a key whose only record has decayed out
        of reach, or whose live records contradict. Use `belief` to tell those apart.

        Raises:
            CapabilityError: If `as_of` is given and this adapter has no history.
            MemoryContradictionError: If more than one record is live for the key. The
                store does not choose, because whichever it chose would be plausible.
            MemoryCorruptionError: If what is stored no longer validates as a record.
        """
        ...

    async def supersede(
        self,
        scope: MemoryScope,
        record: MemoryRecord,
        *,
        expected_version: int | None = None,
        resolves: tuple[str, ...] = (),
    ) -> Supersession:
        """Write a profile record as a new version, closing whatever it replaced.

        Nothing is overwritten. The record it replaces keeps its value and gains a
        `valid_to` and a `superseded_by`, so `history` can say what was believed when.

        Args:
            scope: Whose memory this is.
            record: The new belief. Its `valid_from` decides when the old one closed.
            expected_version: The version the caller read. Where it is given and no
                longer live, the write is refused rather than applied over someone
                else's. None takes whatever is live, for a caller with no race to lose.
            resolves: The ids of live records this write settles, closing them whatever
                the policy would have said. How a branch ends: somebody decided, and the
                decision is recorded rather than inferred.

        Raises:
            CapabilityError: If this adapter does not keep versions.
            MemoryConflictError: If `expected_version` is not what is live.
            MemoryContradictionError: If the policy rejected the change.
            MemoryScopeError: If the record does not belong to `scope`, or is not profile.
            ValueError: If `resolves` names a record that is not live.
        """
        ...

    async def belief(self, scope: MemoryScope, key: str, *, as_of: float | None = None) -> Belief:
        """Return what the scope holds at `key`, including why it holds nothing.

        Where records contradict, the `Belief` carries the `Contradiction` and no record;
        where decay has put a record out of reach, it carries it under `decayed`. Both
        are reported rather than raised, for a caller that wants to show a person.

        Raises:
            CapabilityError: If `as_of` is given and this adapter has no history.
            MemoryCorruptionError: If what is stored no longer validates as a record.
        """
        ...

    async def history(self, scope: MemoryScope, key: str | None = None) -> Sequence[MemoryRecord]:
        """Return every version under `scope`, oldest first, for `key` or for all keys.

        The supersession trail: what was believed, from when, until what replaced it.
        Decay never removes anything from it, because a fact nobody recalls is still a
        fact somebody acted on.
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
