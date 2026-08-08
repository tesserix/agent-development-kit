"""A memory store that needs no network, and behaves exactly like one that does.

The fake is where the protocol's promises are made concrete: scope isolation, ordered
appends, atomic erasure, and a corrupt record that raises rather than disappears. An
adapter that passes `MemoryStoreConformance` against a real server agrees with this.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tesserix_adk.core.errors import (
    CapabilityError,
    EmbeddingDimensionError,
    MemoryCorruptionError,
    MemoryLimitError,
    MemoryScopeError,
)
from tesserix_adk.memory import (
    MemoryCapabilities,
    MemoryHit,
    MemoryKind,
    MemoryRecord,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue

    from tesserix_adk.core.protocols import Clock
    from tesserix_adk.memory import MemoryQuery, MemoryScope

__all__ = ["InMemoryMemoryStore"]

_Key = tuple[tuple[str, str, str, str], MemoryKind, str]


class InMemoryMemoryStore:
    """A dictionary-backed `MemoryStore` for tests and for one-process development.

    Args:
        clock: Where expiry times come from, so a test can advance rather than sleep.
        capabilities: What this instance claims to support, so a test can build a store
            that lacks something and watch the bind fail.
    """

    def __init__(self, *, clock: Clock, capabilities: MemoryCapabilities | None = None) -> None:
        self._clock = clock
        self._capabilities = capabilities or MemoryCapabilities()
        self._items: dict[_Key, Any] = {}
        self._expiry: dict[_Key, float] = {}

    @property
    def capabilities(self) -> MemoryCapabilities:
        """What this instance was constructed to support."""
        return self._capabilities

    def plant(self, scope: MemoryScope, kind: MemoryKind, key: str, payload: Any) -> None:
        """Store a raw payload that never validates, to exercise the corruption path.

        The only way in: an adapter cannot write an unreadable record on purpose, so a
        test needs a door that a caller does not have.
        """
        self._items[(scope.path, kind, key)] = payload

    async def write(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Replace working memory at `record.key`."""
        self._belongs(scope, record, MemoryKind.WORKING)
        self._fits(record.value)
        self._items[(scope.path, MemoryKind.WORKING, record.key)] = record.model_dump()

    async def read(self, scope: MemoryScope, key: str) -> MemoryRecord | None:
        """Return the working record at `key`, or None where it is absent or expired."""
        return self._read(scope, MemoryKind.WORKING, key)

    async def append(self, scope: MemoryScope, key: str, value: JsonValue) -> int:
        """Add `value` to the sequence at `key` and return its position."""
        held = self._read(scope, MemoryKind.WORKING, key)
        values: list[JsonValue] = []
        if held is not None:
            values = list(held.value) if isinstance(held.value, list) else [held.value]
        values.append(value)
        record = MemoryRecord(
            id=f"{key}:{len(values)}",
            kind=MemoryKind.WORKING,
            scope=scope,
            key=key,
            value=values,
            source="append",
            valid_from=self._clock.now(),
        )
        self._items[(scope.path, MemoryKind.WORKING, key)] = record.model_dump()
        return len(values)

    async def expire(self, scope: MemoryScope, key: str, *, ttl_seconds: float) -> None:
        """Have `key` read as absent once `ttl_seconds` have passed."""
        self._expiry[(scope.path, MemoryKind.WORKING, key)] = self._clock.now() + ttl_seconds

    async def upsert(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Write or replace a profile record."""
        self._belongs(scope, record, MemoryKind.PROFILE)
        self._fits(record.value)
        self._items[(scope.path, MemoryKind.PROFILE, record.key)] = record.model_dump()

    async def profile(self, scope: MemoryScope, key: str) -> MemoryRecord | None:
        """Return the profile record at `key`, or None."""
        return self._read(scope, MemoryKind.PROFILE, key)

    async def log(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Record that something happened."""
        self._belongs(scope, record, MemoryKind.EPISODIC)
        self._fits(record.value)
        self._items[(scope.path, MemoryKind.EPISODIC, record.key)] = record.model_dump()

    async def episodes(self, scope: MemoryScope, query: MemoryQuery) -> Sequence[MemoryHit]:
        """Return episodes matching `query`, newest first."""
        if query.as_of is not None and not self._capabilities.supports_as_of:
            raise CapabilityError(
                "this store cannot read memory as of a past time",
                capability="as_of",
                provider=type(self).__name__,
            )
        found = [
            record for record in self._under(scope, MemoryKind.EPISODIC) if _within(record, query)
        ]
        found.sort(key=lambda record: record.valid_from or 0.0, reverse=True)
        return [MemoryHit(record=record) for record in found[: query.limit]]

    async def index(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Add a semantic record to the collection."""
        self._semantic()
        self._belongs(scope, record, MemoryKind.SEMANTIC)
        self._fits(record.value)
        self._width(record.embedding)
        self._items[(scope.path, MemoryKind.SEMANTIC, record.key)] = record.model_dump()

    async def search(self, scope: MemoryScope, query: MemoryQuery) -> Sequence[MemoryHit]:
        """Return semantic records ranked by resemblance, closest first."""
        self._semantic()
        self._width(query.embedding)
        asked = query.embedding or ()
        hits = [
            MemoryHit(record=record, score=_resemblance(asked, record.embedding or ()))
            for record in self._under(scope, MemoryKind.SEMANTIC)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: query.limit]

    async def erase(self, scope: MemoryScope) -> int:
        """Delete every record under `scope` across all four kinds, and return how many."""
        if not self._capabilities.supports_erasure:
            raise CapabilityError(
                "this store cannot erase a scope",
                capability="erasure",
                provider=type(self).__name__,
            )
        doomed = [key for key in self._items if key[0] == scope.path]
        for key in doomed:
            del self._items[key]
            self._expiry.pop(key, None)
        return len(doomed)

    def _read(self, scope: MemoryScope, kind: MemoryKind, key: str) -> MemoryRecord | None:
        at = (scope.path, kind, key)
        expires = self._expiry.get(at)
        if expires is not None and self._clock.now() >= expires:
            return None
        payload = self._items.get(at)
        return None if payload is None else _validated(key, payload)

    def _under(self, scope: MemoryScope, kind: MemoryKind) -> list[MemoryRecord]:
        return [
            _validated(at[2], payload)
            for at, payload in self._items.items()
            if at[0] == scope.path and at[1] is kind
        ]

    def _belongs(self, scope: MemoryScope, record: MemoryRecord, kind: MemoryKind) -> None:
        if record.scope != scope:
            raise MemoryScopeError(
                f"record {record.id!r} belongs to {record.scope.path}, not {scope.path}",
                expected=str(record.scope.path),
                given=str(scope.path),
            )
        if record.kind is not kind:
            raise MemoryScopeError(
                f"record {record.id!r} is {record.kind.value}, filed as {kind.value}",
                expected=record.kind.value,
                given=kind.value,
            )

    def _fits(self, value: JsonValue) -> None:
        limit = self._capabilities.max_value_bytes
        if limit is None:
            return
        size = len(json.dumps(value).encode())
        if size > limit:
            raise MemoryLimitError(
                f"value of {size} bytes is over this store's {limit}", limit=limit, size=size
            )

    def _semantic(self) -> None:
        if not self._capabilities.supports_semantic:
            raise CapabilityError(
                "this store has no vector collection",
                capability="semantic",
                provider=type(self).__name__,
            )

    def _width(self, embedding: tuple[float, ...] | None) -> None:
        expected = self._capabilities.embedding_dimensions
        if embedding is None:
            raise EmbeddingDimensionError(
                "semantic memory needs an embedding", expected=expected or 0, received=0
            )
        if expected is not None and len(embedding) != expected:
            raise EmbeddingDimensionError(
                f"the collection is {expected} wide, this is {len(embedding)}",
                expected=expected,
                received=len(embedding),
            )


def _validated(key: str, payload: Any) -> MemoryRecord:
    """A stored payload as a record, or an error naming the row that has to be fixed."""
    try:
        return MemoryRecord.model_validate(payload)
    except ValidationError as failure:
        raise MemoryCorruptionError(
            f"stored memory {key!r} no longer reads as a record",
            record_id=key,
            payload=payload,
        ) from failure


def _within(record: MemoryRecord, query: MemoryQuery) -> bool:
    """Whether an episode falls inside the window, or the instant, that was asked for."""
    at = record.valid_from or 0.0
    if query.as_of is not None:
        return at <= query.as_of and (record.valid_to is None or query.as_of < record.valid_to)
    if query.since is not None and at < query.since:
        return False
    return not (query.until is not None and at > query.until)


def _resemblance(asked: Sequence[float], held: Sequence[float]) -> float:
    """Cosine similarity, and zero where either side has no length to compare."""
    if len(asked) != len(held):
        return 0.0
    dot = sum(a * b for a, b in zip(asked, held, strict=True))
    scale = math.sqrt(sum(a * a for a in asked)) * math.sqrt(sum(b * b for b in held))
    return dot / scale if scale else 0.0
