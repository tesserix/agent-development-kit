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
    MemoryConflictError,
    MemoryContradictionError,
    MemoryCorruptionError,
    MemoryLimitError,
    MemoryScopeError,
    PartialErasureError,
)
from tesserix_adk.memory import (
    DEFAULT_REDACTOR,
    Belief,
    Contradiction,
    ErasureReceipt,
    MemoryCapabilities,
    MemoryHit,
    MemoryKind,
    MemoryRecord,
    Resolution,
    SupersedeMatching,
    Supersession,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic import JsonValue

    from tesserix_adk.core.protocols import Clock, Tracer
    from tesserix_adk.memory import (
        ContradictionPolicy,
        DecayPolicy,
        Derivation,
        DerivedIndex,
        MemoryQuery,
        MemoryRedactor,
        MemoryScope,
    )

__all__ = ["InMemoryMemoryStore"]

_Key = tuple[tuple[str, str, str, str], MemoryKind, str]


class InMemoryMemoryStore:
    """A dictionary-backed `MemoryStore` for tests and for one-process development.

    Args:
        clock: Where expiry times come from, so a test can advance rather than sleep.
        capabilities: What this instance claims to support, so a test can build a store
            that lacks something and watch the bind fail.
        contradictions: What to do when a new profile record meets a live one.
        decay: What time does to a record's weight. None leaves everything at full
            weight, which is what a store with no opinion about staleness should do.
        redactor: What masks a value on the way in. `None` stores what it is given,
            which has to be asked for rather than arrived at.
        indices: The derived-artefact indices this store is responsible for erasing.
        tracer: Where erasure events go. Ids and counts only, never a value.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        capabilities: MemoryCapabilities | None = None,
        contradictions: ContradictionPolicy | None = None,
        decay: DecayPolicy | None = None,
        redactor: MemoryRedactor | None = DEFAULT_REDACTOR,
        indices: Sequence[DerivedIndex] = (),
        tracer: Tracer | None = None,
    ) -> None:
        self._clock = clock
        self._capabilities = capabilities or MemoryCapabilities()
        self._contradictions = contradictions or SupersedeMatching()
        self._decay = decay
        self._redactor = redactor
        self._indices = tuple(indices)
        self._tracer = tracer
        self._items: dict[_Key, Any] = {}
        self._expiry: dict[_Key, float] = {}
        self._versions: dict[_Key, list[MemoryRecord]] = {}
        self._derived: dict[tuple[str, str, str, str], list[Derivation]] = {}
        self._tombstoned: set[_Key] = set()

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
        held = self._redacted(record)
        self._items[(scope.path, MemoryKind.WORKING, record.key)] = held.model_dump()

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
        self._items[(scope.path, MemoryKind.WORKING, key)] = self._redacted(record).model_dump()
        return len(values)

    async def expire(self, scope: MemoryScope, key: str, *, ttl_seconds: float) -> None:
        """Have `key` read as absent once `ttl_seconds` have passed."""
        self._expiry[(scope.path, MemoryKind.WORKING, key)] = self._clock.now() + ttl_seconds

    async def upsert(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Write or replace a profile record, discarding whatever history it had."""
        self._belongs(scope, record, MemoryKind.PROFILE)
        self._fits(record.value)
        held = self._redacted(record)
        self._items[(scope.path, MemoryKind.PROFILE, record.key)] = held.model_dump()
        self._versions[(scope.path, MemoryKind.PROFILE, record.key)] = [held]

    async def profile(
        self, scope: MemoryScope, key: str, *, as_of: float | None = None
    ) -> MemoryRecord | None:
        """Return the profile record live at `as_of`, or now, or None."""
        held = await self.belief(scope, key, as_of=as_of)
        if held.contradiction is not None:
            raise MemoryContradictionError(
                f"{key!r} holds {len(held.contradiction.holds)} live records that disagree",
                key=key,
                subject=held.contradiction.subject,
                holds=held.contradiction.holds,
            )
        return held.record

    async def belief(self, scope: MemoryScope, key: str, *, as_of: float | None = None) -> Belief:
        """Return what the scope holds at `key`, contradiction and decay included."""
        self._as_of(as_of)
        at = self._clock.now() if as_of is None else as_of
        live = self._live(scope, key, at)
        if len(live) > 1:
            return Belief(contradiction=_disagreement(live))
        if not live:
            return Belief()
        weight = self._weighed(live[0], at)
        if weight <= 0.0:
            return Belief(weight=weight, decayed=(live[0],))
        return Belief(record=live[0], weight=weight)

    async def supersede(
        self,
        scope: MemoryScope,
        record: MemoryRecord,
        *,
        expected_version: int | None = None,
        resolves: tuple[str, ...] = (),
    ) -> Supersession:
        """Write a profile record as a new version, closing whatever it replaced."""
        if not self._capabilities.supports_supersession:
            raise CapabilityError(
                "this store cannot supersede a record, only overwrite it",
                capability="supersession",
                provider=type(self).__name__,
            )
        self._belongs(scope, record, MemoryKind.PROFILE)
        self._fits(record.value)
        record = self._redacted(record)

        at = (scope.path, MemoryKind.PROFILE, record.key)
        versions = self._versions.setdefault(at, self._seeded(scope, record.key))
        now = self._clock.now()
        live = [held for held in versions if _live_at(held, now)]
        self._expected(record.key, live, expected_version)
        _named(live, resolves)

        started = record.valid_from if record.valid_from is not None else now
        closed = [
            held
            for held in live
            if held.id in resolves
            or self._contradictions.resolve(held, record) is Resolution.SUPERSEDE
        ]
        if any(
            self._contradictions.resolve(held, record) is Resolution.REJECT
            for held in live
            if held.id not in resolves
        ):
            raise MemoryContradictionError(
                f"the policy refused to change what is held at {record.key!r}",
                key=record.key,
                subject=record.about,
                holds=tuple(live),
            )

        written = record.model_copy(
            update={
                "recorded_at": now,
                "valid_from": started,
                "version": max((held.version for held in versions), default=0) + 1,
            }
        )
        shut = []
        for held in closed:
            shut.append(held.model_copy(update={"valid_to": started, "superseded_by": written.id}))
            versions[versions.index(held)] = shut[-1]
        versions.append(written)
        self._items[at] = written.model_dump()

        branched = [held for held in live if held not in closed]
        return Supersession(
            record=written,
            superseded=shut[-1] if shut else None,
            resolution=_what_happened(closed, branched),
            contradiction=_disagreement([*branched, written]) if branched else None,
        )

    async def history(self, scope: MemoryScope, key: str | None = None) -> Sequence[MemoryRecord]:
        """Return every version under `scope`, oldest first, for `key` or for all keys."""
        return tuple(
            record
            for at, versions in self._versions.items()
            if at[0] == scope.path and (key is None or at[2] == key) and at not in self._tombstoned
            for record in versions
        )

    async def log(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Record that something happened."""
        self._belongs(scope, record, MemoryKind.EPISODIC)
        self._fits(record.value)
        self._items[(scope.path, MemoryKind.EPISODIC, record.key)] = self._redacted(
            record
        ).model_dump()

    async def episodes(self, scope: MemoryScope, query: MemoryQuery) -> Sequence[MemoryHit]:
        """Return episodes matching `query`, newest first."""
        self._as_of(query.as_of)
        now = self._clock.now() if query.as_of is None else query.as_of
        found = [
            record for record in self._under(scope, MemoryKind.EPISODIC) if _within(record, query)
        ]
        found.sort(key=lambda record: record.valid_from or 0.0, reverse=True)
        hits = [MemoryHit(record=record, score=self._weighed(record, now)) for record in found]
        return [hit for hit in hits if hit.score > 0.0][: query.limit]

    async def index(self, scope: MemoryScope, record: MemoryRecord) -> None:
        """Add a semantic record to the collection."""
        self._semantic()
        self._belongs(scope, record, MemoryKind.SEMANTIC)
        self._fits(record.value)
        self._width(record.embedding)
        self._items[(scope.path, MemoryKind.SEMANTIC, record.key)] = self._redacted(
            record
        ).model_dump()

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

    async def derived(self, scope: MemoryScope, derivation: Derivation) -> None:
        """Record that an artefact was built from a record under `scope`."""
        self._derived.setdefault(scope.path, []).append(derivation)

    async def derivations(
        self, scope: MemoryScope, *, source_id: str | None = None
    ) -> Sequence[Derivation]:
        """Return what has been derived under `scope`, or from one record within it."""
        held = self._derived.get(scope.path, [])
        return tuple(one for one in held if source_id is None or one.source_id == source_id)

    async def erase(
        self,
        scope: MemoryScope,
        *,
        kinds: tuple[MemoryKind, ...] = (),
        dry_run: bool = False,
    ) -> ErasureReceipt:
        """Tombstone every record under `scope`, then purge what was derived from them."""
        if not self._capabilities.supports_erasure:
            raise CapabilityError(
                "this store cannot erase a scope",
                capability="erasure",
                provider=type(self).__name__,
            )
        doomed = [
            key for key in self._items if key[0] == scope.path and (not kinds or key[1] in kinds)
        ]
        counts = self._counted(key for key in doomed if key not in self._tombstoned)
        if dry_run:
            return ErasureReceipt(counts=counts, adapters=self._adapters(), dry_run=True)

        self._tombstoned.update(doomed)
        purged, outstanding = await self._purge(scope)
        if outstanding:
            receipt = ErasureReceipt(
                counts=counts,
                artefacts=purged,
                adapters=self._adapters(),
                outstanding=outstanding,
            )
            self._published(receipt)
            raise PartialErasureError(
                f"{outstanding[0]!r} could not be reached; {scope.path} is tombstoned, not purged",
                adapter=outstanding[0],
                receipt=receipt,
            )

        for key in doomed:
            del self._items[key]
            self._expiry.pop(key, None)
            self._versions.pop(key, None)
            self._tombstoned.discard(key)
        self._derived.pop(scope.path, None)
        receipt = ErasureReceipt(
            counts=counts,
            artefacts=purged,
            adapters=self._adapters(),
            completed_at=self._clock.now(),
            complete=True,
        )
        self._published(receipt)
        return receipt

    def _counted(self, doomed: Iterable[_Key]) -> dict[str, int]:
        """How many records go per kind, counting every version of a superseded key."""
        counts: dict[str, int] = {}
        for key in doomed:
            counts[key[1].value] = counts.get(key[1].value, 0) + max(
                len(self._versions.get(key, ())), 1
            )
        return counts

    def _adapters(self) -> tuple[str, ...]:
        return tuple(index.name for index in self._indices)

    async def _purge(self, scope: MemoryScope) -> tuple[int, tuple[str, ...]]:
        """Drop the artefacts this scope alone derived, reporting whatever went unreachable."""
        mine = self._derived.get(scope.path, [])
        shared = {
            one.artefact_id
            for path, held in self._derived.items()
            if path != scope.path
            for one in held
        }
        purged, outstanding = 0, []
        for index in self._indices:
            ids = tuple(
                one.artefact_id
                for one in mine
                if one.adapter == index.name and one.artefact_id not in shared
            )
            if not ids:
                continue
            try:
                purged += await index.purge(ids)
            # However an index fails to answer, the erasure is incomplete in the same way.
            except Exception:
                outstanding.append(index.name)
        return purged, tuple(outstanding)

    def _published(self, receipt: ErasureReceipt) -> None:
        """Say what an erasure did, in counts. A value here would outlive the erasure."""
        if self._tracer is None:
            return
        self._tracer.event(
            "adk.memory.erased",
            **{
                "adk.memory.records": str(receipt.records),
                "adk.memory.artefacts": str(receipt.artefacts),
                "adk.memory.complete": "true" if receipt.complete else "false",
                "adk.memory.outstanding": ",".join(receipt.outstanding),
            },
        )

    def _redacted(self, record: MemoryRecord) -> MemoryRecord:
        """A record as it should be stored, carrying the paths that were masked."""
        if self._redactor is None:
            return record
        value, masked = self._redactor.redact(record.value)
        return record.model_copy(update={"value": value, "redacted": masked})

    def _seeded(self, scope: MemoryScope, key: str) -> list[MemoryRecord]:
        """A key's versions, starting from whatever a plain upsert left behind."""
        held = self._read(scope, MemoryKind.PROFILE, key)
        return [] if held is None else [held]

    def _live(self, scope: MemoryScope, key: str, at: float) -> list[MemoryRecord]:
        """Every profile record whose window contains `at`."""
        at_key = (scope.path, MemoryKind.PROFILE, key)
        if at_key in self._tombstoned:
            return []
        versions = self._versions.get(at_key) or self._seeded(scope, key)
        return [held for held in versions if _live_at(held, at)]

    def _expected(self, key: str, live: list[MemoryRecord], expected: int | None) -> None:
        """Refuse a write whose caller was reading a version that is no longer live."""
        if expected is None:
            return
        actual = max((held.version for held in live), default=0)
        if actual != expected:
            raise MemoryConflictError(
                f"{key!r} is at version {actual}, not {expected}",
                key=key,
                expected_version=expected,
                actual_version=actual,
            )

    def _as_of(self, as_of: float | None) -> None:
        if as_of is not None and not self._capabilities.supports_as_of:
            raise CapabilityError(
                "this store cannot read memory as of a past time",
                capability="as_of",
                provider=type(self).__name__,
            )

    def _weighed(self, record: MemoryRecord, now: float) -> float:
        """How much of a record survives decay, where the store has a policy at all."""
        return 1.0 if self._decay is None else self._decay.weigh(record, now=now)

    def _read(self, scope: MemoryScope, kind: MemoryKind, key: str) -> MemoryRecord | None:
        at = (scope.path, kind, key)
        if at in self._tombstoned:
            return None
        expires = self._expiry.get(at)
        if expires is not None and self._clock.now() >= expires:
            return None
        payload = self._items.get(at)
        return None if payload is None else _validated(key, payload)

    def _under(self, scope: MemoryScope, kind: MemoryKind) -> list[MemoryRecord]:
        return [
            _validated(at[2], payload)
            for at, payload in self._items.items()
            if at[0] == scope.path and at[1] is kind and at not in self._tombstoned
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


def _live_at(record: MemoryRecord, at: float) -> bool:
    """Whether a record's window contains `at`, so exactly one version is live per instant."""
    started = record.valid_from or 0.0
    return started <= at and (record.valid_to is None or at < record.valid_to)


def _named(live: list[MemoryRecord], resolves: tuple[str, ...]) -> None:
    """Refuse a write that claims to settle a record nobody is holding."""
    held = {record.id for record in live}
    unknown = sorted(set(resolves) - held)
    if unknown:
        raise ValueError(f"nothing live at this key with id {', '.join(unknown)}")


def _what_happened(closed: list[MemoryRecord], branched: list[MemoryRecord]) -> Resolution:
    """What the write did, reported by what it left behind."""
    if closed:
        return Resolution.SUPERSEDE
    return Resolution.BRANCH if branched else Resolution.NOTHING_TO_DO


def _disagreement(holds: list[MemoryRecord]) -> Contradiction:
    """The live records that disagree, and what they disagree about."""
    return Contradiction(
        subject=holds[0].about,
        aspects=tuple(sorted({aspect for record in holds for aspect in record.aspects})),
        holds=tuple(holds),
    )
