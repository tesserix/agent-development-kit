"""What a memory is, what you ask of it, and what comes back.

Four kinds share one record type because they share one lifecycle — written under a
scope, valid over a window, believed to a degree, and traceable to where it came from.
A kind-per-model would repeat that five times and drift four ways.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, JsonValue

from tesserix_adk.core.models import AdkModel
from tesserix_adk.memory.scope import MemoryScope  # noqa: TC001 — pydantic needs it at runtime

__all__ = [
    "MemoryHit",
    "MemoryKind",
    "MemoryQuery",
    "MemoryRecord",
]


class MemoryKind(StrEnum):
    """What kind of remembering a record is, which decides where it is kept."""

    WORKING = "working"
    """The current task's scratch space. Short-lived, appended to, expired."""

    PROFILE = "profile"
    """A durable fact about the user or tenant. Upserted, read by key."""

    EPISODIC = "episodic"
    """Something that happened, at a time. Queried over a window."""

    SEMANTIC = "semantic"
    """Something known, retrieved by resemblance rather than by key."""


class MemoryRecord(AdkModel):
    """One remembered thing, with everything needed to decide whether to believe it.

    Args:
        id: Stable identifier within its scope and kind.
        kind: Which of the four this is.
        scope: Whose memory it is. Carried on the record so that a record moved between
            call sites cannot lose it.
        key: What it is filed under. Unique per scope and kind.
        value: The memory itself, as JSON. Adapters store bytes, not objects.
        source: Where it came from — a tool name, a turn id, an import. Recall without
            provenance is a claim nobody can check.
        valid_from: When it started being true, in epoch seconds.
        valid_to: When it stopped, if it has. Open-ended where None.
        recorded_at: When the system learned it. Separate from `valid_from` because a
            fact backdated to last month was still acted on from today, and an audit
            that cannot tell those apart cannot explain a decision.
        superseded_by: The id of the record that replaced this one, once one has.
        version: Which version of its key this is, counting from 1. What an optimistic
            write checks against.
        subject: What the record is about. Defaults to the key where it is not set.
        predicate: Which aspects of the subject it speaks to. Two records on the same
            subject whose predicates match exactly contradict; overlapping-but-unequal
            ones might, and are not resolved automatically.
        confidence: How much to believe it, 0 to 1. Defaults to certain, because a
            default of "probably" would quietly discount everything written by hand.
        embedding: The vector, for semantic records. None for the other three kinds.
    """

    id: str
    kind: MemoryKind
    scope: MemoryScope
    key: str
    value: JsonValue
    source: str
    valid_from: float | None = None
    valid_to: float | None = None
    recorded_at: float | None = None
    superseded_by: str | None = None
    version: int = Field(default=1, ge=1)
    subject: str | None = None
    predicate: tuple[str, ...] = ()
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    embedding: tuple[float, ...] | None = None

    @property
    def about(self) -> str:
        """What this record is about — its subject, or its key where it has none."""
        return self.subject or self.key

    @property
    def aspects(self) -> frozenset[str]:
        """The aspects it speaks to, falling back to its key so a bare record still pairs."""
        return frozenset(self.predicate) or frozenset({self.key})


class MemoryQuery(AdkModel):
    """What to look for, in one of the four kinds.

    Args:
        kind: Which kind is being asked. A query never spans kinds: they are ranked by
            different things, and a merged ranking is a made-up one.
        key: Exact key, where the caller knows it.
        text: What the caller is looking for, for adapters that search text.
        embedding: The vector to rank by, for semantic search.
        since: Earliest `valid_from` to include.
        until: Latest `valid_from` to include.
        as_of: Read the memory as it stood at this time — records whose window contains
            it. Requires an adapter that declares `supports_as_of`.
        limit: How many to return at most.
    """

    kind: MemoryKind
    key: str | None = None
    text: str | None = None
    embedding: tuple[float, ...] | None = None
    since: float | None = None
    until: float | None = None
    as_of: float | None = None
    limit: int = Field(default=10, ge=1)


class MemoryHit(AdkModel):
    """A record a query matched, and how well.

    Args:
        record: What was found, whole. A hit that carried only an id would send every
            caller back to the store to find out what it had matched.
        score: Higher is closer. Comparable within one result set and not across two,
            since adapters measure distance differently.
    """

    record: MemoryRecord
    score: float = 1.0
