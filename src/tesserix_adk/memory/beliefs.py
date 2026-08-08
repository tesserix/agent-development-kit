"""What to do when the new fact disagrees with the old one, and when to stop believing it.

Last-write-wins destroys the reason an agent believes what it believes, and a store that
never forgets acts on a preference recorded a year ago as confidently as on this morning's.
Both are handled here, and neither by deleting anything: supersession closes a record and
keeps it, decay lowers its weight and keeps it. Deletion is erasure, which is a different
promise made to a different person.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from tesserix_adk.core.models import AdkModel
from tesserix_adk.memory.records import MemoryRecord  # noqa: TC001 — pydantic needs it at runtime

__all__ = [
    "Belief",
    "ConfidenceFloor",
    "Contradiction",
    "ContradictionPolicy",
    "DecayPolicy",
    "HalfLife",
    "Resolution",
    "SupersedeMatching",
    "Supersession",
]


class Resolution(StrEnum):
    """What a policy decided to do with a write that met an existing record."""

    SUPERSEDE = "supersede"
    """The new record replaces the old one, which is closed and kept."""

    BRANCH = "branch"
    """Both stay live and the disagreement is surfaced for someone to settle."""

    REJECT = "reject"
    """The write is refused and the existing belief stands."""

    NOTHING_TO_DO = "nothing-to-do"
    """There was no live record to meet, so the write simply lands."""


@runtime_checkable
class ContradictionPolicy(Protocol):
    """Decides what an incoming record does to the one already held.

    Stability: public API under semver. `resolve` is synchronous on purpose — a policy
    that needs the network to decide belongs in the caller, before the write.
    """

    def resolve(self, existing: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        """Say what `incoming` does to `existing`. Never raises; the store acts on the answer."""
        ...


@dataclass(frozen=True, slots=True)
class SupersedeMatching:
    """The default: replace what says the same thing, branch on a partial overlap.

    Two records under one key, about one subject, speaking to exactly the same aspects
    are the same belief stated twice: the later one wins and the earlier is kept. Anything
    else — a different subject, aspects that overlap only partly, aspects that do not
    overlap at all — is not a restatement, and nothing here can tell which part of the
    old belief changed. That becomes a `Contradiction` for a caller to settle explicitly.
    """

    def resolve(self, existing: MemoryRecord, incoming: MemoryRecord) -> Resolution:
        """Supersede an exact restatement; branch anything that is not one."""
        matches = existing.about == incoming.about and existing.aspects == incoming.aspects
        return Resolution.SUPERSEDE if matches else Resolution.BRANCH


class Contradiction(AdkModel):
    """Live records about one subject that disagree, and that nothing may resolve alone.

    Args:
        subject: What they disagree about.
        aspects: The aspects they have in common, which is where the disagreement is.
        holds: Every live record involved, whole, so the choice is made on the evidence.
    """

    subject: str
    aspects: tuple[str, ...]
    holds: tuple[MemoryRecord, ...]


class Supersession(AdkModel):
    """What a supersessive write did.

    Args:
        record: What is now stored, with its version and `recorded_at` filled in.
        superseded: The record it closed, with `valid_to` and `superseded_by` set, or
            None where there was nothing to close.
        resolution: What the policy decided.
        contradiction: The disagreement left live, where the policy branched.
    """

    record: MemoryRecord
    superseded: MemoryRecord | None = None
    resolution: Resolution = Resolution.SUPERSEDE
    contradiction: Contradiction | None = None


class Belief(AdkModel):
    """What a scope holds about one key at one instant, including what it will not say.

    Args:
        record: The single live record, or None where there is none, where they
            contradict, or where decay has put it out of reach.
        weight: What decay left of it, 0 to 1. 1.0 where nothing decays.
        contradiction: The disagreement, where there is one. A caller that ignores this
            field gets None for `record` rather than an arbitrary side.
        decayed: Records that exist and are no longer recalled. Present so that a decay
            policy aggressive enough to silence a whole scope is visible rather than
            indistinguishable from a scope that was never written to.
    """

    record: MemoryRecord | None = None
    weight: float = 1.0
    contradiction: Contradiction | None = None
    decayed: tuple[MemoryRecord, ...] = ()


@runtime_checkable
class DecayPolicy(Protocol):
    """How much of a record survives the passage of time, or its own uncertainty.

    Stability: public API under semver. A weight of zero means "do not recall", never
    "delete": what a policy stops surfacing, `history` still returns.
    """

    def weigh(self, record: MemoryRecord, *, now: float) -> float:
        """Return how much to believe `record` at `now`, 0 to 1. Zero is not recallable."""
        ...


@dataclass(frozen=True, slots=True)
class HalfLife:
    """Exponential decay by age, ignored below `floor`.

    Args:
        half_life_seconds: The age at which a record counts for half of what it did.
        floor: The weight under which a record stops being recalled at all. Zero keeps
            everything recallable and only re-ranks it.
    """

    half_life_seconds: float
    floor: float = 0.0

    def weigh(self, record: MemoryRecord, *, now: float) -> float:
        """Halve the weight per half-life of age, and return zero once under the floor."""
        if record.valid_from is None:
            return record.confidence
        age = max(now - record.valid_from, 0.0)
        weight = record.confidence * math.pow(0.5, age / self.half_life_seconds)
        return 0.0 if weight < self.floor else weight


@dataclass(frozen=True, slots=True)
class ConfidenceFloor:
    """Recall only what was written with at least `minimum` confidence.

    Args:
        minimum: The confidence below which a record is not recalled. Age is not
            consulted: an uncertain fact does not become certain by getting old.
    """

    minimum: float

    def weigh(self, record: MemoryRecord, *, now: float) -> float:  # noqa: ARG002 — age is not the question here
        """Return the record's own confidence, or zero where it is under the floor."""
        return 0.0 if record.confidence < self.minimum else record.confidence
