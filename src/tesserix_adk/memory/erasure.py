"""Keeping out what should never have been stored, and getting out what must not stay.

Erasure is a promise made to a person, and a promise kept only in the row is not kept at
all: the embedding built from that row, the summary that quoted it and the cache entry
keyed on it all still say the thing. So a derived artefact records what it came from, and
erasure walks the derivations rather than trusting that deleting rows was enough.

The redaction half is the other end of the same promise. A value that never reaches the
store never has to be chased out of six indices later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.redaction import scrub

if TYPE_CHECKING:
    from pydantic import JsonValue

__all__ = [
    "DEFAULT_REDACTOR",
    "Derivation",
    "DerivedIndex",
    "ErasureReceipt",
    "MemoryRedactor",
    "PatternRedactor",
]


class Derivation(AdkModel):
    """An artefact built from a record, and therefore erased with it.

    Args:
        artefact_id: What the adapter calls the thing it holds — a vector id, a summary
            id, a cache key.
        source_id: The record it was derived from. Erasing that record erases this.
        adapter: Which index holds it, matched against `DerivedIndex.name`.
    """

    artefact_id: str
    source_id: str
    adapter: str


class ErasureReceipt(AdkModel):
    """What an erasure actually removed, in a form somebody can be shown.

    Args:
        counts: How many records went, per kind. Only kinds that were in scope appear.
        artefacts: How many derived artefacts were purged alongside them.
        adapters: The indices this erasure was responsible for.
        outstanding: The adapters it could not reach. Empty on a complete erasure.
        completed_at: When it finished, in epoch seconds. None while it has not.
        complete: Whether everything in scope is gone. A dry run is never complete.
        dry_run: Whether anything was actually removed.
    """

    counts: dict[str, int] = Field(default_factory=dict)
    artefacts: int = 0
    adapters: tuple[str, ...] = ()
    outstanding: tuple[str, ...] = ()
    completed_at: float | None = None
    complete: bool = False
    dry_run: bool = False

    @property
    def records(self) -> int:
        """How many records went in total, across every kind."""
        return sum(self.counts.values())


@runtime_checkable
class DerivedIndex(Protocol):
    """Something holding artefacts derived from records, which erasure must reach into.

    A vector collection, a summary store, an embedding cache. It is asked only to purge
    ids it was given, so it never needs to know what a scope or a kind is.
    """

    @property
    def name(self) -> str:
        """What this index is called, matched against `Derivation.adapter`."""
        ...

    async def purge(self, artefact_ids: tuple[str, ...]) -> int:
        """Delete each artefact and return how many were actually held.

        Must be idempotent: erasure resumes by asking again, and an id already gone is
        not an error.
        """
        ...


@runtime_checkable
class MemoryRedactor(Protocol):
    """What decides whether a value is fit to be stored, and masks it where it is not."""

    def redact(self, value: JsonValue) -> tuple[JsonValue, tuple[str, ...]]:
        """Return the value as it should be stored, and the paths that were masked."""
        ...


@dataclass(frozen=True, slots=True)
class PatternRedactor:
    """Masks anything shaped like a secret, wherever it sits inside the value.

    It walks the whole JSON value rather than the top level, because the token is never
    at the top level: it is the third field of the second element of `tool_calls`.

    Args:
        extra_patterns: Shapes a deployment knows about, such as a local account number.
    """

    extra_patterns: tuple[str, ...] = ()

    def redact(self, value: JsonValue) -> tuple[JsonValue, tuple[str, ...]]:
        """Return the value with sensitive runs masked, and the paths that changed."""
        found: list[str] = []
        return self._walk(value, "", found), tuple(found)

    def _walk(self, value: JsonValue, path: str, found: list[str]) -> JsonValue:
        if isinstance(value, str):
            masked = scrub(value, self.extra_patterns)
            if masked != value:
                found.append(path or "value")
            return masked
        if isinstance(value, dict):
            return {key: self._walk(held, _under(path, key), found) for key, held in value.items()}
        if isinstance(value, list):
            return [
                self._walk(held, _under(path, str(index)), found)
                for index, held in enumerate(value)
            ]
        return value


DEFAULT_REDACTOR = PatternRedactor()
"""Applied on every write path unless a store is constructed with `redactor=None`."""


def _under(path: str, part: str) -> str:
    return f"{path}.{part}" if path else part
