"""What never gets stored, and what leaves when somebody asks to be forgotten.

Run it with `python examples/erasure.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import PartialErasureError
from tesserix_adk.memory import (
    Derivation,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
)
from tesserix_adk.testing import FakeClock, FakeTracer, InMemoryMemoryStore

SCOPE = MemoryScope(tenant_id="acme", user_id="u1")
NOW = 1_000.0


class VectorIndex:
    """A stand-in for whatever holds the embeddings built from remembered text."""

    name = "vectors"

    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable
        self.held: dict[str, str] = {"vec-1": "the sentence somebody wants forgotten"}

    async def purge(self, artefact_ids: tuple[str, ...]) -> int:
        """Drop each artefact, or fail the way an unreachable service fails."""
        if not self.reachable:
            raise ConnectionError("vector index unreachable")
        return len([one for one in artefact_ids if self.held.pop(one, None) is not None])


def note(value: str, *, kind: MemoryKind = MemoryKind.EPISODIC) -> MemoryRecord:
    """One record of `kind`, filled in enough to be stored."""
    return MemoryRecord(
        id=f"{kind.value}:note",
        kind=kind,
        scope=SCOPE,
        key="note",
        value=value,
        source="turn",
        valid_from=NOW,
    )


async def filled(index: VectorIndex, tracer: FakeTracer) -> InMemoryMemoryStore:
    """A store holding a record and the embedding derived from it."""
    store = InMemoryMemoryStore(clock=FakeClock(start=NOW), indices=(index,), tracer=tracer)
    await store.log(SCOPE, note("call ada@example.com about the invoice"))
    await store.derived(
        SCOPE, Derivation(artefact_id="vec-1", source_id="episodic:note", adapter="vectors")
    )
    return store


async def never_stored() -> None:
    """The address is masked on the way in, and the record says which field went."""
    store = InMemoryMemoryStore(clock=FakeClock(start=NOW))
    await store.write(SCOPE, note("write to ada@example.com", kind=MemoryKind.WORKING))

    held = await store.read(SCOPE, "note")
    print("stored:", held and held.value)  # noqa: T201
    print("masked:", held and held.redacted)  # noqa: T201


async def asked_to_be_forgotten() -> None:
    """A dry run counts, the real erasure removes the record and its embedding."""
    index, tracer = VectorIndex(), FakeTracer()
    store = await filled(index, tracer)

    planned = await store.erase(SCOPE, dry_run=True)
    print("would remove:", planned.counts, "| complete:", planned.complete)  # noqa: T201

    receipt = await store.erase(SCOPE)
    print("removed:", receipt.counts, "| artefacts:", receipt.artefacts)  # noqa: T201
    print("index now holds:", index.held)  # noqa: T201
    print("audited:", [event.name for event in tracer.recorded])  # noqa: T201


async def index_unreachable() -> None:
    """The rows go out of reach anyway, and the erasure resumes when the index is back."""
    index = VectorIndex(reachable=False)
    store = await filled(index, FakeTracer())

    try:
        await store.erase(SCOPE)
    except PartialErasureError as stalled:
        print("stalled on:", stalled.adapter)  # noqa: T201
        left = await store.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))
        print("still readable:", left)  # noqa: T201

    index.reachable = True
    print("resumed:", (await store.erase(SCOPE)).complete)  # noqa: T201
    print("index now holds:", index.held)  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await never_stored()
    await asked_to_be_forgotten()
    await index_unreachable()


if __name__ == "__main__":
    asyncio.run(main())
