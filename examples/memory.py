"""Four kinds of memory behind one store, scoped, and a capability refused at bind time.

Run it with `python examples/memory.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import CapabilityError
from tesserix_adk.memory import (
    MemoryCapabilities,
    MemoryKind,
    MemoryNeeds,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    require_memory,
)
from tesserix_adk.testing import FakeClock, InMemoryMemoryStore

SCOPE = MemoryScope(tenant_id="acme", user_id="u1", session_id="s1", agent="planner")


def remembered(kind: MemoryKind, key: str, value: str, **rest: float) -> MemoryRecord:
    """One record of `kind`, under the scope everything in this example shares."""
    return MemoryRecord(
        id=f"{kind.value}:{key}",
        kind=kind,
        scope=SCOPE,
        key=key,
        value=value,
        source="example",
        **rest,
    )


async def four_kinds() -> None:
    """Each kind written and read back through the operations that suit it."""
    store = InMemoryMemoryStore(clock=FakeClock())

    await store.write(SCOPE, remembered(MemoryKind.WORKING, "draft", "BOM-DEL, 14 Nov"))
    await store.append(SCOPE, "turns", "asked about baggage")
    position = await store.append(SCOPE, "turns", "asked about seats")
    await store.upsert(SCOPE, remembered(MemoryKind.PROFILE, "seat", "aisle"))
    await store.log(SCOPE, remembered(MemoryKind.EPISODIC, "booked", "PNR X1", valid_from=10.0))

    profile = await store.profile(SCOPE, "seat")
    episodes = await store.episodes(SCOPE, MemoryQuery(kind=MemoryKind.EPISODIC))
    print("turns recorded:", position)  # noqa: T201
    print("seat preference:", profile.value if profile else None)  # noqa: T201
    print("episodes:", [hit.record.value for hit in episodes])  # noqa: T201


async def scoped() -> None:
    """A second tenant sees none of it, and erasure stops where it was told to."""
    store = InMemoryMemoryStore(clock=FakeClock())
    await store.write(SCOPE, remembered(MemoryKind.WORKING, "draft", "BOM-DEL"))

    elsewhere = await store.read(MemoryScope(tenant_id="other"), "draft")
    erased = await store.erase(SCOPE)
    print("another tenant reads:", elsewhere, "| records erased:", erased)  # noqa: T201


def bound() -> None:
    """A plan that needs semantic recall, and a store that cannot do it."""
    store = InMemoryMemoryStore(
        clock=FakeClock(), capabilities=MemoryCapabilities(supports_semantic=False)
    )
    try:
        require_memory(store, MemoryNeeds(semantic=True))
    except CapabilityError as refused:
        print("refused at bind time:", refused)  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await four_kinds()
    await scoped()
    bound()


if __name__ == "__main__":
    asyncio.run(main())
