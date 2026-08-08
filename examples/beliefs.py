"""A preference that changes, kept rather than overwritten.

Run it with `python examples/beliefs.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import MemoryContradictionError
from tesserix_adk.memory import HalfLife, MemoryKind, MemoryRecord, MemoryScope
from tesserix_adk.testing import FakeClock, InMemoryMemoryStore

SCOPE = MemoryScope(tenant_id="acme", user_id="u1")
MARCH = 1_000.0
AUGUST = 100_000.0


def fact(value: str, *, aspects: tuple[str, ...] = ("diet",), at: float = AUGUST) -> MemoryRecord:
    """One profile record about the user's diet."""
    return MemoryRecord(
        id=f"diet:{value}",
        kind=MemoryKind.PROFILE,
        scope=SCOPE,
        key="diet",
        value=value,
        source="turn",
        subject="user",
        predicate=aspects,
        valid_from=at,
    )


def store(*, decay: HalfLife | None = None) -> InMemoryMemoryStore:
    """A store whose clock says it is August."""
    return InMemoryMemoryStore(clock=FakeClock(start=AUGUST), decay=decay)


async def changed_their_mind() -> None:
    """The old fact is closed and kept, and as-of still answers about March."""
    kept = store()
    await kept.supersede(SCOPE, fact("vegetarian", at=MARCH))
    written = await kept.supersede(SCOPE, fact("eats fish"))

    now = await kept.profile(SCOPE, "diet")
    then = await kept.profile(SCOPE, "diet", as_of=MARCH + 1)
    print("now:", now and now.value)  # noqa: T201
    print("in March:", then and then.value)  # noqa: T201
    print("closed at:", written.superseded and written.superseded.valid_to)  # noqa: T201
    print("trail:", [record.value for record in await kept.history(SCOPE, "diet")])  # noqa: T201


async def not_a_restatement() -> None:
    """A partial overlap branches, and recall refuses to pick a side."""
    kept = store()
    await kept.supersede(SCOPE, fact("vegetarian"))
    written = await kept.supersede(SCOPE, fact("no shellfish", aspects=("diet", "allergies")))

    print("resolution:", written.resolution.value)  # noqa: T201
    try:
        await kept.profile(SCOPE, "diet")
    except MemoryContradictionError as unresolved:
        print("recall refused:", unresolved)  # noqa: T201

    held = await kept.belief(SCOPE, "diet")
    disagreement = held.contradiction.holds if held.contradiction else ()
    print("to show a person:", [record.value for record in disagreement])  # noqa: T201


async def gone_stale() -> None:
    """An old fact stops being recalled, and says so rather than vanishing."""
    kept = store(decay=HalfLife(half_life_seconds=10.0, floor=0.5))
    await kept.supersede(SCOPE, fact("vegetarian", at=MARCH))

    held = await kept.belief(SCOPE, "diet")
    print("recalled:", held.record)  # noqa: T201
    print("decayed:", [record.value for record in held.decayed])  # noqa: T201
    print("still on the trail:", len(await kept.history(SCOPE, "diet")))  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await changed_their_mind()
    await not_a_restatement()
    await gone_stale()


if __name__ == "__main__":
    asyncio.run(main())
