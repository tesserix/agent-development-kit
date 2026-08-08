"""What an adapter can do, declared once and checked when it is bound.

Semantic recall against a store with no vector index returns nothing, every run, without
error. Nobody notices for a month. So the adapter declares, the agent states what it needs,
and the mismatch is an exception at bind time.
"""

from __future__ import annotations

from dataclasses import dataclass

from tesserix_adk.core.errors import CapabilityError

__all__ = [
    "MemoryCapabilities",
    "MemoryNeeds",
    "require_memory",
]


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    """What a store supports, in its own words.

    Args:
        supports_semantic: Whether `index` and `search` do anything.
        supports_as_of: Whether a query may ask what was true at a past time.
        supports_erasure: Whether `erase` removes a scope's records.
        supports_supersession: Whether `supersede`, `belief` and `history` keep versions
            rather than overwriting. A store that cannot say what it used to believe
            refuses the write rather than dropping the old record on the floor.
        embedding_dimensions: The width the vector collection was built with, where the
            store has one. None means it does not check.
        max_value_bytes: The largest value the store holds. None means no declared limit.
    """

    supports_semantic: bool = True
    supports_as_of: bool = True
    supports_erasure: bool = True
    supports_supersession: bool = True
    embedding_dimensions: int | None = None
    max_value_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryNeeds:
    """What a consumer's plan requires of whatever store it is given.

    Args:
        semantic: The plan recalls by resemblance.
        as_of: The plan reads memory as it stood at a past time.
        erasure: The plan promises a subject that their memory can be deleted.
        supersession: The plan changes beliefs over time and must be able to say why.
    """

    semantic: bool = False
    as_of: bool = False
    erasure: bool = False
    supersession: bool = False


def require_memory(store: object, needs: MemoryNeeds) -> None:
    """Check `store` against `needs`, or raise naming everything it is missing.

    Args:
        store: The adapter about to be bound. Named in the error, because two adapters
            behind one interface make "unsupported" unactionable.
        needs: What the plan will ask of it.

    Raises:
        CapabilityError: If anything needed is not declared. Every missing capability is
            named at once — one deploy per missing capability is three deploys.
    """
    declared = getattr(store, "capabilities", MemoryCapabilities())
    missing = [
        name
        for name in ("semantic", "as_of", "erasure", "supersession")
        if getattr(needs, name) and not getattr(declared, f"supports_{name}")
    ]
    if not missing:
        return
    adapter = type(store).__name__
    raise CapabilityError(
        f"{adapter} does not support {' '.join(missing)}",
        capability=" ".join(missing),
        provider=adapter,
    )
