"""Finding the booking reference and the paraphrase in one call, and saying which found it.

Four scenarios: an exact identifier only the keyword branch can find; a paraphrase only
the semantic branch can; another tenant's identical passage, which neither may return; and
a branch that is down, which reads as a partial result or as a refusal.

Run it with `python examples/retrieval.py`. Nothing here reaches the network: the store is
the in-process fake from `tesserix_adk.testing`, and a deployment passes a `PgvectorIndex`
or its own `SearchIndex` instead.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import RetrievalDegradedError, tenant_scope
from tesserix_adk.rag import (
    Branch,
    EmbeddedBatch,
    HybridRetriever,
    IndexRetriever,
    RetrievalScope,
)
from tesserix_adk.testing import FakeIndex, Indexed

HANDBOOK = RetrievalScope(collection="handbook")

MEANINGS = {
    "refunds": (1.0, 0.0, 0.0),
    "berths": (0.0, 1.0, 0.0),
    "booking BX-7741": (0.0, 0.0, 1.0),
}


SENSES = {
    "refunds": ("refund", "money", "reimburse"),
    "berths": ("berth", "cabin", "sleep"),
    "booking BX-7741": ("bx-7741", "booking"),
}


class Toy:
    """An embedder over a three-meaning vocabulary, standing in for a real one."""

    async def embed_query(self, text: str) -> tuple[float, ...]:
        """The vector for whichever meaning `text` is nearest."""
        for meaning, words in SENSES.items():
            if any(word in text.lower() for word in words):
                return MEANINGS[meaning]
        return (0.0, 0.0, 0.0)

    async def embed_documents(self, texts: list[str]) -> EmbeddedBatch:
        """Not used here: the passages were embedded at ingest."""
        raise NotImplementedError


def handbook(**overrides: object) -> FakeIndex:
    """The corpus, including one passage belonging to somebody else."""
    return FakeIndex(
        Indexed(
            "refunds",
            "A refund is paid within fourteen days of an approved claim.",
            vector=MEANINGS["refunds"],
        ),
        Indexed(
            "berths",
            "Berths are allocated by seniority at the start of a voyage.",
            vector=MEANINGS["berths"],
        ),
        Indexed(
            "booking",
            "Booking BX-7741 is held until the Friday before departure.",
            vector=MEANINGS["booking BX-7741"],
        ),
        Indexed(
            "globex-refunds",
            "A refund is paid within fourteen days of an approved claim.",
            vector=MEANINGS["refunds"],
            tenant="globex",
        ),
        **overrides,  # type: ignore[arg-type]
    )


def both(store: FakeIndex, **overrides: object) -> HybridRetriever:
    """Both branches over one store, which is the common deployment."""
    return HybridRetriever(
        IndexRetriever(store, branch=Branch.SEMANTIC, embedder=Toy()),
        IndexRetriever(store, branch=Branch.KEYWORD),
        **overrides,  # type: ignore[arg-type]
    )


async def an_identifier_the_vector_cannot_find() -> None:
    """A booking reference is a string, not a meaning: the keyword branch earns its place."""
    with tenant_scope("acme"):
        found = await both(handbook()).retrieve("BX-7741", scope=HANDBOOK)

    top = found.hits[0]
    print(f"BX-7741: {top.chunk_id}, keyword={top.found_by(Branch.KEYWORD)}")  # noqa: T201


async def a_paraphrase_no_keyword_matches() -> None:
    """The question shares no word with the passage that answers it."""
    with tenant_scope("acme"):
        found = await both(handbook()).retrieve("getting my money back", scope=HANDBOOK)

    top = found.hits[0]
    print(f"paraphrase: {top.chunk_id}, semantic={top.found_by(Branch.SEMANTIC)}")  # noqa: T201


async def one_tenants_passage_is_not_anothers() -> None:
    """Identical text under two tenants; the predicate is set from the scope, not the call."""
    with tenant_scope("acme"):
        found = await both(handbook()).retrieve("refund", scope=HANDBOOK, k=10)

    print(f"acme sees: {[hit.chunk_id for hit in found.hits]}")  # noqa: T201


async def a_branch_that_is_down() -> None:
    """Partial by default, and a refusal where the missing branch changes the answer."""
    broken = handbook(fails=True)
    survivor = HybridRetriever(
        IndexRetriever(handbook(), branch=Branch.SEMANTIC, embedder=Toy()),
        IndexRetriever(broken, branch=Branch.KEYWORD),
    )

    with tenant_scope("acme"):
        found = await survivor.retrieve("refund", scope=HANDBOOK)
        print(f"partial={found.partial}, answered={[b.value for b in found.branches]}")  # noqa: T201

        strict = HybridRetriever(
            IndexRetriever(handbook(), branch=Branch.SEMANTIC, embedder=Toy()),
            IndexRetriever(broken, branch=Branch.KEYWORD),
            require=(Branch.KEYWORD,),
        )
        try:
            await strict.retrieve("BX-7741", scope=HANDBOOK)
        except RetrievalDegradedError as refused:
            print(f"refused: missing {refused.missing}, answered {refused.answered}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await an_identifier_the_vector_cannot_find()
    await a_paraphrase_no_keyword_matches()
    await one_tenants_passage_is_not_anothers()
    await a_branch_that_is_down()


if __name__ == "__main__":
    asyncio.run(main())
