"""What two hundred concurrent embedding calls turn into, and who gets which vector.

Every caller here asks for one text and asserts it got the vector for that text and no
other, which is the whole safety argument for coalescing. A local provider stands in for
the vendor, so nothing reaches the network and no key is needed. Run it with
`python examples/embedding_batching.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import InvalidRequestError
from tesserix_adk.models import BatchingEmbedder, EmbeddingLimits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tesserix_adk.models import Vector

MODEL = "text-embedding-3-small"


def vector_for(text: str, dimensions: int = 8) -> Vector:
    """A deterministic stand-in for a real embedding of `text`."""
    digest = hashlib.sha256(text.encode()).digest()
    return tuple(digest[index] / 255 for index in range(dimensions))


class LocalEmbeddings:
    """An embedding provider that answers from a hash and counts its calls."""

    def __init__(self, *, refuses: frozenset[str] = frozenset()) -> None:
        self.batches: list[tuple[str, ...]] = []
        self.refuses = refuses

    @property
    def name(self) -> str:
        """What this provider is called."""
        return "local"

    def limits(self, model: str) -> EmbeddingLimits:
        """What this provider will accept in one call."""
        del model
        return EmbeddingLimits(max_items=32, max_bytes=100_000, max_item_tokens=200, dimensions=8)

    async def embed(self, texts: Sequence[str], *, model: str) -> Sequence[Vector]:
        """Embed a batch, refusing any text this provider was told to refuse."""
        del model
        self.batches.append(tuple(texts))
        refused = [text for text in texts if text in self.refuses]
        if refused:
            raise InvalidRequestError(f"{self.name} refused {refused[0]!r}")
        return [vector_for(text) for text in texts]


async def two_hundred_calls_become_a_handful() -> None:
    """Concurrent single-text calls are coalesced, and each caller keeps its own vector."""
    provider = LocalEmbeddings()
    chunks = [f"paragraph {index}" for index in range(200)]

    async with BatchingEmbedder(provider) as embedding:
        vectors = await asyncio.gather(*(embedding.embed(one, model=MODEL) for one in chunks))

    mismatched = [one for one, got in zip(chunks, vectors, strict=True) if got != vector_for(one)]
    print("\ntwo hundred chunks embedded at once")  # noqa: T201
    print(f"  provider calls: {len(provider.batches)}")  # noqa: T201
    print(f"  largest batch:  {max(len(one) for one in provider.batches)}")  # noqa: T201
    print(f"  wrong vectors:  {len(mismatched)}")  # noqa: T201


async def a_query_does_not_wait_behind_the_bulk() -> None:
    """An interactive request is sent on its own rather than joining the window."""
    provider = LocalEmbeddings()
    async with BatchingEmbedder(provider) as embedding:
        bulk = [embedding.embed(f"paragraph {index}", model=MODEL) for index in range(40)]
        query = embedding.embed("what did the report say", model=MODEL, interactive=True)
        await asyncio.gather(query, *bulk)

    alone = [one for one in provider.batches if one == ("what did the report say",)]
    print("\na query embedded during a bulk flush")  # noqa: T201
    print(f"  sent on its own: {bool(alone)}")  # noqa: T201
    print(f"  bypassed:        {embedding.metrics.bypassed}")  # noqa: T201


async def one_bad_item_loses_only_its_own_caller() -> None:
    """A batch containing a text the provider refuses still answers everyone else."""
    provider = LocalEmbeddings(refuses=frozenset({"poison"}))
    async with BatchingEmbedder(provider) as embedding:
        results = await asyncio.gather(
            embedding.embed("first", model=MODEL),
            embedding.embed("poison", model=MODEL),
            embedding.embed("third", model=MODEL),
            return_exceptions=True,
        )

    print("\none refused item among three")  # noqa: T201
    print(f"  first:  {'vector' if results[0] == vector_for('first') else 'wrong'}")  # noqa: T201
    print(f"  poison: {type(results[1]).__name__}")  # noqa: T201
    print(f"  third:  {'vector' if results[2] == vector_for('third') else 'wrong'}")  # noqa: T201
    print(f"  isolating re-sends: {embedding.metrics.isolated}")  # noqa: T201


async def duplicates_are_sent_once_and_answered_twice() -> None:
    """Two callers asking for the same text cost one slot in the batch."""
    provider = LocalEmbeddings()
    async with BatchingEmbedder(provider) as embedding:
        await asyncio.gather(
            embedding.embed("the same paragraph", model=MODEL),
            embedding.embed("the same paragraph", model=MODEL),
            embedding.embed("a different one", model=MODEL),
        )

    print("\nthree callers, two distinct texts")  # noqa: T201
    print(f"  sent to provider: {provider.batches}")  # noqa: T201
    print(f"  deduplicated:     {embedding.metrics.deduplicated}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await two_hundred_calls_become_a_handful()
    await a_query_does_not_wait_behind_the_bulk()
    await one_bad_item_loses_only_its_own_caller()
    await duplicates_are_sent_once_and_answered_twice()


if __name__ == "__main__":
    asyncio.run(main())
