"""Embedding a corpus twice and paying for it once, and failing without leaving holes.

Four scenarios: the second ingest of an unchanged corpus; a model upgrade that must not
reuse the old vectors; two tenants that must not share an entry; and a provider that stops
answering, which ends with somewhere to resume from rather than a vector nobody asked for.

Run it with `python examples/embedding.py`. Nothing here reaches the network: the vector
source is the deterministic fake from `tesserix_adk.testing`, and a deployment passes its
own `VectorSource` instead.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    EmbeddingUnavailableError,
    RateLimitError,
    RetryConfig,
    tenant_scope,
)
from tesserix_adk.rag import BatchedEmbedder, EmbeddingModel, MemoryEmbeddingCache
from tesserix_adk.testing import FakeEmbedder

HANDBOOK = (
    "Book the overnight service where the meeting starts before ten.",
    "A berth must be booked ahead; there are four to a compartment.",
    "Submit the ticket within thirty days. A claim without a ticket is refused.",
    "Expenses over the limit need a line manager's approval before the trip.",
)

MODEL = EmbeddingModel(name="example-embedder", version="1", dimension=8, max_batch=2)


async def the_second_ingest_is_nearly_free() -> None:
    """The cache is keyed on the text and the model, so unchanged passages are not sent."""
    cache = MemoryEmbeddingCache()
    source = FakeEmbedder(model=MODEL)

    await BatchedEmbedder(source, cache=cache, shared=True).embed_documents(HANDBOOK)
    edited = (*HANDBOOK[:-1], "Expenses over the limit need approval before the trip.")
    again = await BatchedEmbedder(source, cache=cache, shared=True).embed_documents(edited)

    print(  # noqa: T201
        f"re-ingest: {again.stats.cached} cached, {again.stats.embedded} embedded,"
        f" hit rate {again.stats.hit_rate:.0%}"
    )


async def a_model_upgrade_starts_again() -> None:
    """Vectors from two models in one index are a distance nobody can interpret."""
    cache = MemoryEmbeddingCache()
    await BatchedEmbedder(FakeEmbedder(model=MODEL), cache=cache, shared=True).embed_documents(
        HANDBOOK
    )

    upgraded = MODEL.model_copy(update={"version": "2"})
    after = await BatchedEmbedder(
        FakeEmbedder(model=upgraded), cache=cache, shared=True
    ).embed_documents(HANDBOOK)

    print(f"after the upgrade: {after.stats.cached} cached of {after.stats.requested}")  # noqa: T201


async def one_tenants_text_is_not_anothers() -> None:
    """The default: every entry keyed to the tenant in force, and no tenant is a refusal."""
    cache = MemoryEmbeddingCache()
    source = FakeEmbedder(model=MODEL)
    embedder = BatchedEmbedder(source, cache=cache)

    with tenant_scope("acme"):
        await embedder.embed_documents(("the same wording in both corpora",))
    with tenant_scope("globex"):
        crossed = await embedder.embed_documents(("the same wording in both corpora",))

    print(f"across tenants: {crossed.stats.cached} cached, {source.calls} calls made")  # noqa: T201


async def a_provider_that_stops_answering() -> None:
    """No zero vector to keep the pipeline moving: a cursor to resume the ingest from."""
    cache = MemoryEmbeddingCache()
    source = FakeEmbedder(model=MODEL, failures=[None, RateLimitError("slow down")])
    embedder = BatchedEmbedder(
        source, cache=cache, shared=True, concurrency=1, retry=RetryConfig(max_attempts=1)
    )

    try:
        await embedder.embed_documents(HANDBOOK)
    except EmbeddingUnavailableError as stopped:
        print(f"stopped on batch {stopped.batch}; resume from text {stopped.cursor}")  # noqa: T201

    resumed = await BatchedEmbedder(
        FakeEmbedder(model=MODEL), cache=cache, shared=True
    ).embed_documents(HANDBOOK)
    print(f"on resuming: {resumed.stats.cached} already held, {resumed.stats.embedded} sent")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await the_second_ingest_is_nearly_free()
    await a_model_upgrade_starts_again()
    await one_tenants_text_is_not_anothers()
    await a_provider_that_stops_answering()


if __name__ == "__main__":
    asyncio.run(main())
