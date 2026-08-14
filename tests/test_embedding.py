"""Embedding a corpus once, paying for what changed, and never mixing vectors.

Re-ingesting a corpus that is ninety per cent unchanged should cost ten per cent of the
first ingest, and the vectors for the unchanged part should be the ones already in the
index. That needs a cache keyed on what actually determines a vector — the model, its
version and the text — and it needs the failures to be honest: a rate-limited ingest stops
with a cursor to resume from rather than filling the gap with zeros nobody can spot.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    EmbeddingDimensionError,
    EmbeddingUnavailableError,
    MissingTenantContextError,
    RateLimitError,
    RetryConfig,
    SchemaViolationError,
    Usage,
    tenant_scope,
)
from tesserix_adk.rag import (
    BatchedEmbedder,
    EmbeddedBatch,
    Embedder,
    EmbeddingCache,
    EmbeddingModel,
    MemoryEmbeddingCache,
    embedding_key,
)
from tesserix_adk.testing import FakeClock, FakeEmbedder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.rag import Vector

pytestmark = pytest.mark.anyio

CORPUS = tuple(f"passage number {n}" for n in range(10))


def embedder(
    source: FakeEmbedder | None = None,
    *,
    cache: EmbeddingCache | None = None,
    shared: bool = True,
    **overrides: object,
) -> BatchedEmbedder:
    """The unit under test, wired to a fake that never reaches the network.

    One call at a time by default, so a test that scripts a failure per call knows which
    call it scripted; the cap itself has a test of its own.
    """
    return BatchedEmbedder(
        source or FakeEmbedder(),
        cache=cache if cache is not None else MemoryEmbeddingCache(),
        shared=shared,
        clock=FakeClock(),
        **{"concurrency": 1, **overrides},  # type: ignore[arg-type]
    )


class TestEmbeddingACorpus:
    async def test_every_text_gets_its_own_vector_in_the_order_it_was_given(self) -> None:
        """Vectors are matched back to passages by position; one out of order is a citation
        pointing at a passage that does not say it."""
        batch = await embedder().embed_documents(CORPUS)
        singly = [(await embedder().embed_documents((text,))).vectors[0] for text in CORPUS]

        assert list(batch.vectors) == singly
        assert len(set(batch.vectors)) == len(CORPUS)

    async def test_the_same_text_embeds_to_the_same_vector(self) -> None:
        under = embedder()

        first = await under.embed_documents(("a passage",))
        second = await under.embed_documents(("a passage",))

        assert first.vectors == second.vectors

    async def test_a_query_is_embedded_by_the_model_that_embedded_the_documents(self) -> None:
        """A query vector from another model is a search in a space nothing was indexed in."""
        under = embedder()

        documents = await under.embed_documents(("a passage",))
        query = await under.embed_query("a passage")

        assert query == documents.vectors[0]

    async def test_the_work_is_sent_in_batches_the_model_will_accept(self) -> None:
        source = FakeEmbedder(model=EmbeddingModel(name="fake", dimension=8, max_batch=3))
        under = embedder(source)

        batch = await under.embed_documents(CORPUS)

        assert source.calls == 4
        assert batch.stats.batches == 4
        assert all(len(sent) <= 3 for sent in source.batches)

    async def test_a_repeated_text_within_one_call_is_sent_once(self) -> None:
        source = FakeEmbedder()
        under = embedder(source)

        batch = await under.embed_documents(("same", "same", "other"))

        assert [len(sent) for sent in source.batches] == [2]
        assert batch.vectors[0] == batch.vectors[1]

    async def test_no_more_calls_are_in_flight_than_were_allowed(self) -> None:
        """An unbounded ingest rate-limits itself out of the provider it is paying for."""
        source = FakeEmbedder(model=EmbeddingModel(name="fake", dimension=8, max_batch=1))

        await embedder(source, concurrency=2).embed_documents(CORPUS)

        assert source.calls == len(CORPUS)
        assert source.most_in_flight <= 2

    async def test_surrounding_whitespace_is_not_a_different_passage(self) -> None:
        source = FakeEmbedder()
        under = embedder(source)

        batch = await under.embed_documents(("a passage", "  a passage\n"))

        assert batch.vectors[0] == batch.vectors[1]
        assert source.batches == [("a passage",)]

    async def test_an_embedder_satisfies_the_protocol(self) -> None:
        assert isinstance(embedder(), Embedder)


class TestPayingOnlyForWhatChanged:
    async def test_a_reingest_of_an_unchanged_corpus_sends_nothing(self) -> None:
        """The whole point: an unchanged corpus is free to re-index, not full price again."""
        source = FakeEmbedder()
        cache = MemoryEmbeddingCache()
        under = embedder(source, cache=cache)

        first = await under.embed_documents(CORPUS)
        again = await embedder(source, cache=cache).embed_documents(CORPUS)

        assert source.calls == 1
        assert again.stats.cached == len(CORPUS)
        assert again.stats.embedded == 0
        assert again.vectors == first.vectors

    async def test_only_the_changed_passages_are_sent(self) -> None:
        source = FakeEmbedder()
        cache = MemoryEmbeddingCache()
        await embedder(source, cache=cache).embed_documents(CORPUS)

        edited = (*CORPUS[:-1], "passage number 9, rewritten")
        again = await embedder(source, cache=cache).embed_documents(edited)

        assert again.stats.cached == len(CORPUS) - 1
        assert again.stats.embedded == 1
        assert source.batches[-1] == ("passage number 9, rewritten",)

    async def test_a_model_upgrade_does_not_reuse_the_old_models_vectors(self) -> None:
        """Vectors from two models in one index are a distance nobody can interpret."""
        cache = MemoryEmbeddingCache()
        old = FakeEmbedder(model=EmbeddingModel(name="fake", version="1", dimension=8))
        new = FakeEmbedder(model=EmbeddingModel(name="fake", version="2", dimension=8))
        await embedder(old, cache=cache).embed_documents(CORPUS)

        again = await embedder(new, cache=cache).embed_documents(CORPUS)

        assert again.stats.cached == 0
        assert new.calls == 1

    async def test_the_cache_key_is_the_text_and_the_model_and_nothing_else(self) -> None:
        model = EmbeddingModel(name="fake", version="1", dimension=8)
        other = model.model_copy(update={"version": "2"})

        assert embedding_key(model, "text", tenant="acme") == embedding_key(
            model, "text", tenant="acme"
        )
        assert embedding_key(model, "text", tenant="acme") != embedding_key(
            other, "text", tenant="acme"
        )
        assert embedding_key(model, "text", tenant="acme") != embedding_key(
            model, "other", tenant="acme"
        )

    async def test_the_text_itself_is_not_the_key(self) -> None:
        """A cache key is copied into logs and store browsers; the corpus is not."""
        assert "text" not in embedding_key(EmbeddingModel(name="f", dimension=8), "text", tenant="")


class TestOneTenantsVectorsAreNotAnothers:
    async def test_two_tenants_do_not_share_a_cache_entry(self) -> None:
        source = FakeEmbedder()
        cache = MemoryEmbeddingCache()
        under = embedder(source, cache=cache, shared=False)

        with tenant_scope("acme"):
            await under.embed_documents(("shared wording",))
        with tenant_scope("globex"):
            again = await under.embed_documents(("shared wording",))

        assert again.stats.cached == 0
        assert source.calls == 2

    async def test_embedding_for_nobody_in_particular_is_refused(self) -> None:
        """An isolated cache with no tenant in force has nobody to isolate it to."""
        under = embedder(shared=False)

        with pytest.raises(MissingTenantContextError):
            await under.embed_documents(("a passage",))

    async def test_a_public_corpus_can_opt_into_one_shared_cache(self) -> None:
        source = FakeEmbedder()
        cache = MemoryEmbeddingCache()
        under = embedder(source, cache=cache, shared=True)

        with tenant_scope("acme"):
            await under.embed_documents(("public wording",))
        with tenant_scope("globex"):
            again = await under.embed_documents(("public wording",))

        assert again.stats.cached == 1
        assert source.calls == 1


class TestWhatItCost:
    async def test_the_usage_of_every_batch_is_totalled(self) -> None:
        source = FakeEmbedder(
            model=EmbeddingModel(name="fake", dimension=8, max_batch=3),
            usage=Usage(input_tokens=10, output_tokens=0),
        )

        batch = await embedder(source).embed_documents(CORPUS)

        assert batch.usage.input_tokens == 40

    async def test_a_cached_text_costs_nothing(self) -> None:
        source = FakeEmbedder(usage=Usage(input_tokens=10, output_tokens=0))
        cache = MemoryEmbeddingCache()
        await embedder(source, cache=cache).embed_documents(CORPUS)

        again = await embedder(source, cache=cache).embed_documents(CORPUS)

        assert again.usage.input_tokens == 0
        assert again.stats.cached == len(CORPUS)

    async def test_the_stats_say_what_the_hit_rate_was(self) -> None:
        cache = MemoryEmbeddingCache()
        source = FakeEmbedder()
        await embedder(source, cache=cache).embed_documents(CORPUS[:9])

        again = await embedder(source, cache=cache).embed_documents(CORPUS)

        assert again.stats.requested == 10
        assert again.stats.hit_rate == pytest.approx(0.9)

    async def test_an_empty_request_reports_nothing_rather_than_dividing_by_zero(self) -> None:
        batch = await embedder().embed_documents(())

        assert batch.vectors == ()
        assert batch.stats.hit_rate == 0.0


class TestWhenTheProviderWillNotAnswer:
    async def test_a_rate_limit_within_the_retry_budget_is_waited_out(self) -> None:
        source = FakeEmbedder(failures=[RateLimitError("slow down")])

        batch = await embedder(source, retry=RetryConfig(max_attempts=3)).embed_documents(CORPUS)

        assert len(batch.vectors) == len(CORPUS)
        assert batch.stats.retries == 1

    async def test_a_rate_limit_past_the_budget_stops_with_somewhere_to_resume_from(
        self,
    ) -> None:
        """Never a zero vector to keep the pipeline moving: that is a corpus of silent holes."""
        source = FakeEmbedder(
            model=EmbeddingModel(name="fake", dimension=8, max_batch=2),
            failures=[None, None, RateLimitError("no"), RateLimitError("no")],
        )

        with pytest.raises(EmbeddingUnavailableError) as raised:
            await embedder(source, retry=RetryConfig(max_attempts=2)).embed_documents(CORPUS)

        assert raised.value.cursor == 4
        assert raised.value.batch == 2

    async def test_what_was_embedded_before_the_failure_is_kept(self) -> None:
        """Resuming from the cursor must not re-pay for the batches that did land."""
        cache = MemoryEmbeddingCache()
        source = FakeEmbedder(
            model=EmbeddingModel(name="fake", dimension=8, max_batch=2),
            failures=[None, RateLimitError("no")],
        )
        with pytest.raises(EmbeddingUnavailableError) as raised:
            await embedder(source, cache=cache, retry=RetryConfig()).embed_documents(CORPUS)

        resumed = FakeEmbedder(model=source.model)
        after = await embedder(resumed, cache=cache).embed_documents(CORPUS)

        assert raised.value.cursor == 2
        assert after.stats.cached == 8
        assert after.stats.embedded == 2

    async def test_a_failure_that_is_not_worth_retrying_is_not_retried(self) -> None:
        source = FakeEmbedder(failures=[SchemaViolationError("that is not a text")])

        with pytest.raises(EmbeddingUnavailableError):
            await embedder(source, retry=RetryConfig(max_attempts=5)).embed_documents(CORPUS)

        assert source.calls == 1

    async def test_cancelling_an_ingest_cancels_it(self) -> None:
        """A cancellation is not an outage and must not be reported as one."""
        source = FakeEmbedder(failures=[asyncio.CancelledError()])

        with pytest.raises(asyncio.CancelledError):
            await embedder(source).embed_documents(CORPUS)


class TestVectorsOfTheWrongShape:
    async def test_a_vector_of_the_wrong_width_is_refused(self) -> None:
        source = FakeEmbedder(width=4)

        with pytest.raises(EmbeddingDimensionError, match="4"):
            await embedder(source).embed_documents(("a passage",))

    async def test_a_cached_vector_of_a_previous_width_is_refused_rather_than_searched(
        self,
    ) -> None:
        """A stale vector under a live key would be a distance computed over the overlap."""
        cache = MemoryEmbeddingCache()
        model = EmbeddingModel(name="fake", dimension=8)
        await cache.put({embedding_key(model, "a passage", tenant=""): (0.1, 0.2)})

        with pytest.raises(EmbeddingDimensionError):
            await embedder(FakeEmbedder(model=model), cache=cache).embed_documents(("a passage",))

    async def test_a_provider_that_answers_the_wrong_number_of_vectors_is_refused(self) -> None:
        source = FakeEmbedder(short=True)

        with pytest.raises(EmbeddingDimensionError, match="vectors"):
            await embedder(source).embed_documents(CORPUS[:3])


class TestTextThatIsNotWorthEmbedding:
    async def test_an_empty_text_is_refused_naming_which_one(self) -> None:
        """An empty chunk has no meaning to embed, and a vector for it is noise in the index."""
        with pytest.raises(SchemaViolationError, match=r"texts\[1\]"):
            await embedder().embed_documents(("a passage", "", "another"))

    async def test_whitespace_alone_is_the_same_thing(self) -> None:
        with pytest.raises(SchemaViolationError, match=r"texts\[0\]"):
            await embedder().embed_documents(("   \n ",))

    async def test_an_empty_query_is_refused_too(self) -> None:
        with pytest.raises(SchemaViolationError):
            await embedder().embed_query(" ")


class TestWhenTheCacheIsTheThingThatIsDown:
    async def test_an_unreadable_cache_falls_back_to_embedding(self) -> None:
        """Correctness is unaffected and the bill is not: the degradation is counted."""
        source = FakeEmbedder()

        batch = await embedder(source, cache=BrokenCache()).embed_documents(CORPUS)

        assert len(batch.vectors) == len(CORPUS)
        assert batch.stats.cache_failures > 0

    async def test_a_cache_that_cannot_be_written_to_does_not_lose_the_vectors(self) -> None:
        batch = await embedder(cache=BrokenCache(readable=True)).embed_documents(CORPUS[:2])

        assert len(batch.vectors) == 2
        assert batch.stats.cache_failures == 1


class TestTheCacheItself:
    async def test_it_keeps_what_it_was_given(self) -> None:
        cache = MemoryEmbeddingCache()

        await cache.put({"k": (0.1, 0.2)})

        assert await cache.get(["k", "missing"]) == {"k": (0.1, 0.2)}

    async def test_it_is_bounded_so_an_ingest_cannot_exhaust_the_process(self) -> None:
        cache = MemoryEmbeddingCache(keep=2)

        await cache.put({"a": (0.1,), "b": (0.2,)})
        await cache.get(["a"])
        await cache.put({"c": (0.3,)})

        assert sorted(await cache.get(["a", "b", "c"])) == ["a", "c"]

    async def test_a_bound_that_holds_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cache"):
            MemoryEmbeddingCache(keep=0)

    async def test_an_embedder_that_makes_no_calls_at_once_is_refused(self) -> None:
        with pytest.raises(ValueError, match="calls"):
            embedder(concurrency=0)


class BrokenCache:
    """A cache backend that is down, which the kit must survive rather than propagate."""

    def __init__(self, *, readable: bool = False) -> None:
        self._readable = readable

    async def get(self, _keys: Sequence[str]) -> Mapping[str, Vector]:
        """Fail, or answer nothing where only writing is broken."""
        if not self._readable:
            raise ConnectionError("the cache is gone")
        return {}

    async def put(self, _vectors: Mapping[str, Vector]) -> None:
        """Always fail: a write is the half that fails silently in production."""
        raise ConnectionError("the cache is gone")


class TestTheFakeIsWorthTesting:
    async def test_it_answers_the_same_vectors_every_run(self) -> None:
        """A retrieval test whose vectors move between runs asserts nothing."""
        one = await FakeEmbedder().vectors_for(("a passage",))
        other = await FakeEmbedder().vectors_for(("a passage",))

        assert one.vectors == other.vectors

    async def test_its_vectors_are_the_width_it_declares(self) -> None:
        source = FakeEmbedder(model=EmbeddingModel(name="fake", dimension=32))

        answered = await source.vectors_for(("a passage",))

        assert len(answered.vectors[0]) == 32

    async def test_it_needs_no_network(self) -> None:
        batch: EmbeddedBatch = await embedder().embed_documents(CORPUS)

        assert batch.stats.embedded == len(CORPUS)
