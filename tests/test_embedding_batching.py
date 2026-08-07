"""What a caller gets back when hundreds of embeddings are asked for at once.

The bargain of coalescing is that a caller writes the simple thing — one text, one
vector — and the kit turns two hundred of those into a handful of provider calls. The
bargain only holds if every caller gets the vector for its own text, so most of what is
asserted here is identity: whose vector came back, and what happens to the neighbours when
one item in a batch is bad.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core.errors import (
    ContextWindowExceededError,
    InvalidRequestError,
    ModelResponseError,
)
from tesserix_adk.models.embeddings import BatchConfig, BatchingEmbedder, EmbeddingLimits
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from tesserix_adk.models.embeddings import Vector

MODEL = "text-embedding-3-small"
GUARD = 5.0


def vector_for(text: str, dimensions: int = 4) -> Vector:
    """The vector this fake provider always returns for `text`, and only for `text`."""
    digest = hashlib.sha256(text.encode()).digest()
    return tuple(digest[index] / 255 for index in range(dimensions))


class Recorder:
    """An embedding provider that records every batch it was asked for."""

    def __init__(
        self,
        *,
        dimensions: int = 4,
        max_items: int = 8,
        max_bytes: int = 10_000,
        max_item_tokens: int = 100,
        refuses: frozenset[str] = frozenset(),
        held: asyncio.Event | None = None,
    ) -> None:
        self.batches: list[tuple[str, ...]] = []
        self.dimensions = dimensions
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.max_item_tokens = max_item_tokens
        self.refuses = refuses
        self.held = held
        self.short_by = 0
        self.wrong_width = 0

    @property
    def name(self) -> str:
        return "fake"

    def limits(self, model: str) -> EmbeddingLimits:
        del model
        return EmbeddingLimits(
            max_items=self.max_items,
            max_bytes=self.max_bytes,
            max_item_tokens=self.max_item_tokens,
            dimensions=self.dimensions,
        )

    async def embed(self, texts: Sequence[str], *, model: str) -> Sequence[Vector]:
        del model
        self.batches.append(tuple(texts))
        if self.held is not None:
            await self.held.wait()
        refused = [text for text in texts if text in self.refuses]
        if refused:
            raise InvalidRequestError(f"{self.name} refused {refused[0]!r}")
        width = self.dimensions - self.wrong_width
        vectors = [vector_for(text, width) for text in texts]
        return vectors[: len(vectors) - self.short_by]


def embedder(provider: Recorder, clock: FakeClock, **overrides: float | int) -> BatchingEmbedder:
    """A batching embedder over `provider`, whose window only the test moves."""
    return BatchingEmbedder(provider, BatchConfig(**overrides), clock=clock)  # type: ignore[arg-type]


async def driving[T](clock: FakeClock, embedding: asyncio.Future[T]) -> None:
    """Advance the window whenever the batcher is waiting on it, until the work is done."""
    seen = 0
    while not embedding.done():
        await asyncio.sleep(0)
        if len(clock.slept) > seen:
            seen = len(clock.slept)
            clock.advance(60)


async def gathered(clock: FakeClock, *awaited: Awaitable[object]) -> list[object]:
    """Run `awaited` concurrently while the window is advanced under them."""
    work: asyncio.Future[list[object]] = asyncio.ensure_future(asyncio.gather(*awaited))
    await asyncio.wait_for(asyncio.gather(work, driving(clock, work)), GUARD)
    return list(work.result())


class TestCoalescing:
    """Many concurrent single-text requests become a handful of provider calls."""

    async def test_two_hundred_requests_become_batches_of_the_declared_size(self) -> None:
        provider = Recorder(max_items=25)
        clock = FakeClock(auto_advance=False)
        texts = [f"document {index}" for index in range(200)]

        async with embedder(provider, clock) as embedding:
            vectors = await gathered(clock, *(embedding.embed(text, model=MODEL) for text in texts))

        assert len(provider.batches) == 8
        assert max(len(batch) for batch in provider.batches) == 25
        assert vectors == [vector_for(text) for text in texts]

    async def test_every_caller_gets_the_vector_for_its_own_text(self) -> None:
        provider = Recorder(max_items=4)
        clock = FakeClock(auto_advance=False)
        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]

        async with embedder(provider, clock) as embedding:
            vectors = await gathered(clock, *(embedding.embed(text, model=MODEL) for text in texts))

        assert dict(zip(texts, vectors, strict=True)) == {t: vector_for(t) for t in texts}

    async def test_a_batch_still_filling_is_flushed_on_its_deadline(self) -> None:
        provider = Recorder(max_items=64)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock, max_wait_seconds=0.05) as embedding:
            waiting = asyncio.ensure_future(
                asyncio.gather(*(embedding.embed(text, model=MODEL) for text in ("a", "b", "c")))
            )
            await clock.wait_for_sleep(1)
            assert provider.batches == []

            clock.advance(0.05)
            await asyncio.wait_for(waiting, GUARD)

        assert provider.batches == [("a", "b", "c")]

    async def test_a_full_batch_does_not_wait_for_the_window(self) -> None:
        provider = Recorder(max_items=3)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock, max_wait_seconds=30.0) as embedding:
            vectors = await asyncio.wait_for(
                asyncio.gather(*(embedding.embed(t, model=MODEL) for t in ("a", "b", "c"))), GUARD
            )

        assert provider.batches == [("a", "b", "c")]
        assert vectors == [vector_for(t) for t in ("a", "b", "c")]

    async def test_a_batch_too_large_in_bytes_is_split(self) -> None:
        provider = Recorder(max_items=64, max_bytes=20)
        clock = FakeClock(auto_advance=False)
        texts = ["x" * 8, "y" * 8, "z" * 8]

        async with embedder(provider, clock) as embedding:
            await gathered(clock, *(embedding.embed(text, model=MODEL) for text in texts))

        assert [len(batch) for batch in provider.batches] == [2, 1]


class TestBypass:
    """An interactive embedding is a query someone is waiting on, not bulk work."""

    async def test_an_interactive_embedding_skips_the_window(self) -> None:
        provider = Recorder(max_items=64)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock, max_wait_seconds=30.0) as embedding:
            vector = await asyncio.wait_for(
                embedding.embed("what did we decide", model=MODEL, interactive=True), GUARD
            )

        assert vector == vector_for("what did we decide")
        assert provider.batches == [("what did we decide",)]

    async def test_it_does_not_queue_behind_a_bulk_flush(self) -> None:
        held = asyncio.Event()
        provider = Recorder(max_items=2, held=held)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            bulk = asyncio.ensure_future(
                asyncio.gather(*(embedding.embed(t, model=MODEL) for t in ("a", "b")))
            )
            await asyncio.sleep(0)
            query = asyncio.ensure_future(embedding.embed("urgent", model=MODEL, interactive=True))
            await asyncio.sleep(0)
            held.set()

            assert await asyncio.wait_for(query, GUARD) == vector_for("urgent")
            await asyncio.wait_for(bulk, GUARD)


class TestKeying:
    """Vectors from two models, two widths or two tenants are never in one batch."""

    async def test_two_models_are_never_batched_together(self) -> None:
        provider = Recorder(max_items=64)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            await gathered(
                clock,
                embedding.embed("a", model=MODEL),
                embedding.embed("b", model="text-embedding-3-large"),
            )

        assert sorted(provider.batches) == [("a",), ("b",)]

    async def test_two_tenants_are_never_batched_together(self) -> None:
        provider = Recorder(max_items=64)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            await gathered(
                clock,
                embedding.embed("a", model=MODEL, tenant="acme"),
                embedding.embed("b", model=MODEL, tenant="globex"),
            )

        assert sorted(provider.batches) == [("a",), ("b",)]


class TestDeduplication:
    """The provider sees one copy; both callers are answered."""

    async def test_identical_inputs_are_sent_once_and_answered_twice(self) -> None:
        provider = Recorder(max_items=64)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            vectors = await gathered(
                clock,
                embedding.embed("same", model=MODEL),
                embedding.embed("same", model=MODEL),
                embedding.embed("other", model=MODEL),
            )

        assert provider.batches == [("same", "other")]
        assert vectors == [vector_for("same"), vector_for("same"), vector_for("other")]


class TestFailureIsolation:
    """One bad item loses its own caller, and only its own caller."""

    async def test_the_refused_item_fails_and_its_siblings_do_not(self) -> None:
        provider = Recorder(max_items=64, refuses=frozenset({"poison"}))
        clock = FakeClock(auto_advance=False)
        texts = ["a", "b", "poison", "c"]

        async with embedder(provider, clock) as embedding:
            work = asyncio.ensure_future(
                asyncio.gather(
                    *(embedding.embed(t, model=MODEL) for t in texts), return_exceptions=True
                )
            )
            await asyncio.wait_for(asyncio.gather(work, driving(clock, work)), GUARD)
            results = work.result()

        assert isinstance(results[2], InvalidRequestError)
        assert [results[0], results[1], results[3]] == [vector_for(t) for t in ("a", "b", "c")]

    async def test_an_oversized_input_never_reaches_the_provider(self) -> None:
        provider = Recorder(max_items=64, max_item_tokens=4)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            with pytest.raises(ContextWindowExceededError) as refused:
                await embedding.embed("far too many characters to embed", model=MODEL)

            assert provider.batches == []

        assert refused.value.limit == 4

    async def test_a_short_answer_is_refused_rather_than_padded(self) -> None:
        provider = Recorder(max_items=64)
        provider.short_by = 1
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            work = asyncio.ensure_future(
                asyncio.gather(
                    embedding.embed("a", model=MODEL),
                    embedding.embed("b", model=MODEL),
                    return_exceptions=True,
                )
            )
            await asyncio.wait_for(asyncio.gather(work, driving(clock, work)), GUARD)

        assert all(isinstance(one, ModelResponseError) for one in work.result())

    async def test_a_vector_of_the_wrong_width_is_refused(self) -> None:
        provider = Recorder(max_items=64)
        provider.wrong_width = 1
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            work = asyncio.ensure_future(embedding.embed("a", model=MODEL))
            with pytest.raises(ModelResponseError, match="width"):
                await asyncio.wait_for(asyncio.gather(work, driving(clock, work)), GUARD)


class TestCancellation:
    """A caller that gave up takes nothing else with it."""

    async def test_one_cancelled_caller_leaves_its_siblings_answered(self) -> None:
        held = asyncio.Event()
        provider = Recorder(max_items=64, held=held)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            first = asyncio.ensure_future(embedding.embed("a", model=MODEL))
            second = asyncio.ensure_future(embedding.embed("b", model=MODEL))
            await clock.wait_for_sleep(1)
            clock.advance(60)
            await asyncio.sleep(0)
            first.cancel()
            held.set()

            assert await asyncio.wait_for(second, GUARD) == vector_for("b")
            with pytest.raises(asyncio.CancelledError):
                await first

    async def test_a_cancelled_caller_in_a_failed_batch_is_left_alone(self) -> None:
        held = asyncio.Event()
        provider = Recorder(max_items=64, refuses=frozenset({"a", "b"}), held=held)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            first = asyncio.ensure_future(embedding.embed("a", model=MODEL))
            second = asyncio.ensure_future(embedding.embed("b", model=MODEL))
            await clock.wait_for_sleep(1)
            clock.advance(60)
            await asyncio.sleep(0)
            first.cancel()
            held.set()

            with pytest.raises(InvalidRequestError):
                await asyncio.wait_for(second, GUARD)
            with pytest.raises(asyncio.CancelledError):
                await first


class TestDeclaredLimits:
    """The batch size is the provider's to declare, not the kit's to assume."""

    async def test_a_provider_that_raises_its_ceiling_gets_bigger_batches(self) -> None:
        provider = Recorder(max_items=2)
        clock = FakeClock(auto_advance=False)
        texts = ["a", "b", "c", "d"]

        async with embedder(provider, clock) as embedding:
            await gathered(clock, *(embedding.embed(t, model=MODEL) for t in texts))
            provider.max_items = 4
            await gathered(clock, *(embedding.embed(t, model=MODEL) for t in texts))

        assert [len(batch) for batch in provider.batches] == [2, 2, 4]

    async def test_a_configured_ceiling_may_narrow_the_declared_one_and_not_widen_it(self) -> None:
        provider = Recorder(max_items=4)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock, max_items=2) as embedding:
            await gathered(clock, *(embedding.embed(t, model=MODEL) for t in "abcd"))

        assert [len(batch) for batch in provider.batches] == [2, 2]

        provider = Recorder(max_items=2)
        async with embedder(provider, clock, max_items=64) as embedding:
            await gathered(clock, *(embedding.embed(t, model=MODEL) for t in "abcd"))

        assert [len(batch) for batch in provider.batches] == [2, 2]


class TestLifecycle:
    """The embedder's window is a task, and a task nobody stops is a leak."""

    async def test_closing_refuses_further_work(self) -> None:
        provider = Recorder()
        clock = FakeClock(auto_advance=False)
        embedding = embedder(provider, clock)
        await embedding.aclose()

        with pytest.raises(RuntimeError, match="closed"):
            await embedding.embed("a", model=MODEL)

    async def test_closing_twice_is_quiet(self) -> None:
        embedding = embedder(Recorder(), FakeClock(auto_advance=False))
        await embedding.aclose()
        await embedding.aclose()

    async def test_closing_flushes_what_was_still_waiting(self) -> None:
        provider = Recorder(max_items=64)
        clock = FakeClock(auto_advance=False)
        embedding = embedder(provider, clock, max_wait_seconds=30.0)
        waiting = asyncio.ensure_future(embedding.embed("a", model=MODEL))
        await clock.wait_for_sleep(1)

        await embedding.aclose()

        assert await asyncio.wait_for(waiting, GUARD) == vector_for("a")
        assert provider.batches == [("a",)]

        clock.advance(60)
        await asyncio.sleep(0)

        assert provider.batches == [("a",)]

    async def test_a_window_firing_during_close_does_not_send_its_batch_twice(self) -> None:
        held = asyncio.Event()
        provider = Recorder(max_items=64, held=held)
        clock = FakeClock(auto_advance=False)
        embedding = embedder(provider, clock, max_wait_seconds=30.0)
        first = asyncio.ensure_future(embedding.embed("a", model=MODEL, tenant="acme"))
        second = asyncio.ensure_future(embedding.embed("b", model=MODEL, tenant="globex"))
        await clock.wait_for_sleep(2)

        closing = asyncio.ensure_future(embedding.aclose())
        await asyncio.sleep(0)
        clock.advance(60)
        held.set()

        answers = await asyncio.wait_for(asyncio.gather(first, second, closing), GUARD)
        assert list(answers[:2]) == [vector_for("a"), vector_for("b")]
        assert sorted(provider.batches) == [("a",), ("b",)]


class TestMetrics:
    """What the embedder did, for an operator deciding whether the window is right."""

    async def test_the_counters_describe_the_work(self) -> None:
        provider = Recorder(max_items=2)
        clock = FakeClock(auto_advance=False)

        async with embedder(provider, clock) as embedding:
            await gathered(
                clock,
                embedding.embed("a", model=MODEL),
                embedding.embed("a", model=MODEL),
                embedding.embed("b", model=MODEL),
                embedding.embed("c", model=MODEL, interactive=True),
            )

        metrics = embedding.metrics
        assert (metrics.requests, metrics.deduplicated, metrics.bypassed) == (4, 1, 1)
        assert metrics.batches == len(provider.batches)
