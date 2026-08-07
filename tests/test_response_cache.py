"""What may be served from a cache, and what must never be.

A cache that returns the wrong answer is worse than no cache, and the two ways it does
that are serving another tenant's answer and serving a stale one. Most of what is asserted
here is therefore about misses: the determinants that must produce one, the settings that
must refuse caching outright, and the outage that must degrade to a live call rather than
fail the run.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core.capabilities import ModelCapabilities
from tesserix_adk.core.errors import ProviderUnavailableError
from tesserix_adk.core.primitives import Message, TextPart, Usage
from tesserix_adk.core.provider import ModelRequest, ModelResponse, ToolDeclaration
from tesserix_adk.models.cache import (
    CachePolicy,
    CacheStatus,
    CachingProvider,
    MemoryCacheStore,
    MemorySemanticIndex,
    SemanticConfig,
    not_cacheable,
)
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.core.streaming import StreamEvent
    from tesserix_adk.models.embeddings import Vector

MODEL = "gpt-4o"
GUARD = 5.0


class Answering:
    """A provider that answers with a different body every time it is called."""

    def __init__(
        self, *, fails: Exception | None = None, held: asyncio.Event | None = None
    ) -> None:
        self.calls: list[ModelRequest] = []
        self.fails = fails
        self.held = held

    @property
    def name(self) -> str:
        return "answering"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(streaming=True, tool_calling=True)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if self.held is not None:
            await self.held.wait()
        if self.fails is not None:
            raise self.fails
        return ModelResponse(
            content=f"answer {len(self.calls)}",
            usage=Usage(input_tokens=100, output_tokens=20),
        )

    async def stream(self, request: ModelRequest) -> Sequence[StreamEvent]:
        raise NotImplementedError

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return sum(len(str(message.content)) for message in messages)


class Breaking:
    """A cache store where every operation fails, as an unreachable one does."""

    def __init__(self) -> None:
        self.attempts = 0

    async def get(self, digest: str) -> None:
        del digest
        self.attempts += 1
        raise ConnectionError("cache store unreachable")

    async def put(self, entry: object) -> None:
        del entry
        self.attempts += 1
        raise ConnectionError("cache store unreachable")

    async def purge(self, **by: str | None) -> int:
        del by
        raise ConnectionError("cache store unreachable")


class Digests:
    """An embedding provider whose vectors are chosen by the test, not derived."""

    def __init__(self, vectors: Mapping[str, Vector]) -> None:
        self.vectors = vectors
        self.asked: list[str] = []

    @property
    def name(self) -> str:
        return "digests"

    def limits(self, model: str) -> object:
        del model
        raise NotImplementedError

    async def embed(self, texts: Sequence[str], *, model: str) -> Sequence[Vector]:
        del model
        self.asked.extend(texts)
        return [self.vectors[text] for text in texts]


def asked(text: str = "did it rain", *, tools: tuple[ToolDeclaration, ...] = ()) -> ModelRequest:
    """One request, varied by whatever the test is varying."""
    return ModelRequest(
        model=MODEL, messages=(Message(role="user", content=[TextPart(text=text)]),), tools=tools
    )


def caching(
    inner: Answering,
    store: object,
    *,
    tenant: str = "acme",
    clock: FakeClock | None = None,
    **overrides: object,
) -> CachingProvider:
    """A caching provider over `inner`, with the determinants a test wants to vary."""
    return CachingProvider(
        inner,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        tenant=tenant,
        clock=clock or FakeClock(auto_advance=False),
        **overrides,  # type: ignore[arg-type]
    )


class TestExactMatch:
    """The same call twice costs one call, and the second one says so."""

    async def test_a_repeated_call_is_served_from_the_cache(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        first = await provider.complete(asked())
        second = await provider.complete(asked())

        assert len(inner.calls) == 1
        assert second == first

    async def test_the_outcome_names_the_status_and_the_saving(self) -> None:
        seen: list[object] = []
        inner = Answering()
        provider = caching(inner, MemoryCacheStore(), observer=seen.append)

        await provider.complete(asked())
        await provider.complete(asked())

        statuses = [outcome.status for outcome in seen]  # type: ignore[attr-defined]
        assert statuses == [CacheStatus.MISS, CacheStatus.HIT]
        assert seen[1].saved == Usage(input_tokens=100, output_tokens=20)  # type: ignore[attr-defined]
        assert seen[0].saved == Usage(input_tokens=0, output_tokens=0)  # type: ignore[attr-defined]

    async def test_a_live_result_is_never_reported_as_cached(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        await provider.complete(asked())

        assert provider.metrics.hits == 0
        assert provider.metrics.misses == 1


class TestDeterminants:
    """Every part of the key is a part of the key, proven one change at a time."""

    async def test_a_different_prompt_misses(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        await provider.complete(asked("did it rain"))
        await provider.complete(asked("did it snow"))

        assert len(inner.calls) == 2

    async def test_a_different_model_misses(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        await provider.complete(asked())
        await provider.complete(asked().model_copy(update={"model": "gpt-4o-mini"}))

        assert len(inner.calls) == 2

    async def test_a_changed_tool_schema_misses(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())
        before = ToolDeclaration(name="weather", parameters={"type": "object"})
        after = ToolDeclaration(name="weather", parameters={"type": "object", "required": ["city"]})

        await provider.complete(asked(tools=(before,)))
        await provider.complete(asked(tools=(after,)))

        assert len(inner.calls) == 2

    async def test_a_changed_output_schema_misses(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())
        request = asked()

        await provider.complete(request.model_copy(update={"output_schema_hash": "sha256:aa"}))
        await provider.complete(request.model_copy(update={"output_schema_hash": "sha256:bb"}))

        assert len(inner.calls) == 2

    async def test_changed_parameters_miss(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()

        await caching(inner, store, parameters={"top_p": 1.0}).complete(asked())
        await caching(inner, store, parameters={"top_p": 0.5}).complete(asked())

        assert len(inner.calls) == 2

    async def test_a_new_prompt_version_misses(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()

        await caching(inner, store, prompt_version="v1").complete(asked())
        await caching(inner, store, prompt_version="v2").complete(asked())

        assert len(inner.calls) == 2

    async def test_a_new_model_version_misses(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()

        await caching(inner, store, model_version="2026-05-01").complete(asked())
        await caching(inner, store, model_version="2026-06-01").complete(asked())

        assert len(inner.calls) == 2


class TestTenantIsolation:
    """The one hit that must be unreachable, however identical the call."""

    async def test_a_second_tenant_never_reads_the_first_ones_answer(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()
        theirs = caching(inner, store, tenant="acme")
        ours = caching(inner, store, tenant="globex")

        first = await theirs.complete(asked())
        second = await ours.complete(asked())

        assert len(inner.calls) == 2
        assert second != first

    async def test_the_tenant_is_in_the_key_not_only_in_the_wrapper(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()
        await caching(inner, store, tenant="acme").complete(asked())

        held = await store.every()

        assert [entry.key.tenant for entry in held] == ["acme"]

    async def test_no_two_tenants_share_a_digest(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()
        await caching(inner, store, tenant="acme").complete(asked())
        await caching(inner, store, tenant="globex").complete(asked())

        held = await store.every()

        assert len({entry.key.digest for entry in held}) == 2


class TestCacheability:
    """What the rules say may not be cached at all, refused rather than cached anyway."""

    async def test_non_deterministic_sampling_is_refused(self) -> None:
        inner = Answering()
        seen: list[object] = []
        provider = caching(
            inner, MemoryCacheStore(), parameters={"temperature": 0.7}, observer=seen.append
        )

        await provider.complete(asked())
        await provider.complete(asked())

        assert len(inner.calls) == 2
        assert [outcome.status for outcome in seen] == [  # type: ignore[attr-defined]
            CacheStatus.REFUSED,
            CacheStatus.REFUSED,
        ]

    async def test_the_refusal_says_why(self) -> None:
        seen: list[object] = []
        provider = caching(
            Answering(), MemoryCacheStore(), parameters={"temperature": 0.7}, observer=seen.append
        )

        await provider.complete(asked())

        assert "temperature" in seen[0].reason  # type: ignore[attr-defined]

    async def test_zero_temperature_is_cacheable(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore(), parameters={"temperature": 0.0})

        await provider.complete(asked())
        await provider.complete(asked())

        assert len(inner.calls) == 1

    async def test_a_marked_call_is_not_cached(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        with not_cacheable("read personalised memory"):
            await provider.complete(asked())
        await provider.complete(asked())

        assert len(inner.calls) == 2

    async def test_the_mark_does_not_outlive_its_block(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        with not_cacheable("side-effecting tool result"):
            await provider.complete(asked("first"))
        await provider.complete(asked("second"))
        await provider.complete(asked("second"))

        assert len(inner.calls) == 2

    async def test_more_than_one_draw_is_refused(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore(), parameters={"n": 3})

        await provider.complete(asked())
        await provider.complete(asked())

        assert len(inner.calls) == 2

    async def test_an_operator_may_turn_the_determinism_rule_off(self) -> None:
        inner = Answering()
        provider = caching(
            inner,
            MemoryCacheStore(),
            parameters={"temperature": 0.7},
            policy=CachePolicy(deterministic_only=False),
        )

        await provider.complete(asked())
        await provider.complete(asked())

        assert len(inner.calls) == 1

    async def test_a_refused_call_is_never_written_to_the_store(self) -> None:
        store = MemoryCacheStore()
        provider = caching(Answering(), store, parameters={"temperature": 0.9})

        await provider.complete(asked())

        assert await store.every() == []


class TestExpiry:
    """A cached answer has a lifetime, and an expired one is not an answer."""

    async def test_an_entry_past_its_ttl_is_not_served(self) -> None:
        clock = FakeClock(auto_advance=False)
        inner = Answering()
        provider = caching(
            inner, MemoryCacheStore(), clock=clock, policy=CachePolicy(ttl_seconds=60)
        )

        await provider.complete(asked())
        clock.set(61)
        await provider.complete(asked())

        assert len(inner.calls) == 2

    async def test_an_entry_inside_its_ttl_is_served(self) -> None:
        clock = FakeClock(auto_advance=False)
        inner = Answering()
        provider = caching(
            inner, MemoryCacheStore(), clock=clock, policy=CachePolicy(ttl_seconds=60)
        )

        await provider.complete(asked())
        clock.set(59)
        await provider.complete(asked())

        assert len(inner.calls) == 1

    async def test_an_expired_entry_is_dropped_rather_than_kept(self) -> None:
        clock = FakeClock(auto_advance=False)
        store = MemoryCacheStore()
        provider = caching(Answering(), store, clock=clock, policy=CachePolicy(ttl_seconds=60))

        await provider.complete(asked())
        clock.set(61)
        await provider.complete(asked())

        assert len(await store.every()) == 1


class TestInvalidation:
    """Retiring a prompt or a model must remove what it produced, not only stop matching."""

    async def test_forgetting_a_prompt_version_removes_its_entries(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()
        old = caching(inner, store, prompt_version="v1")
        await old.complete(asked())

        removed = await old.forget(prompt_version="v1")

        assert removed == 1
        assert await store.every() == []

    async def test_forgetting_a_tenant_removes_only_that_tenant(self) -> None:
        store = MemoryCacheStore()
        inner = Answering()
        theirs = caching(inner, store, tenant="acme")
        ours = caching(inner, store, tenant="globex")
        await theirs.complete(asked())
        await ours.complete(asked())

        removed = await theirs.forget()

        assert removed == 1
        assert [entry.key.tenant for entry in await store.every()] == ["globex"]

    async def test_forgetting_a_model_version_removes_its_entries(self) -> None:
        store = MemoryCacheStore()
        provider = caching(Answering(), store, model_version="2026-05-01")
        await provider.complete(asked())

        removed = await provider.forget(model_version="2026-05-01")

        assert removed == 1


class TestStoreOutage:
    """An unreachable cache is a slow run, never a failed one."""

    async def test_a_failed_lookup_degrades_to_a_live_call(self) -> None:
        inner = Answering()
        provider = caching(inner, Breaking())

        answer = await provider.complete(asked())

        assert answer.content == "answer 1"
        assert provider.metrics.store_failures >= 1

    async def test_a_failed_write_does_not_fail_the_run(self) -> None:
        provider = caching(Answering(), Breaking())

        answer = await provider.complete(asked())

        assert answer.content == "answer 1"

    async def test_the_outcome_reports_the_outage(self) -> None:
        seen: list[object] = []
        provider = caching(Answering(), Breaking(), observer=seen.append)

        await provider.complete(asked())

        assert seen[0].status is CacheStatus.STORE_UNAVAILABLE  # type: ignore[attr-defined]

    async def test_a_provider_failure_is_still_raised(self) -> None:
        inner = Answering(fails=ProviderUnavailableError("upstream down"))
        provider = caching(inner, MemoryCacheStore())

        with pytest.raises(ProviderUnavailableError):
            await provider.complete(asked())

    async def test_a_failed_call_is_not_cached(self) -> None:
        store = MemoryCacheStore()
        inner = Answering(fails=ProviderUnavailableError("upstream down"))
        provider = caching(inner, store)

        with pytest.raises(ProviderUnavailableError):
            await provider.complete(asked())

        assert await store.every() == []


class TestStampede:
    """A cold key under concurrent load is one call, not one per caller."""

    async def test_concurrent_identical_calls_are_single_flighted(self) -> None:
        held = asyncio.Event()
        inner = Answering(held=held)
        provider = caching(inner, MemoryCacheStore())

        waiting = [asyncio.ensure_future(provider.complete(asked())) for _ in range(10)]
        await asyncio.sleep(0)
        held.set()
        answers = await asyncio.wait_for(asyncio.gather(*waiting), GUARD)

        assert len(inner.calls) == 1
        assert {answer.content for answer in answers} == {"answer 1"}
        assert provider.metrics.coalesced == 9

    async def test_concurrent_calls_on_different_keys_are_not_serialised(self) -> None:
        held = asyncio.Event()
        inner = Answering(held=held)
        provider = caching(inner, MemoryCacheStore())

        waiting = [
            asyncio.ensure_future(provider.complete(asked(f"question {index}")))
            for index in range(4)
        ]
        await asyncio.sleep(0)
        held.set()
        await asyncio.wait_for(asyncio.gather(*waiting), GUARD)

        assert len(inner.calls) == 4

    async def test_a_failure_does_not_leave_the_key_wedged(self) -> None:
        inner = Answering(fails=ProviderUnavailableError("upstream down"))
        provider = caching(inner, MemoryCacheStore())

        with pytest.raises(ProviderUnavailableError):
            await provider.complete(asked())
        inner.fails = None
        answer = await provider.complete(asked())

        assert answer.content == "answer 2"


class TestSemanticTier:
    """Approximate matching, off unless asked for and never looser than its threshold."""

    def semantic(self, vectors: Mapping[str, Vector], threshold: float = 0.95) -> SemanticConfig:
        """A semantic tier over vectors the test chose."""
        return SemanticConfig(
            embedder=Digests(vectors),  # type: ignore[arg-type]
            index=MemorySemanticIndex(),
            model="bge-m3",
            threshold=threshold,
        )

    async def test_it_is_off_by_default(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        await provider.complete(asked("did it rain"))
        await provider.complete(asked("was there rain"))

        assert len(inner.calls) == 2

    async def test_a_near_match_above_the_threshold_is_served(self) -> None:
        inner = Answering()
        vectors = {"did it rain": (1.0, 0.0), "was there rain": (0.999, 0.045)}
        provider = caching(inner, MemoryCacheStore(), semantic=self.semantic(vectors))

        first = await provider.complete(asked("did it rain"))
        second = await provider.complete(asked("was there rain"))

        assert len(inner.calls) == 1
        assert second == first

    async def test_a_near_match_below_the_threshold_is_not_served(self) -> None:
        inner = Answering()
        vectors = {"did it rain": (1.0, 0.0), "did it snow": (0.0, 1.0)}
        provider = caching(inner, MemoryCacheStore(), semantic=self.semantic(vectors))

        await provider.complete(asked("did it rain"))
        await provider.complete(asked("did it snow"))

        assert len(inner.calls) == 2

    async def test_the_entry_records_the_threshold_and_the_embedding_model(self) -> None:
        store = MemoryCacheStore()
        vectors = {"did it rain": (1.0, 0.0)}
        provider = caching(Answering(), store, semantic=self.semantic(vectors))
        await provider.complete(asked("did it rain"))

        entry = (await store.every())[0]

        assert (entry.embedding_model, entry.threshold) == ("bge-m3", 0.95)

    async def test_an_upgraded_embedding_model_never_matches_older_entries(self) -> None:
        inner = Answering()
        store = MemoryCacheStore()
        index = MemorySemanticIndex()
        vectors = {"did it rain": (1.0, 0.0), "was there rain": (0.999, 0.045)}
        before = SemanticConfig(
            embedder=Digests(vectors),  # type: ignore[arg-type]
            index=index,
            model="bge-m3",
            threshold=0.95,
        )
        after = SemanticConfig(
            embedder=Digests(vectors),  # type: ignore[arg-type]
            index=index,
            model="bge-m3-v2",
            threshold=0.95,
        )

        await caching(inner, store, semantic=before).complete(asked("did it rain"))
        await caching(inner, store, semantic=after).complete(asked("was there rain"))

        assert len(inner.calls) == 2

    async def test_a_semantic_hit_is_reported_as_one(self) -> None:
        seen: list[object] = []
        vectors = {"did it rain": (1.0, 0.0), "was there rain": (0.999, 0.045)}
        provider = caching(
            Answering(), MemoryCacheStore(), semantic=self.semantic(vectors), observer=seen.append
        )

        await provider.complete(asked("did it rain"))
        await provider.complete(asked("was there rain"))

        assert seen[1].status is CacheStatus.SEMANTIC_HIT  # type: ignore[attr-defined]
        assert seen[1].similarity == pytest.approx(0.999, abs=0.002)  # type: ignore[attr-defined]

    async def test_a_semantic_near_match_from_another_tenant_is_unreachable(self) -> None:
        inner = Answering()
        store = MemoryCacheStore()
        index = MemorySemanticIndex()
        vectors = {"did it rain": (1.0, 0.0), "was there rain": (0.999, 0.045)}

        def tier() -> SemanticConfig:
            return SemanticConfig(
                embedder=Digests(vectors),  # type: ignore[arg-type]
                index=index,
                model="bge-m3",
                threshold=0.95,
            )

        await caching(inner, store, tenant="acme", semantic=tier()).complete(asked("did it rain"))
        await caching(inner, store, tenant="globex", semantic=tier()).complete(
            asked("was there rain")
        )

        assert len(inner.calls) == 2

    async def test_forgetting_a_tenant_removes_its_vectors_too(self) -> None:
        inner = Answering()
        store = MemoryCacheStore()
        vectors = {"did it rain": (1.0, 0.0), "was there rain": (0.999, 0.045)}
        tier = self.semantic(vectors)
        provider = caching(inner, store, semantic=tier)
        await provider.complete(asked("did it rain"))

        await provider.forget()
        await provider.complete(asked("was there rain"))

        assert len(inner.calls) == 2

    async def test_a_prompt_with_no_messages_is_embedded_as_nothing(self) -> None:
        inner = Answering()
        vectors = {"": (0.0, 0.0)}
        provider = caching(inner, MemoryCacheStore(), semantic=self.semantic(vectors))

        await provider.complete(asked().model_copy(update={"messages": ()}))

        assert len(inner.calls) == 1

    async def test_vectors_of_different_widths_never_match(self) -> None:
        inner = Answering()
        store = MemoryCacheStore()
        index = MemorySemanticIndex()
        wide = SemanticConfig(
            embedder=Digests({"did it rain": (1.0, 0.0, 0.0)}),  # type: ignore[arg-type]
            index=index,
            model="bge-m3",
            threshold=0.95,
        )
        narrow = SemanticConfig(
            embedder=Digests({"was there rain": (1.0, 0.0)}),  # type: ignore[arg-type]
            index=index,
            model="bge-m3",
            threshold=0.95,
        )

        await caching(inner, store, semantic=wide).complete(asked("did it rain"))
        await caching(inner, store, semantic=narrow).complete(asked("was there rain"))

        assert len(inner.calls) == 2

    async def test_a_semantic_entry_expired_is_not_served(self) -> None:
        clock = FakeClock(auto_advance=False)
        inner = Answering()
        vectors = {"did it rain": (1.0, 0.0), "was there rain": (0.999, 0.045)}
        provider = caching(
            inner,
            MemoryCacheStore(),
            clock=clock,
            semantic=self.semantic(vectors),
            policy=CachePolicy(ttl_seconds=60),
        )

        await provider.complete(asked("did it rain"))
        clock.set(61)
        await provider.complete(asked("was there rain"))

        assert len(inner.calls) == 2


class TestMetrics:
    """What an operator reads to decide whether the cache is earning its risk."""

    async def test_the_counters_describe_the_work(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        await provider.complete(asked("a"))
        await provider.complete(asked("a"))
        await provider.complete(asked("b"))

        metrics = provider.metrics
        assert (metrics.hits, metrics.misses, metrics.stores) == (1, 2, 2)

    async def test_the_saving_totals_what_was_not_spent(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        await provider.complete(asked())
        await provider.complete(asked())

        assert provider.metrics.saved == Usage(input_tokens=100, output_tokens=20)

    async def test_refusals_are_counted_apart_from_misses(self) -> None:
        provider = caching(Answering(), MemoryCacheStore(), parameters={"temperature": 1.0})

        await provider.complete(asked())

        assert (provider.metrics.refusals, provider.metrics.misses) == (1, 0)


class TestPassThrough:
    """A caching provider is still the provider it wraps."""

    def test_it_reports_the_wrapped_provider_and_capabilities(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())

        assert provider.name == "answering"
        assert provider.capabilities == inner.capabilities

    def test_it_counts_tokens_the_way_the_wrapped_provider_does(self) -> None:
        inner = Answering()
        provider = caching(inner, MemoryCacheStore())
        messages = (Message(role="user", content=[TextPart(text="hello")]),)

        assert provider.count_tokens(messages) == inner.count_tokens(messages)

    async def test_streaming_is_not_cached(self) -> None:
        provider = caching(Answering(), MemoryCacheStore())

        with pytest.raises(NotImplementedError):
            await provider.stream(asked())


class TestRefusedConfigurations:
    """A cache shape that would behave surprisingly is refused at construction."""

    def test_a_negative_ttl_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ttl"):
            CachePolicy(ttl_seconds=-1)

    def test_a_threshold_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            SemanticConfig(
                embedder=Digests({}),  # type: ignore[arg-type]
                index=MemorySemanticIndex(),
                model="bge-m3",
                threshold=1.5,
            )

    def test_an_empty_tenant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tenant"):
            caching(Answering(), MemoryCacheStore(), tenant="")
