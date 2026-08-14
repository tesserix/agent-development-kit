"""Finding the booking reference and the paraphrase in one call, and saying which found it.

A retrieval that returns the paraphrase and misses the policy code is not half right: the
agent answers confidently from what it got. So both branches run, every hit says which
branch found it and where it ranked, and a branch that did not answer is either flagged or
refused — never quietly absent from a result that reads complete.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    ConfigurationError,
    MissingTenantContextError,
    RetrievalDegradedError,
    SchemaViolationError,
    tenant_scope,
)
from tesserix_adk.rag import (
    Branch,
    EmbeddedBatch,
    EmbeddingModel,
    HybridRetriever,
    IndexQuery,
    IndexRetriever,
    ReciprocalRankFusion,
    RetrievalScope,
    Retriever,
    WeightedSum,
)
from tesserix_adk.testing import FakeIndex, Indexed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.rag import Hit, Vector

pytestmark = pytest.mark.anyio

HANDBOOK = RetrievalScope(collection="handbook")

MEANINGS: dict[str, tuple[float, ...]] = {
    "how do I get my money back for a trip": (1.0, 0.0, 0.0),
    "refunds": (0.9, 0.1, 0.0),
    "sleeper berths": (0.0, 1.0, 0.0),
    "booking BX-7741": (0.0, 0.0, 1.0),
}


class Toy:
    """An embedder over a fixed vocabulary, so a paraphrase is genuinely near its answer."""

    def __init__(self, *, refuses: bool = False) -> None:
        self._refuses = refuses

    @property
    def model(self) -> EmbeddingModel:
        """What it claims to be."""
        return EmbeddingModel(name="toy", dimension=3)

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddedBatch:
        """Embed every text by looking it up."""
        return EmbeddedBatch(vectors=tuple(self._vector(text) for text in texts))

    async def embed_query(self, text: str) -> Vector:
        """Embed one query, or refuse where this embedder was built to.

        Raises:
            SchemaViolationError: Where it was built to refuse, as a real one does for a
                query past its maximum input.
        """
        if self._refuses:
            raise SchemaViolationError("the query is longer than this model accepts")
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        return MEANINGS.get(text, (0.0, 0.0, 0.0))


REFUNDS = Indexed(
    "refunds",
    "Refunds are paid within thirty days of the claim.",
    vector=MEANINGS["refunds"],
    document_id="handbook-2026",
    metadata={"section": "claims"},
)
BERTHS = Indexed(
    "berths",
    "A berth must be booked ahead, four to a compartment.",
    vector=MEANINGS["sleeper berths"],
    document_id="handbook-2026",
    metadata={"section": "travel"},
)
BOOKING = Indexed(
    "booking",
    "Booking BX-7741 was issued against the Lisbon trip.",
    vector=MEANINGS["booking BX-7741"],
    document_id="bookings",
    metadata={"section": "bookings"},
)
OTHER_TENANT = Indexed(
    "globex-refunds",
    "Refunds at Globex are paid on the first of the month.",
    vector=MEANINGS["refunds"],
    tenant="globex",
)


def index(*passages: Indexed, **overrides: object) -> FakeIndex:
    """A store holding the handbook, plus whatever a test adds."""
    return FakeIndex(REFUNDS, BERTHS, BOOKING, *passages, **overrides)  # type: ignore[arg-type]


def hybrid(store: FakeIndex | None = None, **overrides: object) -> HybridRetriever:
    """Both branches over one store, which is the common deployment."""
    held = store or index()
    return HybridRetriever(
        IndexRetriever(held, branch=Branch.SEMANTIC, embedder=Toy()),
        IndexRetriever(held, branch=Branch.KEYWORD),
        **overrides,  # type: ignore[arg-type]
    )


def ids(hits: Sequence[Hit]) -> list[str]:
    """The chunk ids, in the order they came back."""
    return [hit.chunk_id for hit in hits]


class TestFindingBothKindsOfMatch:
    async def test_an_identifier_is_found_that_no_embedding_would_place(self) -> None:
        with tenant_scope("acme"):
            found = await hybrid().retrieve("BX-7741", scope=HANDBOOK)

        assert ids(found.hits) == ["booking"]
        assert found.hits[0].found_by(Branch.KEYWORD)

    async def test_a_paraphrase_is_found_that_shares_no_words(self) -> None:
        with tenant_scope("acme"):
            found = await hybrid().retrieve("how do I get my money back for a trip", scope=HANDBOOK)

        refunds = next(hit for hit in found.hits if hit.chunk_id == "refunds")
        assert refunds.found_by(Branch.SEMANTIC)
        assert not refunds.found_by(Branch.KEYWORD)

    async def test_every_hit_says_which_branch_found_it_and_at_what_rank(self) -> None:
        """Without the breakdown, a bad ranking cannot be attributed to a branch."""
        with tenant_scope("acme"):
            found = await hybrid().retrieve("refunds", scope=HANDBOOK)

        contributions = found.hits[0].contributions
        assert {contribution.branch for contribution in contributions} == {
            Branch.SEMANTIC,
            Branch.KEYWORD,
        }
        assert all(contribution.rank >= 1 for contribution in contributions)

    async def test_a_chunk_both_branches_found_is_returned_once(self) -> None:
        with tenant_scope("acme"):
            found = await hybrid().retrieve("refunds", scope=HANDBOOK)

        assert ids(found.hits).count("refunds") == 1
        assert len(found.hits[0].contributions) == 2

    async def test_both_retrievers_are_retrievers(self) -> None:
        assert isinstance(hybrid(), Retriever)
        assert isinstance(IndexRetriever(index(), branch=Branch.KEYWORD), Retriever)


class TestHowTheBranchesAreCombined:
    async def test_a_chunk_both_branches_ranked_beats_one_that_only_one_did(self) -> None:
        with tenant_scope("acme"):
            found = await hybrid().retrieve("refunds", scope=HANDBOOK)

        assert ids(found.hits)[0] == "refunds"

    async def test_ranks_are_fused_rather_than_scores_by_default(self) -> None:
        """Cosine 0.82 and a word overlap of 0.5 have no exchange rate to add them at."""
        with tenant_scope("acme"):
            found = await hybrid().retrieve("refunds", scope=HANDBOOK)

        assert found.hits[0].score == pytest.approx(2 / 61)

    async def test_a_deployment_can_weigh_the_scores_instead(self) -> None:
        weighted = WeightedSum({Branch.SEMANTIC: 1.0, Branch.KEYWORD: 0.0})

        with tenant_scope("acme"):
            found = await hybrid(fusion=weighted).retrieve("refunds", scope=HANDBOOK)

        assert found.hits[0].score == pytest.approx(1.0, abs=0.02)

    async def test_a_rank_offset_below_one_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            ReciprocalRankFusion(k0=0)

    async def test_weights_that_say_nothing_are_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            WeightedSum({})

    async def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            WeightedSum({Branch.SEMANTIC: -1.0})


class TestTheTenantPredicateIsNotTheCallers:
    async def test_the_tenant_is_put_inside_the_store_query(self) -> None:
        """Filtering in Python is filtering after the rows have left the tenant."""
        store = index(OTHER_TENANT)

        with tenant_scope("acme"):
            found = await hybrid(store).retrieve("refunds", scope=HANDBOOK)

        assert {query.tenant for query in store.queries} == {"acme"}
        assert "globex-refunds" not in ids(found.hits)

    async def test_a_filter_may_not_name_the_tenant(self) -> None:
        """Merging it would let a caller widen the scope by naming every tenant it knows."""
        with tenant_scope("acme"), pytest.raises(SchemaViolationError, match="tenant"):
            await hybrid().retrieve("refunds", scope=HANDBOOK, filters={"tenant": ["globex"]})

    async def test_retrieving_for_nobody_in_particular_is_refused(self) -> None:
        with pytest.raises(MissingTenantContextError):
            await hybrid().retrieve("refunds", scope=HANDBOOK)

    async def test_a_caller_filter_is_pushed_down_rather_than_applied_afterwards(self) -> None:
        store = index()

        with tenant_scope("acme"):
            found = await hybrid(store).retrieve(
                "refunds", scope=HANDBOOK, filters={"section": ["travel"]}
            )

        assert all(query.predicates == {"section": ("travel",)} for query in store.queries)
        assert "refunds" not in ids(found.hits)

    async def test_a_filter_with_no_values_constrains_nothing(self) -> None:
        store = index()

        with tenant_scope("acme"):
            await hybrid(store).retrieve("refunds", scope=HANDBOOK, filters={"section": []})

        assert all(query.predicates == {} for query in store.queries)


class TestWhenABranchDoesNotAnswer:
    async def test_a_branch_that_times_out_is_flagged_rather_than_hidden(self) -> None:
        retriever = HybridRetriever(
            IndexRetriever(index(), branch=Branch.SEMANTIC, embedder=Toy()),
            IndexRetriever(FakeIndex(hangs=True), branch=Branch.KEYWORD),
            timeout_seconds=0.01,
        )

        with tenant_scope("acme"):
            found = await retriever.retrieve("refunds", scope=HANDBOOK)

        assert found.partial
        assert found.branches == (Branch.SEMANTIC,)
        assert ids(found.hits)[0] == "refunds"

    async def test_a_required_branch_that_times_out_is_a_refusal(self) -> None:
        """A narrower result set reads downstream as a complete answer."""
        retriever = HybridRetriever(
            IndexRetriever(index(), branch=Branch.SEMANTIC, embedder=Toy()),
            IndexRetriever(FakeIndex(hangs=True), branch=Branch.KEYWORD),
            require=(Branch.KEYWORD,),
            timeout_seconds=0.01,
        )

        with tenant_scope("acme"), pytest.raises(RetrievalDegradedError) as raised:
            await retriever.retrieve("refunds", scope=HANDBOOK)

        assert raised.value.missing == ("keyword",)
        assert raised.value.answered == ("semantic",)

    async def test_a_branch_whose_store_is_down_is_the_same_thing(self) -> None:
        retriever = HybridRetriever(
            IndexRetriever(index(), branch=Branch.SEMANTIC, embedder=Toy()),
            IndexRetriever(FakeIndex(fails=True), branch=Branch.KEYWORD),
        )

        with tenant_scope("acme"):
            found = await retriever.retrieve("refunds", scope=HANDBOOK)

        assert found.partial

    async def test_no_branch_answering_is_never_an_empty_result(self) -> None:
        """An empty result set and a total outage must not look the same to an agent."""
        retriever = HybridRetriever(
            IndexRetriever(FakeIndex(fails=True), branch=Branch.KEYWORD),
        )

        with tenant_scope("acme"), pytest.raises(RetrievalDegradedError):
            await retriever.retrieve("refunds", scope=HANDBOOK)

    async def test_a_query_the_embedder_will_not_take_degrades_the_semantic_branch(
        self,
    ) -> None:
        store = index()
        retriever = HybridRetriever(
            IndexRetriever(store, branch=Branch.SEMANTIC, embedder=Toy(refuses=True)),
            IndexRetriever(store, branch=Branch.KEYWORD),
        )

        with tenant_scope("acme"):
            found = await retriever.retrieve("refunds " * 4000, scope=HANDBOOK)

        assert found.partial
        assert found.branches == (Branch.KEYWORD,)

    async def test_cancelling_a_retrieval_cancels_it(self) -> None:
        """A cancellation is not a degraded branch and must not be reported as one."""
        retriever = HybridRetriever(IndexRetriever(Cancelling(), branch=Branch.KEYWORD))

        with tenant_scope("acme"), pytest.raises(asyncio.CancelledError):
            await retriever.retrieve("refunds", scope=HANDBOOK)

    async def test_a_branch_names_the_store_behind_it(self) -> None:
        """Which store degraded is the first question asked of a partial result."""
        assert IndexRetriever(index(name="pgvector"), branch=Branch.KEYWORD).name == "pgvector"


class Cancelling:
    """A store whose search is cancelled, as one is when the caller gives up."""

    @property
    def name(self) -> str:
        """What this store is."""
        return "cancelling"

    def supports(self, _branch: Branch) -> bool:
        """It answers anything, briefly."""
        return True

    async def search(self, query: IndexQuery) -> Sequence[Hit]:
        """Raise the cancellation the caller's task would raise.

        Raises:
            asyncio.CancelledError: Always.
        """
        raise asyncio.CancelledError(query.branch)


class TestWhatComesBack:
    async def test_asking_for_more_than_there_is_returns_what_there_is(self) -> None:
        with tenant_scope("acme"):
            found = await hybrid().retrieve("refunds", scope=HANDBOOK, k=500)

        assert len(found.hits) <= 3

    async def test_k_is_honoured_after_fusion_not_before(self) -> None:
        with tenant_scope("acme"):
            found = await hybrid().retrieve("a berth must be booked", scope=HANDBOOK, k=1)

        assert len(found.hits) == 1

    async def test_nothing_found_is_an_explicit_empty_result(self) -> None:
        """So the agent can say it does not know rather than improvise from what it got."""
        with tenant_scope("acme"):
            found = await hybrid().retrieve("quarterly dividend policy", scope=HANDBOOK)

        assert found.hits == ()
        assert not found
        assert not found.partial

    async def test_an_empty_query_is_refused_rather_than_ranked_at_random(self) -> None:
        with tenant_scope("acme"), pytest.raises(SchemaViolationError):
            await hybrid().retrieve("   ", scope=HANDBOOK)

    async def test_asking_for_no_hits_at_all_is_refused(self) -> None:
        with tenant_scope("acme"), pytest.raises(ConfigurationError):
            await hybrid().retrieve("refunds", scope=HANDBOOK, k=0)

    async def test_one_branch_on_its_own_still_reports_its_branch(self) -> None:
        with tenant_scope("acme"):
            found = await IndexRetriever(index(), branch=Branch.KEYWORD).retrieve(
                "refunds", scope=HANDBOOK
            )

        assert found.branches == (Branch.KEYWORD,)
        assert found.hits[0].contributions[0].branch is Branch.KEYWORD


class TestRetrieversThatCannotWork:
    async def test_a_semantic_branch_with_no_embedder_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="embedder"):
            IndexRetriever(index(), branch=Branch.SEMANTIC)

    async def test_a_store_that_does_not_answer_the_branch_is_refused(self) -> None:
        keyword_only = FakeIndex(branches=(Branch.KEYWORD,))

        with pytest.raises(ConfigurationError, match="semantic"):
            IndexRetriever(keyword_only, branch=Branch.SEMANTIC, embedder=Toy())

    async def test_a_hybrid_with_no_branches_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            HybridRetriever()

    async def test_two_retrievers_for_one_branch_are_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            HybridRetriever(
                IndexRetriever(index(), branch=Branch.KEYWORD),
                IndexRetriever(index(), branch=Branch.KEYWORD),
            )

    async def test_requiring_a_branch_that_is_not_there_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="required"):
            HybridRetriever(
                IndexRetriever(index(), branch=Branch.KEYWORD), require=(Branch.SEMANTIC,)
            )


class TestTheQueryAStoreIsGiven:
    async def test_a_semantic_query_carries_a_vector_and_a_keyword_query_carries_text(
        self,
    ) -> None:
        store = index()

        with tenant_scope("acme"):
            await hybrid(store).retrieve("refunds", scope=HANDBOOK)

        by_branch = {query.branch: query for query in store.queries}
        assert by_branch[Branch.SEMANTIC].vector is not None
        assert by_branch[Branch.KEYWORD].text == "refunds"

    async def test_a_semantic_query_without_a_vector_is_not_a_query(self) -> None:
        """It would degrade to whatever the store does with no vector, which is a scan."""
        with pytest.raises(ValueError, match="vector"):
            IndexQuery(tenant="acme", collection="handbook", branch=Branch.SEMANTIC, k=5)

    async def test_a_keyword_query_without_text_is_not_a_query(self) -> None:
        with pytest.raises(ValueError, match="text"):
            IndexQuery(tenant="acme", collection="handbook", branch=Branch.KEYWORD, k=5)


class TestTheFakeIndex:
    async def test_it_holds_what_it_is_given_afterwards(self) -> None:
        store = FakeIndex(branches=(Branch.KEYWORD,))
        store.add(REFUNDS)

        with tenant_scope("acme"):
            found = await IndexRetriever(store, branch=Branch.KEYWORD).retrieve(
                "refunds", scope=HANDBOOK
            )

        assert ids(found.hits) == ["refunds"]

    async def test_it_scores_a_vector_of_the_wrong_width_as_no_match(self) -> None:
        store = FakeIndex(Indexed("odd", "text", vector=(1.0, 0.0)))

        with tenant_scope("acme"):
            found = await IndexRetriever(store, embedder=Toy()).retrieve("refunds", scope=HANDBOOK)

        assert found.hits == ()

    async def test_the_predicates_it_was_asked_are_the_ones_it_applied(self) -> None:
        store = index()
        asked: Mapping[str, Sequence[str]] = {"section": ["claims"]}

        with tenant_scope("acme"):
            found = await IndexRetriever(store, branch=Branch.KEYWORD).retrieve(
                "refunds", scope=HANDBOOK, filters=asked
            )

        assert ids(found.hits) == ["refunds"]
