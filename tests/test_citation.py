"""Citations that resolve, and answers that fail closed when they do not.

The point of every test here is that an answer nobody can check reads exactly like an
answer that has been checked. So a citation pins a version and a span, a claim rests on
citation ids rather than footnote-shaped text, and the grounding check refuses rather than
tidying the offending citation away.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from tesserix_adk.core import (
    ConfigurationError,
    MissingTenantContextError,
    TenantCrossingError,
    UncitedClaimError,
    UngroundedCitationError,
    tenant_scope,
)
from tesserix_adk.memory import MemoryKind, MemoryRecord, MemoryScope
from tesserix_adk.rag import (
    Branch,
    BranchScore,
    Citation,
    CitationResolver,
    CitedAnswer,
    Claim,
    Document,
    Hit,
    ResolvedCitation,
    RetrievalResult,
    SourceLocator,
    Span,
    check_grounding,
    citation_attributes,
    cite,
    excerpt,
)

WHEN = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)

POLICY = Document(
    id="handbook",
    text="Refunds are the subject of this page. A refund is paid within fourteen days.",
    metadata={"version": "v3"},
)


def hit(chunk_id: str = "timing", **overrides: Any) -> Hit:
    """A hit as a well-behaved store returns it, spans and version included."""
    metadata: dict[str, str] = {
        "version": "v3",
        "start": "38",
        "end": "76",
        "uri": "s3://docs/handbook.md",
    }
    metadata.update(overrides.pop("metadata", {}))
    return Hit(
        chunk_id=chunk_id,
        document_id="handbook",
        text=POLICY.text[38:76],
        score=0.8,
        metadata=metadata,
        contributions=(BranchScore(branch=Branch.KEYWORD, score=0.8, rank=1),),
        **overrides,
    )


def found(*hits: Hit) -> RetrievalResult:
    """A retrieval result carrying those hits."""
    return RetrievalResult(query="when is my refund paid", hits=hits, branches=(Branch.KEYWORD,))


def answer(*claims: Claim, citations: tuple[Citation, ...]) -> CitedAnswer:
    """An answer as a model would return it, already parsed into claims."""
    return CitedAnswer(claims=claims, citations=citations)


class TestACitationResolves:
    def test_it_pins_the_document_version_and_the_span(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)

        assert (cited.document_id, cited.document_version) == ("handbook", "v3")
        assert (cited.span.start, cited.span.end) == (38, 76)
        assert cited.retrieved_at == WHEN

    def test_the_span_resolves_back_to_the_exact_sentence(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)

        assert excerpt(cited, POLICY) == "A refund is paid within fourteen days."

    def test_it_carries_the_branch_that_found_it_and_what_it_scored(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)

        assert cited.branches == (Branch.KEYWORD,)
        assert cited.score == 0.8

    def test_a_reranked_hit_is_cited_at_the_score_that_ranked_it(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit(rerank_score=0.42)), retrieved_at=WHEN)

        assert cited.score == 0.42

    def test_it_carries_the_locator_a_reader_would_follow(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(
                found(hit(metadata={"page": "7", "section": "Refunds"})), retrieved_at=WHEN
            )

        assert cited.locator == SourceLocator(
            uri="s3://docs/handbook.md", page=7, section="Refunds"
        )

    def test_it_is_attributed_to_the_tenant_that_retrieved_it(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)

        assert cited.tenant == "acme"

    def test_citing_outside_a_tenant_scope_is_refused(self) -> None:
        with pytest.raises(MissingTenantContextError):
            cite(found(hit()))

    def test_the_time_defaults_to_now(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()))

        assert (datetime.now(UTC) - cited.retrieved_at).total_seconds() < 60

    def test_nothing_retrieved_is_nothing_cited(self) -> None:
        with tenant_scope("acme"):
            assert cite(found()) == ()


class TestAHitThatCannotBeCited:
    def test_a_chunk_with_no_version_is_refused(self) -> None:
        """Resolving against whatever the document says now shows text nobody answered from."""
        with tenant_scope("acme"), pytest.raises(ConfigurationError, match="'version'"):
            cite(found(hit(metadata={"version": ""})), retrieved_at=WHEN)

    def test_a_chunk_with_no_span_is_refused(self) -> None:
        with tenant_scope("acme"), pytest.raises(ConfigurationError, match="character span"):
            cite(found(hit(metadata={"start": ""})), retrieved_at=WHEN)

    def test_a_span_that_runs_backwards_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot end at"):
            Span(start=10, end=4)

    def test_a_span_knows_how_much_it_covers(self) -> None:
        assert Span(start=38, end=76).length == 38


class TestResolvingASourceThatMoved:
    def test_a_document_updated_since_retrieval_will_not_resolve(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)
        updated = POLICY.model_copy(update={"metadata": {"version": "v4"}})

        with pytest.raises(UngroundedCitationError, match="handbook@v3"):
            excerpt(cited, updated)

    def test_another_document_entirely_will_not_resolve(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)
        other = Document(id="tariffs", text=POLICY.text, metadata={"version": "v3"})

        with pytest.raises(UngroundedCitationError):
            excerpt(cited, other)

    def test_a_span_past_the_end_of_the_document_will_not_resolve(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit(metadata={"end": "9000"})), retrieved_at=WHEN)

        with pytest.raises(UngroundedCitationError, match="spans past the end"):
            excerpt(cited, POLICY)

    @pytest.mark.anyio
    async def test_an_erased_chunk_resolves_to_a_tombstone(self) -> None:
        """Right-to-erasure removes the text, not the record that the answer used it."""
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)

        resolved = await Erasing().resolve(cited)

        assert (resolved.erased, resolved.text) == (True, "")
        assert resolved.citation_id == cited.citation_id


class Tombstone:
    """What a citation resolves to once its source has been erased."""

    def __init__(self, citation_id: str) -> None:
        self._id = citation_id

    @property
    def citation_id(self) -> str:
        """Which citation this answers."""
        return self._id

    @property
    def text(self) -> str:
        """Nothing: the source is gone."""
        return ""

    @property
    def erased(self) -> bool:
        """It existed, and was erased on request."""
        return True


class Erasing:
    """A resolver whose corpus has had the cited chunk erased out from under it."""

    async def resolve(self, citation: Citation) -> ResolvedCitation:
        """A tombstone, every time."""
        return Tombstone(citation.citation_id)


class TestTheGroundingCheck:
    def cited(self) -> tuple[Citation, ...]:
        """What this run retrieved."""
        with tenant_scope("acme"):
            return cite(found(hit("timing"), hit("policy")), retrieved_at=WHEN)

    def test_an_answer_from_what_was_retrieved_passes(self) -> None:
        retrieved = self.cited()
        checked = answer(
            Claim(text="A refund is paid within fourteen days.", citation_ids=("timing",)),
            citations=retrieved,
        )

        with tenant_scope("acme"):
            check_grounding(checked, retrieved)

    def test_a_citation_nothing_retrieved_fails_the_run(self) -> None:
        """The kit never returns a fabricated citation, and never strips one to hide it."""
        retrieved = self.cited()
        invented = retrieved[0].model_copy(update={"citation_id": "invented"})
        checked = answer(
            Claim(text="Refunds take a day.", citation_ids=("invented",)), citations=(invented,)
        )

        with tenant_scope("acme"), pytest.raises(UngroundedCitationError) as refused:
            check_grounding(checked, retrieved)

        assert refused.value.missing == ("invented",)
        assert refused.value.available == ("policy", "timing")

    def test_the_same_id_at_another_version_fails_the_run(self) -> None:
        """The document moved between retrieval and answer; the citation must not follow."""
        retrieved = self.cited()
        moved = retrieved[0].model_copy(update={"document_version": "v4"})
        checked = answer(
            Claim(text="A refund is paid within fourteen days.", citation_ids=("timing",)),
            citations=(moved,),
        )

        with tenant_scope("acme"), pytest.raises(UngroundedCitationError, match="did not retrieve"):
            check_grounding(checked, retrieved)

    def test_a_claim_naming_an_id_the_answer_does_not_carry_fails_the_run(self) -> None:
        retrieved = self.cited()
        checked = answer(
            Claim(text="A refund is paid within fourteen days.", citation_ids=("footnote-1",)),
            citations=retrieved,
        )

        with tenant_scope("acme"), pytest.raises(UngroundedCitationError) as refused:
            check_grounding(checked, retrieved)

        assert refused.value.missing == ("footnote-1",)

    def test_a_claim_resting_on_nothing_is_a_refusal(self) -> None:
        """An empty corpus produces a refusal, not an answer with the citations left off."""
        checked = answer(Claim(text="Refunds take a day."), citations=())

        with tenant_scope("acme"), pytest.raises(UncitedClaimError) as refused:
            check_grounding(checked, ())

        assert refused.value.claims == ("Refunds take a day.",)

    def test_a_refusal_with_no_claims_at_all_passes(self) -> None:
        with tenant_scope("acme"):
            check_grounding(CitedAnswer(), ())

    def test_a_citation_into_another_tenant_is_a_scope_violation(self) -> None:
        retrieved = self.cited()
        checked = answer(
            Claim(text="A refund is paid within fourteen days.", citation_ids=("timing",)),
            citations=retrieved,
        )

        with tenant_scope("globex"), pytest.raises(TenantCrossingError, match="another tenant"):
            check_grounding(checked, retrieved)

    def test_several_citations_for_one_claim_and_one_for_several_claims(self) -> None:
        retrieved = self.cited()
        checked = answer(
            Claim(text="A refund is paid within fourteen days.", citation_ids=("timing", "policy")),
            Claim(text="Refunds are covered by the handbook.", citation_ids=("policy",)),
            citations=retrieved,
        )

        with tenant_scope("acme"):
            check_grounding(checked, retrieved)

        assert len(checked.sources(checked.claims[0])) == 2


class TestTheAnswerItself:
    def test_two_citations_may_not_share_an_id(self) -> None:
        with tenant_scope("acme"):
            (cited,) = cite(found(hit()), retrieved_at=WHEN)

        with pytest.raises(ValueError, match="share one id"):
            CitedAnswer(citations=(cited, cited))

    def test_the_claims_read_as_one_piece_of_prose(self) -> None:
        checked = answer(
            Claim(text="Refunds take fourteen days.", citation_ids=("timing",)),
            Claim(text="The handbook says so.", citation_ids=("timing",)),
            citations=(),
        )

        assert checked.text == "Refunds take fourteen days. The handbook says so."

    def test_a_claim_naming_an_unknown_citation_sources_what_it_can(self) -> None:
        checked = answer(Claim(text="Refunds.", citation_ids=("gone",)), citations=())

        assert checked.sources(checked.claims[0]) == ()

    def test_it_reports_every_citation_some_claim_rests_on(self) -> None:
        checked = answer(
            Claim(text="Refunds take fourteen days.", citation_ids=("timing", "policy")),
            Claim(text="The handbook says so.", citation_ids=("policy",)),
            citations=(),
        )

        assert checked.provenance() == ("policy", "timing")


class TestProvenanceReachesMemory:
    def test_a_fact_derived_from_retrieval_records_what_it_came_from(self) -> None:
        """A summary that loses its sources is a claim the corpus cannot be asked about."""
        checked = answer(
            Claim(text="Refunds take fourteen days.", citation_ids=("timing",)), citations=()
        )

        remembered = MemoryRecord(
            id="fact-1",
            kind=MemoryKind.SEMANTIC,
            scope=MemoryScope(tenant_id="acme"),
            key="refund-window",
            value="fourteen days",
            source="retrieval",
            citations=checked.provenance(),
        )

        assert remembered.citations == ("timing",)

    def test_a_record_that_came_from_nowhere_in_particular_cites_nothing(self) -> None:
        remembered = MemoryRecord(
            id="fact-2",
            kind=MemoryKind.PROFILE,
            scope=MemoryScope(tenant_id="acme"),
            key="locale",
            value="en-AU",
            source="signup",
        )

        assert remembered.citations == ()


class TestWhatTheTraceCarries:
    def test_it_names_the_documents_and_versions_and_counts_them(self) -> None:
        with tenant_scope("acme"):
            retrieved = cite(found(hit("timing"), hit("policy")), retrieved_at=WHEN)

        attributes = citation_attributes(retrieved)

        assert attributes["adk.citation.count"] == "2"
        assert attributes["adk.citation.documents"] == "handbook"
        assert attributes["adk.citation.versions"] == "handbook@v3"

    def test_it_never_carries_document_text(self) -> None:
        """A tracing backend is not the corpus, and outlives every redaction rule it has."""
        with tenant_scope("acme"):
            retrieved = cite(found(hit()), retrieved_at=WHEN)

        assert not any(
            POLICY.text[38:76] in value for value in citation_attributes(retrieved).values()
        )

    def test_nothing_cited_is_a_count_of_zero_rather_than_a_missing_attribute(self) -> None:
        assert citation_attributes(()) == {
            "adk.citation.count": "0",
            "adk.citation.documents": "",
            "adk.citation.versions": "",
        }


class TestTheProtocols:
    @pytest.mark.anyio
    async def test_a_resolver_is_a_citation_resolver(self) -> None:
        assert isinstance(Erasing(), CitationResolver)

    def test_a_tombstone_is_a_resolved_citation(self) -> None:
        assert isinstance(Tombstone("timing"), ResolvedCitation)
