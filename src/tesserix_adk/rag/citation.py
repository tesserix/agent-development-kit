"""Answers that can be checked: which document, which version, which characters.

An answer built from three retrieved passages and returned as prose is unauditable. When it
turns out to be wrong, nobody can say whether the corpus was wrong, retrieval was wrong, or
the model made it up, and the support agent holding it cannot show the customer where the
policy came from.

So a citation is a structured thing that resolves to a span of a version of a document, an
answer is claims carrying citation ids rather than footnote-shaped text, and the grounding
check runs before the answer is returned. A citation naming something the run did not
retrieve fails the run; it is never quietly dropped to make the answer look sourced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core import (
    AdkModel,
    ConfigurationError,
    TenantCrossingError,
    UncitedClaimError,
    UngroundedCitationError,
    current_tenant,
)
from tesserix_adk.rag.retrieval import Branch  # noqa: TC001 — pydantic needs the runtime type

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.rag.chunking import Document
    from tesserix_adk.rag.retrieval import Hit, RetrievalResult

__all__ = [
    "Citation",
    "CitationResolver",
    "CitedAnswer",
    "Claim",
    "ResolvedCitation",
    "SourceLocator",
    "Span",
    "check_grounding",
    "citation_attributes",
    "cite",
    "excerpt",
]

VERSION_KEY = "version"
"""The chunk metadata naming the document version it was indexed from."""

SPAN_KEYS = ("start", "end")
"""The chunk metadata giving the character offsets the chunk was taken from."""


class Span(AdkModel):
    """Where in a document a passage was, counted in code points.

    Args:
        start: Offset the passage begins at.
        end: Offset one past its end.
    """

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _it_runs_forwards(self) -> Span:
        if self.end < self.start:
            raise ValueError(f"a span cannot end at {self.end} having started at {self.start}")
        return self

    @property
    def length(self) -> int:
        """How many characters it covers."""
        return self.end - self.start


class SourceLocator(AdkModel):
    """Where to go and look, for a reader who wants the original.

    Args:
        uri: Where the document lives.
        page: The page, for paginated sources.
        section: The heading the passage sits under.
    """

    uri: str = ""
    page: int | None = Field(default=None, ge=1)
    section: str = ""


class Citation(AdkModel):
    """One passage an answer may lean on, pinned hard enough to resolve later.

    The version is part of the citation because the document may be updated between the
    retrieval and the answer; resolving against whatever the document says now would show
    a reader text the answer was never built from.

    Args:
        citation_id: What claims refer to. The chunk id, unless a caller renames it.
        document_id: Which document.
        document_version: Which version of it was retrieved.
        chunk_id: Which chunk of that version.
        tenant: Whose document it is. A citation carries it so a result crossing a call
            boundary cannot lose the only thing that says who may read it.
        span: Which characters of the document the chunk was.
        score: What retrieval scored it.
        branches: Which branches found it.
        retrieved_at: When it was retrieved.
        locator: Where a reader can go and look.
    """

    citation_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    span: Span
    score: float = 0.0
    branches: tuple[Branch, ...] = ()
    retrieved_at: datetime
    locator: SourceLocator = SourceLocator()


class Claim(AdkModel):
    """One statement an answer makes, and what it rests on.

    Args:
        text: The statement itself.
        citation_ids: The citations supporting it. Several may support one claim, and one
            may support several claims.
    """

    text: str = Field(min_length=1)
    citation_ids: tuple[str, ...] = ()


class CitedAnswer(AdkModel):
    """An answer as claims and citations, rather than prose with footnote-shaped strings.

    An answer that carries its citations as text can be checked only by parsing it, which
    means an answer that cites nothing and an answer whose footnotes were invented both
    read as valid.

    Args:
        claims: What the answer says, in order.
        citations: What it says it read.
    """

    claims: tuple[Claim, ...] = ()
    citations: tuple[Citation, ...] = ()

    @model_validator(mode="after")
    def _each_citation_appears_once(self) -> CitedAnswer:
        ids = [held.citation_id for held in self.citations]
        if len(set(ids)) != len(ids):
            raise ValueError("two citations share one id, so a claim citing it is ambiguous")
        return self

    @property
    def text(self) -> str:
        """The claims as one piece of prose, for display."""
        return " ".join(claim.text for claim in self.claims)

    def sources(self, claim: Claim) -> tuple[Citation, ...]:
        """The citations behind `claim`, in the order it named them."""
        found = {held.citation_id: held for held in self.citations}
        return tuple(found[cited] for cited in claim.citation_ids if cited in found)

    def provenance(self) -> tuple[str, ...]:
        """Every citation id some claim rests on, sorted, for recording against a memory."""
        return tuple(sorted({cited for claim in self.claims for cited in claim.citation_ids}))


@runtime_checkable
class ResolvedCitation(Protocol):
    """What a citation resolved to."""

    @property
    def citation_id(self) -> str:
        """Which citation this answers."""
        ...

    @property
    def text(self) -> str:
        """The passage, or empty where it is gone."""
        ...

    @property
    def erased(self) -> bool:
        """Whether the source is a tombstone: it existed, and was erased on request."""
        ...


@runtime_checkable
class CitationResolver(Protocol):
    """Turns a citation back into the text it was made from."""

    async def resolve(self, citation: Citation) -> ResolvedCitation:
        """Fetch the cited span, or a tombstone where the source has been erased."""
        ...


def cite(result: RetrievalResult, *, retrieved_at: datetime | None = None) -> tuple[Citation, ...]:
    """Build a citation per hit, from what the store recorded about each chunk.

    Reads `version`, `start` and `end` from the hit's metadata, and `uri`, `page` and
    `section` where the store carries them.

    Args:
        result: What retrieval found.
        retrieved_at: When, defaulting to now.

    Returns:
        One citation per hit, in the order they were ranked.

    Raises:
        ConfigurationError: If a hit cannot be pinned to a version and a span. A citation
            that resolves to whatever the document says now is worse than no citation.
        MissingTenantContextError: If there is no tenant to attribute the citations to.
    """
    tenant = current_tenant(where="citation").tenant
    when = retrieved_at or datetime.now(UTC)
    return tuple(_citation(hit, tenant=tenant, when=when) for hit in result.hits)


def _citation(hit: Hit, *, tenant: str, when: datetime) -> Citation:
    """One citation, refusing a hit the store did not describe well enough to resolve."""
    version = hit.metadata.get(VERSION_KEY, "")
    if not version:
        raise ConfigurationError(
            f"chunk {hit.chunk_id} carries no {VERSION_KEY!r}, so a citation to it would "
            "resolve against whatever the document says now"
        )
    return Citation(
        citation_id=hit.chunk_id,
        document_id=hit.document_id,
        document_version=version,
        chunk_id=hit.chunk_id,
        tenant=tenant,
        span=_span(hit),
        score=hit.rerank_score if hit.rerank_score is not None else hit.score,
        branches=tuple(scored.branch for scored in hit.contributions),
        retrieved_at=when,
        locator=SourceLocator(
            uri=hit.metadata.get("uri", ""),
            page=int(hit.metadata["page"]) if hit.metadata.get("page") else None,
            section=hit.metadata.get("section", ""),
        ),
    )


def _span(hit: Hit) -> Span:
    """The offsets the chunk was taken from, refusing a hit that does not carry them."""
    offsets = [hit.metadata.get(key, "") for key in SPAN_KEYS]
    if not all(offset.isdigit() for offset in offsets):
        raise ConfigurationError(
            f"chunk {hit.chunk_id} carries no character span, so a citation to it can "
            "point at the document but not at the sentence"
        )
    start, end = (int(offset) for offset in offsets)
    return Span(start=start, end=end)


def check_grounding(answer: CitedAnswer, retrieved: Sequence[Citation]) -> None:
    """Refuse an answer whose citations do not come from this run's retrieval.

    Args:
        answer: What the model produced.
        retrieved: The citations the run actually retrieved, from `cite`.

    Raises:
        UncitedClaimError: If any claim rests on nothing.
        UngroundedCitationError: If a claim names a citation the answer does not carry, or
            the answer carries one this run did not retrieve, or one whose document or
            version differs from what was retrieved under that id.
        TenantCrossingError: If a citation names a tenant other than the caller's.
    """
    uncited = tuple(claim.text for claim in answer.claims if not claim.citation_ids)
    if uncited:
        raise UncitedClaimError(
            f"{len(uncited)} of {len(answer.claims)} claims cite nothing: an answer with "
            "no source is a refusal, not an answer",
            claims=uncited,
        )
    available = {held.citation_id: held for held in retrieved}
    _every_cited_id_is_carried(answer, available)
    _every_carried_citation_was_retrieved(answer, available)


def _every_cited_id_is_carried(answer: CitedAnswer, available: Mapping[str, Citation]) -> None:
    """A claim citing an id the answer does not carry is a fabricated footnote."""
    carried = {held.citation_id for held in answer.citations}
    named = {cited for claim in answer.claims for cited in claim.citation_ids}
    dangling = sorted(named - carried)
    if dangling:
        raise UngroundedCitationError(
            f"{len(dangling)} citation ids are referenced by a claim and defined nowhere",
            missing=dangling,
            available=sorted(available),
        )


def _every_carried_citation_was_retrieved(
    answer: CitedAnswer, available: Mapping[str, Citation]
) -> None:
    """Same id, other document or other version, is the failure this check exists for."""
    tenant = current_tenant(where="grounding").tenant
    missing = sorted(
        held.citation_id
        for held in answer.citations
        if held.citation_id not in available
        or (held.document_id, held.document_version)
        != (available[held.citation_id].document_id, available[held.citation_id].document_version)
    )
    if missing:
        raise UngroundedCitationError(
            f"{len(missing)} citations name something this run did not retrieve",
            missing=missing,
            available=sorted(available),
        )
    crossing = sorted(held.citation_id for held in answer.citations if held.tenant != tenant)
    if crossing:
        raise TenantCrossingError(
            f"citations {', '.join(crossing)} belong to another tenant",
            tenant=tenant,
            into=next(held.tenant for held in answer.citations if held.tenant != tenant),
        )


def citation_attributes(citations: Sequence[Citation]) -> dict[str, str]:
    """Span attributes describing a citation set without carrying any of its text.

    Document text on a span outlives every redaction rule the corpus has, because the
    tracing backend is not the corpus. Counts, ids and versions are enough to tell which
    documents an answer leaned on.

    Args:
        citations: What was cited.

    Returns:
        Attributes for `Tracer.span`, keyed under `adk.citation`.
    """
    return {
        "adk.citation.count": str(len(citations)),
        "adk.citation.documents": ",".join(sorted({held.document_id for held in citations})),
        "adk.citation.versions": ",".join(
            sorted({f"{held.document_id}@{held.document_version}" for held in citations})
        ),
    }


def excerpt(citation: Citation, document: Document) -> str:
    """The exact characters a citation points at, in the version it was taken from.

    Args:
        citation: What to resolve.
        document: The source, carrying its version in `metadata["version"]`.

    Returns:
        The cited span of the document's text.

    Raises:
        UngroundedCitationError: If the document is a different one, or has been updated
            since the citation was made, or is shorter than the span claims.
    """
    version = document.metadata.get(VERSION_KEY, "")
    if document.id != citation.document_id or version != citation.document_version:
        raise UngroundedCitationError(
            f"citation {citation.citation_id} was made against "
            f"{citation.document_id}@{citation.document_version}, not {document.id}@{version}",
            missing=(citation.citation_id,),
        )
    if citation.span.end > len(document.text):
        raise UngroundedCitationError(
            f"citation {citation.citation_id} spans past the end of {document.id}",
            missing=(citation.citation_id,),
        )
    return document.text[citation.span.start : citation.span.end]
