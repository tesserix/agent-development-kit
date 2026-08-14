"""Resolving an answer back to the exact sentence it was built from, and refusing when it will not.

Four scenarios: a citation resolved back to its span in the source document; an answer
citing a document nobody retrieved; an answer whose claim rests on nothing; and a source
updated between the retrieval and the answer, which must not resolve against the new text.

Run it with `python examples/citations.py`. Nothing here reaches the network: the store is
the in-process fake from `tesserix_adk.testing`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    TenantCrossingError,
    UncitedClaimError,
    UngroundedCitationError,
    tenant_scope,
)
from tesserix_adk.rag import (
    Branch,
    Citation,
    CitedAnswer,
    Claim,
    Document,
    IndexRetriever,
    RetrievalScope,
    check_grounding,
    citation_attributes,
    cite,
    excerpt,
)
from tesserix_adk.testing import FakeIndex, Indexed

HANDBOOK = RetrievalScope(collection="handbook")

POLICY = Document(
    id="handbook",
    text="Refunds are the subject of this page. A refund is paid within fourteen days.",
    metadata={"version": "v3"},
)

TIMING = POLICY.text.index("A refund")


def corpus() -> FakeIndex:
    """One chunk, carrying everything a citation needs to resolve later."""
    return FakeIndex(
        Indexed(
            "timing",
            POLICY.text[TIMING:],
            document_id=POLICY.id,
            metadata={
                "version": "v3",
                "start": str(TIMING),
                "end": str(len(POLICY.text)),
                "uri": "s3://docs/handbook.md",
                "section": "Refunds",
            },
        )
    )


async def retrieved() -> tuple[Citation, ...]:
    """What one retrieval offers an answer to lean on."""
    found = await IndexRetriever(corpus(), branch=Branch.KEYWORD).retrieve(
        "when is my refund paid", scope=HANDBOOK
    )
    return cite(found)


async def a_citation_resolves_to_the_sentence() -> None:
    """The whole point: from an answer, back to the characters it was built from."""
    with tenant_scope("acme"):
        citations = await retrieved()

    citation = citations[0]
    where = f"{citation.document_id}@{citation.document_version}"
    print(f"cited: {where} {citation.span.start}:{citation.span.end}")  # noqa: T201
    print(f"says: {excerpt(citation, POLICY)!r}")  # noqa: T201
    print(f"trace: {citation_attributes(citations)}")  # noqa: T201


async def a_citation_nobody_retrieved() -> None:
    """The kit does not strip the offending citation to make the answer look valid."""
    with tenant_scope("acme"):
        citations = await retrieved()
        invented = citations[0].model_copy(
            update={"citation_id": "tariffs-1", "document_id": "tariffs"}
        )
        answer = CitedAnswer(
            claims=(Claim(text="Refunds are paid the same day.", citation_ids=("tariffs-1",)),),
            citations=(invented,),
        )
        try:
            check_grounding(answer, citations)
        except UngroundedCitationError as refused:
            print(f"refused: missing {refused.missing}, available {refused.available}")  # noqa: T201


async def an_answer_that_cites_nothing() -> None:
    """An empty corpus produces a refusal, not an answer with the citations left off."""
    answer = CitedAnswer(claims=(Claim(text="Refunds are paid the same day."),))

    with tenant_scope("acme"):
        try:
            check_grounding(answer, ())
        except UncitedClaimError as refused:
            print(f"refused: {len(refused.claims)} claim resting on nothing")  # noqa: T201


async def a_document_that_moved_underneath() -> None:
    """Resolving against the new version would show text the answer was never built from."""
    with tenant_scope("acme"):
        citations = await retrieved()

    updated = POLICY.model_copy(
        update={
            "text": "Refunds are the subject of this page. A refund is paid within two days.",
            "metadata": {"version": "v4"},
        }
    )
    try:
        excerpt(citations[0], updated)
    except UngroundedCitationError as refused:
        print(f"refused: {refused}")  # noqa: T201


async def a_citation_into_another_tenant() -> None:
    """Whose document it is travels with the citation, not with the call."""
    with tenant_scope("acme"):
        citations = await retrieved()
        answer = CitedAnswer(
            claims=(
                Claim(text="A refund is paid within fourteen days.", citation_ids=("timing",)),
            ),
            citations=citations,
        )

    with tenant_scope("globex"):
        try:
            check_grounding(answer, citations)
        except TenantCrossingError as refused:
            print(f"refused: {refused}")  # noqa: T201


async def main() -> None:
    """Run every scenario in order."""
    await a_citation_resolves_to_the_sentence()
    await a_citation_nobody_retrieved()
    await an_answer_that_cites_nothing()
    await a_document_that_moved_underneath()
    await a_citation_into_another_tenant()


if __name__ == "__main__":
    asyncio.run(main())
