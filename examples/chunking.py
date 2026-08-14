"""Splitting documents into the pieces a retriever returns, sized in tokens.

Four scenarios: structure kept where the author put it; the same handbook chunked the way
one collection asks for and another does not; a blob that will not divide; and the spans
that make a citation point at the source rather than at a copy of it.

Run it with `python examples/chunking.py`. Nothing here reaches the network: the token
counter is a local one, and a deployment swaps in `tokens_via(provider)` to count with the
tokeniser of the model that will read the chunks.
"""

from __future__ import annotations

import re

from tesserix_adk.core import ChunkingError
from tesserix_adk.rag import ChunkerRegistry, ChunkingSpec, Document, Overflow

WORDS = re.compile(r"\S+")

HANDBOOK = Document(
    id="handbook-2026",
    text=(
        "# Travel\n\n"
        "Book the overnight service where the meeting starts before ten. It arrives at "
        "dawn and the fare is lower than the morning train.\n\n"
        "## Sleepers\n\n"
        "A berth must be booked ahead. There are four to a compartment, and a single "
        "compartment can be reserved where the journey runs over two nights.\n\n"
        "## Claims\n\n"
        "Submit the ticket within thirty days. A claim without a ticket is refused.\n"
    ),
    metadata={"tenant": "acme", "source": "handbook"},
)


def counting_words(text: str) -> int:
    """Stands in for a real tokeniser: wrong by a constant, right about boundaries."""
    return len(WORDS.findall(text))


def registry() -> ChunkerRegistry:
    """Which strategy each collection is indexed with, stated once for the deployment."""
    return ChunkerRegistry(
        count=counting_words,
        collections={
            "handbook": ChunkingSpec(strategy="structural", max_tokens=24),
            "search": ChunkingSpec(strategy="fixed", max_tokens=12, overlap_tokens=4),
            "repo": ChunkingSpec(strategy="code", max_tokens=20),
        },
    )


def structure_survives_the_split() -> None:
    """A chunk that ends where the author ended something reads on its own."""
    for chunk in registry().chunker_for("handbook").chunk(HANDBOOK):
        where = " > ".join(chunk.section) or "(no heading)"
        print(f"[{where}] {chunk.tokens:>2} tokens: {chunk.text.strip()[:60]}…")  # noqa: T201


def one_document_two_collections() -> None:
    """The same text is a different set of chunks depending on what the collection is for."""
    settings = registry()
    for collection in ("handbook", "search"):
        chunks = settings.chunker_for(collection).chunk(HANDBOOK)
        widest = max(chunk.tokens for chunk in chunks)
        print(f"{collection}: {len(chunks)} chunks, largest {widest} tokens")  # noqa: T201


def what_will_not_divide() -> None:
    """A refusal names the document and the offset; a policy cuts it where asked."""
    blob = Document(id="export", text="preamble " + "x" * 2_000)
    settings = ChunkerRegistry(count=len, default=ChunkingSpec(strategy="fixed", max_tokens=500))
    try:
        settings.chunker_for("export").chunk(blob)
    except ChunkingError as refused:
        print(f"refused: {refused.document} at offset {refused.offset}")  # noqa: T201

    cutting = ChunkerRegistry(
        count=len,
        default=ChunkingSpec(strategy="fixed", max_tokens=500, overflow=Overflow.SPLIT),
    )
    print(f"with a policy: {len(cutting.chunker_for('export').chunk(blob))} chunks")  # noqa: T201


def a_citation_points_at_the_source() -> None:
    """`text` is exactly `document.text[start:end]`, so the offsets are quotable."""
    chunk = registry().chunker_for("handbook").chunk(HANDBOOK)[-1]
    quoted = HANDBOOK.text[chunk.start : chunk.end]
    print(  # noqa: T201
        f"chunk {chunk.id[:8]} covers {chunk.start}:{chunk.end} of {chunk.document_id}",
        f"and the source still reads {quoted.strip()[:40]!r}",
    )


def main() -> None:
    """Run every scenario in order."""
    structure_survives_the_split()
    one_document_two_collections()
    what_will_not_divide()
    a_citation_points_at_the_source()


if __name__ == "__main__":
    main()
