"""Splitting a document into the pieces a retriever returns and a model reads.

A chunk is the unit of every later decision: it is what gets embedded, what gets scored,
what gets quoted in a citation and what fills the window. Too large and the sentence that
answers the question arrives buried in two pages the model is trusted to ignore; too small
and it arrives without the paragraph that made it mean anything. Neither is recoverable
downstream, which is why the strategy is a choice made per collection rather than a
constant somebody picked once.

Two things every chunker here holds to. Chunks are measured in the tokens of the model
that will read them, not in characters, because a character budget is wrong by a factor
of three the first time a document is not English. And a chunk knows where it came from:
`text` is exactly `document.text[start:end]`, so a citation can point at the source rather
than at a copy of it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.errors import ChunkingError, ConfigurationError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import Message, TextPart

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.core.protocols import ModelProvider

type TokenCount = Callable[[str], int]
"""How many tokens some text occupies, answered by the tokeniser that will read it."""

type ChunkerFactory = Callable[[ChunkingSpec, TokenCount], Chunker]
"""How a registered strategy name becomes a chunker for one collection's settings."""

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkerFactory",
    "ChunkerRegistry",
    "ChunkingSpec",
    "CodeAware",
    "Document",
    "FixedTokens",
    "Overflow",
    "SentenceWindow",
    "Structural",
    "TokenCount",
    "chunk_id",
    "tokens_via",
]

_WORD = re.compile(r"\S+")
# The terminators of the scripts a corpus actually arrives in, ambiguous to a linter.
_SENTENCE = re.compile(r"[.!?…。！？؟।]+[\s\"'”’)\]]*")  # noqa: RUF001
_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")
_TOP_LEVEL = re.compile(r"(?m)^(?=\S)")
_LINE = re.compile(r"\n")


def tokens_via(provider: ModelProvider) -> TokenCount:
    """Return a counter that measures text with `provider`'s own tokeniser.

    The count that matters is the one the model will make, and every vendor makes it
    differently. A chunk sized by a character heuristic is under a window on English and
    over it on Japanese, and nothing finds out until retrieval is already in production.
    """

    def count(text: str) -> int:
        return provider.count_tokens([Message(role="user", content=[TextPart(text=text)])])

    return count


def chunk_id(document_id: str, text: str) -> str:
    """Return the identity of `text` as a chunk of `document_id`.

    Content-addressed, so re-indexing an unchanged document writes the same ids and an
    edit changes only the chunks it touched. Identical text within one document is one id
    by construction: storing the same passage twice under two names is a duplicate hit.
    """
    digest = sha256(f"{document_id}\0{text}".encode())
    return digest.hexdigest()[:32]


class Overflow(StrEnum):
    """What to do with a run of text that cannot be divided small enough.

    Args:
        REFUSE: Raise `ChunkingError` naming the document and the offset. The default:
            a base64 blob or a minified bundle is a loader's problem, and silence about
            it means the chunk that breaks the window is found by a model call.
        SPLIT: Cut it mid-word at the limit. Correct for a document that genuinely has no
            boundaries, and lossy for one that has boundaries the strategy did not know
            about, which is why it is asked for rather than assumed.
    """

    REFUSE = "refuse"
    SPLIT = "split"


class Document(AdkModel):
    """Text to be indexed, with the identity its chunks are attributed to.

    Args:
        id: What citations name. Chunk ids are derived from it, so changing it re-indexes
            the document rather than updating it.
        text: The document itself, already extracted from whatever carried it.
        metadata: Consumer-owned annotations, inherited by every chunk of it. Filters at
            retrieval time read this, so what a chunk may be restricted by belongs here.
    """

    id: str = Field(min_length=1)
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(AdkModel):
    """One piece of a document, and where in it the piece was.

    Args:
        id: Content-addressed — see `chunk_id`.
        document_id: The document this came out of.
        ordinal: Position in the sequence this document produced, from zero.
        text: The piece itself, exactly `document.text[start:end]`.
        start: Character offset the piece begins at, counted in code points.
        end: Character offset one past its end.
        tokens: How many tokens it occupies, by the counter that sized it.
        section: The heading path in force, outermost first, where the strategy found one.
        metadata: The document's metadata, carried down.
    """

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    tokens: int = Field(ge=0)
    section: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _the_span_describes_the_text(self) -> Chunk:
        if self.end < self.start:
            raise ValueError(f"a chunk cannot end at {self.end} having started at {self.start}")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"span {self.start}:{self.end} claims {self.end - self.start} characters "
                f"for text of {len(self.text)}: a citation built on it would point at "
                "the wrong place"
            )
        return self


@runtime_checkable
class Chunker(Protocol):
    """How one collection's documents are divided."""

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Return `document` in order, as pieces that each fit the token limit.

        The pieces cover the document: concatenating a non-overlapping strategy's chunks
        reproduces it. An empty document produces no chunks rather than one empty one.

        Raises:
            ChunkingError: If some run of text cannot be divided under the limit and the
                configured overflow policy is to refuse.
        """
        ...


class ChunkingSpec(AdkModel):
    """One collection's chunking settings, as a deployment states them.

    Args:
        strategy: A name registered with the `ChunkerRegistry` — `fixed`, `structural`,
            `sentence-window`, `code`, or a deployment's own.
        max_tokens: The ceiling no chunk may exceed.
        overlap_tokens: How much of the previous chunk each one repeats, where the
            strategy overlaps at all. Zero means the chunks partition the document.
        window: Sentences per chunk, for `sentence-window`.
        stride: Sentences advanced between windows, for `sentence-window`.
        overflow: What to do with an indivisible run longer than `max_tokens`.
    """

    strategy: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(default=0, ge=0)
    window: int = Field(default=3, gt=0)
    stride: int = Field(default=1, gt=0)
    overflow: Overflow = Overflow.REFUSE

    @model_validator(mode="after")
    def _an_overlap_leaves_room_to_advance(self) -> ChunkingSpec:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens={self.overlap_tokens} does not fit inside "
                f"max_tokens={self.max_tokens}: each chunk would repeat the last entirely"
            )
        if self.stride > self.window:
            raise ValueError(
                f"stride={self.stride} skips past window={self.window}, leaving sentences "
                "in no chunk at all"
            )
        return self


def _spans(length: int, cuts: Iterable[int]) -> list[tuple[int, int]]:
    """Turn cut offsets into contiguous spans covering the whole of `length`.

    Everything here segments by producing cuts rather than by producing pieces, so no
    strategy can drop the whitespace between two units and leave a gap a citation would
    resolve to the wrong text.
    """
    edges = sorted({0, length, *(cut for cut in cuts if 0 < cut < length)})
    return list(pairwise(edges))


def _headings(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(text)]


def _section_at(headings: Sequence[tuple[int, int, str]], offset: int) -> tuple[str, ...]:
    path: list[tuple[int, str]] = []
    for start, level, title in headings:
        if start > offset:
            break
        path = [(lvl, name) for lvl, name in path if lvl < level]
        path.append((level, title))
    return tuple(name for _, name in path)


class _Splitter:
    """The packing every built-in strategy does, over whatever units it segments into."""

    def __init__(self, spec: ChunkingSpec, count: TokenCount) -> None:
        self._spec = spec
        self._count = count

    def _fits(self, text: str, start: int, end: int) -> bool:
        return self._count(text[start:end]) <= self._spec.max_tokens

    def _largest_fit(self, text: str, units: Sequence[tuple[int, int]], first: int) -> int:
        """Return the exclusive unit index of the longest run from `first` that fits.

        Exponential then binary search rather than one unit at a time: a counter is a
        tokeniser call over the candidate, and growing a megabyte one word at a time
        counts the same prefix a hundred thousand times.
        """
        if not self._fits(text, units[first][0], units[first][1]):
            return first
        good, step = first + 1, 1
        while good + step <= len(units) and self._fits(
            text, units[first][0], units[good + step - 1][1]
        ):
            good += step
            step *= 2
        high = min(len(units), good + step)
        if high > good and self._fits(text, units[first][0], units[high - 1][1]):
            return high
        while high - good > 1:
            mid = (good + high) // 2
            if self._fits(text, units[first][0], units[mid - 1][1]):
                good = mid
            else:
                high = mid
        return good

    def _pack(
        self,
        document: Document,
        units: Sequence[tuple[int, int]],
        finer: Sequence[Callable[[str], Sequence[int]]],
    ) -> list[tuple[int, int]]:
        """Group `units` into the largest runs that fit, splitting any that alone do not."""
        out: list[tuple[int, int]] = []
        index = 0
        while index < len(units):
            end = self._largest_fit(document.text, units, index)
            if end == index:
                out.extend(self._divide(document, units[index], finer))
                index += 1
                continue
            out.append((units[index][0], units[end - 1][1]))
            index = self._advance(document.text, units, index, end)
        return out

    def _advance(self, text: str, units: Sequence[tuple[int, int]], first: int, end: int) -> int:
        """Return where the next chunk starts, backing up far enough to overlap."""
        if not self._spec.overlap_tokens:
            return end
        start = end
        while start > first + 1 and (
            self._count(text[units[start - 1][0] : units[end - 1][1]]) <= self._spec.overlap_tokens
        ):
            start -= 1
        return start

    def _divide(
        self,
        document: Document,
        span: tuple[int, int],
        finer: Sequence[Callable[[str], Sequence[int]]],
    ) -> list[tuple[int, int]]:
        """Split one over-long unit using the next-finest segmentation there is."""
        if not finer:
            return self._overflowed(document, span)
        start, end = span
        inner = _spans(end - start, finer[0](document.text[start:end]))
        if len(inner) < 2:
            return self._divide(document, span, finer[1:])
        shifted = [(start + a, start + b) for a, b in inner]
        return self._pack(document, shifted, finer[1:])

    def _overflowed(self, document: Document, span: tuple[int, int]) -> list[tuple[int, int]]:
        start, end = span
        if self._spec.overflow is Overflow.REFUSE:
            raise ChunkingError(
                f"{end - start} characters at offset {start} of document {document.id!r} "
                f"do not divide under {self._spec.max_tokens} tokens; configure an "
                "overflow policy to cut them anyway",
                document=document.id,
                offset=start,
            )
        return self._by_character(document.text, span)

    def _by_character(self, text: str, span: tuple[int, int]) -> list[tuple[int, int]]:
        """Cut mid-word at the limit, one code point at a time so no character is halved."""
        out: list[tuple[int, int]] = []
        start, end = span
        while start < end:
            width, step = 1, 1
            while start + width + step <= end and self._fits(text, start, start + width + step):
                width += step
                step *= 2
            high = min(end, start + width + step)
            if high > start + width and self._fits(text, start, high):
                width = high - start
            while high - (start + width) > 1:
                mid = (start + width + high) // 2
                if self._fits(text, start, mid):
                    width = mid - start
                else:
                    high = mid
            out.append((start, start + width))
            start += width
        return out

    def _assemble(self, document: Document, pieces: Sequence[tuple[int, int]]) -> list[Chunk]:
        headings = _headings(document.text)
        chunks: list[Chunk] = []
        for ordinal, (start, end) in enumerate(pieces):
            text = document.text[start:end]
            chunks.append(
                Chunk(
                    id=chunk_id(document.id, text),
                    document_id=document.id,
                    ordinal=ordinal,
                    text=text,
                    start=start,
                    end=end,
                    tokens=self._count(text),
                    section=_section_at(headings, start),
                    metadata=dict(document.metadata),
                )
            )
        return chunks


def _word_cuts(text: str) -> Sequence[int]:
    return [m.start() for m in _WORD.finditer(text)]


def _sentence_cuts(text: str) -> Sequence[int]:
    return [m.end() for m in _SENTENCE.finditer(text)]


def _paragraph_cuts(text: str) -> Sequence[int]:
    return [m.end() for m in _PARAGRAPH.finditer(text)]


def _heading_cuts(text: str) -> Sequence[int]:
    return [m.start() for m in _HEADING.finditer(text)]


def _block_cuts(text: str) -> Sequence[int]:
    return [m.start() for m in _TOP_LEVEL.finditer(text)]


def _line_cuts(text: str) -> Sequence[int]:
    return [m.end() for m in _LINE.finditer(text)]


class FixedTokens(_Splitter):
    """Fill each chunk to the token limit on word boundaries, repeating an overlap.

    The strategy that assumes nothing about the document, which is its merit and its
    cost: it will cut a table in half without noticing there was a table.
    """

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Return the document packed into token-limited windows."""
        if not document.text:
            return ()
        units = _spans(len(document.text), _word_cuts(document.text))
        return tuple(self._assemble(document, self._pack(document, units, [])))


class Structural(_Splitter):
    """Split on the document's own boundaries, falling to finer ones only as needed.

    Headings first, then paragraphs, then sentences, then words. A chunk that ends where
    the author ended something is a chunk that reads on its own, and the fallbacks mean a
    document with no structure still produces chunks rather than an error.
    """

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Return the document split at the coarsest boundaries that fit the limit."""
        if not document.text:
            return ()
        units = _spans(len(document.text), _heading_cuts(document.text))
        finer = [_paragraph_cuts, _sentence_cuts, _word_cuts]
        return tuple(self._assemble(document, self._pack(document, units, finer)))


class SentenceWindow(_Splitter):
    """Emit overlapping windows of whole sentences.

    Retrieval hits one sentence and answers from its neighbours, so the window is the
    unit and `stride` decides how much context every hit is guaranteed. Chunks overlap by
    construction here — they cover the document, they do not partition it.
    """

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Return windows of `spec.window` sentences, advancing by `spec.stride`."""
        if not document.text:
            return ()
        units = _spans(len(document.text), _sentence_cuts(document.text))
        pieces: list[tuple[int, int]] = []
        index = 0
        while index < len(units):
            end = min(len(units), index + self._spec.window)
            fitting = self._largest_fit(document.text, units[:end], index)
            if fitting == index:
                pieces.extend(self._divide(document, units[index], [_word_cuts]))
            else:
                pieces.append((units[index][0], units[fitting - 1][1]))
            index += self._spec.stride
        return tuple(self._assemble(document, pieces))


class CodeAware(_Splitter):
    """Split source at top-level definitions, then at blank lines, then at lines.

    A function cut in half retrieves as two things that are each wrong. Column zero is
    where a top-level definition starts in every language that indents its bodies, which
    is enough to keep whole definitions together without a parser per language.
    """

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Return the source packed into chunks that end at a definition or a line."""
        if not document.text:
            return ()
        units = _spans(len(document.text), _block_cuts(document.text))
        return tuple(self._assemble(document, self._pack(document, units, [_line_cuts])))


_BUILT_IN: dict[str, ChunkerFactory] = {
    "fixed": FixedTokens,
    "structural": Structural,
    "sentence-window": SentenceWindow,
    "code": CodeAware,
}


class ChunkerRegistry:
    """Which strategy each collection is chunked by, resolved by name.

    Chunking settings belong beside the collection they produce, because a collection
    chunked one way on Monday and another way on Tuesday is a collection whose retrieval
    quality nobody can reason about. The registry is where a deployment states that, and
    where its own strategies join the built-in ones.

    Args:
        count: How tokens are counted for every collection here.
        collections: Settings per collection name.
        default: Settings for a collection not named above. Absent means an unnamed
            collection is a configuration error rather than a silent guess.
    """

    def __init__(
        self,
        *,
        count: TokenCount,
        collections: Mapping[str, ChunkingSpec] | None = None,
        default: ChunkingSpec | None = None,
    ) -> None:
        self._count = count
        self._collections = dict(collections or {})
        self._default = default
        self._factories: dict[str, ChunkerFactory] = dict(_BUILT_IN)

    def register(self, strategy: str, build: ChunkerFactory) -> None:
        """Add a strategy under `strategy`, replacing any of that name.

        Raises:
            ConfigurationError: If `strategy` is empty.
        """
        if not strategy:
            raise ConfigurationError("a chunking strategy needs a name to be selected by")
        self._factories[strategy] = build

    def spec_for(self, collection: str) -> ChunkingSpec:
        """Return the settings `collection` is chunked by.

        Raises:
            ConfigurationError: If the collection is not configured and there is no
                default.
        """
        spec = self._collections.get(collection, self._default)
        if spec is None:
            raise ConfigurationError(
                f"collection {collection!r} has no chunking settings and no default is "
                "configured; chunking it by guesswork would index it differently from "
                "everything already in it"
            )
        return spec

    def chunker_for(self, collection: str) -> Chunker:
        """Return the chunker `collection` is indexed with.

        Raises:
            ConfigurationError: If the collection is not configured, or names a strategy
                nobody registered.
        """
        spec = self.spec_for(collection)
        build = self._factories.get(spec.strategy)
        if build is None:
            known = ", ".join(sorted(self._factories)) or "none"
            raise ConfigurationError(
                f"collection {collection!r} asks for chunking strategy "
                f"{spec.strategy!r}, which is not registered; known: {known}"
            )
        return build(spec, self._count)
