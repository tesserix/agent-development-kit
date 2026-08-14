"""Reading scanned documents and recordings into text an agent can cite.

Document and call intake is the same job in every product that has it, and doing it per
product means each one re-invents provenance. A page of OCR without a page number is a
paragraph an answer cannot be traced back to, and a transcript without timestamps is a
quote nobody can check.

Both have mature CPU backends — PaddleOCR and whisper.cpp — and neither is a dependency
here, for the reason `tesserix_adk.models.onnx` gives: they are large native installs, and
a consumer who does not read documents should not carry one. The kit takes `OcrBackend` and
`TranscriptionBackend` protocols, validates what crosses into them, and refuses to call an
unread file empty.

Reading streams. A long PDF is yielded a page at a time, so a hundred-page contract costs
one page of memory and the caller can stop after the page it wanted.
"""

from __future__ import annotations

import unicodedata
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.errors import MediaIntakeError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.tools.context import (
    ToolContext,  # noqa: TC001 — the decorator reads this annotation at runtime
)
from tesserix_adk.tools.decorator import Tool, tool
from tesserix_adk.tools.errors import ToolRefusal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

__all__ = [
    "DEFAULT_INTAKE_CHARS",
    "SUPPORTED_AUDIO",
    "SUPPORTED_DOCUMENTS",
    "OcrBackend",
    "OcrPage",
    "Region",
    "RegionKind",
    "Segment",
    "TranscriptionBackend",
    "ocr_document",
    "ocr_tool",
    "transcribe_audio",
    "transcribe_tool",
]

SUPPORTED_DOCUMENTS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})
"""Document types a CPU OCR backend reads. Anything else is refused before it is opened."""

SUPPORTED_AUDIO = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg"})
"""Recording types a CPU transcription backend reads."""

DEFAULT_INTAKE_CHARS = 4_096
"""How much one tool call returns. A window onto the document, not the document."""

_SCRIPT_NAMES = {"CJK": "Han", "IDEOGRAPHIC": "Han", "FULLWIDTH": "Latin"}


class RegionKind(StrEnum):
    """What a region is on the page, as the layout pass saw it.

    Kept because an answer built from a footer and one built from a clause are not equally
    trustworthy, and only the layout knows which is which.
    """

    HEADING = "heading"
    BODY = "body"
    TABLE = "table"
    CAPTION = "caption"
    FOOTER = "footer"


class Region(AdkModel):
    """One laid-out block of text on a page.

    Args:
        text: What was read there.
        box: Where it sat, as (left, top, width, height) fractions of the page, so a
            reference survives a re-render at a different resolution.
        confidence: What the backend thought of its own reading, 0 to 1.
        kind: What the layout pass took it for.
    """

    text: str = ""
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    kind: RegionKind = RegionKind.BODY

    def reference(self, page: int) -> str:
        """Render the citation for this region: page, then where on it."""
        left, top, _, _ = self.box
        return f"p{page}@{left:.2f},{top:.2f}"


class OcrPage(AdkModel):
    """One page, read.

    Args:
        number: Its page number, one-based, as the document numbers it.
        regions: What was on it, in reading order.
    """

    number: int = Field(ge=1)
    regions: tuple[Region, ...] = ()

    @property
    def text(self) -> str:
        """The page as one string, in reading order."""
        return "\n".join(region.text for region in self.regions if region.text)

    @property
    def scripts(self) -> tuple[str, ...]:
        """Every writing system on the page, sorted, for a mixed-script document."""
        return tuple(sorted({_script(char) for char in self.text if _script(char)}))

    @property
    def reference(self) -> str:
        """The citation for the page as a whole."""
        return f"p{self.number}"


class Segment(AdkModel):
    """One span of a recording, transcribed.

    Args:
        start: Where it begins, in seconds from the start of the recording.
        end: Where it ends.
        text: What was said.
        speaker: Who said it, where the backend labelled it. Diarisation quality is the
            backend's business; carrying the label is this module's.
    """

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str = ""
    speaker: str = ""

    def reference(self) -> str:
        """Render the citation: the span it was said in, to the second."""
        return f"{_timestamp(self.start)}-{_timestamp(self.end)}"


@runtime_checkable
class OcrBackend(Protocol):
    """Turns a document on disk into pages. PaddleOCR on the CPU, or anything shaped so."""

    def pages(self, path: Path) -> AsyncIterator[OcrPage]:
        """Yield each page in order, without reading the whole document first."""
        ...


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Turns a recording on disk into segments. whisper.cpp on the CPU, or its shape."""

    def segments(self, path: Path) -> AsyncIterator[Segment]:
        """Yield each transcribed span in order."""
        ...


async def ocr_document(path: Path, *, backend: OcrBackend) -> AsyncIterator[OcrPage]:
    """Read `path` page by page.

    Args:
        path: The document. Its suffix decides whether the backend is asked at all.
        backend: What does the reading.

    Yields:
        Each page in order.

    Raises:
        MediaIntakeError: If the type is unsupported, the file is missing or empty, the
            backend failed, or the document turned out to hold no pages at all. The last is
            the important one: a document that read as nothing is not a document that read
            as blank.
    """
    _readable(path, SUPPORTED_DOCUMENTS, "document")
    read = 0
    try:
        async for page in backend.pages(path):
            read += 1
            yield page
    except MediaIntakeError:
        raise
    except Exception as broke:
        raise MediaIntakeError(
            f"{path} could not be read as a document: {broke}",
            path=str(path),
            reason="corrupt",
        ) from broke
    if not read:
        raise MediaIntakeError(f"{path} produced no pages at all", path=str(path), reason="empty")


async def transcribe_audio(path: Path, *, backend: TranscriptionBackend) -> AsyncIterator[Segment]:
    """Transcribe `path` segment by segment.

    Args:
        path: The recording.
        backend: What does the transcribing.

    Yields:
        Each segment in order, with the span it was said in.

    Raises:
        MediaIntakeError: If the type is unsupported, the file is missing or empty, the
            backend failed, or nothing at all came back.
    """
    _readable(path, SUPPORTED_AUDIO, "recording")
    read = 0
    try:
        async for segment in backend.segments(path):
            read += 1
            yield segment
    except MediaIntakeError:
        raise
    except Exception as broke:
        raise MediaIntakeError(
            f"{path} could not be transcribed: {broke}", path=str(path), reason="corrupt"
        ) from broke
    if not read:
        raise MediaIntakeError(
            f"{path} produced no transcript at all", path=str(path), reason="empty"
        )


def ocr_tool(
    backend: OcrBackend,
    *,
    root: Path,
    name: str = "read_document",
    max_chars: int = DEFAULT_INTAKE_CHARS,
) -> Tool[..., str]:
    """Build the tool that lets an agent read a scanned document.

    Args:
        backend: What does the reading.
        root: The only directory this tool will read from. A model names a file; what that
            is allowed to mean is decided here, not by the model.
        name: What the model calls it.
        max_chars: How much one call returns. A window, so a long contract cannot fill the
            context in a single call.

    Returns:
        The tool, holding the name until it is released.

    Example:
        >>> from pathlib import Path
        >>> class OnePage:
        ...     async def pages(self, path):
        ...         yield OcrPage(number=1, regions=(Region(text="clause one"),))
        >>> read = ocr_tool(OnePage(), root=Path("."), name="read_document_doc")
        >>> read.name
        'read_document_doc'
        >>> read.release()
    """

    @tool(name=name, idempotency="read_only")
    async def read_document(document: str, context: ToolContext, first_page: int = 1) -> str:
        """Read a scanned document, returning its text with a page reference per page.

        Args:
            document: The filename, inside the directory this tool reads from.
            context: The run this call belongs to.
            first_page: Which page to start at, for a document longer than one window.
        """
        del context
        path = _within(root, document, name)
        parts: list[str] = []
        length = 0
        try:
            async for page in ocr_document(path, backend=backend):
                if page.number < first_page:
                    continue
                rendered = f"[{page.reference}] {page.text}"
                if length + len(rendered) > max_chars:
                    break
                parts.append(rendered)
                length += len(rendered) + 1
        except MediaIntakeError as unreadable:
            raise ToolRefusal(name, "unreadable_document", str(unreadable)) from unreadable
        return "\n".join(parts)

    return read_document


def transcribe_tool(
    backend: TranscriptionBackend,
    *,
    root: Path,
    name: str = "transcribe_recording",
    max_chars: int = DEFAULT_INTAKE_CHARS,
) -> Tool[..., str]:
    """Build the tool that lets an agent read a recording.

    Args:
        backend: What does the transcribing.
        root: The only directory this tool will read from.
        name: What the model calls it.
        max_chars: How much one call returns.

    Returns:
        The tool, holding the name until it is released.

    Example:
        >>> from pathlib import Path
        >>> class OneSpan:
        ...     async def segments(self, path):
        ...         yield Segment(start=0.0, end=2.0, text="I want a refund")
        >>> listen = transcribe_tool(OneSpan(), root=Path("."), name="transcribe_doc")
        >>> listen.name
        'transcribe_doc'
        >>> listen.release()
    """

    @tool(name=name, idempotency="read_only")
    async def transcribe_recording(recording: str, context: ToolContext) -> str:
        """Transcribe a recording, returning each span with the time it was said.

        Args:
            recording: The filename, inside the directory this tool reads from.
            context: The run this call belongs to.
        """
        del context
        path = _within(root, recording, name)
        parts: list[str] = []
        length = 0
        try:
            async for segment in transcribe_audio(path, backend=backend):
                who = f" {segment.speaker}:" if segment.speaker else ""
                rendered = f"[{segment.reference()}]{who} {segment.text}"
                if length + len(rendered) > max_chars:
                    break
                parts.append(rendered)
                length += len(rendered) + 1
        except MediaIntakeError as unreadable:
            raise ToolRefusal(name, "unreadable_recording", str(unreadable)) from unreadable
        return "\n".join(parts)

    return transcribe_recording


def _within(root: Path, given: str, tool_name: str) -> Path:
    """Resolve `given` inside `root`, or refuse. The model does not get to leave."""
    path = (root / given).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ToolRefusal(
            tool_name, "path_not_permitted", f"{given!r} is not inside the readable directory"
        )
    return path


def _readable(path: Path, supported: frozenset[str], what: str) -> None:
    """Refuse before the backend is asked, so an unsupported type costs nothing."""
    if path.suffix.lower() not in supported:
        offered = ", ".join(sorted(supported))
        raise MediaIntakeError(
            f"{path.suffix or 'a file with no suffix'} is not a {what} this reads; "
            f"supported: {offered}",
            path=str(path),
            reason="unsupported",
        )
    if not path.is_file():
        raise MediaIntakeError(f"no {what} at {path}", path=str(path), reason="missing")
    if path.stat().st_size == 0:
        raise MediaIntakeError(f"{path} is empty", path=str(path), reason="empty")


def _script(char: str) -> str:
    """The writing system one character belongs to, or "" for punctuation and digits.

    Read off the character's Unicode name, which starts with its script for every alphabet
    that matters here; `_SCRIPT_NAMES` covers the ones whose name is not the script.
    """
    if not char.isalpha():
        return ""
    first = unicodedata.name(char, "").split(" ")[0]
    return _SCRIPT_NAMES.get(first, first.title())


def _timestamp(seconds: float) -> str:
    """Whole seconds, as hh:mm:ss, which is what a person scrubs a recording by."""
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"
