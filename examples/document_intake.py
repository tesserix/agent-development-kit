"""Reading a scanned document and a recorded call, and refusing what cannot be read.

Four scenarios: a multi-page document with page references; a mixed-script page; a file the
kit will not pretend it read; and the two tools a model calls, with the directory they are
allowed to read from.

Run it with `python examples/document_intake.py`. The backends here are local stand-ins —
a real deployment passes PaddleOCR and whisper.cpp, neither of which the kit installs.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.core import MediaIntakeError
from tesserix_adk.tools import (
    OcrPage,
    Region,
    RegionKind,
    Segment,
    ToolContext,
    ToolRefusal,
    ocr_document,
    ocr_tool,
    transcribe_audio,
    transcribe_tool,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CONTEXT = ToolContext(run_id="run-1", tenant="acme")


class Paddle:
    """A stand-in for PaddleOCR: three pages, one of them mixed-script."""

    async def pages(self, path: Path) -> AsyncIterator[OcrPage]:
        """Yield each page as it is read, never the whole document at once."""
        del path
        yield OcrPage(
            number=1,
            regions=(
                Region(
                    text="Refund policy",
                    box=(0.1, 0.05, 0.8, 0.04),
                    kind=RegionKind.HEADING,
                ),
                Region(text="Refunds are issued within 14 days.", box=(0.1, 0.12, 0.8, 0.05)),
            ),
        )
        mixed = Region(text="退款政策 refund policy", box=(0.1, 0.1, 0.8, 0.05))
        yield OcrPage(number=2, regions=(mixed,))
        yield OcrPage(number=3, regions=())


class Whisper:
    """A stand-in for whisper.cpp: two spans of a support call."""

    async def segments(self, path: Path) -> AsyncIterator[Segment]:
        """Yield each transcribed span with the time it was said."""
        del path
        yield Segment(start=83.0, end=89.5, text="I want a refund", speaker="caller")
        yield Segment(start=90.0, end=95.0, text="I can do that today", speaker="agent")


class Shredder:
    """A backend that gets part-way and gives up, the way a truncated PDF makes one."""

    async def pages(self, path: Path) -> AsyncIterator[OcrPage]:
        """Read one page, then fail."""
        del path
        yield OcrPage(number=1, regions=(Region(text="page one"),))
        raise ValueError("the page tree ends here")


def scanned(directory: Path, name: str = "contract.pdf") -> Path:
    """A file that exists and is not empty, which is all the kit checks itself."""
    path = directory / name
    path.write_bytes(b"%PDF-1.7 scanned")
    return path


async def a_document_reads_page_by_page(directory: Path) -> None:
    """Every page carries the reference an answer would have to cite."""
    async for page in ocr_document(scanned(directory), backend=Paddle()):
        print(f"{page.reference}: {page.text!r} scripts={page.scripts}")  # noqa: T201


async def a_recording_reads_span_by_span(directory: Path) -> None:
    """Same shape, timestamps instead of page numbers."""
    path = directory / "call.wav"
    path.write_bytes(b"RIFF....WAVE")

    async for segment in transcribe_audio(path, backend=Whisper()):
        print(f"{segment.reference()} {segment.speaker}: {segment.text}")  # noqa: T201


async def what_cannot_be_read_says_so(directory: Path) -> None:
    """Three refusals, each naming which check failed."""
    for path, backend in (
        (scanned(directory, "notes.docx"), Paddle()),
        (directory / "gone.pdf", Paddle()),
        (scanned(directory, "truncated.pdf"), Shredder()),
    ):
        try:
            async for _ in ocr_document(path, backend=backend):
                pass
        except MediaIntakeError as refused:
            print(f"{path.name}: {refused.reason}")  # noqa: T201


async def the_tools_a_model_calls(directory: Path) -> None:
    """A model names a file; the tool decides what that is allowed to mean."""
    scanned(directory)
    (directory / "call.wav").write_bytes(b"RIFF....WAVE")
    read = ocr_tool(Paddle(), root=directory, max_chars=200)
    listen = transcribe_tool(Whisper(), root=directory)

    print(await read.invoke({"document": "contract.pdf"}, CONTEXT))  # noqa: T201
    print(await listen.invoke({"recording": "call.wav"}, CONTEXT))  # noqa: T201
    try:
        await read.invoke({"document": "../../etc/passwd"}, CONTEXT)
    except ToolRefusal as refused:
        print(f"refused: {refused.code}")  # noqa: T201
    read.release()
    listen.release()


async def main() -> None:
    """Run every scenario in order, in a directory that goes away afterwards."""
    with tempfile.TemporaryDirectory() as made:
        directory = Path(made)
        await a_document_reads_page_by_page(directory)
        await a_recording_reads_span_by_span(directory)
        await what_cannot_be_read_says_so(directory)
        await the_tools_a_model_calls(directory)


if __name__ == "__main__":
    asyncio.run(main())
