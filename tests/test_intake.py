"""Reading documents and audio, and refusing to call an unread file empty."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import MediaIntakeError
from tesserix_adk.tools import ToolContext, ToolRefusal
from tesserix_adk.tools.intake import (
    SUPPORTED_AUDIO,
    SUPPORTED_DOCUMENTS,
    OcrPage,
    Region,
    RegionKind,
    Segment,
    ocr_document,
    ocr_tool,
    transcribe_audio,
    transcribe_tool,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio

CONTEXT = ToolContext(run_id="r", tenant="acme")


class Paddle:
    """An OCR backend standing in for PaddleOCR, yielding a page at a time."""

    def __init__(self, pages: Sequence[OcrPage] = (), fails: Exception | None = None) -> None:
        self._pages = pages
        self._fails = fails
        self.opened: list[str] = []

    async def pages(self, path: Path) -> AsyncIterator[OcrPage]:
        self.opened.append(path.name)
        if self._fails is not None:
            raise self._fails
        for page in self._pages:
            yield page


class Whisper:
    """A transcription backend standing in for whisper.cpp."""

    def __init__(self, segments: Sequence[Segment] = (), fails: Exception | None = None) -> None:
        self._segments = segments
        self._fails = fails

    async def segments(self, path: Path) -> AsyncIterator[Segment]:
        del path
        if self._fails is not None:
            raise self._fails
        for segment in self._segments:
            yield segment


def page(number: int, *text: str) -> OcrPage:
    """One page of body text, laid out top to bottom."""
    return OcrPage(
        number=number,
        regions=tuple(
            Region(text=line, box=(0.1, 0.1 * index, 0.8, 0.08), confidence=0.97)
            for index, line in enumerate(text)
        ),
    )


def scanned(tmp_path: Path, name: str = "contract.pdf") -> Path:
    """A file that exists and is not empty, which is all the kit checks itself."""
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.7 scanned")
    return path


class TestReadingAMultiPageDocument:
    """The primary scenario: text out, with page and region references."""

    async def test_every_page_comes_back_in_order(self, tmp_path: Path) -> None:
        backend = Paddle([page(1, "clause one"), page(2, "clause two")])

        read = [p async for p in ocr_document(scanned(tmp_path), backend=backend)]

        assert [p.number for p in read] == [1, 2]
        assert read[0].text == "clause one"

    async def test_a_region_cites_its_page_and_place_on_it(self, tmp_path: Path) -> None:
        backend = Paddle([page(3, "the refund clause")])

        read = [p async for p in ocr_document(scanned(tmp_path), backend=backend)]

        assert read[0].regions[0].reference(3) == "p3@0.10,0.00"

    async def test_layout_kinds_survive_the_read(self, tmp_path: Path) -> None:
        heading = Region(
            text="Refunds", box=(0.1, 0.0, 0.8, 0.05), confidence=0.99, kind=RegionKind.HEADING
        )
        backend = Paddle([OcrPage(number=1, regions=(heading,))])

        read = [p async for p in ocr_document(scanned(tmp_path), backend=backend)]

        assert read[0].regions[0].kind is RegionKind.HEADING

    async def test_pages_are_yielded_not_accumulated(self, tmp_path: Path) -> None:
        backend = Paddle([page(n, f"page {n}") for n in range(1, 6)])

        seen = []
        async for read in ocr_document(scanned(tmp_path), backend=backend):
            seen.append(read.number)
            if len(seen) == 2:
                break

        assert seen == [1, 2]

    async def test_a_mixed_script_page_names_every_script_on_it(self, tmp_path: Path) -> None:
        backend = Paddle([page(1, "退款政策", "refund policy")])

        read = [p async for p in ocr_document(scanned(tmp_path), backend=backend)]

        assert read[0].scripts == ("Han", "Latin")

    async def test_a_blank_page_is_still_a_page(self, tmp_path: Path) -> None:
        backend = Paddle([OcrPage(number=1, regions=()), page(2, "clause")])

        read = [p async for p in ocr_document(scanned(tmp_path), backend=backend)]

        assert [p.number for p in read] == [1, 2]
        assert read[0].text == ""


class TestADocumentThatCannotBeRead:
    """The failure scenario: a typed error, never empty text passed off as a result."""

    async def test_an_unsupported_type_is_refused_before_the_backend(self, tmp_path: Path) -> None:
        backend = Paddle([page(1, "never reached")])
        path = scanned(tmp_path, "contract.docx")

        with pytest.raises(MediaIntakeError) as refused:
            [p async for p in ocr_document(path, backend=backend)]

        assert refused.value.reason == "unsupported"
        assert backend.opened == []

    async def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(MediaIntakeError) as refused:
            [p async for p in ocr_document(tmp_path / "gone.pdf", backend=Paddle())]

        assert refused.value.reason == "missing"

    async def test_an_empty_file_is_not_a_document(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")

        with pytest.raises(MediaIntakeError) as refused:
            [p async for p in ocr_document(path, backend=Paddle())]

        assert refused.value.reason == "empty"

    async def test_a_backend_that_falls_over_is_a_corrupt_document(self, tmp_path: Path) -> None:
        backend = Paddle(fails=ValueError("not a page tree"))

        with pytest.raises(MediaIntakeError) as refused:
            [p async for p in ocr_document(scanned(tmp_path), backend=backend)]

        assert refused.value.reason == "corrupt"

    async def test_a_document_with_no_pages_at_all_is_not_a_success(self, tmp_path: Path) -> None:
        with pytest.raises(MediaIntakeError) as refused:
            [p async for p in ocr_document(scanned(tmp_path), backend=Paddle([]))]

        assert refused.value.reason == "empty"

    async def test_a_backend_that_typed_the_problem_itself_is_not_relabelled(
        self, tmp_path: Path
    ) -> None:
        backend = Paddle(fails=MediaIntakeError("page 4 is encrypted", reason="corrupt"))

        with pytest.raises(MediaIntakeError) as refused:
            [p async for p in ocr_document(scanned(tmp_path), backend=backend)]

        assert str(refused.value) == "page 4 is encrypted"

    async def test_the_supported_types_are_stated(self) -> None:
        assert ".pdf" in SUPPORTED_DOCUMENTS
        assert ".wav" in SUPPORTED_AUDIO


class TestTranscription:
    """The same shape for audio: text with a timestamp to cite it by."""

    async def test_segments_carry_the_span_they_were_said_in(self, tmp_path: Path) -> None:
        path = tmp_path / "call.wav"
        path.write_bytes(b"RIFF....WAVE")
        backend = Whisper([Segment(start=83.0, end=89.5, text="I want a refund")])

        read = [s async for s in transcribe_audio(path, backend=backend)]

        assert read[0].reference() == "00:01:23-00:01:29"
        assert read[0].text == "I want a refund"

    async def test_a_speaker_is_carried_where_the_backend_gives_one(self, tmp_path: Path) -> None:
        path = tmp_path / "call.wav"
        path.write_bytes(b"RIFF....WAVE")
        backend = Whisper([Segment(start=0.0, end=1.0, text="hello", speaker="agent")])

        read = [s async for s in transcribe_audio(path, backend=backend)]

        assert read[0].speaker == "agent"

    async def test_an_unsupported_audio_type_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "call.mov"
        path.write_bytes(b"....")

        with pytest.raises(MediaIntakeError) as refused:
            [s async for s in transcribe_audio(path, backend=Whisper())]

        assert refused.value.reason == "unsupported"

    async def test_silence_is_not_a_transcript(self, tmp_path: Path) -> None:
        path = tmp_path / "call.wav"
        path.write_bytes(b"RIFF....WAVE")

        with pytest.raises(MediaIntakeError) as refused:
            [s async for s in transcribe_audio(path, backend=Whisper([]))]

        assert refused.value.reason == "empty"

    async def test_a_backend_that_typed_the_problem_itself_is_not_relabelled(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "call.wav"
        path.write_bytes(b"RIFF....WAVE")
        backend = Whisper(fails=MediaIntakeError("the codec is not one we read", reason="corrupt"))

        with pytest.raises(MediaIntakeError) as refused:
            [s async for s in transcribe_audio(path, backend=backend)]

        assert str(refused.value) == "the codec is not one we read"

    async def test_a_backend_that_falls_over_is_a_corrupt_recording(self, tmp_path: Path) -> None:
        path = tmp_path / "call.wav"
        path.write_bytes(b"RIFF....WAVE")
        backend = Whisper(fails=RuntimeError("no frames"))

        with pytest.raises(MediaIntakeError) as refused:
            [s async for s in transcribe_audio(path, backend=backend)]

        assert refused.value.reason == "corrupt"


class TestTheToolsAModelCalls:
    """A model names a file, not a path: the tool decides what that is allowed to mean."""

    async def test_a_document_comes_back_with_its_page_references(self, tmp_path: Path) -> None:
        scanned(tmp_path)
        read = ocr_tool(Paddle([page(1, "clause one"), page(2, "clause two")]), root=tmp_path)

        answer = await read.invoke({"document": "contract.pdf"}, CONTEXT)

        assert "[p1]" in answer
        assert "clause two" in answer
        read.release()

    async def test_only_the_asked_for_pages_are_returned(self, tmp_path: Path) -> None:
        scanned(tmp_path)
        read = ocr_tool(Paddle([page(n, f"page {n}") for n in range(1, 6)]), root=tmp_path)

        answer = await read.invoke({"document": "contract.pdf", "first_page": 2}, CONTEXT)

        assert "page 1" not in answer
        assert "page 2" in answer
        read.release()

    async def test_a_long_document_is_windowed_not_dumped(self, tmp_path: Path) -> None:
        scanned(tmp_path)
        backend = Paddle([page(n, "clause " * 200) for n in range(1, 40)])
        read = ocr_tool(backend, root=tmp_path, max_chars=500)

        answer = await read.invoke({"document": "contract.pdf"}, CONTEXT)

        assert len(answer) <= 500
        read.release()

    async def test_a_path_out_of_the_root_is_refused(self, tmp_path: Path) -> None:
        read = ocr_tool(Paddle([page(1, "secret")]), root=tmp_path / "docs")
        (tmp_path / "docs").mkdir()

        with pytest.raises(ToolRefusal) as refused:
            await read.invoke({"document": "../contract.pdf"}, CONTEXT)

        assert refused.value.code == "path_not_permitted"
        read.release()

    async def test_a_document_that_cannot_be_read_is_a_refusal_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        scanned(tmp_path, "contract.docx")
        read = ocr_tool(Paddle(), root=tmp_path)

        with pytest.raises(ToolRefusal) as refused:
            await read.invoke({"document": "contract.docx"}, CONTEXT)

        assert refused.value.code == "unreadable_document"
        read.release()

    async def test_transcription_comes_back_with_timestamps(self, tmp_path: Path) -> None:
        (tmp_path / "call.wav").write_bytes(b"RIFF....WAVE")
        listen = transcribe_tool(
            Whisper([Segment(start=0.0, end=2.0, text="I want a refund")]), root=tmp_path
        )

        answer = await listen.invoke({"recording": "call.wav"}, CONTEXT)

        assert "[00:00:00-00:00:02]" in answer
        assert "I want a refund" in answer
        listen.release()

    async def test_a_long_recording_is_windowed_too(self, tmp_path: Path) -> None:
        (tmp_path / "call.wav").write_bytes(b"RIFF....WAVE")
        heard = [
            Segment(start=float(n), end=float(n + 1), text="a long sentence " * 10)
            for n in range(60)
        ]
        listen = transcribe_tool(Whisper(heard), root=tmp_path, max_chars=400)

        answer = await listen.invoke({"recording": "call.wav"}, CONTEXT)

        assert len(answer) <= 400
        listen.release()

    async def test_an_unreadable_recording_is_a_refusal(self, tmp_path: Path) -> None:
        (tmp_path / "call.wav").write_bytes(b"RIFF....WAVE")
        listen = transcribe_tool(Whisper(fails=RuntimeError("no frames")), root=tmp_path)

        with pytest.raises(ToolRefusal) as refused:
            await listen.invoke({"recording": "call.wav"}, CONTEXT)

        assert refused.value.code == "unreadable_recording"
        listen.release()
