"""Reading a trace file a colleague attached to a bug report."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from tesserix_adk.cli.trace import MISSING, MISUSED, OK, UNREADABLE, main
from tesserix_adk.observability import FILE_VERSION, RecordedSpan, TraceFile

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _file() -> TraceFile:
    return TraceFile.of(
        (
            RecordedSpan(span_id="root", name="adk.run", ended=2.0),
            RecordedSpan(
                span_id="call",
                parent_span_id="root",
                name="adk.tool",
                ended=2.0,
                attributes={"adk.error.type": "ToolTimeout", "http.authorization": "Bearer x"},
            ),
        )
    )


def _written(tmp_path: Path, document: str | None = None) -> Path:
    path = tmp_path / "trace.json"
    path.write_text(document if document is not None else _file().model_dump_json())
    return path


class TestReadingAFile:
    def test_a_saved_trace_is_drawn(self, tmp_path: Path) -> None:
        out = io.StringIO()
        assert main([str(_written(tmp_path))], out=out) == OK
        assert "adk.run" in out.getvalue()

    def test_the_failure_in_it_is_visible(self, tmp_path: Path) -> None:
        out = io.StringIO()
        main([str(_written(tmp_path))], out=out)
        assert "ToolTimeout" in out.getvalue()

    def test_a_secret_the_writer_never_redacted_is_not_in_the_file(self, tmp_path: Path) -> None:
        assert "Bearer x" not in _written(tmp_path).read_text()

    def test_the_view_states_the_redaction_that_produced_the_file(self, tmp_path: Path) -> None:
        out = io.StringIO()
        main([str(_written(tmp_path))], out=out)
        assert FILE_VERSION in out.getvalue()


class TestOptions:
    def test_depth_can_be_capped(self, tmp_path: Path) -> None:
        out = io.StringIO()
        main([str(_written(tmp_path)), "--depth", "1"], out=out)
        assert "hidden" in out.getvalue()

    def test_output_can_be_machine_readable(self, tmp_path: Path) -> None:
        out = io.StringIO()
        main([str(_written(tmp_path)), "--json"], out=out)
        assert json.loads(out.getvalue())[0]["name"] == "adk.run"

    def test_a_view_can_be_narrowed_to_one_kind_of_step(self, tmp_path: Path) -> None:
        out = io.StringIO()
        assert main([str(_written(tmp_path)), "--only", "adk.tool"], out=out) == OK

    def test_writing_defaults_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([str(_written(tmp_path))]) == OK
        assert "adk.run" in capsys.readouterr().out


class TestWhatCanGoWrong:
    def test_a_file_that_is_not_there_is_reported_rather_than_raised(self, tmp_path: Path) -> None:
        out = io.StringIO()
        assert main([str(tmp_path / "absent.json")], out=out) == MISSING
        assert "absent.json" in out.getvalue()

    def test_a_file_that_is_not_a_trace_is_reported(self, tmp_path: Path) -> None:
        out = io.StringIO()
        assert main([str(_written(tmp_path, "{]"))], out=out) == UNREADABLE

    def test_a_file_from_a_newer_version_says_so_rather_than_misreading_it(
        self, tmp_path: Path
    ) -> None:
        document = json.loads(_file().model_dump_json())
        document["version"] = "adk-trace/99"
        out = io.StringIO()
        assert main([str(_written(tmp_path, json.dumps(document)))], out=out) == UNREADABLE
        assert "version" in out.getvalue()

    def test_a_command_line_this_cannot_read_is_its_own_exit_code(self) -> None:
        assert main([]) == MISUSED
