"""Run artefacts are streamable, private, versioned and visibly complete."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tesserix_adk.cli.artifacts import (
    ArtifactHeader,
    ArtifactTruncatedError,
    ArtifactVersionError,
    ArtifactWriter,
    scan_artifact,
)
from tesserix_adk.core import Run, RunState, Usage
from tesserix_adk.runtime import AnswerDelta, RunCompleted

if TYPE_CHECKING:
    from pathlib import Path


def finished() -> Run:
    """One small authoritative run record."""
    return Run(
        id="run-1",
        tenant="local-dev",
        user="developer@example.com",
        agent_name="planner",
        agent_version="1.0.0",
        model="fake",
        state=RunState.COMPLETED,
    )


def header(*, version: int = 1) -> ArtifactHeader:
    """A header containing values that may never survive unredacted."""
    return ArtifactHeader(
        version=version,
        kit_version="0.52.0",
        target="demo:agent",
        input="use Bearer opaque-secret for developer@example.com",
        tenant="local-dev",
        user="developer@example.com",
        agent="planner",
    )


def test_a_complete_artifact_streams_events_and_contains_no_sensitive_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    writer = ArtifactWriter(path, header())
    writer.append(AnswerDelta(run_id="run-1", sequence=0, text="token sk-live-0123456789"))
    writer.append(
        RunCompleted(
            run_id="run-1",
            sequence=1,
            state=RunState.COMPLETED,
            usage=Usage(input_tokens=0, output_tokens=0),
        )
    )
    writer.finish(finished())
    kinds: list[str] = []

    summary = scan_artifact(path, on_event=lambda event: kinds.append(event.kind))

    assert kinds == ["answer_delta", "run_completed"]
    assert summary.event_count == 2
    assert summary.run["state"] == "completed"
    contents = path.read_text(encoding="utf-8")
    assert "opaque-secret" not in contents
    assert "developer@example.com" not in contents
    assert "sk-live-0123456789" not in contents


def test_an_uncommitted_artifact_is_typed_as_truncated(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = ArtifactWriter(path, header())
    writer.append(AnswerDelta(run_id="run-1", text="partial"))
    writer.close()

    with pytest.raises(ArtifactTruncatedError, match="final run is missing"):
        scan_artifact(path)


def test_a_newer_wire_version_names_the_compatibility_window(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    writer = ArtifactWriter(path, header(version=99))
    writer.finish(finished())

    with pytest.raises(ArtifactVersionError, match=r"99.*installed CLI.*1\.\.1"):
        scan_artifact(path)


def test_a_changed_committed_record_fails_integrity_before_rendering(tmp_path: Path) -> None:
    path = tmp_path / "changed.jsonl"
    writer = ArtifactWriter(path, header())
    writer.finish(finished())
    path.write_text(
        path.read_text(encoding="utf-8").replace('"agent":"planner"', '"agent":"other"', 1),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactTruncatedError, match="checksum"):
        scan_artifact(path)
