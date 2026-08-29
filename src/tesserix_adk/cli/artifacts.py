"""Versioned, redacted run artefacts shared by local runs, inspection and evals.

The format is newline-delimited JSON so an inspector can stream a run with tens of
thousands of events.  A checksum footer is the commit marker: without it the artefact is
typed as truncated and no reader is allowed to present its partial contents as complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, overload

from pydantic import BaseModel, Field, JsonValue, ValidationError

from tesserix_adk import __version__
from tesserix_adk.core import AdkError, AdkModel, Run
from tesserix_adk.core.redaction import scrub
from tesserix_adk.runtime import ProgressEvent, decode_progress
from tesserix_adk.testing import Cassette

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

__all__ = [
    "ARTIFACT_VERSION",
    "ArtifactHeader",
    "ArtifactSummary",
    "ArtifactTruncatedError",
    "ArtifactVersionError",
    "ArtifactWriter",
    "redacted_json",
    "scan_artifact",
]

FORMAT: Final = "tesserix-adk-run"
ARTIFACT_VERSION: Final = 1
SUPPORTED_MIN: Final = 1
SUPPORTED_MAX: Final = 1
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT_DIGEST_KEYS = frozenset({"messages", "tools", "output_schema", "hooks"})


class ArtifactVersionError(AdkError):
    """The artefact version lies outside this CLI's compatibility window.

    Args:
        found: Version declared by the artefact.
        minimum: Oldest version this CLI reads.
        maximum: Newest version this CLI reads.
    """

    def __init__(
        self, found: int, minimum: int = SUPPORTED_MIN, maximum: int = SUPPORTED_MAX
    ) -> None:
        self.found = found
        self.minimum = minimum
        self.maximum = maximum
        super().__init__(
            f"artefact version {found} is not supported by installed CLI {__version__}; "
            f"supported versions are {minimum}..{maximum}"
        )


class ArtifactTruncatedError(AdkError):
    """An artefact has no valid commit footer or ends part-way through a record.

    Args:
        path: File that did not contain one complete committed run.
        reason: Exact integrity check that failed.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"artefact {path} is truncated or corrupt: {reason}")


class ArtifactHeader(AdkModel):
    """Identity and replay input written before the first progress event.

    Args:
        format: Stable format discriminator.
        version: Integer wire-format version.
        kit_version: Tesserix Agent Development Kit version that wrote the run.
        target: Import reference used to resolve the local agent.
        input: Original input after mandatory secret and personal-data redaction.
        tenant: Isolation boundary used for the run.
        user: Acting principal after redaction, where supplied.
        agent: Agent declaration name.
    """

    format: Literal["tesserix-adk-run"] = FORMAT
    version: int = Field(ge=1)
    kit_version: str
    target: str
    input: str
    tenant: str
    user: str | None = None
    agent: str


@dataclass(frozen=True)
class ArtifactSummary:
    """A complete artefact's bounded metadata and final records.

    Progress events are deliberately not retained here; ``scan_artifact`` sends them to a
    callback as each line is read, keeping inspection memory bounded.
    """

    header: ArtifactHeader
    run: Mapping[str, object]
    cassette: Cassette | None
    event_count: int


class ArtifactWriter:
    """Append one redacted run artefact and commit it with an integrity footer.

    Raises:
        FileExistsError: The target already exists. Recording never overwrites incident
            evidence without an explicit consumer-side decision.
        OSError: The target cannot be opened, flushed or committed.
    """

    def __init__(self, path: Path, header: ArtifactHeader) -> None:
        self.path = path
        self._stream = path.open("x", encoding="utf-8", newline="\n")
        self._digest = hashlib.sha256()
        self._events = 0
        self._finished = False
        self._write({"type": "header", "header": redacted_json(header.model_dump(mode="json"))})

    def append(self, event: ProgressEvent) -> None:
        """Write one progress event after recursively scrubbing every string value."""
        if self._finished:
            raise RuntimeError("a committed artefact cannot accept another event")
        self._write({"type": "progress", "event": redacted_json(event.model_dump(mode="json"))})
        self._events += 1

    def finish[OutputT: BaseModel](
        self,
        run: Run[OutputT],
        *,
        cassette: Cassette | None = None,
    ) -> None:
        """Write the authoritative run and cassette, then its checksum commit marker."""
        if self._finished:
            raise RuntimeError("this artefact is already committed")
        self._write(
            {
                "type": "run",
                "run": redacted_json(run.model_dump(mode="json")),
                "cassette": redacted_json(cassette.model_dump(mode="json"))
                if cassette is not None
                else None,
            }
        )
        footer = _line(
            {"type": "complete", "events": self._events, "sha256": self._digest.hexdigest()}
        )
        self._stream.write(footer)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._finished = True

    def close(self) -> None:
        """Flush an unfinished artefact without inventing a completion marker."""
        if self._stream.closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()

    def _write(self, record: Mapping[str, object]) -> None:
        encoded = _line(record)
        self._stream.write(encoded)
        self._stream.flush()
        self._digest.update(encoded.encode())


def scan_artifact(
    path: Path, *, on_event: Callable[[ProgressEvent], None] | None = None
) -> ArtifactSummary:
    """Validate and stream a committed artefact.

    Args:
        path: JSONL artefact to read.
        on_event: Optional callback invoked once per decoded progress event.

    Returns:
        Bounded metadata plus the final run and optional cassette.

    Raises:
        ArtifactVersionError: The header is newer or older than the compatibility window.
        ArtifactTruncatedError: JSON, ordering, counts, checksum or the completion marker
            is invalid. No partial summary is returned.
        OSError: The file cannot be read.
    """
    digest = hashlib.sha256()
    header: ArtifactHeader | None = None
    run: Mapping[str, object] | None = None
    cassette: Cassette | None = None
    events = 0
    committed = False
    try:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ArtifactTruncatedError(
                        path, f"line {number} is incomplete JSON"
                    ) from error
                if not isinstance(record, dict):
                    raise ArtifactTruncatedError(path, f"line {number} is not an object")
                kind = record.get("type")
                if kind == "complete":
                    if committed or run is None:
                        raise ArtifactTruncatedError(path, "completion marker is out of order")
                    expected = record.get("sha256")
                    count = record.get("events")
                    if expected != digest.hexdigest():
                        raise ArtifactTruncatedError(path, "checksum does not match its contents")
                    if count != events:
                        raise ArtifactTruncatedError(path, "footer event count does not match")
                    committed = True
                    if next(source, None) is not None:
                        raise ArtifactTruncatedError(
                            path, "records exist after the completion marker"
                        )
                    break
                digest.update(line.encode())
                if kind == "header":
                    if header is not None or events or run is not None:
                        raise ArtifactTruncatedError(path, "header is missing or out of order")
                    try:
                        header = ArtifactHeader.model_validate(record.get("header"))
                    except ValidationError as error:
                        raise ArtifactTruncatedError(path, "header does not validate") from error
                    if not SUPPORTED_MIN <= header.version <= SUPPORTED_MAX:
                        raise ArtifactVersionError(header.version)
                elif kind == "progress":
                    if header is None or run is not None:
                        raise ArtifactTruncatedError(path, "progress event is out of order")
                    payload = record.get("event")
                    if not isinstance(payload, dict):
                        raise ArtifactTruncatedError(path, f"event {events} is not an object")
                    try:
                        decoded = decode_progress(payload)
                    except ValueError as error:
                        raise ArtifactTruncatedError(
                            path, f"event {events} does not validate"
                        ) from error
                    if decoded is not None and on_event is not None:
                        on_event(decoded)
                    events += 1
                elif kind == "run":
                    if header is None or run is not None:
                        raise ArtifactTruncatedError(path, "final run is missing or duplicated")
                    payload = record.get("run")
                    if not isinstance(payload, dict):
                        raise ArtifactTruncatedError(path, "final run is not an object")
                    run = payload
                    recorded = record.get("cassette")
                    if recorded is not None:
                        try:
                            cassette = Cassette.model_validate_json(json.dumps(recorded))
                        except ValidationError as error:
                            raise ArtifactTruncatedError(
                                path, "cassette does not validate"
                            ) from error
                else:
                    raise ArtifactTruncatedError(path, f"unknown record type {kind!r}")
    except UnicodeDecodeError as error:
        raise ArtifactTruncatedError(path, "file is not UTF-8") from error
    if header is None:
        raise ArtifactTruncatedError(path, "header is missing")
    if run is None:
        raise ArtifactTruncatedError(path, "final run is missing")
    if not committed:
        raise ArtifactTruncatedError(path, "completion marker is missing")
    return ArtifactSummary(header=header, run=run, cassette=cassette, event_count=events)


@overload
def redacted_json(value: Mapping[str, object]) -> dict[str, JsonValue]: ...


@overload
def redacted_json(value: list[object] | tuple[object, ...]) -> list[JsonValue]: ...


@overload
def redacted_json(value: str) -> str: ...


@overload
def redacted_json(value: bool) -> bool: ...


@overload
def redacted_json(value: int) -> int: ...


@overload
def redacted_json(value: float) -> float: ...


@overload
def redacted_json(value: None) -> None: ...


def redacted_json(value: object) -> JsonValue:
    """Recursively scrub strings while retaining their JSON structure for replay."""
    return _redacted_json(value, key="")


def _redacted_json(value: object, *, key: str) -> JsonValue:
    """Scrub a JSON node while preserving non-secret cassette fingerprints."""
    if isinstance(value, str):
        if key in _FINGERPRINT_DIGEST_KEYS and _DIGEST.fullmatch(value):
            return value
        return scrub(value)
    if isinstance(value, dict):
        return {
            str(child_key): _redacted_json(item, key=str(child_key))
            for child_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redacted_json(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [_redacted_json(item, key=key) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub(str(value))


def _line(record: Mapping[str, object]) -> str:
    """Encode one canonical line; its bytes are what the footer authenticates."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n"
