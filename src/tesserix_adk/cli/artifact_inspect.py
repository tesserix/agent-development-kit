"""Inspect, diff, replay and export a committed local run artefact offline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.cli.artifacts import (
    ArtifactSummary,
    ArtifactTruncatedError,
    ArtifactVersionError,
    redacted_json,
    scan_artifact,
)
from tesserix_adk.cli.run_agent import LocalAgent
from tesserix_adk.core import ApprovalDecision, ApprovalGate, ApprovalRecord
from tesserix_adk.runtime import ProgressEvent, decode_progress
from tesserix_adk.testing import ReplayingProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import TextIO

__all__ = ["Resolve", "main"]

OK = 0
MISSING = 1
MISUSED = 2
UNREADABLE = 3
DIVERGED = 4

type Resolve = Callable[[str, ApprovalGate], LocalAgent]
"""Resolve the recorded target while replay remains bound to the recorded provider."""


async def main(
    argv: Sequence[str], *, resolve: Resolve | None = None, out: TextIO | None = None
) -> int:
    """Inspect one artefact and return a stable diagnostic exit code.

    Args:
        argv: Arguments after ``inspect``.
        resolve: Target resolver, required only by ``--replay``.
        out: Render destination. Defaults to stdout.

    Returns:
        ``0`` for a successful operation, ``1`` for a missing path, ``2`` for command
        misuse, ``3`` for an unreadable/incompatible artefact and ``4`` for replay or diff
        divergence.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    path = Path(parsed.artefact)
    if not await asyncio.to_thread(path.is_file):
        writer.write(f"no run artefact at {path}\n")
        return MISSING
    summary = _validated(path, writer)
    if summary is None:
        return UNREADABLE
    if parsed.diff:
        other_path = Path(parsed.diff)
        if not await asyncio.to_thread(other_path.is_file):
            writer.write(f"no comparison artefact at {other_path}\n")
            return MISSING
        other = _validated(other_path, writer)
        if other is None:
            return UNREADABLE
        difference = _first_difference(path, summary, other_path, other)
        writer.write(difference + "\n" if difference else "runs do not diverge\n")
        if difference:
            return DIVERGED
    else:
        _render(path, summary, parsed, writer)
    if parsed.export_eval_case is not None:
        destination = _export_path(path, parsed.export_eval_case)
        try:
            _export(destination, summary)
        except OSError as error:
            writer.write(f"eval case was not exported: {error}\n")
            return MISUSED
        writer.write(f"exported replay-backed eval case to {destination}\n")
    if parsed.replay:
        if resolve is None:
            writer.write("--replay needs an agent target resolver\n")
            return MISUSED
        matched = await _replay(summary, resolve=resolve, writer=writer)
        if not matched:
            return DIVERGED
    return OK


def _validated(path: Path, writer: TextIO) -> ArtifactSummary | None:
    """Validate before rendering any record, so partial evidence never looks complete."""
    try:
        return scan_artifact(path)
    except (ArtifactVersionError, ArtifactTruncatedError, OSError) as error:
        writer.write(f"cannot inspect {path}: {error}\n")
        return None


def _render(
    path: Path, summary: ArtifactSummary, parsed: argparse.Namespace, writer: TextIO
) -> None:
    """Stream matching events and then the bounded authoritative summary."""
    writer.write(
        f"format={summary.header.format}/{summary.header.version} "
        f"kit={summary.header.kit_version} agent={summary.header.agent} "
        f"tenant={summary.header.tenant}\n"
    )
    for event in _events(path):
        if _selected(event, parsed):
            writer.write(_event_line(event) + "\n")
    run = summary.run
    usage = run.get("usage", {})
    budget = run.get("budget")
    writer.write(
        f"terminal state={run.get('state')} input_tokens={_at(usage, 'input_tokens')} "
        f"output_tokens={_at(usage, 'output_tokens')} budget={budget}\n"
    )


def _selected(event: ProgressEvent, parsed: argparse.Namespace) -> bool:
    """Apply all requested filters conjunctively."""
    payload = event.model_dump(mode="json")
    if parsed.step is not None and event.sequence != parsed.step:
        return False
    if parsed.tool is not None and payload.get("tool") != parsed.tool:
        return False
    if parsed.since is not None and (event.at is None or event.at < parsed.since):
        return False
    contextual = event.kind in {"tool_call_started", "guardrail_decision"}
    failed = any(
        word in event.kind
        for word in ("failed", "error", "denied", "refused", "cancelled", "indeterminate")
    )
    return not parsed.errors_only or contextual or failed


def _event_line(event: ProgressEvent) -> str:
    """Render one already-redacted, validated progress event."""
    payload = redacted_json(event.model_dump(mode="json"))
    hidden = {"kind", "run_id", "sequence", "at"}
    detail = " ".join(f"{key}={value}" for key, value in payload.items() if key not in hidden)
    return f"#{event.sequence:04d} {event.kind}{' ' + detail if detail else ''}"


def _events(path: Path) -> Iterator[ProgressEvent]:
    """Yield events from an artefact that has already passed its full integrity scan."""
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("type") != "progress":
                continue
            payload = record.get("event")
            if not isinstance(payload, dict):  # pragma: no cover — full scan proved it
                continue
            event = decode_progress(payload)
            if event is not None:
                yield event


def _first_difference(
    first_path: Path,
    first: ArtifactSummary,
    second_path: Path,
    second: ArtifactSummary,
) -> str:
    """Name the first input, message, tool or progress divergence."""
    if first.header.input != second.header.input:
        return "first divergence: recorded inputs differ"
    for field in ("messages", "tool_calls"):
        left = first.run.get(field, [])
        right = second.run.get(field, [])
        if isinstance(left, list) and isinstance(right, list):
            for index, pair in enumerate(zip_longest(left, right, fillvalue=None)):
                if pair[0] != pair[1]:
                    return f"first divergence: {field}[{index}] {pair[0]!r} != {pair[1]!r}"
    for index, pair in enumerate(
        zip_longest(_signatures(first_path), _signatures(second_path), fillvalue=None)
    ):
        if pair[0] != pair[1]:
            return f"first divergence: progress[{index}] {pair[0]!r} != {pair[1]!r}"
    if first.run.get("state") != second.run.get("state"):
        return (
            f"first divergence: terminal state {first.run.get('state')!r} "
            f"!= {second.run.get('state')!r}"
        )
    return ""


def _signatures(path: Path) -> Iterator[tuple[str, object, object]]:
    """Yield behaviour-bearing progress fields without timings or run ids."""
    for event in _events(path):
        payload = event.model_dump(mode="json")
        yield event.kind, payload.get("tool"), payload.get("error")


async def _replay(summary: ArtifactSummary, *, resolve: Resolve, writer: TextIO) -> bool:
    """Replay recorded provider traffic against the currently installed runtime only."""
    if summary.cassette is None:
        writer.write("artefact has no provider cassette; replay cannot run offline\n")
        return False
    gate = _ReplayGate()
    try:
        target = resolve(summary.header.target, gate)
        provider = ReplayingProvider(summary.cassette)
        replayed = await target.runner(provider, gate).run(
            target.agent,
            summary.header.input,
            tenant=summary.header.tenant,
            user=summary.header.user,
        )
    except Exception as error:  # replay converts any current-code mismatch to evidence
        writer.write(f"replay diverged with {type(error).__name__}: {error}\n")
        return False
    recorded_state = summary.run.get("state")
    matched = replayed.state.value == recorded_state
    writer.write(
        f"replay terminal={replayed.state.value} recorded={recorded_state} "
        f"matched={'yes' if matched else 'no'} provider_calls={len(provider.served)}\n"
    )
    return matched


class _ReplayGate:
    """Replays never invent an old human decision that the artefact did not encode."""

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Deny a fresh approval request because offline replay has no approver."""
        return ApprovalDecision(
            record_id=record.id,
            granted=False,
            decided_by="system:offline-replay",
            decided_at=time.time(),
            reason="offline replay cannot manufacture an approval",
        )


def _export_path(source: Path, requested: str) -> Path:
    """Resolve the optional flag value to a deterministic adjacent path."""
    if requested:
        return Path(requested)
    return source.with_suffix(source.suffix + ".eval.json")


def _export(path: Path, summary: ArtifactSummary) -> None:
    """Write one self-contained, replay-backed golden eval case without overwriting."""
    if summary.cassette is None:
        raise OSError("artefact has no cassette to embed in an offline eval case")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": "tesserix-adk-eval-case",
        "version": 1,
        "id": f"{summary.header.agent}-{summary.run.get('id', 'recorded')}",
        "target": summary.header.target,
        "input": summary.header.input,
        "tenant": summary.header.tenant,
        "user": summary.header.user,
        "expected_state": summary.run.get("state"),
        "expected_output": summary.run.get("output"),
        "cassette": summary.cassette.model_dump(mode="json"),
    }
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(redacted_json(document), destination, sort_keys=True, indent=2, default=str)
        destination.write("\n")


def _at(value: object, key: str) -> object:
    """Read a key from a JSON mapping without trusting its shape."""
    return value.get(key, 0) if isinstance(value, dict) else 0


def _parser() -> argparse.ArgumentParser:
    """Build the ``tesserix-adk inspect`` command line."""
    parser = argparse.ArgumentParser(prog="tesserix-adk inspect", description=__doc__)
    parser.add_argument("artefact", help="committed run JSONL path")
    parser.add_argument("--step", type=int, help="only one progress sequence")
    parser.add_argument("--tool", help="only progress for this tool")
    parser.add_argument("--errors-only", action="store_true", help="only failure events")
    parser.add_argument("--since", type=float, help="only events at or after this Unix time")
    parser.add_argument("--diff", help="compare with another committed artefact")
    parser.add_argument("--replay", action="store_true", help="re-run from the embedded cassette")
    parser.add_argument(
        "--export-eval-case",
        nargs="?",
        const="",
        metavar="PATH",
        help="write a self-contained golden eval case (default: beside artefact)",
    )
    return parser
