"""`tesserix-adk trace` — read a recorded run, including one from a bug report.

A trace attached to a report is often the only account of a failure nobody can reproduce.
Reading it should not require standing up a collector, so this takes the file and draws it.

The file is redacted when it is written rather than when it is read, because a reader who
has the file already has whatever it carries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.observability.local_view import (
    TraceFile,
    assembled,
    machine_readable,
    rendered,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

__all__ = ["MISSING", "MISUSED", "OK", "UNREADABLE", "main"]

OK = 0
MISSING = 1
MISUSED = 2
UNREADABLE = 3


def main(argv: Sequence[str], *, out: TextIO | None = None) -> int:
    """Draw a recorded trace and return an exit code.

    Args:
        argv: Arguments after the program name, e.g. `["trace.json", "--depth", "3"]`.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the trace was drawn, `1` where the file is not there, `2` for a command
        line this could not read, and `3` where the file is not a trace this build reads —
        distinct from `1`, because a file written by a newer version is a different problem
        from a path somebody typed wrong.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    path = Path(parsed.path)
    try:
        document = path.read_text()
    except OSError:
        writer.write(f"no trace file at {parsed.path}\n")
        return MISSING
    try:
        recorded = TraceFile.model_validate_json(document)
    except ValueError as problem:
        writer.write(f"{parsed.path} is not a trace this build reads: {problem}\n")
        return UNREADABLE
    nodes = assembled(recorded.spans)
    if parsed.json:
        writer.write(machine_readable(nodes))
        return OK
    writer.write(f"{recorded.version} redacted={list(recorded.redaction.dropped)}\n")
    writer.write(rendered(nodes, depth=parsed.depth, only=parsed.only))
    return OK


def _parser() -> argparse.ArgumentParser:
    """The `trace` command line."""
    parser = argparse.ArgumentParser(prog="tesserix-adk trace", description="draw a recorded run")
    parser.add_argument("path", help="the trace file to read")
    parser.add_argument("--depth", type=int, default=None, help="how many levels to show")
    parser.add_argument(
        "--only",
        nargs="+",
        default=(),
        metavar="SPAN",
        help="keep only these span names; a failing step is kept regardless",
    )
    parser.add_argument("--json", action="store_true", help="write the tree as JSON")
    return parser
