"""Runnable command modules that need no application-specific storage wiring."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from tesserix_adk.cli.evals import main as evals_main
from tesserix_adk.cli.trace import main as trace_main

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]

MISUSED = 2


def main(argv: Sequence[str]) -> int:
    """Dispatch a self-contained command and return its exit code."""
    if not argv:
        sys.stderr.write("usage: python -m tesserix_adk.cli {evals,trace} ...\n")
        return MISUSED
    command, *arguments = argv
    if command == "evals":
        return evals_main(arguments)
    if command == "trace":
        return trace_main(arguments)
    sys.stderr.write(f"unknown command {command!r}; choose evals or trace\n")
    return MISUSED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
