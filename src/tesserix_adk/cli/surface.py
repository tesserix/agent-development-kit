"""`adk surface` — the tools an agent actually has, and where each of them came from.

When an agent calls a tool nobody expected it to have, the question is which server put it
there and under what name. Reading the servers' own tool lists does not answer it: the
answer is what discovery resolved, after allowlists, prefixes and sanitisation. So the
resolved surface is printed, and `--pin` prints it in the form that gets committed next to
the agent, which is what makes a later change to a server an error rather than a surprise.

No third-party argument parser and no console-script entry point: a kit that installs a
global `adk` binary fights with whatever the consumer already ships. Where the servers are
and how they are reached is the deployment's business, so it supplies the resolution and
this supplies the rendering and the exit codes.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import TextIO

    from tesserix_adk.adapters.mcp_surface import ToolSurface

__all__ = ["Resolve", "main"]

OK = 0
MISSING = 1
MISUSED = 2

type Resolve = Callable[[], Awaitable[ToolSurface]]
"""Resolve the agent's whole tool surface, however its servers are reached."""


async def main(argv: Sequence[str], *, resolve: Resolve, out: TextIO | None = None) -> int:
    """Print one agent's resolved tool surface and return an exit code.

    Args:
        argv: Arguments after the program name, e.g. `["--server", "handbook"]`.
        resolve: Where the deployment resolves its surface.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the surface was printed, `1` where `--server` named one that contributed
        nothing, and `2` for a command line this could not read.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    surface = await resolve()
    if parsed.server:
        surface = surface.of_server(parsed.server)
        if not surface.entries:
            writer.write(f"no tools resolved from {parsed.server!r}\n")
            return MISSING
    writer.write(surface.pin().model_dump_json() + "\n" if parsed.pin else surface.report())
    return OK


def _parser() -> argparse.ArgumentParser:
    """The `surface` command line."""
    parser = argparse.ArgumentParser(
        prog="adk surface", description="show the tools an agent resolved, and from where"
    )
    parser.add_argument("--server", default="", help="only what this server contributed")
    parser.add_argument(
        "--pin", action="store_true", help="print the surface as a pin, to commit and hold to"
    )
    return parser
