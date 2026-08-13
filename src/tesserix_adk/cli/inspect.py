"""`adk inspect` — one multi-agent run drawn as a tree, with the money on every node.

The question after a run that spanned a supervisor, three workers and a peer is always the
same: which agent spent it and where did the time go. Answering it from a trace UI means
having kept the trace and having sampled it; answering it here means having the tree, which
the kit builds whether or not anything was exported.

No third-party argument parser and no console-script entry point: a kit that installs a
global `adk` binary fights with whatever the consumer already ships. Where the trees are
kept is the deployment's business, so it supplies the lookup and this supplies the
rendering and the exit codes.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from tesserix_adk.observability.tree import render

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import TextIO

    from tesserix_adk.observability.tree import RunTree

__all__ = ["Lookup", "main"]

OK = 0
MISSING = 1
MISUSED = 2

type Lookup = Callable[[str], Awaitable[RunTree | None]]
"""Take a root run id and give back its tree, or None where nothing is kept under it."""


async def main(argv: Sequence[str], *, lookup: Lookup, out: TextIO | None = None) -> int:
    """Draw one run and return an exit code.

    Args:
        argv: Arguments after the program name, e.g. `["run_1"]`.
        lookup: Where the deployment keeps assembled trees.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the run was drawn, `1` where nothing is kept under that id — which is
        not the same as a run that spent nothing — and `2` for a command line this could
        not read.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    assembled = await lookup(parsed.run_id)
    if assembled is None:
        writer.write(f"no run is kept under {parsed.run_id!r}\n")
        return MISSING
    writer.write(render(assembled))
    if parsed.by:
        writer.write("\n")
        for key, totals in sorted(assembled.totals_by(*parsed.by).items()):
            writer.write(
                f"{'/'.join(key)}  {totals.cost.quantised(6).total} {totals.cost.currency}  "
                f"{totals.calls} calls\n"
            )
    return OK


def _parser() -> argparse.ArgumentParser:
    """The `inspect` command line."""
    parser = argparse.ArgumentParser(prog="adk inspect", description="draw one run as a tree")
    parser.add_argument("run_id", help="the root run to draw")
    parser.add_argument(
        "--by",
        nargs="+",
        default=(),
        metavar="DIMENSION",
        help="also roll the local participants up by these attribution dimensions",
    )
    return parser
