"""`adk card` — the card this agent would publish, checked before a peer depends on it.

A card is generated, so the way it goes wrong is at generation: a skill exported that the
agent may not call, arguments that are not an object, a card over the published ceiling.
Every one of those is a startup failure in the deployment, which is a slow place to find
out. Rendering and linting it here makes publishing checkable from a terminal.

No third-party argument parser and no console-script entry point, as with `adk surface`:
where the definition and its exports come from is the deployment's business, so it supplies
the build and this supplies the rendering and the exit codes.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from tesserix_adk.a2a import MAX_CARD_BYTES, AgentCardError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TextIO

    from tesserix_adk.a2a import AgentCard

__all__ = ["Build", "main"]

OK = 0
INVALID = 1
MISUSED = 2

type Build = Callable[[], AgentCard]
"""Generate the card this deployment would serve, however its definition is assembled."""


async def main(argv: Sequence[str], *, build: Build, out: TextIO | None = None) -> int:
    """Render or lint one agent card and return an exit code.

    Args:
        argv: Arguments after the program name, e.g. `["--lint"]`.
        build: Where the deployment generates its card.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the card was rendered or passed the lint, `1` where it could not be
        generated as published, and `2` for a command line this could not read.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    try:
        card = build()
    except AgentCardError as refused:
        writer.write(f"{refused.skill}: {refused}\n" if refused.skill else f"{refused}\n")
        return INVALID
    writer.write(_report(card) if parsed.lint else card.rendered().decode() + "\n")
    return OK


def _report(card: AgentCard) -> str:
    """What a peer would receive, in the terms a reviewer checks before publishing."""
    lines = [
        f"{card.agent}@{card.version} for {card.audience}",
        f"{len(card.skills)} skills, {len(card.rendered())} of {MAX_CARD_BYTES} bytes",
    ]
    lines += [
        f"  {skill.name}"
        + (f" needs {' '.join(skill.required_scopes)}" if skill.required_scopes else "")
        + (" (approval)" if skill.requires_approval else "")
        for skill in card.skills
    ]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    """The `card` command line."""
    parser = argparse.ArgumentParser(
        prog="adk card", description="render or lint the agent card this deployment publishes"
    )
    parser.add_argument(
        "--lint", action="store_true", help="report what a peer would receive, rather than the card"
    )
    return parser
