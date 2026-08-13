"""`adk approvals` — see what is waiting on a person, and answer it from a terminal.

An approval that stops a run for three days has to be answerable without the process that
asked still being alive, and a terminal is the one channel every operator already has. This
is the command surface for that; the kit cannot know where a deployment keeps its
suspensions or which agent to carry on, so the consumer supplies both and this supplies the
parsing, the rendering and the exit codes.

No third-party argument parser and no console-script entry point: a kit that installs a
global `adk` binary fights with whatever the consumer already ships.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Protocol

from tesserix_adk.core.errors import ApprovalTokenError
from tesserix_adk.core.suspension import PendingDecision

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import TextIO

    from tesserix_adk.core.suspension import SuspendedRun

__all__ = ["Answering", "Waiting", "main"]

OK = 0
REFUSED = 1
MISUSED = 2


class Waiting(Protocol):
    """Whatever can say which runs in a tenant are waiting on somebody."""

    async def pending(self, *, tenant: str) -> tuple[SuspendedRun, ...]:
        """Every run in `tenant` waiting on somebody, oldest first."""
        ...


type Answering = Callable[[str, bool, str, str], Awaitable[object]]
"""Take a decision: `(token, granted, decided_by, reason)`.

Bind it to `AgentRunner.resume_with_decision` with the agent and tenant the deployment
knows and this command cannot.
"""


async def main(
    argv: Sequence[str],
    *,
    waiting: Waiting,
    answering: Answering,
    out: TextIO | None = None,
) -> int:
    """Run one `approvals` command and return its exit code.

    Args:
        argv: Arguments after the program name, e.g. `["list", "--tenant", "acme"]`.
        waiting: Where suspended runs are kept.
        answering: What carries a run on once somebody has decided.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the command did what it says, `1` where a token bought nothing, and `2`
        for a command line this could not read.
    """
    writer = out if out is not None else sys.stdout
    parser = _parser()
    try:
        parsed = parser.parse_args(argv)
    except SystemExit:
        return MISUSED
    if parsed.command == "list":
        return await _list(parsed.tenant, waiting=waiting, writer=writer)
    return await _decide(parsed, answering=answering, writer=writer)


async def _list(tenant: str, *, waiting: Waiting, writer: TextIO) -> int:
    """Print what is on this tenant's rota, payload digests and not payloads."""
    held = await waiting.pending(tenant=tenant)
    if not held:
        writer.write(f"nothing is waiting on {tenant}\n")
        return OK
    for one in held:
        shown = PendingDecision.of(one)
        writer.write(
            f"{shown.run_id}  {shown.tool_name}  asked by {shown.agent_name}  "
            f"closes {shown.expires_at:.0f}  {shown.arguments_digest[:12]}\n"
        )
        writer.write(f"    {shown.reason}\n")
    return OK


async def _decide(parsed: argparse.Namespace, *, answering: Answering, writer: TextIO) -> int:
    """Answer one held call, and say plainly where the token bought nothing."""
    granted = parsed.command == "approve"
    try:
        await answering(parsed.token, granted, parsed.by, parsed.reason)
    except ApprovalTokenError as refused:
        writer.write(f"refused: {refused}\n")
        return REFUSED
    writer.write(f"{'approved' if granted else 'denied'} by {parsed.by}\n")
    return OK


def _parser() -> argparse.ArgumentParser:
    """The `approvals` command line."""
    parser = argparse.ArgumentParser(prog="adk approvals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="show the runs waiting on somebody")
    listing.add_argument("--tenant", required=True, help="the isolation boundary to look in")

    for name, what in (("approve", "let the held call run"), ("deny", "refuse the held call")):
        answering = commands.add_parser(name, help=what)
        answering.add_argument("--token", required=True, help="what the approver was handed")
        answering.add_argument("--by", required=True, help="who is deciding")
        answering.add_argument("--reason", default="", help="why, where there is one")
    return parser
