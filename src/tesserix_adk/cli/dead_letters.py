"""`tesserix-adk dead-letters` — inspect and replay failed work.

The recovery at three in the morning is where the second incident comes from, so this
command makes the safe thing the easy one: a listing that prints identifiers rather than
payloads, a dry run that is one flag away, and a replay that will not run without a name to
put in the audit record.

This application-wired command is not part of the self-contained dispatcher. Applications
may expose it under the project-qualified command.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from tesserix_adk.adapters.dead_letters import DeadLetterQuery
from tesserix_adk.core.errors import ScopeViolationError
from tesserix_adk.core.events import EventType

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

    from tesserix_adk.adapters.dead_letters import Replayer

__all__ = ["main"]

OK = 0
MISSING = 1
MISUSED = 2

_REFUSED = 3
"""A selection that reached past its tenant, which is an operator error worth its own code."""


async def main(argv: Sequence[str], *, replayer: Replayer, out: TextIO | None = None) -> int:
    """Run one `dead-letters` command and return its exit code.

    Args:
        argv: Arguments after the program name, e.g. `["list", "--tenant", "acme"]`.
        replayer: Bound to the deployment's store and its live consumer path.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the command did what it says, `1` where there is no such record, `2` for
        a command line this could not read, and `3` where the selection crossed a tenant.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    query = DeadLetterQuery(
        tenant=parsed.tenant,
        event_type=EventType(parsed.type) if getattr(parsed, "type", "") else None,
        group=getattr(parsed, "group", "") or "",
        limit=getattr(parsed, "limit", 100),
    )
    try:
        if parsed.command == "list":
            return await _list(query, replayer=replayer, writer=writer)
        if parsed.command == "show":
            return await _show(query, parsed.event_id, replayer=replayer, writer=writer)
        return await _replay(query, parsed, replayer=replayer, writer=writer)
    except ScopeViolationError as crossed:
        writer.write(f"refused: {crossed}\n")
        return _REFUSED


async def _list(query: DeadLetterQuery, *, replayer: Replayer, writer: TextIO) -> int:
    """Print the backlog as identifiers and counts, never as bodies."""
    records = await replayer.records(query)
    if not records:
        writer.write(f"nothing is dead-lettered for {query.tenant}\n")
        return OK
    for record in records:
        shown = record.inspected()
        writer.write(
            f"{shown['event_id']}  {shown['type']}  {shown['group']}  "
            f"{shown['reason']}  {shown['last_error']}  x{shown['attempts']}\n"
        )
    return OK


async def _show(
    query: DeadLetterQuery, event_id: str, *, replayer: Replayer, writer: TextIO
) -> int:
    """Print one record field by field, attribute names and not attribute values."""
    wider = query.model_copy(update={"limit": 500})
    found = [
        record for record in await replayer.records(wider) if record.envelope.event_id == event_id
    ]
    if not found:
        writer.write(f"no record is kept under {event_id!r}\n")
        return MISSING
    for field, value in found[0].inspected().items():
        writer.write(f"{field:<16}{value}\n")
    return OK


async def _replay(
    query: DeadLetterQuery,
    parsed: argparse.Namespace,
    *,
    replayer: Replayer,
    writer: TextIO,
) -> int:
    """Redeliver the selection, or say what a replay would have done."""
    if parsed.dry_run:
        plan = await replayer.plan(query)
        writer.write(f"would replay {plan.replayable}, {plan.remaining} beyond this batch\n")
        _refusals(plan.refusals, writer)
        return OK
    report = await replayer.replay(query, operator=parsed.by, reason=parsed.reason)
    writer.write(
        f"replayed {report.replayed}, suppressed {report.suppressed}, "
        f"failed {report.failed}, refused {report.refused}, "
        f"{report.remaining} beyond this batch  [{report.replay_id}]\n"
    )
    _refusals(report.refusals, writer)
    return OK


def _refusals(refusals: tuple[tuple[str, str], ...], writer: TextIO) -> None:
    """Name what was not redelivered, so an operator has something to act on."""
    for event_id, why in refusals:
        writer.write(f"    refused {event_id}  {why}\n")


def _parser() -> argparse.ArgumentParser:
    """The `dead-letters` command line."""
    parser = argparse.ArgumentParser(
        prog="tesserix-adk dead-letters", description="inspect and replay"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, what in (("list", "show the backlog"), ("replay", "redeliver the selection")):
        command = commands.add_parser(name, help=what)
        command.add_argument("--tenant", required=True, help="the isolation boundary")
        command.add_argument("--type", default="", help="one event type")
        command.add_argument("--group", default="", help="one consumer group")
        command.add_argument("--limit", type=int, default=100, help="how many, at most")

    replay = commands.choices["replay"]
    replay.add_argument("--by", required=True, help="who is running it, for the audit record")
    replay.add_argument("--reason", default="", help="why, for the audit record")
    replay.add_argument("--dry-run", action="store_true", help="report without sending")

    showing = commands.add_parser("show", help="print one record")
    showing.add_argument("--tenant", required=True, help="the isolation boundary")
    showing.add_argument("--event-id", required=True, help="the record to print")
    return parser
