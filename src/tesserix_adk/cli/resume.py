"""`tesserix-adk run --resume` — continue a checkpointed run.

The run that has been sitting on an approval for two days is usually resumed by an on-call
engineer, not by the service that started it, and what they need first is the answer to
"is this safe to carry on": which iteration it stopped at, what it is waiting for, and
whether anything outstanding cannot be decided. That summary is `describe`, and
`tesserix-adk inspect`
prints the same one off the same frontier, so the operator who looks before resuming and the
command that resumes are never reading two different things.

This application-wired command is not part of the self-contained dispatcher. Applications
may expose it under the project-qualified command, which avoids collisions with other
consumer already ships.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Protocol

from tesserix_adk.core.errors import (
    CheckpointFormatError,
    HistoryUnavailableError,
    RunLeaseError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import TextIO

    from tesserix_adk.core.checkpoint import Checkpoint
    from tesserix_adk.runtime.resume import ResumedRun

__all__ = ["Frontiers", "Resumes", "describe", "main", "show"]

OK = 0
MISSING = 1
MISUSED = 2
HELD = 3
UNSAFE = 4

type Frontiers = Callable[[str, str], Awaitable[Checkpoint | None]]
"""Take a run id and a tenant and give back its frontier, or None where there is not one."""


class Resumes(Protocol):
    """What this command needs of a resumer, satisfied by `tesserix_adk.runtime.Resumer`."""

    async def resume(self, run_id: str, *, tenant: str, worker: str) -> ResumedRun | None:
        """Take the run over, or return `None` because nothing was ever checkpointed."""
        ...


def describe(checkpoint: Checkpoint) -> str:
    """One frontier as an operator needs to read it before deciding anything.

    Example:
        >>> from tesserix_adk.core import Checkpoint
        >>> print(describe(Checkpoint(
        ...     run_id="r1", tenant="acme", agent_name="booking", iterations=3)))
        run r1  tenant acme  agent booking
        iteration 3  0 in, 0 out  0 micros
        waiting on nothing
    """
    waiting = (
        f"waiting on approval {checkpoint.pending_approval}"
        if checkpoint.pending_approval
        else "waiting on nothing"
    )
    return (
        f"run {checkpoint.run_id}  tenant {checkpoint.tenant}  agent {checkpoint.agent_name}\n"
        f"iteration {checkpoint.iterations}  {checkpoint.usage.input_tokens} in, "
        f"{checkpoint.usage.output_tokens} out  {checkpoint.cost_micros} micros\n"
        f"{waiting}"
    )


async def main(
    argv: Sequence[str], *, resumer: Resumes, out: TextIO | None = None, worker: str = "cli"
) -> int:
    """Resume one run and return an exit code.

    Args:
        argv: Arguments after the program name, e.g. `["--resume", "r1", "--tenant", "acme"]`.
        resumer: What takes the lease, reads the frontier and plans the outstanding calls.
        out: Where to write. Absent, stdout.
        worker: What this terminal calls itself in the lease, so the next operator to be
            refused is told who has the run.

    Returns:
        `0` where the run was taken over and is safe to carry on, `1` where nothing is
        checkpointed under that id, `2` for a command line this could not read, `3` where
        another worker holds the run, and `4` where the run must not be carried on — an
        undecidable call, an evicted transcript, or a frontier this kit cannot read.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    try:
        carried = await resumer.resume(parsed.run_id, tenant=parsed.tenant, worker=worker)
    except RunLeaseError as refused:
        writer.write(f"{parsed.run_id} is held by {refused.holder}\n")
        return HELD
    except CheckpointFormatError as refused:
        writer.write(
            f"{parsed.run_id} was checkpointed at format {refused.format_version}, "
            f"which this kit does not read\n"
        )
        return UNSAFE
    except HistoryUnavailableError as refused:
        writer.write(f"{parsed.run_id} points at transcript {refused.handle}, which has gone\n")
        return UNSAFE
    if carried is None:
        writer.write(f"no checkpoint is kept under {parsed.run_id!r}\n")
        return MISSING
    writer.write(describe(carried.checkpoint) + "\n")
    if not carried.plan.safe:
        undecided = ", ".join(one.call.name for one in carried.plan.indeterminate)
        writer.write(f"not safe to carry on: nothing can say whether {undecided} ran\n")
        return UNSAFE
    writer.write(f"held with fence {carried.lease.fence}\n")
    return OK


async def show(argv: Sequence[str], *, frontiers: Frontiers, out: TextIO | None = None) -> int:
    """Print a run's frontier without taking the lease or resuming anything.

    What `tesserix-adk inspect` calls for a run that has not finished. Reading a frontier is not
    resuming one, so this takes no lease and cannot refuse an operator who only wants to look.

    Returns:
        `0` where the frontier was printed, `1` where nothing is checkpointed under that
        id, and `2` for a command line this could not read.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser(resuming=False).parse_args(argv)
    except SystemExit:
        return MISUSED
    checkpoint = await frontiers(parsed.run_id, parsed.tenant)
    if checkpoint is None:
        writer.write(f"no checkpoint is kept under {parsed.run_id!r}\n")
        return MISSING
    writer.write(describe(checkpoint) + "\n")
    return OK


def _parser(*, resuming: bool = True) -> argparse.ArgumentParser:
    """The `run --resume` command line, and the read-only half of it."""
    parser = argparse.ArgumentParser(
        prog="tesserix-adk run" if resuming else "tesserix-adk inspect",
        description="carry a checkpointed run on" if resuming else "show a run's frontier",
    )
    if resuming:
        parser.add_argument("--resume", dest="run_id", required=True, help="the run")
    else:
        parser.add_argument("run_id", help="the run")
    parser.add_argument("--tenant", required=True, help="the tenant the run belongs to")
    return parser
