"""`tesserix-adk evals` — compare, bootstrap, or promote an evaluation baseline.

This is the part of the eval gate that CI runs. The consumer's own harness measures the
suite and writes a baseline artefact; this reads two of them, applies the project's policy
and returns the exit code that blocks the merge, plus the comment that explains it.

It fails closed by design. A missing baseline, one recorded on another dataset version and
a declared metric nobody measured each exit non-zero with the command that fixes them,
because a gate that clears itself when the comparison is impossible is not a gate.

This application-wired command is not part of the self-contained dispatcher. `python -m
tesserix_adk.cli.evals` is what the reusable workflow calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import BaselineUnusableError, ConfigurationError
from tesserix_adk.evals.baseline import Baseline, BaselinePolicy, compare, promote
from tesserix_adk.evals.gate import Bypass

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

__all__ = ["main"]

OK = 0
REGRESSED = 1
MISUSED = 2
UNUSABLE = 3


def main(argv: Sequence[str], *, out: TextIO | None = None) -> int:
    """Run one `evals` command and return its exit code.

    Args:
        argv: Arguments after the program name, e.g. `["compare", "--baseline", "b.json"]`.
        out: Where to write. Absent, stdout.

    Returns:
        `0` where the merge may proceed, `1` where a declared metric regressed beyond its
        band, `2` for a command line this could not read, and `3` where the baseline could
        not decide anything — which is a refusal, not a pass.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    handlers = {"compare": _compare, "bootstrap": _bootstrap, "promote": _promote}
    try:
        return handlers[parsed.command](parsed, writer)
    except (BaselineUnusableError, ConfigurationError) as refused:
        remedy = getattr(refused, "remedy", "")
        writer.write(f"refused: {refused}\n" + (f"{remedy}\n" if remedy else ""))
        return UNUSABLE


def _compare(parsed: argparse.Namespace, writer: TextIO) -> int:
    """Judge a candidate against the baseline and report what moved."""
    report = compare(
        Baseline.read(parsed.baseline),
        Baseline.read(parsed.candidate),
        policy=_policy(parsed.policy),
        override=_override(parsed),
    )
    comment = report.comment(artefacts=parsed.artefacts)
    if parsed.comment is not None:
        parsed.comment.parent.mkdir(parents=True, exist_ok=True)
        parsed.comment.write_text(comment + "\n", encoding="utf-8")
    writer.write(json.dumps(report.as_dict(), indent=2) + "\n" if parsed.json else comment + "\n")
    return report.exit_code


def _bootstrap(parsed: argparse.Namespace, writer: TextIO) -> int:
    """Record the first baseline for a suite, which the gate will not infer."""
    if parsed.baseline.exists() and not parsed.force:
        writer.write(f"refused: {parsed.baseline} already exists; pass --force to replace it\n")
        return UNUSABLE
    candidate = Baseline.read(parsed.candidate)
    candidate.write(parsed.baseline)
    writer.write(
        f"recorded {candidate.provenance.suite} on dataset "
        f"{candidate.provenance.dataset_version} at {parsed.baseline}\n"
    )
    return OK


def _promote(parsed: argparse.Namespace, writer: TextIO) -> int:
    """Make the merged run the baseline, keeping the one it replaces for a rollback."""
    candidate = Baseline.read(parsed.candidate)
    previous = promote(candidate, parsed.baseline)
    kept = f", previous kept at {previous}" if previous else ""
    writer.write(f"promoted {candidate.provenance.agent_version or 'candidate'}{kept}\n")
    return OK


def _policy(path: Path | None) -> BaselinePolicy:
    """The project's policy, or an empty one that judges nothing and says so by passing."""
    if path is None:
        return BaselinePolicy()
    if not path.exists():
        raise BaselineUnusableError(
            f"no policy at {path}", path=str(path), reason="format", remedy="declare one in JSON"
        )
    return BaselinePolicy.model_validate_json(path.read_text(encoding="utf-8"))


def _override(parsed: argparse.Namespace) -> Bypass | None:
    """The recorded exception, refused unless it says who took it and why."""
    if not parsed.override:
        return None
    return Bypass(
        metrics=tuple(parsed.override),
        by=parsed.override_by,
        reason=parsed.override_reason,
        incident=parsed.incident,
    )


def _parser() -> argparse.ArgumentParser:
    """The command line, one subcommand per thing CI does with a baseline."""
    parser = argparse.ArgumentParser(prog="tesserix-adk evals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    judge = commands.add_parser("compare", help="judge a candidate against the baseline")
    judge.add_argument("--baseline", type=Path, required=True, help="the stored baseline")
    judge.add_argument("--candidate", type=Path, required=True, help="this run's baseline artefact")
    judge.add_argument("--policy", type=Path, default=None, help="the gate policy, as JSON")
    judge.add_argument("--comment", type=Path, default=None, help="where to write the PR comment")
    judge.add_argument("--artefacts", default="", help="base URL of this run's artefacts")
    judge.add_argument("--json", action="store_true", help="machine-readable output")
    judge.add_argument(
        "--override", action="append", default=[], help="excuse this metric, once per metric"
    )
    judge.add_argument("--override-by", default="", help="who took the override")
    judge.add_argument("--override-reason", default="", help="why they took it")
    judge.add_argument("--incident", default="", help="the incident it belongs to")

    for name, what in (
        ("bootstrap", "record the first baseline for a suite"),
        ("promote", "make this run the baseline, keeping the previous one"),
    ):
        command = commands.add_parser(name, help=what)
        command.add_argument("--candidate", type=Path, required=True, help="the measured run")
        command.add_argument("--baseline", type=Path, required=True, help="where it is stored")
        if name == "bootstrap":
            command.add_argument(
                "--force", action="store_true", help="replace a baseline that already exists"
            )
    return parser


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
