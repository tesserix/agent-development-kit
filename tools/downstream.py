"""Decide whether an alpha broke a consumer, or the consumer was already broken.

The consumer's suite runs twice — once against the last stable, once against the alpha —
and only a test that is newly failing is attributed to the kit. A job that goes red on
the consumer's own unrelated breakage is a job the team learns to ignore.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import ParseError

from defusedxml.ElementTree import parse as parse_xml

if TYPE_CHECKING:
    from collections.abc import Collection

__all__ = ["DownstreamError", "attributable", "failing"]

FAILED = ("failure", "error")


class DownstreamError(Exception):
    """Raised when a results file cannot be read as a verdict."""


def failing(results: Path) -> set[str]:
    """The `class::name` of every test that failed or errored in a JUnit XML report."""
    try:
        tree = parse_xml(results)
    except (OSError, ParseError) as err:
        raise DownstreamError(f"{results} cannot be read as a JUnit report") from err

    return {
        f"{case.get('classname', '')}::{case.get('name', '')}"
        for case in tree.iter("testcase")
        if any(case.find(outcome) is not None for outcome in FAILED)
    }


def attributable(*, baseline: Collection[str], candidate: Collection[str]) -> list[str]:
    """Tests that pass on the last stable and fail on the alpha — ours to explain."""
    return sorted(set(candidate) - set(baseline))


def _report(regressions: list[str]) -> str:
    listed = "\n".join(f"- `{test}`" for test in regressions)
    return (
        "# Downstream regressions\n\n"
        f"Failing under the alpha but not the last stable:\n\n{listed}\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Compare two consumer runs and fail only on a regression the alpha introduced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="JUnit XML from the last stable")
    parser.add_argument("--candidate", required=True, help="JUnit XML from the alpha")
    parser.add_argument("--report", help="write the regressions here for the release notes")
    args = parser.parse_args(argv)

    try:
        before = failing(Path(args.baseline))
        after = failing(Path(args.candidate))
    except DownstreamError as err:
        sys.stderr.write(f"{err}\n")
        return 1

    regressions = attributable(baseline=before, candidate=after)
    if not regressions:
        already = f"; {len(after)} still failing, not attributable to the alpha" if after else ""
        sys.stdout.write(f"no regression against the last stable{already}\n")
        return 0

    if args.report:
        Path(args.report).write_text(_report(regressions), encoding="utf-8")

    sys.stderr.write("the alpha broke tests that pass on the last stable:\n")
    sys.stderr.write("".join(f"  {test}\n" for test in regressions))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
