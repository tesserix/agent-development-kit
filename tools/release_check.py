"""Refuse a release that breaks the versioning policy.

Compares the surface and deprecations published at the last tag with the ones about to
ship. A symbol that disappeared or changed shape needs a deprecation record that came
due, and a release that breaks anything needs the version bump the policy demands —
otherwise the breakage reaches consumers as a patch upgrade. See docs/versioning.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import TYPE_CHECKING

from tesserix_adk import __version__ as VERSION  # noqa: N812 — patched in tests
from tools.api_surface import collect_surface
from tools.deprecations import parse as parse_deprecations
from tools.release_guard import version_of
from tools.versions import parts as _parts

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.core.deprecation import Deprecation

SURFACE_PATH = "docs/api-surface.txt"
DEPRECATIONS_PATH = "docs/deprecations.md"


class ReleaseCheckError(Exception):
    """Raised when the released tree cannot be read well enough to check against."""


def parse_surface(snapshot: str) -> dict[str, str]:
    """Read a committed API snapshot back into the mapping the check compares."""
    return dict(line.split(" :: ", 1) for line in snapshot.splitlines() if " :: " in line)


def latest_tag() -> str | None:
    """The most recent release tag, or None before the first release."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def read_at(ref: str, path: str) -> str:
    """Read one path as it was at a git ref."""
    result = subprocess.run(  # noqa: S603
        ["git", "show", f"{ref}:{path}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ReleaseCheckError(f"cannot read {path} at {ref}: {result.stderr.strip()}")
    return result.stdout


def _breaking_release(old: tuple[int, ...], new: tuple[int, ...]) -> bool:
    """Before 1.0 the minor is the breaking channel; from 1.0 onward only a major is."""
    if new[0] == 0:
        return new[1] > old[1] and new[2] == 0
    return new[0] > old[0] and new[1:] == (0, 0)


def check(
    *,
    baseline: Mapping[str, str],
    current: Mapping[str, str],
    records: Sequence[Deprecation],
    baseline_version: str,
    version: str,
) -> list[str]:
    """Return every reason this release may not ship, in the order a releaser fixes them."""
    removed = sorted(set(baseline) - set(current))
    reshaped = sorted(k for k in set(baseline) & set(current) if baseline[k] != current[k])
    added = sorted(set(current) - set(baseline))
    breaking = removed + reshaped

    old, new = _parts(baseline_version), _parts(version)
    problems = []
    if (breaking or added) and new <= old:
        problems.append(f"version must increase: {baseline_version} is already released")
    if breaking and not _breaking_release(old, new):
        channel = "a minor" if new[0] == 0 else "a major"
        problems.append(
            f"removing or changing public surface requires {channel} release; {version} is not one"
        )
    if added and not (new[0] > old[0] or new[1] > old[1]):
        problems.append(f"adding public surface requires at least a minor release, not {version}")

    scheduled = {record.name: record for record in records}
    for name in breaking:
        record = scheduled.get(name)
        if record is None:
            problems.append(
                f"{name} is removed or changed with no deprecation record: consumers were "
                f"never warned, so this cannot ship"
            )
        elif _parts(record.removal) > new:
            problems.append(
                f"{name} promised removal in {record.removal}, but this release is {version}"
            )
    return problems


def _released_version(ref: str) -> str:
    """The version a ref released. A tag is the version; anything else has to say so."""
    try:
        return version_of(ref)
    except ValueError as err:
        raise ReleaseCheckError(f"{ref} is not a release tag, so it names no version") from err


def main(argv: list[str] | None = None) -> int:
    """Check the working tree against the last release, or an explicit `--ref`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="git ref of the released version to compare against")
    args = parser.parse_args(argv)

    ref = args.ref or latest_tag()
    if ref is None:
        sys.stdout.write("no released version to compare against; nothing to check.\n")
        return 0

    problems = check(
        baseline=parse_surface(read_at(ref, SURFACE_PATH)),
        current=collect_surface(),
        records=parse_deprecations(read_at(ref, DEPRECATIONS_PATH)),
        baseline_version=_released_version(ref),
        version=VERSION,
    )
    if problems:
        sys.stderr.write(f"release blocked against {ref}:\n  " + "\n  ".join(problems) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
