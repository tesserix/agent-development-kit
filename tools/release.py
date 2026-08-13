"""Pick the version a release is allowed to be, and drive the manual steps that cut it.

docs/versioning.md is unambiguous about which digit moves, and still easy to get wrong at
the one moment nobody can take it back: an index artefact is that version forever. So the
number is derived from what is actually pending — the change fragments and the public
surface diff — and the releaser confirms an answer rather than inventing one.

The bias is upward. Breaking work shipped as a patch reaches consumers as a broken build;
a fix shipped as a minor costs a version number. Where the fragments and the snapshot
disagree, the snapshot wins: a fragment is a claim, the snapshot is evidence, and a
consumer meets the evidence.

`--apply` does the two irreversible-ish local steps — folding the notes into the changelog
and consuming the fragments — and then prints the commit and tag commands rather than
running them. Pushing the tag is the publish, and that stays a decision somebody makes.
"""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from tools.release_check import latest_tag
from tools.release_guard import version_of
from tools.release_notes import BREAKING_KINDS, FRAGMENTS, read_fragments
from tools.release_notes import _surface_diff_against_last_release as surface_diff
from tools.release_notes import main as assemble
from tools.versions import ReleaseVersionError, parts

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tools.release_notes import Fragment

__all__ = ["Bump", "ReleaseError", "next_version", "plan", "why"]

FIRST_RELEASE = "0.1.0"

# How many non-breaking reasons the plan lists before it starts counting instead.
LISTED = 5

# Kinds that move more than the patch digit. `changed` does not appear: a behavioural
# change with an unchanged surface is either breaking, and says so, or it is a fix.
_SURFACE_KINDS = frozenset({"added", "deprecated"})


class Bump(IntEnum):
    """How far the number has to move. Ordered, because the largest reason decides."""

    PATCH = 0
    SURFACE = 1
    BREAKING = 2


class ReleaseError(Exception):
    """Raised when no version can be chosen for what is pending."""


def why(
    *,
    fragments: Sequence[Fragment],
    diff: Mapping[str, list[str]],
) -> dict[Bump, list[str]]:
    """Every reason the number has to move, grouped by how far.

    Returns:
        A mapping from bump to the reasons demanding it, empty where nothing is pending.
        A number nobody can argue with is a number nobody checks, so the reasons travel
        with it into the plan the releaser reads.
    """
    reasons: dict[Bump, list[str]] = {}
    for fragment in fragments:
        if fragment.kind in BREAKING_KINDS:
            reasons.setdefault(Bump.BREAKING, []).append(
                f"changes/{fragment.id}.{fragment.kind}.md is breaking"
            )
        elif fragment.kind in _SURFACE_KINDS:
            reasons.setdefault(Bump.SURFACE, []).append(
                f"changes/{fragment.id}.{fragment.kind}.md adds surface"
            )
        else:
            reasons.setdefault(Bump.PATCH, []).append(f"changes/{fragment.id}.{fragment.kind}.md")

    for name in diff["removed"]:
        reasons.setdefault(Bump.BREAKING, []).append(f"{name} is gone from the API snapshot")
    for name in diff["changed"]:
        reasons.setdefault(Bump.BREAKING, []).append(f"{name} changed shape in the API snapshot")
    if diff["added"]:
        reasons.setdefault(Bump.SURFACE, []).append(
            f"{len(diff['added'])} new names in the API snapshot"
        )
    return reasons


def next_version(
    *,
    current: str | None,
    fragments: Sequence[Fragment],
    diff: Mapping[str, list[str]],
) -> str:
    """The version this release is allowed to be.

    Args:
        current: The last released version, or None before the first release.
        fragments: What is pending in `changes/`.
        diff: The public API snapshot diff against the last release.

    Returns:
        The next version, per docs/versioning.md — before 1.0 the minor is the breaking
        channel, from 1.0 onward the major is, and new surface is never a patch.

    Raises:
        ReleaseError: Where `current` names no release, or a breaking fragment carries no
            migration note. A breaking entry without instructions is the failure the whole
            mechanism exists to prevent, so it stops the release rather than the review.
    """
    for fragment in fragments:
        if fragment.kind in BREAKING_KINDS and not fragment.migration:
            raise ReleaseError(
                f"changes/{fragment.id}.{fragment.kind}.md is breaking and has no migration "
                f"note; consumers cannot be told what to do"
            )
    if current is None:
        return FIRST_RELEASE

    try:
        major, minor, patch = parts(current)
    except ReleaseVersionError as err:
        raise ReleaseError(f"{current!r} names no released version to bump from") from err

    bump = max(why(fragments=fragments, diff=diff), default=Bump.PATCH)
    if bump is Bump.BREAKING and major == 0:
        return f"0.{minor + 1}.0"
    if bump is Bump.BREAKING:
        return f"{major + 1}.0.0"
    if bump is Bump.SURFACE:
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def plan(
    *,
    current: str | None,
    fragments: Sequence[Fragment],
    diff: Mapping[str, list[str]],
) -> str:
    """What the releaser reads before committing to a number.

    Breaking reasons are listed in full and the rest are counted. A hundred lines of
    `adds surface` is a page nobody reads, and the entry that needed reading is in it.
    """
    version = next_version(current=current, fragments=fragments, diff=diff)
    reasons = why(fragments=fragments, diff=diff)
    lines = [f"{current or 'first release'} -> {version}", ""]
    if not reasons:
        lines.append("nothing pending; the patch digit moves so the number is never reused.")
    for bump in sorted(reasons, reverse=True):
        listed = sorted(reasons[bump])
        lines.append(f"{bump.name.lower()}: {len(listed)}")
        shown = listed if bump is Bump.BREAKING else listed[:LISTED]
        lines += [f"  {reason}" for reason in shown]
        if len(listed) > len(shown):
            lines.append(f"  ... and {len(listed) - len(shown)} more")
    return "\n".join(lines).rstrip() + "\n"


def _released_version(tag: str | None) -> str | None:
    """The version the last tag released, or None before the first release."""
    if tag is None:
        return None
    try:
        return version_of(tag)
    except ValueError as err:
        raise ReleaseError(f"{tag} is not a release tag, so it names no version") from err


def _checked(version: str) -> str:
    """The version about to be tagged, or a refusal. `make release` defaults it to a word."""
    try:
        parts(version)
    except ReleaseVersionError as err:
        raise ReleaseError(f"{version!r} is not a version to release") from err
    return version


def _fold(version: str, *, notes: Path | None) -> int:
    """Fold the notes into the changelog and consume the fragments, via the one tool."""
    argv = ["--version", version, "--release"]
    if notes is not None:
        argv += ["--output", str(notes)]
    return assemble(argv)


def main(argv: list[str] | None = None) -> int:
    """Show the plan, or apply it with `--apply`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="release this version instead of the derived one")
    parser.add_argument("--apply", action="store_true", help="fold the notes and consume fragments")
    parser.add_argument("--notes", help="also write the release body to this file")
    args = parser.parse_args(argv)

    try:
        current = _released_version(latest_tag())
        fragments = read_fragments(FRAGMENTS)
        diff = surface_diff()
        version = args.version or next_version(current=current, fragments=fragments, diff=diff)
        _checked(version)
    except ReleaseError as err:
        sys.stderr.write(f"{err}\n")
        return 1

    sys.stdout.write(plan(current=current, fragments=fragments, diff=diff))
    if args.version:
        sys.stdout.write(f"\noverridden: releasing {version}\n")
    if not args.apply:
        sys.stdout.write(f"\nrun `make release VERSION={version}` to cut it.\n")
        return 0

    failed = _fold(version, notes=Path(args.notes) if args.notes else None)
    if failed:
        return failed

    # Neither is run here: the commit goes through review, and pushing the tag *is* the
    # publish. See docs/releasing.md.
    sys.stdout.write(
        f"\nchangelog folded and fragments consumed. To finish:\n"
        f"  git commit -am 'chore(release): notes for {version}'\n"
        f"  git tag -a v{version} -m 'v{version}'\n"
        f"  git push origin main && git push origin v{version}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
