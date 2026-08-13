"""Assemble release notes from the repository rather than from memory.

Hand-written notes are always incomplete, and the entries that get left out are the
breaking ones — a consumer then meets the change as a failing test. So the notes come
from change fragments, commit subjects, the public API snapshot and the live deprecation
records, and a change nobody documented fails the release instead of shipping silently.

Fragment format is in docs/releasing.md. Run `make notes` to preview.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tools.api_surface import collect_surface
from tools.deprecations import collect as collect_deprecations
from tools.release_check import (
    SURFACE_PATH,
    ReleaseCheckError,
    latest_tag,
    parse_surface,
    read_at,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tesserix_adk.core.deprecation import Deprecation

ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS = ROOT / "changes"
CHANGELOG = ROOT / "CHANGELOG.md"

EXPERIMENTAL = "tesserix_adk.experimental"
UNRELEASED = "## [Unreleased]"

# Where the Unreleased body ends: the previous release, or the link definitions at the foot.
_AFTER_UNRELEASED = re.compile(r"^(?:## |\[[^\]]+\]: )", re.MULTILINE)

# The section each fragment kind is announced under. `removed` is breaking by definition,
# so it shares the section with the migration notes rather than getting its own.
SECTIONS = {
    "breaking": "Breaking changes",
    "removed": "Breaking changes",
    "added": "Added",
    "changed": "Changed",
    "fixed": "Fixed",
    "deprecated": "Deprecated",
    "reverted": "Reverted",
    "experimental": "Experimental",
}
BREAKING_KINDS = frozenset({"breaking", "removed"})
ORDER = (
    "Breaking changes",
    "Added",
    "Changed",
    "Fixed",
    "Deprecated",
    "Reverted",
    "Experimental",
)

# `<issue>.<kind>.md`, so the kind is greppable and two fragments cannot collide.
_FILENAME = re.compile(r"^(\d+)\.([a-z]+)\.md$")
_HEADER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<bang>!)?: (?P<text>.+)$")
_ISSUE = re.compile(r"#(\d+)")

# Commit messages contain newlines, so the log is delimited by characters they cannot.
_UNIT = "\x1f"
_RECORD = "\x1e"

_INTERNAL = "internal"
# An unrecognised type is not housekeeping, it is an unreadable subject: treating
# `core: tidy up` as internal would let an undocumented change through the gate.
_TYPES = {
    "feat": "added",
    "fix": "fixed",
    "refactor": "changed",
    "perf": "changed",
    "revert": "reverted",
    "docs": _INTERNAL,
    "chore": _INTERNAL,
    "test": _INTERNAL,
    "ci": _INTERNAL,
    "build": _INTERNAL,
    "style": _INTERNAL,
}


class NoteError(Exception):
    """Raised when a fragment cannot be read, which blocks the release."""


@dataclass(frozen=True)
class Fragment:
    """One documented, consumer-visible change, written when the change was made."""

    id: str
    kind: str
    text: str
    surface: str | None = None
    migration: str | None = None

    @property
    def experimental(self) -> bool:
        """Changes under the experimental namespace carry none of the stability promises."""
        return self.kind == "experimental" or (self.surface or "").startswith(EXPERIMENTAL)


@dataclass(frozen=True)
class Commit:
    """A commit in the range being released."""

    sha: str
    subject: str
    body: str = ""

    @property
    def issues(self) -> frozenset[str]:
        """Issue numbers this commit references, which a fragment can document.

        The body counts as well as the subject: `Closes #42` is where the convention puts
        the link, and a subject already pushed cannot be edited to move it there.
        """
        return frozenset(_ISSUE.findall(f"{self.subject}\n{self.body}"))


@dataclass(frozen=True)
class Subject:
    """A commit subject that followed the convention well enough to place."""

    kind: str
    text: str
    scope: str | None = None


def _fields(header: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in header.splitlines())
        if key.strip()
    }


def _fragment(path: Path) -> Fragment:
    matched = _FILENAME.match(path.name)
    if matched is None:
        raise NoteError(f"{path.name} is not named <issue>.<kind>.md")
    issue, kind = matched.groups()
    if kind not in SECTIONS:
        raise NoteError(f"{path.name}: {kind!r} is not a change kind ({', '.join(SECTIONS)})")

    body = path.read_text(encoding="utf-8")
    header = _HEADER.match(body)
    fields = _fields(header.group(1)) if header else {}
    text = body[header.end() :].strip() if header else body.strip()
    if not text:
        raise NoteError(
            f"{path.name} has no text: a fragment with nothing to say documents nothing"
        )
    if kind in BREAKING_KINDS and not fields.get("migration"):
        raise NoteError(
            f"{path.name} is breaking and has no `migration:` telling consumers what to do"
        )
    return Fragment(
        id=issue,
        kind=kind,
        text=text,
        surface=fields.get("surface"),
        migration=fields.get("migration"),
    )


def read_fragments(directory: Path) -> tuple[Fragment, ...]:
    """Every fragment in a directory, in issue order so two runs agree."""
    if not directory.is_dir():
        return ()
    paths = sorted(
        (p for p in directory.iterdir() if p.suffix == ".md" and p.name != "README.md"),
        key=lambda p: (int(m.group(1)) if (m := _FILENAME.match(p.name)) else 0, p.name),
    )
    return tuple(_fragment(path) for path in paths)


def parse_subject(subject: str) -> Subject | None:
    """Place a conventional commit subject, or None if it followed no convention."""
    matched = _SUBJECT.match(subject)
    if matched is None:
        return None
    if matched["bang"]:
        return Subject(kind="breaking", text=matched["text"], scope=matched["scope"])
    kind = _TYPES.get(matched["type"])
    if kind is None:
        return None
    return Subject(kind=kind, text=matched["text"], scope=matched["scope"])


def undocumented(commits: Sequence[Commit], fragments: Sequence[Fragment]) -> list[str]:
    """Return one message per change that would ship without a consumer-facing note."""
    documented = {fragment.id for fragment in fragments}
    with_migration = {f.id for f in fragments if f.migration}
    problems = []
    for commit in commits:
        parsed = parse_subject(commit.subject)
        if commit.issues & documented:
            missing = parsed is not None and parsed.kind == "breaking"
            if missing and not commit.issues & with_migration:
                problems.append(
                    f"{commit.sha} is breaking and its fragment carries no migration note"
                )
            continue
        if parsed is None:
            problems.append(
                f"{commit.sha} {commit.subject!r}: no change fragment and no conventional "
                f"subject, so nothing describes it to a consumer"
            )
        elif parsed.kind == "breaking":
            problems.append(
                f"{commit.sha} {commit.subject!r} is breaking and has no migration note: "
                f"add a changes/<issue>.breaking.md with a `migration:` field"
            )
    return problems


def assemble(*, fragments: Sequence[Fragment], commits: Sequence[Commit]) -> dict[str, list[str]]:
    """Group every entry under its section, once per change rather than once per commit."""
    sections: dict[str, list[str]] = {}
    covered: set[str] = set()

    for fragment in fragments:
        covered |= {fragment.id}
        section = "Experimental" if fragment.experimental else SECTIONS[fragment.kind]
        entry = fragment.text
        if fragment.surface:
            entry = f"**{fragment.surface}**: {entry}"
        if fragment.migration:
            entry = f"{entry}\n\n  *Migration:* {fragment.migration}"
        sections.setdefault(section, []).append(entry)

    for commit in commits:
        if commit.issues & covered:
            continue
        parsed = parse_subject(commit.subject)
        if parsed is None or parsed.kind == _INTERNAL:
            continue
        entry = f"**{parsed.scope}**: {parsed.text}" if parsed.scope else parsed.text
        sections.setdefault(SECTIONS[parsed.kind], []).append(entry)

    return sections


def surface_diff(*, baseline: dict[str, str], current: dict[str, str]) -> dict[str, list[str]]:
    """What the public API snapshot says changed, for attaching to the notes."""
    return {
        "added": sorted(set(current) - set(baseline)),
        "removed": sorted(set(baseline) - set(current)),
        "changed": sorted(k for k in set(baseline) & set(current) if baseline[k] != current[k]),
    }


def render(
    *,
    version: str,
    sections: dict[str, list[str]],
    diff: dict[str, list[str]],
    deprecations: Sequence[Deprecation],
) -> str:
    """Render the notes for one version. The same text goes to the release and the changelog."""
    lines = [f"## {version}", ""]
    for name in ORDER:
        entries = sections.get(name)
        if entries:
            lines += [f"### {name}", ""]
            lines += [f"- {entry}" for entry in entries]
            lines.append("")
    if not any(sections.get(name) for name in ORDER):
        lines += ["No consumer-visible changes.", ""]

    if any(diff.values()):
        lines += ["### Public API surface", ""]
        for label, names in (
            ("Added", diff["added"]),
            ("Removed", diff["removed"]),
            ("Changed", diff["changed"]),
        ):
            lines += [f"- {label}: `{name}`" for name in names]
        lines.append("")

    if deprecations:
        lines += ["### Deprecated, scheduled for removal", ""]
        lines += [
            f"- `{r.name}` — removed in {r.removal}, use `{r.alternative}`" for r in deprecations
        ]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def update_changelog(changelog: str, *, notes: str) -> str:
    """Put a rendered release section below the Unreleased heading, in place of its body."""
    if UNRELEASED not in changelog:
        raise NoteError(f"CHANGELOG.md has no {UNRELEASED} heading to release from")
    head, _, tail = changelog.partition(UNRELEASED)
    return f"{head}{UNRELEASED}\n\n{notes.rstrip()}\n\n{_kept(tail)}"


def _kept(tail: str) -> str:
    """Everything after the Unreleased body: earlier releases, and the link definitions.

    The body itself goes. Entries are written there by hand as each change merges and
    describe the same work the fragments do, so carrying them into the released section
    ships every entry twice — once derived, once as it was typed.
    """
    boundary = _AFTER_UNRELEASED.search(tail)
    return tail[boundary.start() :] if boundary else ""


def _history() -> tuple[Commit, ...]:
    """Commits since the last release tag, which is the range this version ships."""
    tag = latest_tag()
    span = f"{tag}..HEAD" if tag else "HEAD"
    result = subprocess.run(  # noqa: S603
        ["git", "log", "--no-merges", f"--format=%h %s{_UNIT}%b{_RECORD}", span],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    if result.returncode:
        return ()
    return tuple(
        Commit(sha=sha, subject=subject, body=body)
        for sha, subject, body in (_split(record) for record in result.stdout.split(_RECORD))
        if sha
    )


def _split(record: str) -> tuple[str, str, str]:
    sha, _, rest = record.strip().partition(" ")
    subject, _, body = rest.partition(_UNIT)
    return sha, subject, body.strip()


def _surface_diff_against_last_release() -> dict[str, list[str]]:
    """Empty before the first release, and empty rather than fatal if the snapshot is new."""
    tag = latest_tag()
    if tag is None:
        return {"added": [], "removed": [], "changed": []}
    try:
        baseline = parse_surface(read_at(tag, SURFACE_PATH))
    except ReleaseCheckError:
        return {"added": [], "removed": [], "changed": []}
    return surface_diff(baseline=baseline, current=collect_surface())


def _consumed(fragments: Iterable[Fragment]) -> None:
    """Delete released fragments: one left behind is announced again next release."""
    ids = {fragment.id for fragment in fragments}
    for path in FRAGMENTS.glob("*.md"):
        matched = _FILENAME.match(path.name)
        if matched and matched.group(1) in ids:
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    """Render the notes, or fold them into the changelog with `--release`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="the version being released")
    parser.add_argument("--dry-run", action="store_true", help="render to stdout and stop")
    parser.add_argument("--output", help="write the notes to a file, for the release body")
    parser.add_argument("--release", action="store_true", help="fold into CHANGELOG.md")
    args = parser.parse_args(argv)

    try:
        fragments = read_fragments(FRAGMENTS)
    except NoteError as err:
        sys.stderr.write(f"{err}\n")
        return 1

    commits = _history()
    problems = undocumented(commits, fragments)
    if problems:
        sys.stderr.write(
            "release blocked, undocumented changes:\n  " + "\n  ".join(problems) + "\n"
        )
        return 1

    notes = render(
        version=args.version,
        sections=assemble(fragments=fragments, commits=commits),
        diff=_surface_diff_against_last_release(),
        deprecations=collect_deprecations(),
    )
    if args.output:
        Path(args.output).write_text(notes, encoding="utf-8")
    if args.release:
        CHANGELOG.write_text(
            update_changelog(CHANGELOG.read_text(encoding="utf-8"), notes=notes), encoding="utf-8"
        )
        _consumed(fragments)
    if args.dry_run or not (args.output or args.release):
        sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
