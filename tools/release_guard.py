"""Refuse a tag that must not become a release.

Runs before the build, because everything after it is irreversible: an index artefact
cannot be replaced, only yanked and superseded, so a wrong version costs a version
number and a note to every consumer. See docs/releasing.md.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from tools.versions import release_segment

if TYPE_CHECKING:
    from collections.abc import Sequence

DISTRIBUTION = "tesserix-adk"
INDEX_JSON = f"https://pypi.org/pypi/{DISTRIBUTION}/json"
MAIN = "origin/main"
TIMEOUT_SECONDS = 15.0

# The only tag shape that releases: v1.2.3, optionally an alpha, beta or release candidate.
TAG = re.compile(r"v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)$")


class IndexUnavailableError(Exception):
    """Raised when the index cannot be asked what it already holds."""


def version_of(tag: str) -> str:
    """The version the tag names, without its `v`.

    Raises:
        ValueError: If the tag is not the documented release format.
    """
    found = TAG.fullmatch(tag)
    if found is None:
        raise ValueError(f"{tag!r} is not the documented tag format")
    return found.group(1)


def on_main(tag: str) -> bool:
    """Is the tagged commit an ancestor of main? A release off main is unreviewed code."""
    result = subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", f"{tag}^{{commit}}", MAIN],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _fetch(url: str, timeout: float) -> Any:  # noqa: ANN401
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
        raise IndexUnavailableError(url) from err


def released_versions() -> tuple[str, ...]:
    """Versions the index already holds. Empty before the first release."""
    try:
        payload = _fetch(INDEX_JSON, TIMEOUT_SECONDS)
    except IndexUnavailableError:
        return ()
    return tuple(sorted(payload.get("releases", {})))


def check(*, tag: str, on_main: bool, released: Sequence[str]) -> list[str]:
    """Return every reason this tag may not be released."""
    problems = []
    try:
        version = version_of(tag)
    except ValueError:
        return [
            f"{tag} is not the documented tag format: v1.2.3, optionally with an a1, b1 "
            f"or rc1 suffix"
        ]

    if not on_main:
        problems.append(f"{tag} is not on main: a release must be built from reviewed code")

    already = {release_segment(existing) for existing in released}
    if release_segment(version) in already:
        problems.append(
            f"{release_segment(version)} is already on the index, and index artefacts are "
            f"immutable: yank and supersede it with a new version instead"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Check a tag before anything irreversible happens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v0.3.0")
    args = parser.parse_args(argv)

    problems = check(tag=args.tag, on_main=on_main(args.tag), released=released_versions())
    if problems:
        sys.stderr.write(f"{args.tag} is not releasable:\n  " + "\n  ".join(problems) + "\n")
        return 1

    sys.stdout.write(f"releasing {version_of(args.tag)} from {args.tag}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
