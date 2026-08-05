"""Compare versions by the release they belong to.

A development build carries a suffix (`0.1.dev7+g1a2b3c4`) and a pre-release carries
another (`0.2.0rc1`), but every policy question — has this shipped, is this bump big
enough — is about the release numbers underneath.
"""

from __future__ import annotations

import re

__all__ = ["ReleaseVersionError", "parts", "release_segment"]

_RELEASE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


class ReleaseVersionError(ValueError):
    """Raised when a string names no release at all."""


def parts(version: str) -> tuple[int, int, int]:
    """Return `(major, minor, patch)`, padding a truncated development version with zeros."""
    found = _RELEASE.match(version)
    if found is None:
        raise ReleaseVersionError(f"{version!r} does not start with a release number")
    major, minor, patch = found.groups()
    return int(major), int(minor or 0), int(patch or 0)


def release_segment(version: str) -> str:
    """Render just the release numbers, dropping any pre-release or development suffix."""
    return ".".join(str(part) for part in parts(version))
