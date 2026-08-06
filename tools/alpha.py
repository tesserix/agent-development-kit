"""Number the alpha that main publishes, and say which old ones to retire.

The alpha channel exists to get the kit in front of a consumer while the API can still
change. It is opt-in by PEP 440's own rule — a stable specifier never resolves a
pre-release — so the only way to land on one is to ask for it. See docs/stability.md.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from tools.release_guard import released_versions

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "STABILITY_LEVELS",
    "AlphaError",
    "next_alpha",
    "next_base",
    "released_versions",
    "stale",
]

FIRST_RELEASE = "0.1.0"
KEEP = 10

STABILITY_LEVELS = frozenset({"stable", "beta", "alpha", "experimental", "internal"})


class AlphaError(Exception):
    """Raised when no alpha can be numbered for the requested version."""


def _versions(released: Iterable[str]) -> list[Version]:
    parsed = []
    for text in released:
        try:
            parsed.append(Version(text))
        except InvalidVersion:
            continue
    return parsed


def _is_alpha(version: Version) -> bool:
    return version.pre is not None and version.pre[0] == "a"


def next_alpha(base: str, *, released: Sequence[str]) -> str:
    """The next alpha of `base`, following the highest alpha of that same base.

    Raises:
        AlphaError: If `base` has already shipped as a stable release.
    """
    target = Version(base)
    versions = _versions(released)

    if any(version == target and not version.is_prerelease for version in versions):
        raise AlphaError(f"{base} is already released: an alpha of it describes shipped work")

    numbers = [
        version.pre[1]
        for version in versions
        if version.pre is not None and version.pre[0] == "a" and version.base_version == base
    ]
    return f"{base}a{max(numbers, default=0) + 1}"


def next_base(released: Sequence[str]) -> str:
    """The version main is heading for: the next minor after the last stable release.

    Pre-1.0 the minor is the breaking channel, and after 1.0 a breaking change waits for a
    major that no alpha off main can decide on its own.
    """
    stable = [version for version in _versions(released) if not version.is_prerelease]
    if not stable:
        return FIRST_RELEASE
    latest = max(stable)
    return f"{latest.major}.{latest.minor + 1}.0"


def stale(*, released: Sequence[str], keep: int = KEEP) -> list[str]:
    """Alphas the index should stop carrying, oldest first.

    An alpha of a version that has since shipped is obsolete outright; otherwise only the
    most recent `keep` of the open version are worth keeping. Stable releases and release
    candidates are never touched.
    """
    versions = _versions(released)
    shipped = {version.base_version for version in versions if not version.is_prerelease}

    alphas = sorted(version for version in versions if _is_alpha(version))
    obsolete = [version for version in alphas if version.base_version in shipped]
    live = [version for version in alphas if version.base_version not in shipped]

    return [str(version) for version in obsolete + live[: max(len(live) - keep, 0)]]


def main(argv: list[str] | None = None) -> int:
    """Print the next alpha version, or the retention report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retention", action="store_true", help="name the alphas that should be yanked"
    )
    parser.add_argument("--keep", type=int, default=KEEP, help="alphas of the open version to keep")
    args = parser.parse_args(argv)

    released = released_versions()

    if args.retention:
        retire = stale(released=released, keep=args.keep)
        listed = ", ".join(retire) if retire else "nothing"
        sys.stdout.write(f"yank: {listed}\n")
        return 0

    try:
        version = next_alpha(next_base(released=released), released=released)
    except AlphaError as err:
        sys.stderr.write(f"{err}\n")
        return 1

    sys.stdout.write(f"{version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
