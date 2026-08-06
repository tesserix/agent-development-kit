"""A bill of materials for what a consumer actually receives.

Built from `uv.lock`, so it describes the resolved graph rather than whatever happened to
be installed on the machine that ran it, and generated inside the release build so it
describes the artefact that was published. Development-only packages are left out: they
are not part of a consumer's exposure, and listing them overstates the surface.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.licences import declared, spdx
from tools.lockfile import LOCK, PROJECT, RUNTIME, Graph

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Difference", "build", "diff", "lock", "render"]

SPEC_VERSION = "1.6"
UNKNOWN_LICENCE = "NOASSERTION"


@dataclass(frozen=True)
class Difference:
    """What changed in the dependency graph between two releases."""

    total: int
    added: dict[str, str] = field(default_factory=dict)
    removed: dict[str, str] = field(default_factory=dict)
    changed: dict[str, tuple[str, str]] = field(default_factory=dict)


def lock() -> dict[str, Any]:
    """This repository's own parsed lock."""
    with LOCK.open("rb") as handle:
        parsed: dict[str, Any] = tomllib.load(handle)
        return parsed


def build(
    parsed: dict[str, Any],
    *,
    version: str,
    licence_of: Callable[[str], str | None] = declared,
    built_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """A CycloneDX document for one release, from a parsed lock."""
    graph = Graph.from_lock(parsed)
    packages = {package["name"]: package for package in parsed.get("package", [])}
    shipped = sorted(name for name, labels in graph.reach.items() if _profiles(labels))

    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "version": 1,
        "metadata": {
            "timestamp": (built_at or dt.datetime.now(dt.UTC)).isoformat(),
            "component": {
                "type": "library",
                "name": PROJECT,
                "version": version,
                "purl": f"pkg:pypi/{PROJECT}@{version}",
            },
        },
        "components": [
            _component(packages[name], graph.reach[name], licence_of) for name in shipped
        ],
    }


def _profiles(labels: frozenset[str]) -> list[str]:
    """Which install profiles reach a package, in the words a consumer installs with."""
    if RUNTIME in labels:
        return ["base"]
    return sorted(label for label in labels if label.startswith("extra:"))


def _component(
    package: dict[str, Any], labels: frozenset[str], licence_of: Callable[[str], str | None]
) -> dict[str, Any]:
    name, version = package["name"], package.get("version", "")
    wheels = package.get("wheels", [])

    properties = [{"name": "tesserix:profile", "value": profile} for profile in _profiles(labels)]
    properties += [
        {"name": "tesserix:platform", "value": platform} for platform in _platforms(wheels)
    ]
    properties += [
        {"name": "tesserix:artefact", "value": f"{_filename(wheel['url'])} {wheel['hash']}"}
        for wheel in wheels
    ]

    return {
        "type": "library",
        "name": name,
        "version": version,
        "purl": f"pkg:pypi/{name}@{version}",
        "licenses": [{"expression": _licence(name, licence_of)}],
        "hashes": _hashes(package.get("sdist")),
        "properties": properties,
    }


def _licence(name: str, licence_of: Callable[[str], str | None]) -> str:
    declaration = licence_of(name)
    return spdx(declaration) if declaration else UNKNOWN_LICENCE


def _hashes(sdist: dict[str, Any] | None) -> list[dict[str, str]]:
    if not sdist or "hash" not in sdist:
        return []
    algorithm, _, digest = str(sdist["hash"]).partition(":")
    return [{"alg": algorithm.upper().replace("SHA", "SHA-"), "content": digest}]


def _platforms(wheels: list[dict[str, Any]]) -> list[str]:
    """The platform tag of every wheel, so a native component is not described by one.

    A wheel filename ends `-<python>-<abi>-<platform>.whl`; `any` is pure Python.
    """
    return sorted({_filename(wheel["url"]).removesuffix(".whl").split("-")[-1] for wheel in wheels})


def _filename(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def diff(old: dict[str, Any], new: dict[str, Any]) -> Difference:
    """What a consumer's dependency set gains, loses and moves between two releases."""
    before = {item["name"]: item["version"] for item in old.get("components", [])}
    after = {item["name"]: item["version"] for item in new.get("components", [])}

    return Difference(
        total=len(after),
        added={name: version for name, version in after.items() if name not in before},
        removed={name: version for name, version in before.items() if name not in after},
        changed={
            name: (before[name], version)
            for name, version in after.items()
            if name in before and before[name] != version
        },
    )


def render(difference: Difference) -> str:
    """The dependency summary that goes into the release notes."""
    counts = ", ".join(
        f"{len(group)} {label}"
        for label, group in (
            ("added", difference.added),
            ("removed", difference.removed),
            ("updated", difference.changed),
        )
        if group
    )
    if not counts:
        return f"{difference.total} components, no dependency changes since the last release."

    lines = [f"{difference.total} components: {counts}.", ""]
    lines += [f"  + {name} {version}" for name, version in sorted(difference.added.items())]
    lines += [f"  - {name} {version}" for name, version in sorted(difference.removed.items())]
    lines += [
        f"  ~ {name} {was} → {now}" for name, (was, now) in sorted(difference.changed.items())
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the document for a release, or diff two of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="the release the document describes")
    parser.add_argument("--output", type=Path, help="where to write the document")
    parser.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"), type=Path)
    args = parser.parse_args(argv)

    if args.diff:
        old, new = (json.loads(path.read_text(encoding="utf-8")) for path in args.diff)
        sys.stdout.write(render(diff(old, new)) + "\n")
        return 0

    if not args.version or not args.output:
        parser.error("--version and --output are required unless --diff is given")

    document = build(lock(), version=args.version)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"{len(document['components'])} components written to {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
