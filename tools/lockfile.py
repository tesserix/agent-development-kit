"""The resolved dependency graph, read from `uv.lock`.

Read from the lock rather than from a resolver run: no network, and it is the same graph
CI installs. Used to say who a finding actually reaches, which is the difference between
a release and a note in the tracker.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["Graph", "LockfileError", "project_graph"]

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "uv.lock"
PROJECT = "tesserix-adk"

RUNTIME = "runtime"


class LockfileError(Exception):
    """Raised when the lock cannot be read as a dependency graph."""


@dataclass(frozen=True)
class Graph:
    """Locked versions, and the labels each package is reachable from."""

    versions: dict[str, str]
    reach: dict[str, frozenset[str]]

    @classmethod
    def from_lock(cls, lock: dict[str, Any], root: str = PROJECT) -> Graph:
        """Build the graph from a parsed `uv.lock`.

        Raises:
            LockfileError: If the root package is absent, or an edge names a package the
                lock does not hold.
        """
        packages = {package["name"]: package for package in lock.get("package", [])}
        if root not in packages:
            raise LockfileError(f"{root} is not in the lock")

        versions = {
            name: package["version"]
            for name, package in packages.items()
            if name != root and "version" in package
        }

        reach: dict[str, set[str]] = {}
        for label, edges in _entry_points(packages[root]):
            _walk(packages, edges, label, reach)

        return cls(versions=versions, reach={name: frozenset(v) for name, v in reach.items()})

    def blast_radius(self, package: str) -> str:
        """Who receives this package, in the terms a consuming team needs to act on."""
        labels = self.reach.get(package)
        if labels is None:
            return f"{package} is not in the lock"
        if RUNTIME in labels:
            return "every consumer"

        extras = sorted(
            label.removeprefix("extra:") for label in labels if label.startswith("extra:")
        )
        if extras:
            return "consumers of the " + ", ".join(extras) + " extra"
        return "development only, not shipped to consumers"


def _entry_points(root: dict[str, Any]) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    yield RUNTIME, root.get("dependencies", [])
    for extra, edges in root.get("optional-dependencies", {}).items():
        yield f"extra:{extra}", edges
    for group, edges in root.get("dev-dependencies", {}).items():
        yield f"group:{group}", edges


def _walk(
    packages: dict[str, Any],
    edges: list[dict[str, Any]],
    label: str,
    reach: dict[str, set[str]],
) -> None:
    queue = [edge["name"] for edge in edges]
    while queue:
        name = queue.pop()
        if name not in packages:
            raise LockfileError(f"{name} is depended on but not in the lock")
        if label in reach.setdefault(name, set()):
            continue
        reach[name].add(label)
        queue.extend(edge["name"] for edge in packages[name].get("dependencies", []))


def project_graph() -> Graph:
    """The graph of this repository's own lock."""
    with LOCK.open("rb") as handle:
        return Graph.from_lock(tomllib.load(handle))
