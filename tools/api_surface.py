"""Collect, render and check the kit's declared public API surface.

Run `python -m tools.api_surface --write` (or `make api-snapshot`) to regenerate
`docs/api-surface.txt`. CI runs the same collection and fails on any difference, so
a consumer-visible change cannot land without appearing in a reviewed diff.

The snapshot is deliberately textual rather than pickled: a reviewer reads the diff.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import tesserix_adk

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "api-surface.txt"

# Not part of the stability promise, so pinning it in the snapshot would be a lie.
EXCLUDED_PACKAGES = frozenset({"tesserix_adk.experimental"})

# A vendor type in a public signature couples every consumer to that vendor's releases.
VENDOR_MARKERS = (
    "httpx",
    "requests",
    "aiohttp",
    "openai",
    "anthropic",
    "boto3",
    "redis",
    "asyncpg",
    "psycopg",
    "sqlalchemy",
    "socket",
)

# Fakes are the testing package's product; anywhere else they are a substitutability leak.
CONCRETE_MARKERS = ("Fake",)

# Third-party names the kit re-exports on purpose. Adding one adopts that project's
# compatibility problems, so the list is short and reviewed.
RE_EXPORT_ALLOWLIST: frozenset[str] = frozenset()


class LeakError(Exception):
    """Raised when a public signature exposes a vendor or concrete implementation type."""

    def __init__(self, leaks: tuple[str, ...]) -> None:
        self.leaks = leaks
        super().__init__("public API surface leaks implementation types:\n  " + "\n  ".join(leaks))


def is_public_module(name: str) -> bool:
    """Is this dotted module name part of the promised surface?"""
    if any(part.startswith("_") for part in name.split(".")[1:]):
        return False
    return not any(name == p or name.startswith(f"{p}.") for p in EXCLUDED_PACKAGES)


def public_modules() -> list[ModuleType]:
    """Return every importable public module of the kit, excluding experimental."""
    modules: list[ModuleType] = [tesserix_adk]
    modules.extend(
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(tesserix_adk.__path__, "tesserix_adk.")
        if is_public_module(info.name)
    )
    return modules


def _describe(name: str, obj: object) -> str:
    # A typing alias is callable, so it must be recognised before the callable branch or it
    # renders as `def Layer(*args, **kwargs)` and the snapshot stops describing the contract.
    if getattr(obj, "__module__", "") == "typing":
        return f"{name} = {obj}"
    if inspect.isclass(obj):
        bases = ", ".join(b.__name__ for b in obj.__bases__)
        members = [
            f"{n}{inspect.signature(m)}"
            for n, m in sorted(vars(obj).items())
            if not n.startswith("_") and callable(m)
        ]
        body = "; ".join(members)
        return f"class {name}({bases})" + (f" {{{body}}}" if body else "")
    if callable(obj):
        return f"def {name}{inspect.signature(obj)}"
    return f"{name}: {type(obj).__name__}"


def collect_surface(modules: Sequence[ModuleType] | None = None) -> dict[str, str]:
    """Map every exported dotted name to a stable one-line description of its signature."""
    surface: dict[str, str] = {}
    for module in public_modules() if modules is None else modules:
        module_name = module.__name__
        for name in sorted(getattr(module, "__all__", [])):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            surface[f"{module_name}.{name}"] = _describe(name, obj)
    return surface


def render(surface: dict[str, str]) -> str:
    """Render the surface as the committed snapshot: one sorted line per symbol."""
    return "".join(f"{key} :: {surface[key]}\n" for key in sorted(surface))


def find_leaks(surface: dict[str, str]) -> list[str]:
    """Return one message per public symbol whose signature exposes an internal type."""
    leaks = []
    for key, description in sorted(surface.items()):
        if " = typing.Literal[" in description:
            continue
        in_testing = key.startswith("tesserix_adk.testing")
        markers = VENDOR_MARKERS if in_testing else VENDOR_MARKERS + CONCRETE_MARKERS
        symbol_name = key.rpartition(".")[2]
        signature = description.replace(symbol_name, "", 1)
        found = [m for m in markers if m in signature]
        if found:
            leaks.append(f"{key} exposes {', '.join(found)} in: {description}")
    return leaks


def third_party_re_exports(modules: Sequence[ModuleType] | None = None) -> set[str]:
    """Return exported names whose definition lives outside the kit.

    A type alias built from `typing` is defined here, not re-exported: `typing` is not a
    vendor whose release cadence we would be adopting.
    """
    outside = set()
    for module in public_modules() if modules is None else modules:
        module_name = module.__name__
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name, None)
            origin = getattr(obj, "__module__", "tesserix_adk")
            if not origin.startswith("tesserix_adk") and origin != "typing":
                outside.add(f"{module_name}.{name}")
    return outside


def main(argv: list[str] | None = None) -> int:
    """Check the snapshot, or rewrite it with `--write`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the snapshot")
    args = parser.parse_args(argv)

    surface = collect_surface()
    leaks = find_leaks(surface)
    if leaks:
        raise LeakError(tuple(leaks))

    rendered = render(surface)
    if args.write:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered, encoding="utf-8")
        return 0

    committed = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
    if rendered != committed:
        diff = difflib.unified_diff(
            committed.splitlines(), rendered.splitlines(), "committed", "current", lineterm=""
        )
        sys.stderr.write("\n".join(diff) + "\n")
        sys.stderr.write("\npublic API surface changed: run `make api-snapshot`.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
