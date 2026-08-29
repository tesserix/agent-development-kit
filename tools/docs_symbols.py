"""Fail when an explicit ``tesserix_adk`` symbol in public documentation cannot resolve."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SYMBOL = re.compile(r"`(tesserix_adk(?:\.[A-Za-z_]\w*)+)`")


def _root(name: str) -> tuple[object, tuple[str, ...]] | None:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        try:
            return importlib.import_module(candidate), tuple(parts[end:])
        except ModuleNotFoundError as failure:
            if failure.name is not None and not candidate.startswith(failure.name):
                raise
    return None


def resolves(name: str) -> bool:
    """Whether a dotted module, export, model field or class member exists."""
    rooted = _root(name)
    if rooted is None:
        return False
    value, remaining = rooted
    for part in remaining:
        if hasattr(value, part):
            value = getattr(value, part)
            continue
        fields = getattr(value, "model_fields", {})
        if part in fields:
            value = fields[part]
            continue
        annotations = getattr(value, "__annotations__", {})
        if part in annotations:
            value = annotations[part]
            continue
        return False
    return True


def unresolved(pages: Sequence[Path]) -> tuple[str, ...]:
    """Return page-and-line findings for explicit ADK names that do not resolve."""
    findings: list[str] = []
    for page in pages:
        for line_number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for name in SYMBOL.findall(line):
                if not resolves(name):
                    findings.append(f"{page}:{line_number}: unresolved API symbol {name}")
    return tuple(findings)


def main() -> int:
    """Check every public Markdown page and print actionable findings."""
    findings = unresolved(tuple(sorted(DOCS.rglob("*.md"))))
    if findings:
        sys.stderr.write("\n".join(findings) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
