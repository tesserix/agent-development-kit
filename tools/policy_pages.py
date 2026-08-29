"""Project canonical repository policies into versioned documentation pages."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECURITY_SOURCE = ROOT / "SECURITY.md"
SECURITY_PAGE = ROOT / "docs" / "security-policy.md"
GENERATED = "<!-- generated from SECURITY.md by tools.policy_pages; do not edit -->\n"


def render_security() -> str:
    """Render the root policy with repository-relative doc links made site-relative."""
    source = SECURITY_SOURCE.read_text(encoding="utf-8")
    return GENERATED + source.replace("](docs/", "](")


def main(argv: list[str] | None = None) -> int:
    """Check the projected page, or regenerate it with ``--write``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    rendered = render_security()
    if args.write:
        SECURITY_PAGE.write_text(rendered, encoding="utf-8")
        return 0
    committed = SECURITY_PAGE.read_text(encoding="utf-8") if SECURITY_PAGE.exists() else ""
    if committed == rendered:
        return 0
    diff = difflib.unified_diff(
        committed.splitlines(),
        rendered.splitlines(),
        fromfile=str(SECURITY_PAGE),
        tofile="generated/security-policy.md",
    )
    sys.stderr.write("\n".join(diff) + "\nrun `make policy-pages` to regenerate.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
