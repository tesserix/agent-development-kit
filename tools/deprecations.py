"""Generate and check `docs/deprecations.md` from the `@deprecate` records.

Run `make deprecations` to regenerate. CI runs the same collection and fails on any
difference, so the published list cannot disagree with the code, and fails again once a
promised removal version has shipped, so the list cannot become a graveyard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk import __version__ as VERSION  # noqa: N812 — patched in tests
from tesserix_adk.core.deprecation import Deprecation, deprecations
from tools.api_surface import public_modules

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "deprecations.md"

PREAMBLE = """# Deprecations

Every public name scheduled for removal, generated from the `@deprecate` records by
`make deprecations`. Do not edit by hand.

Set `TESSERIX_ADK_DEPRECATIONS_AS_ERRORS=1` in your CI to fail on these before the
removal lands. The policy behind the dates is in [versioning.md](versioning.md).

"""

_COLUMNS = (
    "| Symbol | Deprecated in | Removed in | Use instead | Why |\n| --- | --- | --- | --- | --- |\n"
)
_EMPTY = "No deprecations are live.\n"


def collect() -> tuple[Deprecation, ...]:
    """Import the public surface so every decorator registers, then read the registry."""
    public_modules()
    return deprecations()


def render(records: Sequence[Deprecation]) -> str:
    """Render the page. Reversible by `parse`, which the release check depends on."""
    if not records:
        return PREAMBLE + _EMPTY
    rows = "".join(
        f"| `{r.name}` | {r.since} | {r.removal} | `{r.alternative}` | {r.reason or '—'} |\n"
        for r in records
    )
    return PREAMBLE + _COLUMNS + rows


def parse(page: str) -> tuple[Deprecation, ...]:
    """Read back a rendered page, so a released page can be compared with a new one."""
    records = []
    for line in page.splitlines():
        if not line.startswith("| `"):
            continue
        name, since, removal, alternative, reason = (c.strip() for c in line.strip("|").split("|"))
        records.append(
            Deprecation(
                name=name.strip("`"),
                since=since,
                removal=removal,
                alternative=alternative.strip("`"),
                reason=None if reason == "—" else reason,
            )
        )
    return tuple(records)


def stale(records: Sequence[Deprecation], version: str) -> list[str]:
    """Return one message per deprecation whose promised removal has already shipped."""
    shipped = tuple(int(part) for part in version.split("."))
    return [
        f"{r.name} promised removal in {r.removal} but {version} has shipped: remove it, "
        f"or move the removal out and say why in the changelog"
        for r in records
        if tuple(int(part) for part in r.removal.split(".")) <= shipped
    ]


def main(argv: list[str] | None = None) -> int:
    """Check the page against the decorators, or rewrite it with `--write`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the page")
    args = parser.parse_args(argv)

    records = collect()
    overdue = stale(records, VERSION)
    if overdue:
        sys.stderr.write("\n".join(overdue) + "\n")
        return 1

    rendered = render(records)
    if args.write:
        PAGE.parent.mkdir(parents=True, exist_ok=True)
        PAGE.write_text(rendered, encoding="utf-8")
        return 0

    committed = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
    if rendered != committed:
        sys.stderr.write("docs/deprecations.md is out of date: run `make deprecations`.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
