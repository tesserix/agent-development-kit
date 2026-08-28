"""Public documentation must resolve to the repository it describes."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS_MARKDOWN = tuple(sorted(DOCS.rglob("*.md")))
MARKDOWN = tuple(sorted((*ROOT.glob("*.md"), *DOCS_MARKDOWN)))
LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")
REPOSITORY_BLOB = "https://github.com/tesserix/agent-development-kit/blob/main/"


def local_links(path: Path) -> list[tuple[str, Path]]:
    found = []
    for raw in LINK.findall(path.read_text(encoding="utf-8")):
        destination = raw.strip().strip("<>").split("#", 1)[0]
        if not destination or urlsplit(destination).scheme:
            continue
        relative = Path(unquote(destination))
        target = (
            ROOT / str(relative).lstrip("/") if relative.is_absolute() else path.parent / relative
        )
        found.append((raw, target.resolve()))
    return found


@pytest.mark.parametrize("page", MARKDOWN, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_every_local_documentation_link_has_a_target(page: Path) -> None:
    broken = [raw for raw, target in local_links(page) if not target.exists()]
    assert broken == []


@pytest.mark.parametrize("page", DOCS_MARKDOWN, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_deployed_documentation_does_not_link_outside_the_site(page: Path) -> None:
    escaped = [raw for raw, target in local_links(page) if not target.is_relative_to(DOCS)]
    assert escaped == []


@pytest.mark.parametrize("page", DOCS_MARKDOWN, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_repository_links_from_the_site_still_name_existing_files(page: Path) -> None:
    destinations = LINK.findall(page.read_text(encoding="utf-8"))
    broken = [
        raw
        for raw in destinations
        if raw.startswith(REPOSITORY_BLOB)
        and not (ROOT / unquote(raw.removeprefix(REPOSITORY_BLOB)).split("#", 1)[0]).exists()
    ]
    assert broken == []
