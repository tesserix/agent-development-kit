"""Public documentation must resolve to the repository it describes."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from tools.docs_symbols import unresolved
from tools.policy_pages import SECURITY_PAGE, render_security

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"
FRAMEWORK_INTEROP = DOCS / "framework-interop.md"
DOCS_MARKDOWN = tuple(sorted(DOCS.rglob("*.md")))
MARKDOWN = tuple(sorted((*ROOT.glob("*.md"), *DOCS_MARKDOWN)))
LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")
NAV_LINK = re.compile(r"^\s*-\s+[^:]+:\s+([^\s]+\.md)\s*$", re.MULTILINE)
REPOSITORY_BLOB = "https://github.com/tesserix/agent-development-kit/blob/main/"
NAMING_SURFACES = (
    ROOT / "README.md",
    DOCS / "agent-lifecycle.md",
    FRAMEWORK_INTEROP,
    DOCS / "google-adk.md",
    DOCS / "index.md",
    DOCS / "integrations.md",
    DOCS / "migration.md",
    ROOT / "src" / "tesserix_adk" / "cli" / "__main__.py",
)

INTEROP_ENTRY_POINTS = (
    ROOT / "README.md",
    DOCS / "index.md",
    DOCS / "integrations.md",
    DOCS / "agent-lifecycle.md",
    DOCS / "migration.md",
)


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


def test_every_documentation_page_is_reachable_from_the_navigation() -> None:
    navigation = MKDOCS.read_text(encoding="utf-8")
    reachable = {(DOCS / target).resolve() for target in NAV_LINK.findall(navigation)}
    pending = list(reachable)

    while pending:
        page = pending.pop()
        if not page.exists():
            continue
        for _raw, target in local_links(page):
            if target.suffix == ".md" and target.is_relative_to(DOCS) and target not in reachable:
                reachable.add(target)
                pending.append(target)

    orphaned = [
        page.relative_to(ROOT).as_posix()
        for page in DOCS_MARKDOWN
        if page.resolve() not in reachable
    ]
    assert orphaned == []


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


def test_every_explicit_adk_symbol_in_public_docs_resolves() -> None:
    assert unresolved(DOCS_MARKDOWN) == ()


def test_a_renamed_symbol_names_the_page_line_and_missing_name(tmp_path: Path) -> None:
    page = tmp_path / "guide.md"
    page.write_text("Use `tesserix_adk.core.SymbolRemovedInThisRelease`.\n", encoding="utf-8")
    findings = unresolved((page,))
    assert len(findings) == 1
    assert "guide.md:1" in findings[0]
    assert "tesserix_adk.core.SymbolRemovedInThisRelease" in findings[0]


def test_the_site_security_policy_is_derived_from_the_repository_policy() -> None:
    assert SECURITY_PAGE.read_text(encoding="utf-8") == render_security()


def test_framework_interop_is_reachable_from_every_adoption_path() -> None:
    assert FRAMEWORK_INTEROP.exists()
    for page in INTEROP_ENTRY_POINTS:
        assert "framework-interop.md" in page.read_text(encoding="utf-8"), page


def test_framework_interop_names_each_supported_boundary() -> None:
    text = FRAMEWORK_INTEROP.read_text(encoding="utf-8")
    expected = (
        "import_tool",
        "import_google_adk_tool",
        "wrap_agent_as_tool",
        "wrap_agent_as_subagent",
        "wrap_google_adk_agent",
        "export_as_tool",
        "export_as_mcp_tool",
        "export_as_a2a",
    )
    assert all(f"`{name}`" in text for name in expected)


def test_readme_offers_pip_and_uv_install_paths() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m pip install" in text
    assert "uv add" in text
    assert "import tesserix_adk" in text


@pytest.mark.parametrize(
    "page", NAMING_SURFACES, ids=lambda path: path.relative_to(ROOT).as_posix()
)
def test_public_naming_surfaces_do_not_use_the_ambiguous_adk_initialism(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    prose = re.sub(r"`[^`]+`", "", prose)
    assert not re.search(r"\bADK\b", prose)
