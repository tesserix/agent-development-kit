"""Public adoption guidance and repository ownership stay enforceable."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_keeping_current_uses_reviewed_reproducible_updates() -> None:
    guide = (ROOT / "docs" / "keeping-current.md").read_text(encoding="utf-8")
    required = (
        "uv lock --upgrade-package tesserix-adk",
        ".github/dependabot.yml",
        "PUBLISH_ALPHAS",
        "DOWNSTREAM_REPO",
        "DeprecationWarning",
        "git revert",
    )

    assert [item for item in required if item not in guide] == []


def test_keeping_current_is_discoverable_from_public_entry_points() -> None:
    entry_points = (ROOT / "README.md", ROOT / "docs" / "index.md", ROOT / "mkdocs.yml")

    assert [
        path.name for path in entry_points if "keeping-current.md" not in path.read_text()
    ] == []


def test_install_guides_do_not_claim_an_unavailable_pypi_channel() -> None:
    pages = (ROOT / "README.md", ROOT / "docs" / "getting-started.md")

    assert [
        path.name
        for path in pages
        if "PyPI trusted publishing is not enabled" not in path.read_text()
    ] == []


def test_integration_guides_use_the_available_source_install_path() -> None:
    pages = (
        ROOT / "docs" / "a2a.md",
        ROOT / "docs" / "agentgateway.md",
        ROOT / "docs" / "integrations.md",
        ROOT / "docs" / "mcp-client.md",
    )

    assert [path.name for path in pages if "uv add 'tesserix-adk[" in path.read_text()] == []


def test_main_is_owned_and_documented_as_pull_request_only() -> None:
    owners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    governance = (ROOT / "docs" / "repository-governance.md").read_text(encoding="utf-8")

    assert "* @sam123ben @mahesh-sangawar" in owners
    assert "Do not push directly to `main`" in contributing
    assert "code-owner approval" in governance
    assert "no bypass actors" in governance


def test_alpha_guidance_distinguishes_build_from_publication() -> None:
    pages = (
        ROOT / "docs" / "stability.md",
        ROOT / "docs" / "contributing.md",
        ROOT / "docs" / "releasing.md",
    )

    assert [path.name for path in pages if "PUBLISH_ALPHAS" not in path.read_text()] == []
    assert [
        path.name for path in pages if "Every merge to `main` publishes" in path.read_text()
    ] == []


def test_public_issue_forms_route_bugs_features_and_security_reports() -> None:
    templates = ROOT / ".github" / "ISSUE_TEMPLATE"
    bug = yaml.safe_load((templates / "bug.yml").read_text(encoding="utf-8"))
    feature = yaml.safe_load((templates / "feature.yml").read_text(encoding="utf-8"))
    config = yaml.safe_load((templates / "config.yml").read_text(encoding="utf-8"))

    assert bug["name"] == "Bug report"
    assert feature["name"] == "Feature request"
    assert config["blank_issues_enabled"] is False
    assert any("security/advisories/new" in link["url"] for link in config["contact_links"])
