"""The public documentation path is built, checked, and deployed with narrow authority."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from tests.ci_config import load_yaml, triggers

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
DOCS = WORKFLOWS / "docs.yml"
RELEASE = WORKFLOWS / "release.yml"
CI = WORKFLOWS / "ci.yml"
MAKEFILE = ROOT / "Makefile"
MKDOCS = ROOT / "mkdocs.yml"
RELEASING = ROOT / "docs" / "releasing.md"
SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _uses(value: object) -> list[str]:
    if isinstance(value, dict):
        found = [str(item) for key, item in value.items() if key == "uses"]
        return found + [use for item in value.values() for use in _uses(item)]
    if isinstance(value, list):
        return [use for item in value for use in _uses(item)]
    return []


EXTERNAL_ACTIONS = tuple(
    (workflow.name, use)
    for workflow in sorted(WORKFLOWS.glob("*.yml"))
    for use in _uses(load_yaml(workflow))
    if not use.startswith("./")
)


@pytest.mark.parametrize(
    ("workflow", "action"),
    EXTERNAL_ACTIONS,
    ids=lambda value: value.replace("/", "-"),
)
def test_every_external_action_is_pinned_to_a_commit(workflow: str, action: str) -> None:
    del workflow
    assert SHA_PIN.fullmatch(action)


def test_pull_requests_and_main_build_docs_without_deploying() -> None:
    declared = triggers(DOCS)
    assert "pull_request" in declared
    assert declared["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in declared

    jobs: dict[str, Any] = load_yaml(DOCS)["jobs"]
    assert set(jobs) == {"build"}
    assert "upload-pages-artifact" not in str(jobs)
    assert "deploy-pages" not in str(jobs)


def test_the_build_checks_links_and_mkdocs_strictly() -> None:
    jobs: dict[str, Any] = load_yaml(DOCS)["jobs"]
    commands = [step["run"] for step in jobs["build"]["steps"] if "run" in step]
    assert any("tests/test_documentation.py" in command for command in commands)
    assert any("mkdocs build --strict" in command for command in commands)


def test_the_build_has_no_write_permissions() -> None:
    workflow = load_yaml(DOCS)
    assert workflow["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in workflow["jobs"].values())


def test_releases_publish_versioned_docs_and_preserve_earlier_versions() -> None:
    jobs: dict[str, Any] = load_yaml(RELEASE)["jobs"]
    publish = jobs["docs-publish"]
    commands = "\n".join(
        step["run"] for step in publish["steps"] if isinstance(step, dict) and "run" in step
    )

    assert publish["needs"] == ["build", "mirror"]
    assert publish["permissions"] == {"contents": "write"}
    assert "mike deploy" in commands
    assert "--push" in commands
    assert "--update-aliases" in commands
    assert "stable" in commands
    assert "(unstable)" in commands
    assert "mike set-default" in commands
    assert "git archive" in commands

    deploy = jobs["docs-deploy"]
    assert deploy["needs"] == "docs-publish"
    assert deploy["permissions"] == {
        "pages": "write",
        "id-token": "write",
    }


def test_the_release_runbook_names_every_release_job() -> None:
    documented = RELEASING.read_text(encoding="utf-8")
    missing = [job for job in load_yaml(RELEASE)["jobs"] if f"`{job}`" not in documented]
    assert missing == []


def test_generated_api_reference_is_a_local_and_ci_drift_gate() -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "api-reference:" in makefile
    assert "api-reference-check:" in makefile
    assert "-m tools.api_reference --write" in makefile
    assert "-m tools.api_reference" in makefile

    jobs: dict[str, Any] = load_yaml(CI)["jobs"]
    commands = [step["run"] for step in jobs["api-reference"]["steps"] if "run" in step]
    assert commands[-1] == "uv run python -m tools.api_reference"


def test_generated_reference_is_in_the_published_navigation() -> None:
    navigation = MKDOCS.read_text(encoding="utf-8")
    assert "reference/index.md" in navigation
    assert "reference/core.md" in navigation
    assert "reference/protocols.md" in navigation


def test_versioning_deprecation_and_security_policies_are_first_class_pages() -> None:
    navigation = MKDOCS.read_text(encoding="utf-8")
    assert "Versioning: versioning.md" in navigation
    assert "Deprecations: deprecations.md" in navigation
    assert "Security policy: security-policy.md" in navigation
