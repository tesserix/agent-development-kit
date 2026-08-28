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


def test_pull_requests_build_docs_without_deploying() -> None:
    declared = triggers(DOCS)
    assert "pull_request" in declared
    assert declared["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in declared


def test_the_build_checks_links_and_mkdocs_strictly() -> None:
    jobs: dict[str, Any] = load_yaml(DOCS)["jobs"]
    commands = [step["run"] for step in jobs["build"]["steps"] if "run" in step]
    assert any("tests/test_documentation.py" in command for command in commands)
    assert any("mkdocs build --strict" in command for command in commands)


def test_write_permissions_exist_only_on_the_deploy_job() -> None:
    workflow = load_yaml(DOCS)
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["deploy"]["permissions"] == {
        "pages": "write",
        "id-token": "write",
    }
