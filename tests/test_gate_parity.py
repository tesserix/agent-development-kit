"""The local gate and the CI gate must be the same gate.

A pre-commit hook pinned independently of the project's own toolchain drifts, and
the first symptom is a PR that is green locally and red in CI. These tests assert
the two run the same commands from the same resolved environment.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"


def _yaml(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _ci_jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = _yaml(CI)["jobs"]
    return jobs


def _ci_run_steps(job: str) -> list[str]:
    return [step["run"] for step in _ci_jobs()[job]["steps"] if "run" in step]


@pytest.mark.parametrize("job", ["lint", "typecheck", "test-fast", "layering", "lockfile"])
def test_ci_runs_every_gate(job: str) -> None:
    assert job in _ci_jobs()


def test_ci_lint_annotates_the_pull_request_diff() -> None:
    """Concise output in a log costs a click per finding; GitHub format lands inline."""
    assert any("--output-format=github" in step for step in _ci_run_steps("lint"))


def test_ci_lint_checks_formatting_as_well_as_rules() -> None:
    assert any("ruff format --check" in step for step in _ci_run_steps("lint"))


def test_ci_typecheck_is_strict() -> None:
    assert any("mypy" in step and "--strict" in step for step in _ci_run_steps("typecheck"))


def test_ci_reports_annotation_coverage_of_the_public_surface() -> None:
    """A shrinking `Any` surface is only visible if something measures it."""
    assert any("any-exprs-report" in step for step in _ci_run_steps("typecheck"))


def _hooks() -> list[dict[str, Any]]:
    config = _yaml(PRE_COMMIT)
    return [hook for repo in config["repos"] for hook in repo["hooks"]]


def test_every_pre_commit_hook_runs_from_the_project_environment() -> None:
    """`rev`-pinned mirrors drift from the lockfile; `uv run` cannot."""
    for repo in _yaml(PRE_COMMIT)["repos"]:
        assert repo["repo"] == "local", f"{repo['repo']} pins a tool version outside uv.lock"
    for hook in _hooks():
        assert hook["entry"].startswith("uv run "), hook["entry"]


@pytest.mark.parametrize("hook_id", ["ruff", "ruff-format", "mypy", "import-linter"])
def test_pre_commit_covers_every_static_gate(hook_id: str) -> None:
    assert hook_id in {hook["id"] for hook in _hooks()}
