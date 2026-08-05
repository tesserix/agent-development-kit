"""Loaders shared by the tests that assert things about the project's own configuration."""

import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as handle:
        parsed: dict[str, Any] = tomllib.load(handle)
    return parsed


def ci_jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = load_yaml(CI)["jobs"]
    return jobs


def ci_run_steps(job: str) -> list[str]:
    return [step["run"] for step in ci_jobs()[job]["steps"] if "run" in step]


def supported_minors() -> list[str]:
    """The supported versions as declared to consumers, i.e. the trove classifiers."""
    prefix = "Programming Language :: Python :: 3."
    classifiers: list[str] = pyproject()["project"]["classifiers"]
    return sorted(
        (f"3.{c.removeprefix(prefix)}" for c in classifiers if c.startswith(prefix)),
        key=lambda v: int(v.split(".")[1]),
    )
