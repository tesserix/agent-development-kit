"""The lint and type gates are themselves behaviour, so they are tested like behaviour.

A rule that is merely configured is a rule nobody has proven fires. Each test here
runs the real tool over a crafted sample and asserts the finding.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def ruff_findings(source: str, tmp_path: Path, name: str = "sample.py") -> str:
    sample = tmp_path / name
    sample.write_text(source, encoding="utf-8")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format=concise",
            str(sample),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    return result.stdout


def test_a_bare_type_ignore_is_rejected(tmp_path: Path) -> None:
    findings = ruff_findings("x: int = 'no'  # type: ignore\n", tmp_path)
    assert "PGH003" in findings


def test_a_type_ignore_with_a_specific_code_is_accepted(tmp_path: Path) -> None:
    findings = ruff_findings(
        "x: int = 'no'  # type: ignore[assignment]  # upstream stub\n", tmp_path
    )
    assert "PGH003" not in findings


def test_a_bare_noqa_is_rejected(tmp_path: Path) -> None:
    findings = ruff_findings("import os  # noqa\n", tmp_path)
    assert "PGH004" in findings


def test_a_public_function_without_annotations_is_rejected(tmp_path: Path) -> None:
    findings = ruff_findings(
        '"""Doc."""\n\n\ndef public(value):\n    """Doc."""\n    return value\n', tmp_path
    )
    assert "ANN" in findings


def test_a_public_function_without_a_docstring_is_rejected(tmp_path: Path) -> None:
    findings = ruff_findings(
        '"""Doc."""\n\n\ndef public(value: int) -> int:\n    return value\n', tmp_path
    )
    assert "D103" in findings


def test_a_stray_print_is_rejected(tmp_path: Path) -> None:
    findings = ruff_findings('"""Doc."""\n\nprint("debug")\n', tmp_path)
    assert "T201" in findings


def test_an_unsafe_subprocess_call_is_flagged(tmp_path: Path) -> None:
    findings = ruff_findings(
        '"""Doc."""\n\nimport subprocess\n\nsubprocess.run("ls", shell=True)\n', tmp_path
    )
    assert "S" in findings


def _selected_rules() -> list[str]:
    with PYPROJECT.open("rb") as handle:
        selected: list[str] = tomllib.load(handle)["tool"]["ruff"]["lint"]["select"]
    return selected


@pytest.mark.parametrize("rule", ["S", "ASYNC", "T20", "PGH", "D", "ANN"])
def test_required_rule_families_are_enabled(rule: str) -> None:
    assert rule in _selected_rules()


def test_mypy_is_strict_with_no_blanket_module_relaxations() -> None:
    with PYPROJECT.open("rb") as handle:
        mypy = tomllib.load(handle)["tool"]["mypy"]
    assert mypy["strict"] is True
    assert "overrides" not in mypy, (
        "a per-module override is a blanket relaxation; use a local stub or a coded "
        "suppression with a justification instead"
    )


def test_every_lint_relaxation_carries_an_owner_and_a_reason() -> None:
    """The allowlist is meant to shrink, which requires knowing who owns each entry."""
    source = PYPROJECT.read_text(encoding="utf-8")
    block = source.split("[tool.ruff.lint.per-file-ignores]", 1)[1].split("\n[", 1)[0]

    entries, comment_before = [], False
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            comment_before = True
        elif line:
            entries.append((line, comment_before))
            comment_before = False

    assert entries, "no per-file-ignores found to check"
    undocumented = [line for line, documented in entries if not documented]
    assert undocumented == [], f"lint relaxations without an owner and reason: {undocumented}"
