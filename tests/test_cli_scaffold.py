"""Scaffolding writes a complete typed start or writes nothing."""

from __future__ import annotations

import io
import subprocess
import sys
from typing import TYPE_CHECKING

from tesserix_adk.cli.scaffold import main

if TYPE_CHECKING:
    from pathlib import Path


def project(tmp_path: Path) -> Path:
    """Create the only project file scaffolding requires."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "consumer"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n',
        encoding="utf-8",
    )
    return tmp_path


def test_templates_are_discoverable_without_writing(tmp_path: Path) -> None:
    output = io.StringIO()
    assert main(["new", "--list"], cwd=project(tmp_path), out=output) == 0
    assert output.getvalue().splitlines() == ["mcp-client", "multi-agent", "single", "tool-using"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["pyproject.toml"]


def test_tool_using_agent_is_generated_and_works_without_network(tmp_path: Path) -> None:
    root = project(tmp_path)
    output = io.StringIO()
    assert (
        main(
            ["new", "agent", "trip-planner", "--template", "tool-using"],
            cwd=root,
            out=output,
        )
        == 0
    )
    module = root / "trip_planner_agent.py"
    test = root / "test_trip_planner_agent.py"
    assert module.exists()
    assert test.exists()
    source = module.read_text(encoding="utf-8")
    assert "class TripPlannerInput" in source
    assert "class TripPlannerOutput" in source
    assert "@tool" in source
    assert "BudgetLimits" in source
    assert "load_config" in source
    assert "Instrumentation" in source
    assert "ADK_TEMPLATE_VERSION" in source

    checked = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "-q", test.name],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stderr
    typed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "mypy", "--strict", module.name, test.name],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
        timeout=30,
    )
    assert typed.returncode == 0, typed.stdout + typed.stderr
    linted = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "check", module.name, test.name],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
        timeout=30,
    )
    assert linted.returncode == 0, linted.stdout + linted.stderr


def test_any_conflict_aborts_before_every_write(tmp_path: Path) -> None:
    root = project(tmp_path)
    existing = root / "trip_planner_agent.py"
    existing.write_bytes(b"consumer bytes\n")
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    output = io.StringIO()
    assert main(["new", "agent", "trip-planner"], cwd=root, out=output) == 1
    assert "trip_planner_agent.py" in output.getvalue()
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_every_conflict_is_reported_together(tmp_path: Path) -> None:
    root = project(tmp_path)
    for name in ("trip_planner_agent.py", "test_trip_planner_agent.py"):
        (root / name).write_text("owned\n", encoding="utf-8")
    output = io.StringIO()

    assert main(["new", "agent", "trip-planner"], cwd=root, out=output) == 1

    assert "trip_planner_agent.py" in output.getvalue()
    assert "test_trip_planner_agent.py" in output.getvalue()


def test_force_replaces_the_complete_template_set(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "search_tool.py").write_text("old\n", encoding="utf-8")
    output = io.StringIO()
    assert main(["new", "tool", "search", "--force"], cwd=root, out=output) == 0
    assert "@tool" in (root / "search_tool.py").read_text(encoding="utf-8")
    assert (root / "test_search_tool.py").exists()


def test_an_invalid_or_reserved_name_is_refused_before_writing(tmp_path: Path) -> None:
    root = project(tmp_path)
    for name in ("two words", "core"):
        output = io.StringIO()
        assert main(["new", "agent", name], cwd=root, out=output) == 2
        assert "name" in output.getvalue().lower()
    assert sorted(path.name for path in root.iterdir()) == ["pyproject.toml"]


def test_an_unsupported_project_python_is_refused_before_writing(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "consumer"\nversion = "0.1.0"\nrequires-python = ">=99"\n',
        encoding="utf-8",
    )
    output = io.StringIO()

    assert main(["new", "tool", "search"], cwd=root, out=output) == 2

    assert "Python" in output.getvalue()
    assert sorted(path.name for path in root.iterdir()) == ["pyproject.toml"]
