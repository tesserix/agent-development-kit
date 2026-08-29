"""Scaffolding writes a complete typed start or writes nothing."""

from __future__ import annotations

import io
import subprocess
import sys
from typing import TYPE_CHECKING

from tesserix_adk import __version__
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


def test_each_agent_template_generates_its_advertised_composition(tmp_path: Path) -> None:
    expected = {
        "single": ("TypedAgent",),
        "tool-using": ("TypedAgent", "@tool"),
        "multi-agent": ("TypedAgent", "Roster", "Specialist", "build_roster"),
        "mcp-client": ("TypedAgent", "McpClient", "McpServerConfig", "build_mcp_client"),
    }

    for template, markers in expected.items():
        root = tmp_path / template
        root.mkdir()
        assert (
            main(
                ["new", "agent", "trip-planner", "--template", template],
                cwd=project(root),
                out=io.StringIO(),
            )
            == 0
        )
        source = (root / "trip_planner_agent.py").read_text(encoding="utf-8")
        tools = (root / "trip_planner_tools.py").read_text(encoding="utf-8")
        assert f'TEMPLATE_KIND: Final[str] = "{template}"' in source
        assert all(marker in source + tools for marker in markers)
        assert source.count("from tesserix_adk import") == 1


def test_default_agent_creates_complete_file_set_and_next_command(tmp_path: Path) -> None:
    root = project(tmp_path)
    output = io.StringIO()

    assert main(["new", "agent", "order-reviewer"], cwd=root, out=output) == 0

    assert sorted(path.name for path in root.iterdir()) == [
        "order_reviewer.adk.toml",
        "order_reviewer_agent.py",
        "order_reviewer_tools.py",
        "pyproject.toml",
        "test_order_reviewer_agent.py",
    ]
    assert "next: python -m pytest -q test_order_reviewer_agent.py" in output.getvalue()
    config = (root / "order_reviewer.adk.toml").read_text(encoding="utf-8")
    assert f"Tesserix ADK template version {__version__}" in config
    assert "api_key" not in config


def test_every_agent_template_is_offline_typed_and_linted(tmp_path: Path) -> None:
    for template in ("single", "tool-using", "multi-agent", "mcp-client"):
        root = tmp_path / template
        root.mkdir()
        assert (
            main(
                ["new", "agent", "trip-planner", "--template", template],
                cwd=project(root),
                out=io.StringIO(),
            )
            == 0
        )
        files = (
            "trip_planner_agent.py",
            "trip_planner_tools.py",
            "test_trip_planner_agent.py",
        )
        commands = (
            [sys.executable, "-m", "pytest", "-q", files[2]],
            [sys.executable, "-m", "mypy", "--strict", *files],
            [sys.executable, "-m", "ruff", "check", *files],
        )
        for command in commands:
            checked = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=root,
                timeout=30,
            )
            assert checked.returncode == 0, (
                f"{template}: {' '.join(command)}\n{checked.stdout}{checked.stderr}"
            )


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
    tools = root / "trip_planner_tools.py"
    test = root / "test_trip_planner_agent.py"
    assert module.exists()
    assert tools.exists()
    assert test.exists()
    source = module.read_text(encoding="utf-8")
    assert "class TripPlannerInput" in source
    assert "class TripPlannerOutput" in source
    assert "@tool" in tools.read_text(encoding="utf-8")
    assert "BudgetLimits" in source
    assert "load_typed_config" in source
    assert "Instrumentation" in source
    assert "ADK_TEMPLATE_VERSION" in source
    assert "run_typed_sync" in test.read_text(encoding="utf-8")

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
        [sys.executable, "-m", "mypy", "--strict", module.name, tools.name, test.name],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
        timeout=30,
    )
    assert typed.returncode == 0, typed.stdout + typed.stderr
    linted = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "check", module.name, tools.name, test.name],
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
