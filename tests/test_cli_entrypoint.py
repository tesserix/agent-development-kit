"""The installed command identifies the Tesserix project unambiguously."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tesserix_adk.cli.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_help_names_both_agent_development_kits_in_full(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "usage: tesserix-adk" in output
    assert "Tesserix Agent Development Kit" in output
    assert "Google Agent Development Kit" in output


def test_the_cli_guide_covers_every_installed_command(
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> None:
    assert main(["--help"]) == 0
    usage = capsys.readouterr().out.splitlines()[0]
    match = re.search(r"\{([^}]+)\}", usage)
    assert match is not None

    root: Path = request.config.rootpath
    guide = (root / "docs" / "cli.md").read_text(encoding="utf-8")
    commands = (command.strip() for command in match.group(1).split(","))
    undocumented = [command for command in commands if f"`{command}`" not in guide]
    assert undocumented == []
