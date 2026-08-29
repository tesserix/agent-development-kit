"""The installed command identifies the Tesserix project unambiguously."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.cli.__main__ import main

if TYPE_CHECKING:
    import pytest


def test_help_names_both_agent_development_kits_in_full(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "usage: tesserix-adk" in output
    assert "Tesserix Agent Development Kit" in output
    assert "Google Agent Development Kit" in output
