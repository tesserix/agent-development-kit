"""An example nobody runs rots into a bug report from the first consumer to try it.

The smoke job runs the same file against the published wheel; this runs it against the
working tree, so a broken example fails at review instead of after the release.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.stem)
def test_every_example_runs(example: Path) -> None:
    """In a subprocess: an example that only works with the test suite's imports is a lie."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(example)], capture_output=True, text=True, check=False, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr


def test_there_is_a_getting_started_example() -> None:
    """Named in the README and the smoke job, so its absence has to fail here."""
    assert (EXAMPLES / "getting_started.py").exists()


def test_the_getting_started_example_needs_no_credentials() -> None:
    """A first run that demands an API key is a first run nobody completes."""
    source = (EXAMPLES / "getting_started.py").read_text(encoding="utf-8")
    assert "api_key" not in source
