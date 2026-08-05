"""The kit's own suite runs under the isolation it publishes to consumers."""

import os
from pathlib import Path

pytest_plugins = ["tesserix_adk.testing.pytest_plugin"]

# pytester subprocesses start coverage from a temporary rootdir, where they cannot find
# this project's settings; statement-only data will not combine with branch data.
os.environ.setdefault(
    "COVERAGE_RCFILE", str(Path(__file__).resolve().parents[1] / "pyproject.toml")
)
