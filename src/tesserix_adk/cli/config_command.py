"""Show configuration provenance or validate every typed value before startup."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.core import (
    ConfigError,
    ConfigOverrides,
    ConfigResolution,
    resolve_typed_config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

__all__ = ["main"]

OK = 0
INVALID = 1
MISUSED = 2


def main(
    argv: Sequence[str],
    *,
    overrides: ConfigOverrides | None = None,
    env: dict[str, str] | None = None,
    start: Path | str | None = ".",
    out: TextIO | None = None,
) -> int:
    """Resolve one configuration and show it or report every validation problem.

    Args:
        argv: Arguments after ``config``.
        overrides: Explicit code-layer values, useful to application-owned entrypoints.
        env: Environment mapping. Defaults to the process environment.
        start: Discovery root for ``adk.toml`` or ``pyproject.toml``.
        out: Report destination. Defaults to stdout.

    Returns:
        ``0`` for a usable configuration, ``1`` for invalid configuration and ``2`` for
        command misuse. Secret values are masked in every output mode.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    try:
        resolution = resolve_typed_config(
            overrides,
            env=env,
            path=parsed.config,
            start=start,
        )
    except ConfigError as error:
        _problems(error, writer=writer, json_mode=parsed.json)
        return INVALID
    except (tomllib.TOMLDecodeError, OSError) as error:
        location = parsed.config or start or "configuration"
        message = f"{location}: {error}"
        if parsed.json:
            writer.write(json.dumps({"valid": False, "problems": [message]}) + "\n")
        else:
            writer.write(f"invalid: {message}\n")
        return INVALID
    if parsed.command == "validate":
        writer.write(json.dumps({"valid": True}) + "\n" if parsed.json else "valid\n")
        return OK
    _show(resolution, writer=writer, json_mode=parsed.json)
    return OK


def _show(resolution: ConfigResolution, *, writer: TextIO, json_mode: bool) -> None:
    """Render every key, winner and lower-precedence value without raw secrets."""
    if not json_mode:
        writer.write(resolution.explain() + "\n")
        return
    document = {
        key: {
            "source": provenance.layer,
            "value": provenance.value,
            "overridden": [
                {"source": layer, "value": value} for layer, value in provenance.overridden
            ],
        }
        for key, provenance in sorted(resolution.provenance.items())
    }
    writer.write(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _problems(error: ConfigError, *, writer: TextIO, json_mode: bool) -> None:
    """Report every collected problem rather than stopping at the first key."""
    problems = [str(problem) for problem in error.problems]
    if json_mode:
        writer.write(json.dumps({"valid": False, "problems": problems}, indent=2) + "\n")
        return
    writer.write("invalid configuration:\n")
    for problem in problems:
        writer.write(f"  {problem}\n")


def _parser() -> argparse.ArgumentParser:
    """Build the ``tesserix-adk config`` command line."""
    parser = argparse.ArgumentParser(prog="tesserix-adk config", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("show", "show each resolved value and winning source"),
        ("validate", "validate all configuration before startup"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, help="explicit TOML configuration path")
        command.add_argument("--json", action="store_true", help="machine-readable output")
    return parser
