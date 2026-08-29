"""Runnable command modules that need no application-specific storage wiring."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

from tesserix_adk.cli.artifact_inspect import main as artifact_inspect_main
from tesserix_adk.cli.config_command import main as config_main
from tesserix_adk.cli.doctor import (
    CheckRegistry,
    CredentialPresenceCheck,
    DoctorContext,
    PythonVersionCheck,
)
from tesserix_adk.cli.doctor import (
    main as doctor_main,
)
from tesserix_adk.cli.eval_run import load_target as load_eval_target
from tesserix_adk.cli.eval_run import main as eval_run_main
from tesserix_adk.cli.evals import main as evals_main
from tesserix_adk.cli.run_agent import load_target as load_run_target
from tesserix_adk.cli.run_agent import main as run_main
from tesserix_adk.cli.scaffold import main as scaffold_main
from tesserix_adk.cli.trace import main as trace_main
from tesserix_adk.core import ConfigError, load_config

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]

MISUSED = 2
_COMMANDS = "config, doctor, eval, evals, inspect, new, run, trace"
_HELP = f"""usage: tesserix-adk {{{_COMMANDS}}} ...

Tesserix Agent Development Kit project commands.

This command belongs to the tesserix-adk distribution and tesserix_adk Python package.
It is distinct from Google Agent Development Kit; interoperability is available through
the optional google-adk adapter and the independent Agent2Agent protocol.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a self-contained command and return its exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        sys.stderr.write(_HELP.splitlines()[0] + "\n")
        return MISUSED
    if arguments == ["--help"] or arguments == ["-h"]:
        sys.stdout.write(_HELP)
        return 0
    command, *remaining = arguments
    if command == "config":
        return config_main(remaining)
    if command == "doctor":
        try:
            config = load_config()
        except ConfigError as error:
            sys.stderr.write(
                f"configuration is invalid: {error}\nrun: tesserix-adk config validate\n"
            )
            return MISUSED
        registry = CheckRegistry(
            (
                PythonVersionCheck(),
                CredentialPresenceCheck(
                    "TESSERIX_ADK_PROVIDER__API_KEY",
                    required=False,
                ),
            )
        )
        return asyncio.run(
            doctor_main(
                remaining,
                registry=registry,
                context=DoctorContext(config=config, environ=dict(os.environ)),
            )
        )
    if command == "eval":
        return asyncio.run(eval_run_main(remaining, resolve=load_eval_target))
    if command == "evals":
        return evals_main(remaining)
    if command == "inspect":
        return asyncio.run(artifact_inspect_main(remaining, resolve=load_run_target))
    if command == "new":
        return scaffold_main([command, *remaining])
    if command == "run":
        return asyncio.run(run_main(remaining, resolve=load_run_target))
    if command == "trace":
        return trace_main(remaining)
    sys.stderr.write(f"unknown command {command!r}; choose {_COMMANDS}\n")
    return MISUSED


if __name__ == "__main__":
    raise SystemExit(main())
