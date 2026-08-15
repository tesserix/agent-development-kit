"""Command-line entrypoints."""

from tesserix_adk.cli.approvals import Answering, Waiting
from tesserix_adk.cli.approvals import main as approvals_main
from tesserix_adk.cli.inspect import Lookup
from tesserix_adk.cli.inspect import main as inspect_main
from tesserix_adk.cli.prompts import Aliases, Prompts
from tesserix_adk.cli.prompts import main as prompts_main

__all__ = [
    "Aliases",
    "Answering",
    "Lookup",
    "Prompts",
    "Waiting",
    "approvals_main",
    "inspect_main",
    "prompts_main",
]
