"""Command-line entrypoints."""

from tesserix_adk.cli.approvals import Answering, Waiting
from tesserix_adk.cli.approvals import main as approvals_main

__all__ = ["Answering", "Waiting", "approvals_main"]
