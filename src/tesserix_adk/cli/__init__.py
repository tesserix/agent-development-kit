"""Command-line entrypoints."""

from tesserix_adk.cli.approvals import Answering, Waiting
from tesserix_adk.cli.approvals import main as approvals_main
from tesserix_adk.cli.evals import main as evals_main
from tesserix_adk.cli.inspect import Lookup
from tesserix_adk.cli.inspect import main as inspect_main
from tesserix_adk.cli.prompts import Aliases, Prompts
from tesserix_adk.cli.prompts import main as prompts_main
from tesserix_adk.cli.resume import Frontiers, Resumes, describe
from tesserix_adk.cli.resume import main as resume_main
from tesserix_adk.cli.resume import show as frontier_main
from tesserix_adk.cli.surface import Resolve
from tesserix_adk.cli.surface import main as surface_main
from tesserix_adk.cli.trace import main as trace_main

__all__ = [
    "Aliases",
    "Answering",
    "Frontiers",
    "Lookup",
    "Prompts",
    "Resolve",
    "Resumes",
    "Waiting",
    "approvals_main",
    "describe",
    "evals_main",
    "frontier_main",
    "inspect_main",
    "prompts_main",
    "resume_main",
    "surface_main",
    "trace_main",
]
