"""Agent development kit: typed primitives, a substitutable runtime, CPU-first inference."""

from importlib.metadata import PackageNotFoundError, version

from tesserix_adk.core import Agent
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.tools import ToolRegistry, tool

__all__ = ["Agent", "AgentRunner", "ToolRegistry", "__version__", "tool"]

try:
    __version__ = version("tesserix-adk")
except PackageNotFoundError:  # pragma: no cover — only when running from an uninstalled tree
    __version__ = "0.0.0"
