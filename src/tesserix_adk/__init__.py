"""Agent development kit: typed primitives, a substitutable runtime, CPU-first inference."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("tesserix-adk")
except PackageNotFoundError:  # pragma: no cover — only when running from an uninstalled tree
    __version__ = "0.0.0"
