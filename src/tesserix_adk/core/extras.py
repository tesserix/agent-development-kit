"""Lazy access to optional integrations.

The kit's base install is pydantic, httpx and opentelemetry-api. Every provider and
store SDK sits behind an extra and is imported through `require_extra`, so a consumer
who reaches past their install gets a message naming the extra rather than a traceback
from a transitive module they have never heard of.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import MissingExtraError

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["require_extra"]


def _is_the_target_itself(missing: str | None, module: str) -> bool:
    """Is the module Python could not find the requested one, or a package containing it?"""
    return missing is not None and (module == missing or module.startswith(f"{missing}."))


def require_extra(extra: str, module: str) -> ModuleType:
    """Import an optional dependency, or say which extra provides it.

    Args:
        extra: The extra that installs the dependency, e.g. `redis`.
        module: The importable module, e.g. `redis.asyncio`.

    Returns:
        The imported module.

    Raises:
        MissingExtraError: The dependency is not installed. The message names the extra
            and the exact install command.
        ModuleNotFoundError: The dependency is installed but its own import failed. That
            is the dependency's problem, and reporting it as a missing extra would send
            the consumer to install something they already have.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as err:
        if _is_the_target_itself(err.name, module):
            raise MissingExtraError(extra=extra, module=module) from err
        raise
