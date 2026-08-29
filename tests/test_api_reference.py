"""The generated reference is checked against the same surface consumers install."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from tools.api_reference import REFERENCE, main, validate
from tools.api_reference import ReferenceError as ApiReferenceError

from tesserix_adk.core import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, **members: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__["__all__"] = list(members)
    for member, value in members.items():
        setattr(module, member, value)
    return module


def test_the_committed_reference_matches_the_typed_public_surface() -> None:
    assert main([]) == 0
    index = (REFERENCE / "index.md").read_text(encoding="utf-8")
    core = (REFERENCE / "core.md").read_text(encoding="utf-8")
    assert "Generated API reference" in index
    assert "tesserix_adk.core.Agent" in core
    assert "**Stability:** `beta`" in core
    assert "Runnable recipe" in core


def test_a_stale_docstring_parameter_is_named() -> None:
    def sample(current: int) -> int:
        """Return the value.

        Args:
            old: A parameter that no longer exists.
        """
        return current

    with pytest.raises(ApiReferenceError, match="old"):
        validate((_module("tesserix_adk.sample", sample=sample),))


def test_a_public_callable_without_a_docstring_is_named() -> None:
    def undocumented() -> None:
        pass

    undocumented.__doc__ = None
    with pytest.raises(ApiReferenceError, match="undocumented"):
        validate((_module("tesserix_adk.sample", undocumented=undocumented),))


def test_a_direct_typed_error_must_be_in_the_raises_clause() -> None:
    def refused() -> None:
        """Refuse the operation."""
        raise ConfigurationError("no")

    with pytest.raises(ApiReferenceError, match="ConfigurationError"):
        validate((_module("tesserix_adk.sample", refused=refused),))


def test_an_error_inside_a_nested_callable_is_not_attributed_to_its_factory() -> None:
    def factory() -> object:
        """Build a callback without invoking it."""

        def callback() -> None:
            raise ConfigurationError("no")

        return callback

    validate((_module("tesserix_adk.sample", factory=factory),))
