"""A docstring example that no longer runs is documentation that lies.

Doctests are collected here rather than through `--doctest-modules` so that importing a
module for its examples cannot change how the rest of the suite collects.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import tesserix_adk


def modules() -> list[str]:
    return [
        module.name
        for module in pkgutil.walk_packages(tesserix_adk.__path__, "tesserix_adk.")
        if not module.ispkg
    ]


@pytest.mark.parametrize("name", modules())
def test_every_docstring_example_still_runs(name: str) -> None:
    module = importlib.import_module(name)
    results = doctest.testmod(module, verbose=False, report=False)
    assert results.failed == 0, f"{results.failed} doctest failure(s) in {name}"


def test_the_primitives_carry_examples() -> None:
    """The types a consumer meets first are the ones an example is worth most on."""
    finder = doctest.DocTestFinder()
    module = importlib.import_module("tesserix_adk.core.primitives")
    assert any(test.examples for test in finder.find(module))
