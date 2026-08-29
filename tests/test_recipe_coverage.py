"""Runnable recipes are part of the public contract, not optional prose."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.api_surface import collect_surface
from tools.recipe_coverage import MAPPING, ROUTES, CoverageError, main, mapped, routes

ROOT = Path(__file__).resolve().parents[1]


def test_every_public_symbol_has_a_committed_runnable_recipe() -> None:
    committed = json.loads(MAPPING.read_text(encoding="utf-8"))["symbols"]
    assert set(committed) == set(collect_surface())
    assert main([]) == 0


def test_a_new_export_fails_until_the_recipe_mapping_is_regenerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.recipe_coverage.collect_surface",
        lambda: {"tesserix_adk.NewPrimitive": "class NewPrimitive(object)"},
    )
    assert main([]) == 1


def test_a_route_to_a_missing_example_is_refused_before_mapping() -> None:
    declared = routes(ROUTES)
    broken = declared[0].__class__(
        prefix="tesserix_adk",
        recipe="examples/does-not-exist.py",
        page="docs/cookbook/index.md",
        rule="fail closed",
    )
    with pytest.raises(CoverageError, match="does-not-exist"):
        mapped({"tesserix_adk.Agent": "class Agent"}, (broken,))


def test_the_cookbook_covers_every_required_composition() -> None:
    declared = {route.id for route in routes(ROUTES)}
    assert {
        "agent",
        "tool",
        "allowlist",
        "memory-working",
        "memory-profile",
        "memory-episodic",
        "memory-semantic",
        "guardrails",
        "budget",
        "mcp-client",
        "mcp-server",
        "peer",
        "retrieval",
        "workflow",
        "evals",
    } <= declared
