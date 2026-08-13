"""A review instrument is only real if the thing it asks for can be obtained.

The ladder in `docs/escalation-ladder.md` rejects designs for want of a number, so what it
fails here is what would make it unusable in the review it exists for: a rung with no
stated threshold, a metric that no shipped surface produces, a link to a page that is not
there. Prose is not tested; obligations are.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from tesserix_adk.testing.benchmarks import Metric

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "docs" / "escalation-ladder.md"
TEXT = LADDER.read_text(encoding="utf-8")

RUNGS = (
    "One agent with tools",
    "One agent plus a deterministic workflow",
    "A router with specialist agents",
    "Collaborating agents",
)
STEPS = ("Rung 1 → 2", "Rung 2 → 3", "Rung 3 → 4")
SYMBOL = re.compile(r"`(tesserix_adk\.[A-Za-z0-9_.]+)`")
LINK = re.compile(r"\]\((?!https?:)([^)#]+)")


def test_the_ladder_names_its_rungs_in_order() -> None:
    """A ladder read out of order is four options, which is what it replaces."""
    assert [rung for rung in RUNGS if rung in TEXT] == list(RUNGS)
    assert sorted(TEXT.index(rung) for rung in RUNGS) == [TEXT.index(rung) for rung in RUNGS]


@pytest.mark.parametrize("step", STEPS)
def test_every_step_up_states_a_threshold(step: str) -> None:
    """A bar without a number is taste with a heading over it."""
    section = TEXT.split(step, 1)[1].split("\n### ", 1)[0]
    assert "**Bar" in section
    assert re.search(r"\d+(\.\d+)?\s*(percentage points|%)", section), step


@pytest.mark.parametrize("step", STEPS)
def test_every_step_up_prices_quality_cost_and_latency(step: str) -> None:
    """One of the three improving at the expense of another is not an improvement."""
    section = TEXT.split(step, 1)[1].split("\n### ", 1)[0]
    assert "suite" in section
    assert "cost" in section
    assert "latency" in section


def _resolves(dotted: str) -> bool:
    """Whether a dotted path names a module, or an attribute reached from the longest one."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            found: object = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attribute in parts[cut:]:
            fields = getattr(found, "model_fields", {})
            if attribute in fields:
                return True
            if not hasattr(found, attribute):
                return False
            found = getattr(found, attribute)
        return True
    return False


def test_every_measurement_it_asks_for_resolves_to_something_shipped() -> None:
    """A threshold measured with a symbol nobody exports cannot be cleared or refused."""
    for dotted in sorted(set(SYMBOL.findall(TEXT))):
        assert _resolves(dotted), dotted


def test_the_benchmark_metrics_it_cites_are_ones_the_harness_reports() -> None:
    """`latency_p95` has to be the harness's name for it, not a plausible one."""
    reported = {metric.value for metric in Metric}
    for metric in ("latency_p50", "latency_p95", "tokens"):
        assert metric in TEXT
        assert metric in reported


def test_it_records_the_reasons_that_need_no_measurement() -> None:
    """Least privilege and a trust boundary are not quality wins and no suite shows them."""
    reasons = TEXT.split("## Reasons that need no measurement", 1)[1].split("\n## ", 1)[0]
    assert "tool grant" in reasons
    assert "trust boundary" in reasons
    assert "still cost a hop" in reasons


def test_it_keeps_deterministic_rules_out_of_the_model() -> None:
    """The rule that saves the most money is the one that was never a model call."""
    assert "## Deterministic rules stay in code" in TEXT
    assert "never a model call" in TEXT


def test_the_worked_example_compares_two_rungs_on_all_three_figures() -> None:
    """A worked example without the numbers is an anecdote, which is what it argues against."""
    example = TEXT.split("## A worked example", 1)[1].split("\n## ", 1)[0]
    assert "Rung 2" in example
    assert "Rung 4" in example
    assert "Suite pass rate" in example
    assert "Cost per task" in example
    assert "latency_p95" in example
    assert "Verdict" in example


def test_the_worked_examples_figures_say_where_they_came_from() -> None:
    """Unsourced figures in a page about sourcing figures would teach the wrong lesson."""
    example = TEXT.split("## A worked example", 1)[1].split("\n## ", 1)[0]
    assert "scripted providers" in example


def test_it_does_not_forbid_the_shapes_it_gates() -> None:
    """The ladder requires multi-agent designs to be earned, not avoided."""
    assert "It does not forbid any shape" in TEXT


def test_it_records_its_own_limitations() -> None:
    assert "## Known limitations" in TEXT
    assert "chosen, not derived" in TEXT


def test_every_page_it_sends_a_reader_to_exists() -> None:
    for target in sorted(set(LINK.findall(TEXT))):
        assert (LADDER.parent / target).exists(), target


def test_the_readme_lists_it() -> None:
    """A review instrument nobody can find is not one."""
    assert "docs/escalation-ladder.md" in (ROOT / "README.md").read_text(encoding="utf-8")
