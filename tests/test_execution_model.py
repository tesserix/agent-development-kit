"""A decision record is only real if the thing it decides can be checked.

`docs/execution-model.md` sends teams to one of three rungs and rejects designs at review.
What it fails here is what would make it unusable: a rung with no stated bar, a claim about
the package footprint that the packaging does not hold to, a named extra that pyproject does
not declare, a link to a page that is not there. Prose is not tested; obligations are.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.ci_config import pyproject

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "execution-model.md"
TEXT = RECORD.read_text(encoding="utf-8")

RUNGS = (
    "Rung 1 — in-process",
    "Rung 2 — checkpointed",
    "Rung 3 — durable workflow",
)
QUESTIONS = (
    "How long does the run take",
    "Is there a human gate",
    "Are the side effects irreversible",
    "How wide does it fan out",
    "What does a restart cost",
)
LINK = re.compile(r"\]\((?!https?:)([^)#]+)")
EXTRA = re.compile(r"tesserix-adk\[([a-z,]+)\]")


class TestTheLadder:
    def test_it_names_its_rungs_in_order(self) -> None:
        """A ladder read out of order is three options, which is what it replaces."""
        found = [TEXT.index(rung) for rung in RUNGS if rung in TEXT]

        assert len(found) == len(RUNGS)
        assert found == sorted(found)

    def test_the_default_is_the_bottom_rung(self) -> None:
        """A default that is not stated is decided by whoever writes the first design doc."""
        assert "default" in TEXT.split(RUNGS[0], 1)[1].split("\n### ", 1)[0].lower()

    @pytest.mark.parametrize("rung", RUNGS[1:])
    def test_every_step_up_states_what_earns_it(self, rung: str) -> None:
        """A rung without a bar is taste with a heading over it."""
        section = TEXT.split(rung, 1)[1].split("\n### ", 1)[0]

        assert "**Earned by" in section, rung


class TestTheDecisionTable:
    @pytest.mark.parametrize("question", QUESTIONS)
    def test_the_table_asks_the_question(self, question: str) -> None:
        """Five inputs, or the table answers a different question than it was built for."""
        assert question in TEXT

    def test_a_short_ungated_run_is_answered_without_a_workflow(self) -> None:
        row = next(line for line in TEXT.splitlines() if "under five minutes" in line)

        assert "Rung 1" in row

    def test_an_hours_long_human_gate_stops_at_the_checkpoint_store(self) -> None:
        """The edge case the record exists to settle: waiting is not transacting."""
        row = next(line for line in TEXT.splitlines() if "days, no transactions" in line)

        assert "Rung 2" in row
        assert "Rung 3" not in row


class TestWhatIsNotAReasonToReachForAWorkflow:
    @pytest.mark.parametrize(
        "wrong", ["observability", "retries", "an approval gate", "an autonomy grant"]
    )
    def test_the_anti_pattern_is_named(self, wrong: str) -> None:
        section = TEXT.split("## Not a reason", 1)[1].split("\n## ", 1)[0]

        assert wrong in section

    def test_durability_is_never_authority(self) -> None:
        """The rule a review cites when a design moves money because it can resume."""
        assert "resumable, never authorised" in TEXT


class TestWhatThePackagingHasToHold:
    def test_every_extra_the_record_names_is_declared(self) -> None:
        """A record naming an install command nobody can run is worse than no record."""
        declared = set(pyproject()["project"].get("optional-dependencies", {}))

        for named in EXTRA.findall(TEXT):
            assert set(named.split(",")) <= declared, named

    def test_it_promises_no_workflow_import_at_package_import_time(self) -> None:
        """Held by tests/test_extras.py; the record states what that test defends."""
        assert "temporalio" in TEXT
        assert "tests/test_extras.py" in TEXT


class TestTheReviewChecklist:
    def test_the_checklist_is_there_to_be_copied_into_a_review(self) -> None:
        section = TEXT.split("## Review checklist", 1)[1].split("\n## ", 1)[0]

        assert section.count("- [ ]") >= 5

    def test_it_is_guidance_rather_than_a_gate(self) -> None:
        """A team with a legitimate unusual case must be able to write the reason down."""
        assert "record the reason" in TEXT


def test_every_page_it_sends_a_reader_to_exists() -> None:
    """A decision record that dead-ends is not read twice."""
    for link in LINK.findall(TEXT):
        assert (RECORD.parent / link).resolve().exists(), link
