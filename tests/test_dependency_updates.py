"""How dependency updates arrive and what has to pass before one merges.

Updates that arrive at no fixed time arrive when somebody remembers, which is after the
advisory. Updates that arrive as forty separate pull requests are not reviewed, they are
approved. Both failure modes are configuration, so both are asserted here.
"""

from __future__ import annotations

import json
from typing import Any

from tests.ci_config import CI, ROOT, ci_jobs, ci_run_steps, triggers

RENOVATE = ROOT / "renovate.json"
CODEOWNERS = ROOT / ".github" / "CODEOWNERS"
DOCS = ROOT / "docs" / "dependencies.md"
POLICY_JOB = "dependency-policy"
FULL_MATRIX = "test-matrix"


def _config() -> dict[str, Any]:
    config: dict[str, Any] = json.loads(RENOVATE.read_text(encoding="utf-8"))
    return config


class TestCadence:
    def test_python_dependencies_are_checked_on_a_fixed_cadence(self) -> None:
        config = _config()
        assert "pep621" in config["enabledManagers"]
        assert config["schedule"] == ["before 6am on monday"]
        assert config["timezone"] == "Australia/Sydney"

    def test_the_actions_the_workflows_pin_are_updated_too(self) -> None:
        assert "github-actions" in _config()["enabledManagers"]


class TestGrouping:
    def test_routine_updates_arrive_as_one_change(self) -> None:
        for manager in ("pep621", "github-actions"):
            groups = [
                rule
                for rule in _config()["packageRules"]
                if manager in rule.get("matchManagers", []) and rule.get("groupName")
            ]
            assert len(groups) == 1
            assert set(groups[0]["matchUpdateTypes"]) == {"minor", "patch", "pin", "digest"}

    def test_a_major_upgrade_is_not_folded_into_the_batch(self) -> None:
        rules = _config()["packageRules"]
        assert rules[-1]["matchUpdateTypes"] == ["major"]
        assert rules[-1]["groupName"] is None
        assert not _config()["automerge"]

    def test_a_dependency_change_is_labelled_so_the_full_gates_can_find_it(self) -> None:
        assert "dependencies" in _config()["labels"]


class TestWhatMustPassBeforeMerge:
    def test_the_published_requirements_are_checked_on_every_pull_request(self) -> None:
        assert "tools.dependency_policy" in " ".join(ci_run_steps(POLICY_JOB))

    def test_the_full_matrix_runs_on_a_dependency_change(self) -> None:
        """The fast lane proves two ends of the range; an update has to prove all of it."""
        assert "dependencies" in str(ci_jobs()[FULL_MATRIX]["if"])

    def test_labelling_an_open_pull_request_starts_the_full_matrix(self) -> None:
        """Otherwise a hand-labelled bump keeps the fast lane's verdict from before."""
        assert "labeled" in triggers(CI)["pull_request"]["types"]

    def test_the_lowest_declared_versions_are_proved_on_every_pull_request(self) -> None:
        """A floor nothing resolves against is a floor nobody has checked."""
        assert "--resolution lowest-direct" in " ".join(ci_run_steps("lowest-direct"))


class TestReviewRota:
    def test_a_dependency_change_has_a_named_reviewer(self) -> None:
        """Unowned update pull requests accumulate until somebody merges the pile."""
        owners = CODEOWNERS.read_text(encoding="utf-8")
        assert "uv.lock" in owners
        assert "pyproject.toml" in owners


class TestInjection:
    def test_no_event_input_reaches_a_shell_in_the_new_job(self) -> None:
        assert "${{" not in " ".join(ci_run_steps(POLICY_JOB))
        assert CI.exists()


class TestDocumentation:
    def test_the_policy_is_written_down_where_a_consumer_can_read_it(self) -> None:
        """A resolution error against the kit is debugged from the outside."""
        page = DOCS.read_text(encoding="utf-8").lower()
        assert "lowest-direct" in page
        assert "upper bound" in page

    def test_the_security_fast_track_is_documented(self) -> None:
        """The weekly cadence must not be what an advisory waits for."""
        assert "advisor" in DOCS.read_text(encoding="utf-8").lower()
