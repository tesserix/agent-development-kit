"""The scanning workflow's shape, asserted rather than reviewed by eye.

Two properties matter and neither shows up in a green run: that the scan repeats on a
schedule, so an advisory published after a merge is still caught, and that it needs no
secrets, so it runs on a pull request from a fork instead of silently skipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.ci_config import load_yaml, triggers

SECURITY = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "security.yml"


def jobs() -> dict[str, Any]:
    listed: dict[str, Any] = load_yaml(SECURITY)["jobs"]
    return listed


def run_steps(job: str) -> list[str]:
    return [step["run"] for step in jobs()[job].get("steps", []) if "run" in step]


class TestTriggers:
    def test_every_pull_request_is_scanned(self) -> None:
        assert "pull_request" in triggers(SECURITY)

    def test_the_scan_repeats_on_a_schedule(self) -> None:
        """An advisory published the day after a merge is not caught by a merge trigger."""
        assert triggers(SECURITY)["schedule"]

    def test_main_is_scanned_on_push_so_the_default_branch_has_a_current_verdict(self) -> None:
        assert triggers(SECURITY)["push"]["branches"] == ["main"]


class TestForkedPullRequests:
    def test_the_workflow_needs_no_secrets(self) -> None:
        """A scan that skips on a fork's pull request is a scan with a hole in it, and
        one that exposes a token to fork-controlled code is worse than no scan."""
        assert "secrets." not in SECURITY.read_text(encoding="utf-8")

    def test_the_workflow_is_read_only(self) -> None:
        assert load_yaml(SECURITY)["permissions"] == {"contents": "read"}


class TestAdvisories:
    def test_the_lockfile_is_audited(self) -> None:
        assert any("tools.audit" in step for step in run_steps("advisories"))

    def test_the_frozen_lock_is_installed_so_the_scan_matches_what_ci_ships(self) -> None:
        assert any("--frozen" in step for step in run_steps("advisories"))


class TestSecrets:
    def test_the_tree_is_scanned_for_credential_shapes(self) -> None:
        assert any("tools.secret_scan" in step for step in run_steps("secrets"))

    def test_the_full_history_is_available_to_the_scan(self) -> None:
        """A credential removed in a later commit is still in the history and still live."""
        checkout = next(
            step for step in jobs()["secrets"]["steps"] if "checkout" in str(step.get("uses"))
        )
        assert checkout["with"]["fetch-depth"] == 0

    def test_the_history_itself_is_scanned_not_only_the_working_tree(self) -> None:
        assert any("gitleaks" in step for step in run_steps("secrets"))


class TestInjection:
    def test_no_event_input_is_interpolated_into_a_shell(self) -> None:
        for job in jobs():
            assert not any("${{" in step for step in run_steps(job))


class TestLicences:
    """A licence obligation arrives silently and surfaces in a legal review years later."""

    def test_every_pull_request_is_checked_against_the_licence_policy(self) -> None:
        assert any("tools.licences" in step for step in run_steps("licences"))

    def test_the_check_covers_the_extras_as_well_as_the_base_install(self) -> None:
        assert any("--all-extras" in step for step in run_steps("licences"))
