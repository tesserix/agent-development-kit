"""Business rules belong in code. The lint is what keeps them from drifting into prose."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core import (
    RULES,
    LintReport,
    lint_directory,
    lint_prompt,
)

if TYPE_CHECKING:
    from pathlib import Path

REFUND = "If the booking was made more than 24 hours ago, refund 50% of the fare."


class TestWhatMustNotBeInAPrompt:
    """Every rule here was a real production incident somewhere before it was a rule."""

    def test_a_monetary_threshold_is_flagged(self) -> None:
        (finding,) = lint_prompt("Waive the fee for orders over $500.")

        assert finding.code == "ADK-P001"
        assert finding.line == 1

    def test_a_percentage_of_money_is_flagged(self) -> None:
        codes = {finding.code for finding in lint_prompt(REFUND)}

        assert "ADK-P001" in codes

    def test_a_conditional_rule_chain_is_flagged(self) -> None:
        codes = {finding.code for finding in lint_prompt(REFUND)}

        assert "ADK-P002" in codes

    def test_a_time_window_rule_is_flagged(self) -> None:
        codes = {finding.code for finding in lint_prompt(REFUND)}

        assert "ADK-P003" in codes

    def test_authorisation_language_is_flagged(self) -> None:
        (finding,) = lint_prompt("You may approve the discount yourself.")

        assert finding.code == "ADK-P004"

    def test_an_irreversible_action_is_flagged(self) -> None:
        (finding,) = lint_prompt("Issue the refund to the customer's card.")

        assert finding.code == "ADK-P005"
        assert "approval" in finding.remedy

    def test_an_embedded_endpoint_is_flagged(self) -> None:
        (finding,) = lint_prompt("Call POST /v1/refunds when they ask.")

        assert finding.code == "ADK-P006"

    def test_every_finding_names_the_line_it_came_from(self) -> None:
        (finding,) = lint_prompt("You are helpful.\nWaive the fee for orders over $500.")

        assert finding.line == 2
        assert finding.text == "Waive the fee for orders over $500."

    def test_a_finding_reads_as_a_line_a_reviewer_can_act_on(self) -> None:
        (finding,) = lint_prompt("Issue the refund.", source="prompts/refunds.toml")

        assert str(finding).startswith("prompts/refunds.toml:1: ADK-P005")
        assert str(lint_prompt("Issue the refund.")[0]).startswith("line 1:")

    def test_every_rule_has_a_code_a_severity_and_a_remedy(self) -> None:
        assert {rule.code for rule in RULES} == {rule.code for rule in RULES if rule.remedy}
        assert all(rule.severity in {"error", "warning"} for rule in RULES)


class TestWhatIsLeftAlone:
    """A lint routinely disabled protects nothing, so ordinary prose has to pass."""

    def test_a_prompt_that_only_frames_the_task_is_clean(self) -> None:
        assert lint_prompt("You are a support agent. Answer briefly and cite the policy.") == ()

    def test_an_output_shaping_number_is_not_a_business_rule(self) -> None:
        assert lint_prompt("Return at most 5 suggestions, in 2 sentences each.") == ()

    def test_asking_for_a_decision_rather_than_making_one_is_clean(self) -> None:
        assert lint_prompt("Ask the refund tool whether this booking qualifies.") == ()


class TestSuppression:
    """A suppression is a decision with an owner, not a way to make the check quiet."""

    def test_a_justified_suppression_removes_the_finding(self) -> None:
        text = (
            "Waive the fee for orders over $500."
            "  # adk-lint: allow ADK-P001 — legacy tariff, ticket PLAT-91"
        )

        assert lint_prompt(text) == ()

    def test_a_suppression_without_a_reason_is_itself_a_finding(self) -> None:
        (finding,) = lint_prompt("Waive the fee for orders over $500.  # adk-lint: allow ADK-P001")

        assert finding.code == "ADK-P000"

    def test_a_suppression_only_covers_the_code_it_names(self) -> None:
        text = "Issue the refund.  # adk-lint: allow ADK-P001 — unrelated, kept for the audit"

        assert {finding.code for finding in lint_prompt(text)} == {"ADK-P005"}

    def test_a_suppression_on_the_line_above_covers_the_line_below(self) -> None:
        text = (
            "# adk-lint: allow ADK-P001 — legacy tariff, ticket PLAT-91\nWaive the fee over $500."
        )

        assert lint_prompt(text) == ()


class TestRunningItOverAProject:
    """The check runs in CI over a prompt directory, and says what it found."""

    def test_every_prompt_file_in_the_tree_is_read(self, tmp_path: Path) -> None:
        (tmp_path / "greeting.toml").write_text('body = "Greet them."', encoding="utf-8")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "refunds.toml").write_text(REFUND, encoding="utf-8")

        report = lint_directory(tmp_path)

        assert report.files == 2
        assert report.ok is False

    def test_a_clean_tree_passes(self, tmp_path: Path) -> None:
        (tmp_path / "greeting.toml").write_text('body = "Greet them."', encoding="utf-8")

        report = lint_directory(tmp_path)

        assert report.ok is True
        assert report.exit_code == 0

    def test_a_file_that_is_not_a_prompt_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.7")

        assert lint_directory(tmp_path).files == 0

    def test_a_finding_names_the_file_it_came_from(self, tmp_path: Path) -> None:
        (tmp_path / "refunds.toml").write_text(REFUND, encoding="utf-8")

        (finding, *_) = lint_directory(tmp_path).findings

        assert finding.source.endswith("refunds.toml")

    def test_warnings_alone_do_not_fail_the_check(self, tmp_path: Path) -> None:
        (tmp_path / "hints.toml").write_text(
            "Call POST /v1/refunds when they ask.", encoding="utf-8"
        )

        report = lint_directory(tmp_path)

        assert report.findings
        assert report.ok is True


class TestWhatTheSummarySays:
    """A project suppressing everything should be visible, not silently green."""

    def test_the_summary_counts_findings_by_code(self, tmp_path: Path) -> None:
        (tmp_path / "refunds.toml").write_text(REFUND, encoding="utf-8")

        summary = lint_directory(tmp_path).summary()

        assert "ADK-P001" in summary
        assert "1 file" in summary

    def test_the_summary_counts_suppressions_so_abuse_is_visible(self, tmp_path: Path) -> None:
        (tmp_path / "refunds.toml").write_text(
            "Waive the fee over $500.  # adk-lint: allow ADK-P001 — legacy tariff, PLAT-91",
            encoding="utf-8",
        )

        report = lint_directory(tmp_path)

        assert report.suppressed == 1
        assert "1 suppressed" in report.summary()

    def test_an_empty_report_says_so(self) -> None:
        assert "clean" in LintReport().summary()
