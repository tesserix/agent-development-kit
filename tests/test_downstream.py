"""The downstream job has to answer one question: did *we* break them?

A consumer's suite is red for their own reasons half the time. A job that fails whenever
their suite is red teaches the team to ignore it, and then it catches nothing. So the
suite runs twice — against the last stable and against the alpha — and only a failure
that is new under the alpha is attributed to the kit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tools import downstream

if TYPE_CHECKING:
    from pathlib import Path


def junit(*cases: tuple[str, str, bool]) -> str:
    body = "".join(
        f'<testcase classname="{cls}" name="{name}">'
        + ("<failure>boom</failure>" if failed else "")
        + "</testcase>"
        for cls, name, failed in cases
    )
    return f'<?xml version="1.0"?><testsuite name="s">{body}</testsuite>'


class TestReadingResults:
    def test_a_failing_case_is_reported_by_its_full_name(self, tmp_path: Path) -> None:
        path = tmp_path / "results.xml"
        path.write_text(junit(("tests.test_agent", "test_runs", True)), encoding="utf-8")
        assert downstream.failing(path) == {"tests.test_agent::test_runs"}

    def test_a_passing_suite_has_no_failures(self, tmp_path: Path) -> None:
        path = tmp_path / "results.xml"
        path.write_text(junit(("tests.test_agent", "test_runs", False)), encoding="utf-8")
        assert downstream.failing(path) == set()

    def test_an_error_counts_as_a_failure(self, tmp_path: Path) -> None:
        """A collection error is how a removed symbol usually shows up."""
        path = tmp_path / "results.xml"
        path.write_text(
            '<testsuite><testcase classname="t" name="a"><error>ImportError</error>'
            "</testcase></testsuite>",
            encoding="utf-8",
        )
        assert downstream.failing(path) == {"t::a"}

    def test_a_skipped_case_is_not_a_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "results.xml"
        path.write_text(
            '<testsuite><testcase classname="t" name="a"><skipped/></testcase></testsuite>',
            encoding="utf-8",
        )
        assert downstream.failing(path) == set()

    def test_nested_suites_are_read(self, tmp_path: Path) -> None:
        """pytest writes a testsuites wrapper; junit-xml writers differ on this."""
        path = tmp_path / "results.xml"
        path.write_text(
            '<testsuites><testsuite><testcase classname="t" name="a">'
            "<failure/></testcase></testsuite></testsuites>",
            encoding="utf-8",
        )
        assert downstream.failing(path) == {"t::a"}

    def test_a_results_file_that_is_not_readable_is_an_error_not_an_empty_suite(
        self, tmp_path: Path
    ) -> None:
        """An empty result set would silently read as "the consumer is fine"."""
        path = tmp_path / "results.xml"
        path.write_text("not xml at all", encoding="utf-8")
        with pytest.raises(downstream.DownstreamError):
            downstream.failing(path)

    def test_a_missing_results_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(downstream.DownstreamError, match=r"absent\.xml"):
            downstream.failing(tmp_path / "absent.xml")


class TestAttribution:
    def test_a_failure_that_is_new_under_the_alpha_is_attributed_to_the_kit(self) -> None:
        assert downstream.attributable(baseline=set(), candidate={"t::a"}) == ["t::a"]

    def test_a_failure_the_consumer_already_had_is_not_ours(self) -> None:
        """Otherwise the job fails on their unrelated breakage and stops being believed."""
        assert downstream.attributable(baseline={"t::a"}, candidate={"t::a"}) == []

    def test_a_failure_the_alpha_fixed_is_not_reported_as_a_problem(self) -> None:
        assert downstream.attributable(baseline={"t::a"}, candidate=set()) == []

    def test_every_new_failure_is_reported_in_a_stable_order(self) -> None:
        found = downstream.attributable(baseline=set(), candidate={"t::b", "t::a"})
        assert found == ["t::a", "t::b"]


class TestCommandLine:
    def _results(self, tmp_path: Path, name: str, failed: bool) -> Path:
        path = tmp_path / name
        path.write_text(junit(("tests.test_agent", "test_runs", failed)), encoding="utf-8")
        return path

    def test_a_regression_fails_the_job_and_names_the_symbol(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        baseline = self._results(tmp_path, "stable.xml", failed=False)
        candidate = self._results(tmp_path, "alpha.xml", failed=True)

        assert downstream.main(["--baseline", str(baseline), "--candidate", str(candidate)]) == 1
        assert "tests.test_agent::test_runs" in capsys.readouterr().err

    def test_a_consumer_suite_that_was_already_red_does_not_fail_the_job(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        baseline = self._results(tmp_path, "stable.xml", failed=True)
        candidate = self._results(tmp_path, "alpha.xml", failed=True)

        assert downstream.main(["--baseline", str(baseline), "--candidate", str(candidate)]) == 0
        assert "not attributable" in capsys.readouterr().out

    def test_a_clean_run_reports_the_alpha_as_safe_for_that_consumer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        baseline = self._results(tmp_path, "stable.xml", failed=False)
        candidate = self._results(tmp_path, "alpha.xml", failed=False)

        assert downstream.main(["--baseline", str(baseline), "--candidate", str(candidate)]) == 0
        assert "no regression" in capsys.readouterr().out.lower()

    def test_an_unreadable_result_fails_the_job_rather_than_passing_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        baseline = self._results(tmp_path, "stable.xml", failed=False)
        assert downstream.main(["--baseline", str(baseline), "--candidate", "absent.xml"]) == 1
        assert "absent.xml" in capsys.readouterr().err

    def test_the_regression_is_written_where_the_release_notes_can_pick_it_up(
        self, tmp_path: Path
    ) -> None:
        """A finding nobody records is a finding the next release repeats."""
        baseline = self._results(tmp_path, "stable.xml", failed=False)
        candidate = self._results(tmp_path, "alpha.xml", failed=True)
        report = tmp_path / "regressions.md"

        downstream.main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--report",
                str(report),
            ]
        )
        assert "tests.test_agent::test_runs" in report.read_text(encoding="utf-8")
