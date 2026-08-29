"""The command a contributor and CI both run, and what it refuses to do on their behalf.

The one thing this must never do is record a baseline while checking against it. A harness
that re-records what it just measured ratchets performance downwards and reports green
every time, so these pin that the check path leaves the file exactly as it found it.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest
from tools.benchmark import main, scenarios_from

from tesserix_adk.testing.benchmarks import Measurement, Metric, write_baseline

if TYPE_CHECKING:
    from pathlib import Path

# The harness collects garbage while it measures, which finalises whatever else the test
# session left unclosed. Those finalisers are not this module's to answer for.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


SUITE = "tests.benchmark_suite_fixture"


def recorded(where: Path, **values: float) -> None:
    """Put a baseline on disk for the fixture suite, on this interpreter."""
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    write_baseline(
        where,
        (
            Measurement(
                scenario="counting",
                python=python,
                values={Metric(name): value for name, value in values.items()},
                spread=0.01,
                rounds=2,
                iterations=2,
            ),
        ),
    )


class TestChecking:
    """What the exit code says, and what the contributor reads on the way past."""

    def test_a_run_with_nothing_recorded_reports_it_without_failing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--suite", SUITE, "--baseline", str(tmp_path / "baseline.json")])

        assert code == 0
        assert "no baseline" in capsys.readouterr().out

    def test_a_metric_far_past_its_baseline_fails_and_names_the_scenario(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        where = tmp_path / "baseline.json"
        recorded(where, throughput=1e20)

        code = main(["--suite", SUITE, "--baseline", str(where)])

        assert code == 1
        assert "counting" in capsys.readouterr().out

    def test_a_generous_baseline_passes(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"
        recorded(where, latency_p95=100.0)

        assert main(["--suite", SUITE, "--baseline", str(where)]) == 0

    def test_the_rounds_and_iterations_can_be_cut_for_a_local_run(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"

        code = main(
            ["--suite", SUITE, "--baseline", str(where), "--rounds", "1", "--iterations", "1"]
        )

        assert code == 0


class TestRecording:
    """A baseline moves in a reviewed commit, never as a side effect of checking."""

    def test_writing_records_what_was_measured(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"

        assert main(["--suite", SUITE, "--baseline", str(where), "--write"]) == 0
        assert "counting" in json.loads(where.read_text())["scenarios"]

    def test_a_check_never_writes_the_baseline_even_when_it_fails(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"
        recorded(where, throughput=1e20)
        before = where.read_text()

        assert main(["--suite", SUITE, "--baseline", str(where)]) == 1
        assert where.read_text() == before

    def test_a_check_does_not_create_a_baseline_that_was_absent(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"

        main(["--suite", SUITE, "--baseline", str(where)])

        assert not where.exists()

    def test_only_the_metrics_asked_for_are_recorded(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"

        main(["--suite", SUITE, "--baseline", str(where), "--write", "--only", "allocations"])

        held = json.loads(where.read_text())["scenarios"]["counting"]
        assert [list(one["metrics"]) for one in held.values()] == [["allocations"]]

    def test_what_was_recorded_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        where = tmp_path / "baseline.json"

        main(["--suite", SUITE, "--baseline", str(where), "--write"])

        assert "recorded" in capsys.readouterr().out


class TestResolvingTheSuite:
    """A suite is a module naming its scenarios, so a typo must not read as an empty run."""

    def test_the_shipped_suite_names_its_scenarios(self) -> None:
        assert [one.name for one in scenarios_from("benchmarks.suite")]

    def test_a_module_that_is_not_a_suite_is_refused(self) -> None:
        with pytest.raises(ValueError, match="scenarios"):
            scenarios_from("json")

    def test_a_missing_module_is_reported_rather_than_traced(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--suite", "nothing.here", "--baseline", str(tmp_path / "baseline.json")])

        assert code == 3
        assert "nothing.here" in capsys.readouterr().err


class TestTheBaselineIsComparable:
    """Baselines are keyed by interpreter, so a run on another one compares against nothing."""

    def test_another_interpreters_numbers_are_not_borrowed(self, tmp_path: Path) -> None:
        where = tmp_path / "baseline.json"
        write_baseline(
            where,
            (
                Measurement(
                    scenario="counting",
                    python="2.7",
                    values={Metric.TOKENS: 1.0},
                    spread=0.0,
                    rounds=2,
                    iterations=2,
                ),
            ),
        )

        assert main(["--suite", SUITE, "--baseline", str(where)]) == 0
