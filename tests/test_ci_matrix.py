"""The supported-versions table is a promise; the matrix is what makes it true.

Every one of these tests fails when the promise and the matrix disagree, in either
direction — an untested claim and an unclaimed test leg are both defects.
"""

from typing import Any

import pytest

from tests.ci_config import ci_jobs, ci_run_steps, pyproject, supported_minors

FAST_LANE = "test-fast"
FULL_MATRIX = "test-matrix"
EXTRAS = "test-extras"
ADVISORY = "test-advisory"


def _matrix(job: str) -> dict[str, list[str]]:
    strategy: dict[str, list[str]] = ci_jobs()[job]["strategy"]["matrix"]
    return strategy


def test_the_full_matrix_covers_every_supported_minor() -> None:
    assert _matrix(FULL_MATRIX)["python-version"] == supported_minors()


def test_the_fast_lane_runs_the_oldest_and_newest_supported_minors() -> None:
    """A leak of newer syntax and a leak of removed behaviour sit at opposite ends."""
    minors = supported_minors()
    assert set(_matrix(FAST_LANE)["python-version"]) == {minors[0], minors[-1]}


def test_the_fast_lane_includes_a_macos_leg() -> None:
    """Contributors develop on macOS; a Linux-only lane exports their breakage to them."""
    assert any("macos" in os for os in _matrix(FAST_LANE)["os"])


def test_the_full_matrix_does_not_stop_at_the_first_failing_leg() -> None:
    """One red leg tells you far less than knowing which legs are red."""
    assert ci_jobs()[FULL_MATRIX]["strategy"]["fail-fast"] is False


def test_the_advisory_leg_cannot_block_a_merge() -> None:
    """Pre-release and free-threaded interpreters are information, not a gate."""
    assert ci_jobs()[ADVISORY]["continue-on-error"] is True


def test_the_full_matrix_does_not_run_on_every_pull_request() -> None:
    """The fast lane keeps review quick; the matrix runs where latency does not matter."""
    condition = ci_jobs()[FULL_MATRIX]["if"]
    assert "pull_request" in condition


def test_every_declared_extra_has_a_standalone_matrix_leg() -> None:
    """Proves optional imports are genuinely optional, one extra at a time."""
    # `all` is the union and gets a leg of its own; this test is about the parts.
    declared = set(pyproject()["project"].get("optional-dependencies", {})) - {"all"}
    legs = set(_matrix(EXTRAS)["extra"]) - {"none", "all"}
    assert legs == declared


def test_the_extras_leg_covers_the_bare_install_and_the_union() -> None:
    covered = set(_matrix(EXTRAS)["extra"])
    assert {"none", "all"} <= covered


@pytest.mark.parametrize("job", [FAST_LANE, FULL_MATRIX, EXTRAS])
def test_every_test_leg_records_the_random_seed(job: str) -> None:
    """A failing order is only useful if the log says how to reproduce it."""
    assert any("randomly-seed" in step for step in ci_run_steps(job))


def _pytest_config() -> dict[str, Any]:
    config: dict[str, Any] = pyproject()["tool"]["pytest"]["ini_options"]
    return config


def test_warnings_from_the_kit_are_errors() -> None:
    """A DeprecationWarning nobody sees is a breakage scheduled for a consumer."""
    assert "error" in _pytest_config()["filterwarnings"]


def test_test_order_is_randomised() -> None:
    assert "pytest-randomly" in " ".join(pyproject()["dependency-groups"]["test"])
    assert "no:randomly" not in _pytest_config()["addopts"]


def test_the_coverage_floor_lives_in_one_reviewed_place() -> None:
    """A floor repeated in a workflow file is a floor that can be lowered quietly."""
    assert pyproject()["tool"]["coverage"]["report"]["fail_under"] >= 90
    hardcoded = [
        step
        for job in (FAST_LANE, FULL_MATRIX, EXTRAS)
        for step in ci_run_steps(job)
        if "--cov-fail-under" in step
    ]
    assert hardcoded == []
