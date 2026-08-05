"""Quarantine is a loan against the suite, and every loan needs a due date.

An indefinitely retried or skipped test is a test nobody will ever fix; it stays
green in the report while covering nothing.
"""

from typing import cast

import pytest
from _pytest.outcomes import Skipped

from tesserix_adk.testing import QuarantineError
from tesserix_adk.testing.pytest_plugin import _quarantine_reason, pytest_runtest_setup

pytest_plugins = ["pytester"]

CONFTEST = 'pytest_plugins = ["tesserix_adk.testing.pytest_plugin"]'


def test_a_quarantined_test_does_not_fail_the_run(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.quarantine(owner="@sam123ben", expires="2099-01-01", reason="flaky")
        def test_flaky():
            raise AssertionError("intermittent")
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:randomly", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*@sam123ben*"])


def test_an_expired_quarantine_fails_the_run(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.quarantine(owner="@sam123ben", expires="2000-01-01", reason="flaky")
        def test_flaky():
            pass
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:randomly")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*expired*"])


@pytest.mark.parametrize("missing", ["owner", "expires", "reason"])
def test_a_quarantine_without_full_provenance_is_rejected(
    pytester: pytest.Pytester, missing: str
) -> None:
    args = {"owner": '"@sam123ben"', "expires": '"2099-01-01"', "reason": '"flaky"'}
    del args[missing]
    marker = ", ".join(f"{k}={v}" for k, v in args.items())
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        f"""
        import pytest

        @pytest.mark.quarantine({marker})
        def test_flaky():
            pass
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:randomly")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines([f"*{missing}*"])


@pytest.mark.parametrize("missing", ["owner", "expires", "reason"])
def test_the_reason_builder_names_the_missing_field(missing: str) -> None:
    kwargs = {"owner": "@sam123ben", "expires": "2099-01-01", "reason": "flaky"}
    del kwargs[missing]
    with pytest.raises(QuarantineError, match=missing):
        _quarantine_reason(kwargs)


def test_the_reason_builder_states_owner_expiry_and_reason() -> None:
    reason = _quarantine_reason({"owner": "@sam123ben", "expires": "2099-01-01", "reason": "flaky"})
    assert "@sam123ben" in reason
    assert "2099-01-01" in reason
    assert "flaky" in reason


def test_the_reason_builder_rejects_an_expired_quarantine() -> None:
    with pytest.raises(QuarantineError, match="expired"):
        _quarantine_reason({"owner": "@sam123ben", "expires": "2000-01-01", "reason": "flaky"})


class _Item:
    """The two members of `pytest.Item` the hook actually uses."""

    def __init__(self, marker: pytest.Mark | None) -> None:
        self._marker = marker

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        return self._marker if name == "quarantine" else None


def _item(**kwargs: str) -> pytest.Item:
    marker = pytest.mark.quarantine(**kwargs).mark if kwargs else None
    return cast("pytest.Item", _Item(marker))


def test_the_hook_ignores_a_test_with_no_quarantine_marker() -> None:
    pytest_runtest_setup(_item())  # must not raise


def test_the_hook_skips_a_live_quarantine() -> None:
    with pytest.raises(Skipped, match="@sam123ben"):
        pytest_runtest_setup(_item(owner="@sam123ben", expires="2099-01-01", reason="flaky"))


def test_the_hook_fails_an_expired_quarantine() -> None:
    with pytest.raises(QuarantineError, match="expired"):
        pytest_runtest_setup(_item(owner="@sam123ben", expires="2000-01-01", reason="flaky"))


def test_the_suite_has_no_quarantined_tests_of_its_own() -> None:
    """Recorded so that adding the first one is a visible decision."""
    from pathlib import Path

    tests = Path(__file__).resolve().parent
    quarantined = [p.name for p in tests.rglob("test_*.py") if "mark.quarantine(" in p.read_text()]
    assert quarantined == ["test_quarantine.py"], quarantined
