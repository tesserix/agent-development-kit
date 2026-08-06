"""A suppression is a decision with an owner and an end date, or it is a way to lose.

The failure mode of every scanner is the same: a finding is inconvenient, someone silences
it, and two years later nobody knows whether it still applies. So a suppression that has
expired is itself a build failure — the same weight as the finding it was hiding.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest
from tools import security_policy as policy

if TYPE_CHECKING:
    from pathlib import Path

TODAY = dt.date(2026, 8, 6)

ENTRY = """
[[suppression]]
id = "GHSA-aaaa-bbbb-cccc"
kind = "advisory"
owner = "@sam123ben"
reason = "Reachable only from the docs build, which runs on no case data."
expires = "2026-10-01"
"""


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "policy.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestLoading:
    def test_a_complete_suppression_loads(self, tmp_path: Path) -> None:
        loaded = policy.load(write(tmp_path, ENTRY), today=TODAY)
        assert loaded.suppressions[0].id == "GHSA-aaaa-bbbb-cccc"
        assert loaded.suppressions[0].owner == "@sam123ben"

    def test_an_expiry_written_as_a_toml_date_is_accepted(self, tmp_path: Path) -> None:
        """TOML has a date type, and someone will use it rather than quoting the string."""
        native = ENTRY.replace('expires = "2026-10-01"', "expires = 2026-10-01")
        loaded = policy.load(write(tmp_path, native), today=TODAY)
        assert loaded.suppressions[0].expires == dt.date(2026, 10, 1)

    def test_a_policy_that_is_not_valid_toml_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(policy.PolicyError, match="valid TOML"):
            policy.load(write(tmp_path, "[[suppression]\nid = "), today=TODAY)

    def test_an_empty_policy_is_valid(self, tmp_path: Path) -> None:
        """The state to aim for; a scanner with nothing suppressed is the goal, not a bug."""
        assert policy.load(write(tmp_path, ""), today=TODAY).suppressions == ()

    def test_a_missing_policy_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(policy.PolicyError, match=r"policy\.toml"):
            policy.load(tmp_path / "policy.toml")

    @pytest.mark.parametrize("field", ["owner", "reason", "expires", "id", "kind"])
    def test_a_suppression_missing_any_required_field_is_rejected(
        self, tmp_path: Path, field: str
    ) -> None:
        body = "\n".join(line for line in ENTRY.splitlines() if not line.startswith(f"{field} "))
        with pytest.raises(policy.PolicyError, match=field):
            policy.load(write(tmp_path, body), today=TODAY)

    def test_an_unknown_field_is_rejected_rather_than_ignored(self, tmp_path: Path) -> None:
        """A typo in `expires` that is silently ignored is a suppression with no end date."""
        with pytest.raises(policy.PolicyError, match="expiry"):
            policy.load(write(tmp_path, ENTRY + 'expiry = "2027-01-01"\n'))

    def test_an_unknown_kind_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(policy.PolicyError, match="vibes"):
            policy.load(write(tmp_path, ENTRY.replace('"advisory"', '"vibes"')), today=TODAY)

    def test_a_date_that_is_not_a_date_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(policy.PolicyError, match="expires"):
            policy.load(write(tmp_path, ENTRY.replace("2026-10-01", "soon")), today=TODAY)

    def test_a_reason_that_says_nothing_is_rejected(self, tmp_path: Path) -> None:
        """`reason = "needed"` is not a reason anyone can review later."""
        short = write(tmp_path, ENTRY.replace(ENTRY.split("\n")[5], 'reason = "wip"'))
        with pytest.raises(policy.PolicyError, match="reason"):
            policy.load(short, today=TODAY)

    def test_an_expiry_further_out_than_the_maximum_is_rejected(self, tmp_path: Path) -> None:
        """An open-ended suppression with a date on it is still open-ended."""
        with pytest.raises(policy.PolicyError, match="90 days"):
            policy.load(write(tmp_path, ENTRY.replace("2026-10-01", "2030-01-01")), today=TODAY)

    def test_two_suppressions_for_the_same_finding_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(policy.PolicyError, match="twice"):
            policy.load(write(tmp_path, ENTRY + ENTRY), today=TODAY)


class TestUnfixedAdvisories:
    def test_an_advisory_with_no_fix_needs_a_stated_mitigation(self, tmp_path: Path) -> None:
        """There is nothing to upgrade to, so the only real answer is what we did instead."""
        loaded = policy.load(
            write(tmp_path, ENTRY + 'mitigation = "Egress blocked."\n'), today=TODAY
        )
        assert loaded.suppressions[0].mitigation == "Egress blocked."

    def test_a_suppression_may_omit_a_mitigation_when_a_fix_exists(self, tmp_path: Path) -> None:
        assert policy.load(write(tmp_path, ENTRY), today=TODAY).suppressions[0].mitigation is None


class TestExpiry:
    def test_a_live_suppression_applies(self, tmp_path: Path) -> None:
        loaded = policy.load(write(tmp_path, ENTRY), today=TODAY)
        assert loaded.suppresses("GHSA-aaaa-bbbb-cccc", kind="advisory", today=TODAY)

    def test_an_expired_suppression_does_not_apply(self, tmp_path: Path) -> None:
        loaded = policy.load(write(tmp_path, ENTRY), today=TODAY)
        assert not loaded.suppresses(
            "GHSA-aaaa-bbbb-cccc", kind="advisory", today=dt.date(2027, 1, 1)
        )

    def test_a_suppression_expires_at_the_end_of_its_last_day(self, tmp_path: Path) -> None:
        loaded = policy.load(write(tmp_path, ENTRY), today=TODAY)
        assert loaded.suppresses("GHSA-aaaa-bbbb-cccc", kind="advisory", today=dt.date(2026, 10, 1))

    def test_a_suppression_of_one_kind_does_not_silence_another(self, tmp_path: Path) -> None:
        """An accepted advisory must not also excuse a committed credential."""
        loaded = policy.load(write(tmp_path, ENTRY), today=TODAY)
        assert not loaded.suppresses("GHSA-aaaa-bbbb-cccc", kind="secret", today=TODAY)

    def test_expired_suppressions_are_reported_for_the_build_to_fail_on(
        self, tmp_path: Path
    ) -> None:
        loaded = policy.load(write(tmp_path, ENTRY), today=TODAY)
        assert loaded.expired(dt.date(2027, 1, 1)) == [loaded.suppressions[0]]

    def test_nothing_is_expired_while_it_is_live(self, tmp_path: Path) -> None:
        assert policy.load(write(tmp_path, ENTRY), today=TODAY).expired(TODAY) == []


class TestSeverity:
    @pytest.mark.parametrize("level", ["critical", "high"])
    def test_a_serious_finding_blocks(self, level: str) -> None:
        assert policy.blocks(level)

    @pytest.mark.parametrize("level", ["moderate", "low"])
    def test_a_lesser_finding_is_tracked_rather_than_blocking(self, level: str) -> None:
        assert not policy.blocks(level)

    def test_an_unrated_finding_blocks(self) -> None:
        """Fail closed: an advisory nobody has scored yet is not evidence of safety."""
        assert policy.blocks("unknown")

    def test_severity_is_read_case_insensitively(self) -> None:
        assert policy.blocks("CRITICAL")


class TestTheRepositoryPolicy:
    def test_the_committed_policy_is_valid_today(self) -> None:
        """Runs in CI, so an expiry passing silently is not possible."""
        loaded = policy.load(policy.POLICY)
        assert loaded.expired(dt.date.today()) == []
