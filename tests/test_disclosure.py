"""The coordinated disclosure process, as far as it can be checked by a machine.

The prose in `SECURITY.md` is a promise: a private channel, an acknowledgement inside a
stated window, and a patched release for every supported minor landing with the advisory
rather than before it. A promise nothing checks is a promise that quietly stops being
true, so the dates and the version coverage live in records the release gates read.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest
from tools import disclosure

if TYPE_CHECKING:
    from pathlib import Path

TODAY = dt.date(2026, 8, 6)

MITIGATION = "Set the retrieval guardrail to strict until the patch is installed."

RECORD = f"""
id = "ADK-2026-0001"
title = "Retrieved content escapes the boundary guardrail"
severity = "high"
reported = 2026-07-01
acknowledged = 2026-07-02
published = 2026-07-10
notified = 2026-07-10
profiles = ["base"]
affected = ["0.1", "0.2"]
fixed = ["0.1.4", "0.2.1"]
mitigation = "{MITIGATION}"
credit = "Reported privately by a security researcher."
"""


def write(tmp_path: Path, body: str = RECORD, name: str = "ADK-2026-0001.toml") -> Path:
    directory = tmp_path / "advisories"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(body, encoding="utf-8")
    return directory


def check(tmp_path: Path, body: str = RECORD) -> list[str]:
    advisories = disclosure.advisories(write(tmp_path, body))
    return disclosure.violations(advisories, targets=disclosure.targets(), today=TODAY)


class TestAcknowledgement:
    def test_a_report_acknowledged_inside_the_published_target_passes(self, tmp_path: Path) -> None:
        assert check(tmp_path) == []

    def test_an_acknowledgement_later_than_the_target_is_a_violation(self, tmp_path: Path) -> None:
        """The published target is what a reporter decides to report privately against."""
        late = RECORD.replace("acknowledged = 2026-07-02", "acknowledged = 2026-07-30")
        found = check(tmp_path, late)
        assert found
        assert "acknowledg" in found[0]

    def test_an_acknowledgement_before_the_report_is_a_violation(self, tmp_path: Path) -> None:
        early = RECORD.replace("acknowledged = 2026-07-02", "acknowledged = 2026-06-01")
        found = check(tmp_path, early)
        assert found


class TestSupportedVersionCoverage:
    def test_every_affected_minor_gets_a_patched_release(self, tmp_path: Path) -> None:
        """A fix for the newest minor only leaves the other supported minor exposed."""
        found = check(tmp_path, RECORD.replace('fixed = ["0.1.4", "0.2.1"]', 'fixed = ["0.2.1"]'))
        assert found
        assert "0.1" in found[0]

    def test_a_fix_outside_the_affected_minors_is_a_violation(self, tmp_path: Path) -> None:
        """A version nobody said was affected in the fix list means one list is wrong."""
        stray = RECORD.replace('fixed = ["0.1.4", "0.2.1"]', 'fixed = ["0.1.4", "0.2.1", "0.3.0"]')
        found = check(tmp_path, stray)
        assert found


class TestPublication:
    def test_consumers_are_notified_no_later_than_publication(self, tmp_path: Path) -> None:
        """Embedding products should learn from the process, not from an index diff."""
        found = check(tmp_path, RECORD.replace("notified = 2026-07-10", "notified = 2026-07-14"))
        assert found
        assert "notif" in found[0]

    def test_an_advisory_published_before_it_was_acknowledged_is_a_violation(
        self, tmp_path: Path
    ) -> None:
        found = check(tmp_path, RECORD.replace("published = 2026-07-10", "published = 2026-07-01"))
        assert found

    def test_an_unfixed_advisory_inside_its_window_is_not_yet_a_violation(
        self, tmp_path: Path
    ) -> None:
        """Work in progress is not a failure until the published target passes."""
        body = RECORD.replace("reported = 2026-07-01", "reported = 2026-08-05")
        body = body.replace("acknowledged = 2026-07-02", "acknowledged = 2026-08-05")
        body = body.replace("published = 2026-07-10\n", "")
        body = body.replace("notified = 2026-07-10\n", "")
        body = body.replace('fixed = ["0.1.4", "0.2.1"]', "fixed = []")
        assert check(tmp_path, body) == []

    def test_an_unfixed_advisory_past_its_window_is_a_violation(self, tmp_path: Path) -> None:
        body = RECORD.replace("published = 2026-07-10\n", "")
        body = body.replace("notified = 2026-07-10\n", "")
        body = body.replace('fixed = ["0.1.4", "0.2.1"]', "fixed = []")
        found = check(tmp_path, body)
        assert found
        assert "ADK-2026-0001" in found[0]

    def test_a_fix_with_no_published_advisory_is_a_violation(self, tmp_path: Path) -> None:
        """A patch on the index with no advisory is a fix consumers cannot act on."""
        body = RECORD.replace("published = 2026-07-10\n", "")
        found = check(tmp_path, body)
        assert found
        assert "published advisory" in found[0]

    def test_a_fix_with_no_record_of_notifying_consumers_is_a_violation(
        self, tmp_path: Path
    ) -> None:
        found = check(tmp_path, RECORD.replace("notified = 2026-07-10\n", ""))
        assert found
        assert "notifying" in found[0]


class TestTheEmergencyPath:
    def test_a_flaw_disclosed_before_a_fix_must_carry_an_interim_mitigation(
        self, tmp_path: Path
    ) -> None:
        """Otherwise every consuming product invents its own workaround."""
        body = RECORD.replace('fixed = ["0.1.4", "0.2.1"]', "fixed = []")
        body = body.replace(f'mitigation = "{MITIGATION}"\n', "")
        body += "disclosed_publicly = true\n"
        found = check(tmp_path, body)
        assert any("mitigation" in violation for violation in found)


class TestSeverityIsNotDiscountedByScope:
    def test_a_flaw_in_an_optional_extra_is_triaged_like_any_other(self) -> None:
        """Products that install the extra are exposed exactly as much as anyone else."""
        assert disclosure.target("high", profiles=("extra:mcp",)) == disclosure.target(
            "high", profiles=("base",)
        )

    def test_an_unknown_severity_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="severity"):
            disclosure.advisories(write(tmp_path, RECORD.replace('"high"', '"spicy"')))


class TestRecords:
    def test_a_record_missing_a_required_field_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="reported"):
            disclosure.advisories(write(tmp_path, RECORD.replace("reported = 2026-07-01\n", "")))

    def test_a_record_with_a_field_nobody_recognises_is_rejected(self, tmp_path: Path) -> None:
        """A misspelt field is a commitment that silently does nothing."""
        with pytest.raises(disclosure.DisclosureError, match="notifed"):
            disclosure.advisories(write(tmp_path, RECORD + "notifed = 2026-07-10\n"))

    def test_a_record_that_is_not_valid_toml_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="TOML"):
            disclosure.advisories(write(tmp_path, "id = \n"))

    def test_no_advisories_is_a_valid_state(self, tmp_path: Path) -> None:
        directory = tmp_path / "advisories"
        directory.mkdir()
        assert disclosure.advisories(directory) == ()


class TestThePublishedPolicy:
    def test_the_response_targets_in_the_policy_are_the_ones_rendered(self) -> None:
        """One source: prose that disagrees with the gate is prose a reporter relies on."""
        page = disclosure.PAGE.read_text(encoding="utf-8")
        for severity, target in disclosure.targets().items():
            assert severity in page
            assert str(target.acknowledge_days) in page

    def test_the_page_names_a_private_reporting_channel(self) -> None:
        assert disclosure.channel().private in disclosure.PAGE.read_text(encoding="utf-8")

    def test_the_page_is_current(self) -> None:
        """`make disclosure` regenerates the tables; CI runs this check."""
        page = disclosure.PAGE.read_text(encoding="utf-8")
        assert disclosure.render(page) == page

    def test_the_repository_advisories_satisfy_the_process(self) -> None:
        assert disclosure.violations(disclosure.advisories(), targets=disclosure.targets()) == []


class TestTheThreatModel:
    THREAT_MODEL = disclosure.ROOT / "docs" / "threat-model.md"

    def test_it_states_what_the_kit_does_not_defend_against(self) -> None:
        """A consumer who over-trusts a boundary has a vulnerability the kit cannot fix."""
        page = self.THREAT_MODEL.read_text(encoding="utf-8")
        assert "## What the kit does not defend against" in page

    def test_every_guarantee_carries_the_assumption_behind_it(self) -> None:
        page = self.THREAT_MODEL.read_text(encoding="utf-8")
        assert page.count("Assumption") >= page.count("### Guarantee")
        assert page.count("### Guarantee") >= 4

    def test_the_security_policy_points_at_it(self) -> None:
        assert "threat-model.md" in disclosure.PAGE.read_text(encoding="utf-8")


class TestCommandLine:
    def test_a_clean_state_passes(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert disclosure.main([]) == 0
        assert "disclosure" in capsys.readouterr().out

    def test_a_stale_page_fails_the_job(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(disclosure, "render", lambda page: page + "drift\n")
        assert disclosure.main([]) == 1
        assert "make disclosure" in capsys.readouterr().out

    def test_write_regenerates_the_page(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = tmp_path / "SECURITY.md"
        page.write_text(disclosure.PAGE.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr(disclosure, "PAGE", page)
        assert disclosure.main(["--write"]) == 0
        assert page.read_text(encoding="utf-8") == disclosure.render(
            disclosure.PAGE.read_text(encoding="utf-8")
        )


POLICY = """
[channel]
private = "https://example.invalid/report"
rota = ["@maintainer"]

[[target]]
severity = "high"
acknowledge_days = 2
fix_days = 14
"""


def policy(tmp_path: Path, body: str = POLICY) -> Path:
    path = tmp_path / "disclosure.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestReadingThePolicy:
    def test_a_missing_policy_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="cannot be read"):
            disclosure.targets(tmp_path / "absent.toml")

    def test_a_policy_that_is_not_valid_toml_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="TOML"):
            disclosure.targets(policy(tmp_path, "[[target]\n"))

    def test_a_channel_without_a_private_route_is_rejected(self, tmp_path: Path) -> None:
        """A policy with no private channel sends the reporter to a public issue."""
        with pytest.raises(disclosure.DisclosureError, match="private"):
            disclosure.channel(
                policy(tmp_path, POLICY.replace('private = "https://example.invalid/report"', ""))
            )

    def test_a_target_missing_its_window_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="fix_days"):
            disclosure.targets(policy(tmp_path, POLICY.replace("fix_days = 14", "")))

    def test_a_policy_declaring_no_targets_is_rejected(self, tmp_path: Path) -> None:
        """A policy with no commitment is not a policy."""
        with pytest.raises(disclosure.DisclosureError, match="no response targets"):
            disclosure.targets(policy(tmp_path, "[channel]\nprivate = 'x'\nrota = ['@a']\n"))

    def test_asking_for_a_severity_the_policy_does_not_declare_is_an_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(disclosure.DisclosureError, match="spicy"):
            disclosure.target("spicy", path=policy(tmp_path))

    def test_a_date_field_holding_something_else_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="reported"):
            disclosure.advisories(
                write(tmp_path, RECORD.replace("reported = 2026-07-01", 'reported = "soon"'))
            )


class TestRendering:
    PAGE = (
        "# Security policy\n\n"
        "<!-- generated: response-targets -->\nold\n<!-- end generated: response-targets -->\n\n"
        "<!-- generated: advisories -->\nold\n<!-- end generated: advisories -->\n"
    )

    def test_a_published_advisory_reaches_the_table(self, tmp_path: Path) -> None:
        """The table is what a consumer checks their pinned version against."""
        rendered = disclosure.render(self.PAGE, directory=write(tmp_path))
        assert "| ADK-2026-0001 |" in rendered
        assert "0.1.4, 0.2.1" in rendered

    def test_an_embargoed_advisory_is_shown_as_embargoed(self, tmp_path: Path) -> None:
        body = RECORD.replace("published = 2026-07-10\n", "")
        rendered = disclosure.render(self.PAGE, directory=write(tmp_path, body))
        assert "embargoed" in rendered

    def test_the_prose_around_a_generated_block_is_left_alone(self, tmp_path: Path) -> None:
        assert disclosure.render(self.PAGE, directory=write(tmp_path)).startswith(
            "# Security policy"
        )

    def test_a_page_without_the_generated_block_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(disclosure.DisclosureError, match="response-targets"):
            disclosure.render("# Security policy\n", directory=write(tmp_path))


class TestReportingAMissedTarget:
    def test_a_missed_target_fails_the_job(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        late = RECORD.replace("acknowledged = 2026-07-02", "acknowledged = 2026-07-30")
        records = disclosure.advisories(write(tmp_path, late))
        monkeypatch.setattr(disclosure, "advisories", lambda *_, **__: records)
        assert disclosure.main([]) == 1
        assert "ADK-2026-0001" in capsys.readouterr().out
