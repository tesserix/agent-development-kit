"""Credential shapes in anything the repository ships.

The kit ships recorded provider traffic as fixtures, which is the obvious route for a
real key to be committed and then distributed to every consumer with the sdist. A test
fixture that deliberately looks like a credential is legitimate — but it has to be
declared, not inferred, because "it's probably fake" is how the real one gets through.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest
from tools import secret_scan
from tools import security_policy as policy

if TYPE_CHECKING:
    from pathlib import Path

TODAY = dt.date(2026, 8, 6)
EMPTY = policy.Policy(suppressions=())

LIVE_LOOKING = {
    "an OpenAI key": "sk-" + "a" * 48,
    "an Anthropic key": "sk-ant-api03-" + "b" * 40,
    "an AWS access key id": "AKIA" + "C" * 16,
    "a GitHub token": "ghp_" + "d" * 36,
    "a Slack bot token": "xox" + "b-123456789012-123456789012-" + "e" * 24,
    "a private key block": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "a JWT": "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxIn0." + "a" * 28,
}

# Every value above is assembled at runtime rather than written out: a literal here would
# be a credential shape in a tracked file, and the scanner would be right to flag it.

INNOCENT = {
    "a redacted value": "sk-***REDACTED***",
    "prose about keys": "Set OPENAI_API_KEY in your environment before running this.",
    "an environment reference": 'api_key = os.environ["ANTHROPIC_API_KEY"]',
    "a short hex string": "deadbeef",
    "a placeholder": "sk-YOUR-KEY-HERE",
}


class TestCredentialShapes:
    @pytest.mark.parametrize("text", LIVE_LOOKING.values(), ids=list(LIVE_LOOKING))
    def test_a_live_looking_credential_is_found(self, text: str) -> None:
        assert secret_scan.matches(text)

    @pytest.mark.parametrize("text", INNOCENT.values(), ids=list(INNOCENT))
    def test_a_value_that_only_talks_about_credentials_is_not_a_finding(self, text: str) -> None:
        """False positives are not free: they are what gets the scanner turned off."""
        assert not secret_scan.matches(text)

    def test_the_rule_that_matched_is_named(self) -> None:
        """ "Something looks like a secret" is not actionable; the rule name is."""
        assert secret_scan.matches("sk-ant-api03-" + "b" * 40)[0].rule == "anthropic-key"

    def test_the_matched_value_is_never_echoed_in_full(self) -> None:
        """A scanner that prints the credential has published it to the build log."""
        found = secret_scan.matches("sk-" + "a" * 48)[0]
        assert "a" * 48 not in found.evidence
        assert found.evidence.endswith("…")


class TestScanningFiles:
    def test_a_credential_in_a_file_is_found_with_its_line(self, tmp_path: Path) -> None:
        target = tmp_path / "cassette.json"
        target.write_text('{\n "authorization": "Bearer sk-' + "a" * 48 + '"\n}', encoding="utf-8")
        found = secret_scan.scan([target], policy=EMPTY, today=TODAY)
        assert found[0].path == target
        assert found[0].line == 2

    def test_a_clean_file_produces_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.md"
        target.write_text("Set OPENAI_API_KEY in the environment.", encoding="utf-8")
        assert secret_scan.scan([target], policy=EMPTY, today=TODAY) == []

    def test_a_binary_file_is_skipped_rather_than_crashing(self, tmp_path: Path) -> None:
        target = tmp_path / "logo.png"
        target.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
        assert secret_scan.scan([target], policy=EMPTY, today=TODAY) == []

    def test_a_declared_fixture_value_is_allowed(self, tmp_path: Path) -> None:
        """Redaction fixtures have to contain credential shapes to test redaction."""
        target = tmp_path / "redaction_fixture.py"
        target.write_text("LEAKED = 'sk-" + "a" * 48 + "'", encoding="utf-8")
        allowed = policy.Policy(
            suppressions=(
                policy.Suppression(
                    id="openai-key:redaction_fixture.py",
                    kind="secret",
                    owner="@sam123ben",
                    reason="The redaction suite needs a value shaped like a real key.",
                    expires=dt.date(2026, 9, 1),
                ),
            )
        )
        assert secret_scan.scan([target], policy=allowed, today=TODAY) == []

    def test_an_expired_allowance_stops_allowing(self, tmp_path: Path) -> None:
        target = tmp_path / "redaction_fixture.py"
        target.write_text("LEAKED = 'sk-" + "a" * 48 + "'", encoding="utf-8")
        allowed = policy.Policy(
            suppressions=(
                policy.Suppression(
                    id="openai-key:redaction_fixture.py",
                    kind="secret",
                    owner="@sam123ben",
                    reason="The redaction suite needs a value shaped like a real key.",
                    expires=dt.date(2026, 1, 1),
                ),
            )
        )
        assert secret_scan.scan([target], policy=allowed, today=TODAY)

    def test_an_advisory_suppression_does_not_allow_a_credential(self, tmp_path: Path) -> None:
        target = tmp_path / "redaction_fixture.py"
        target.write_text("LEAKED = 'sk-" + "a" * 48 + "'", encoding="utf-8")
        wrong_kind = policy.Policy(
            suppressions=(
                policy.Suppression(
                    id="openai-key:redaction_fixture.py",
                    kind="advisory",
                    owner="@sam123ben",
                    reason="An advisory acceptance is not a credential acceptance.",
                    expires=dt.date(2026, 9, 1),
                ),
            )
        )
        assert secret_scan.scan([target], policy=wrong_kind, today=TODAY)


class TestWhatIsScanned:
    def test_the_working_tree_is_walked(self) -> None:
        paths = secret_scan.tracked_files()
        assert any(path.name == "pyproject.toml" for path in paths)

    def test_the_git_directory_is_not_walked(self) -> None:
        assert not any(".git" in path.parts for path in secret_scan.tracked_files())


class TestRecordedTraffic:
    """Cassettes are recorded from something. Whatever that was, it was not synthetic."""

    def test_a_personal_identifier_in_recorded_traffic_is_a_finding(self, tmp_path: Path) -> None:
        cassette = tmp_path / "cassettes" / "chat.json"
        cassette.parent.mkdir()
        cassette.write_text('{"user": "priya.sharma@example.org"}', encoding="utf-8")
        found = secret_scan.scan([cassette], policy=EMPTY, today=TODAY, personal=True)
        assert found[0].rule == "email-address"

    def test_a_personal_identifier_elsewhere_is_not_a_finding(self, tmp_path: Path) -> None:
        """Maintainer addresses in CODEOWNERS and docs are the point of those files."""
        target = tmp_path / "CODEOWNERS"
        target.write_text("* maintainer@example.org", encoding="utf-8")
        assert secret_scan.scan([target], policy=EMPTY, today=TODAY) == []

    def test_a_phone_number_in_recorded_traffic_is_a_finding(self, tmp_path: Path) -> None:
        cassette = tmp_path / "cassettes" / "chat.json"
        cassette.parent.mkdir()
        cassette.write_text('{"contact": "+44 7700 900123"}', encoding="utf-8")
        found = secret_scan.scan([cassette], policy=EMPTY, today=TODAY, personal=True)
        assert found[0].rule == "phone-number"

    def test_the_repository_has_no_recorded_traffic_carrying_identifiers(self) -> None:
        found = secret_scan.scan(
            secret_scan.recorded_traffic(),
            policy=policy.load(),
            today=dt.date.today(),
            personal=True,
        )
        assert found == [], secret_scan.render(found)


class TestReporting:
    def test_a_finding_names_the_file_line_and_rule(self, tmp_path: Path) -> None:
        target = tmp_path / "cassette.json"
        target.write_text("token: ghp_" + "d" * 36, encoding="utf-8")
        found = secret_scan.scan([target], policy=EMPTY, today=TODAY)
        rendered = secret_scan.render(found)
        assert "cassette.json" in rendered
        assert "github-token" in rendered
        assert ":1" in rendered

    def test_rotation_comes_before_anything_else_in_the_guidance(self, tmp_path: Path) -> None:
        """History rewriting is not remediation: the credential is already gone."""
        target = tmp_path / "cassette.json"
        target.write_text("token: ghp_" + "d" * 36, encoding="utf-8")
        rendered = secret_scan.render(secret_scan.scan([target], policy=EMPTY, today=TODAY))
        assert "rotate" in rendered.lower()

    def test_a_clean_report_says_so(self) -> None:
        assert "no credential" in secret_scan.render([]).lower()


class TestCommandLine:
    def test_a_finding_fails_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "cassette.json"
        target.write_text("token: ghp_" + "d" * 36, encoding="utf-8")
        monkeypatch.setattr(secret_scan, "tracked_files", lambda: [target])
        assert secret_scan.main([]) == 1
        assert "github-token" in capsys.readouterr().out

    def test_a_clean_tree_passes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(secret_scan, "tracked_files", list)
        assert secret_scan.main([]) == 0
        assert "no credential" in capsys.readouterr().out.lower()

    def test_recorded_traffic_is_also_checked_for_identifiers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cassette = tmp_path / "cassettes" / "chat.json"
        cassette.parent.mkdir()
        cassette.write_text('{"user": "priya.sharma@example.org"}', encoding="utf-8")
        monkeypatch.setattr(secret_scan, "tracked_files", lambda: [cassette])
        monkeypatch.setattr(secret_scan, "recorded_traffic", lambda: [cassette])
        assert secret_scan.main([]) == 1
        assert "email-address" in capsys.readouterr().out

    def test_named_paths_can_be_scanned_on_their_own(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "cassette.json"
        target.write_text("token: ghp_" + "d" * 36, encoding="utf-8")
        assert secret_scan.main([str(target)]) == 1
        assert "cassette.json" in capsys.readouterr().out


class TestTheRepositoryItself:
    def test_this_repository_carries_no_credentials(self) -> None:
        """The gate itself, run as a test so it fails locally before it fails in CI."""
        found = secret_scan.scan(
            secret_scan.tracked_files(), policy=policy.load(), today=dt.date.today()
        )
        assert found == [], secret_scan.render(found)
