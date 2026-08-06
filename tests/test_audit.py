"""Advisories against the resolved lock, reported so a consuming team can act.

A finding is only useful with three things attached: which package and version, what to
upgrade to, and who receives it. Without the third, every product downstream repeats the
same reachability analysis, unevenly.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import subprocess
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest
from tools import audit, lockfile
from tools import security_policy as policy

TODAY = dt.date(2026, 8, 6)

GRAPH = lockfile.Graph.from_lock(
    {
        "package": [
            {
                "name": "tesserix-adk",
                "dependencies": [{"name": "httpx"}],
                "optional-dependencies": {"redis": [{"name": "redis"}]},
                "dev-dependencies": {"test": [{"name": "pytest"}]},
            },
            {"name": "httpx", "version": "0.28.1"},
            {"name": "redis", "version": "6.0.0"},
            {"name": "pytest", "version": "8.3.4"},
        ]
    }
)

EMPTY = policy.Policy(suppressions=())


def report(name: str, version: str, vulns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"dependencies": [{"name": name, "version": version, "vulns": vulns}]}


def _response(payload: dict[str, Any]) -> contextlib.closing[io.BytesIO]:
    return contextlib.closing(io.BytesIO(json.dumps(payload).encode()))


def vuln(vid: str, fixes: list[str] | None = None) -> dict[str, Any]:
    return {"id": vid, "fix_versions": fixes if fixes is not None else ["9.9.9"], "aliases": []}


class TestReadingFindings:
    def test_a_vulnerable_dependency_is_a_finding(self) -> None:
        found = audit.findings(report("httpx", "0.28.1", [vuln("GHSA-1")]))
        assert (found[0].package, found[0].version, found[0].id) == ("httpx", "0.28.1", "GHSA-1")

    def test_a_clean_dependency_produces_nothing(self) -> None:
        assert audit.findings(report("httpx", "0.28.1", [])) == []

    def test_the_first_fixed_version_is_carried_through(self) -> None:
        found = audit.findings(report("httpx", "0.28.1", [vuln("GHSA-1", ["1.0.0", "0.29.0"])]))
        assert found[0].fixed == "0.29.0"

    def test_an_advisory_with_no_fix_yet_says_so(self) -> None:
        """The case that needs a mitigation rather than an upgrade."""
        found = audit.findings(report("httpx", "0.28.1", [vuln("GHSA-1", [])]))
        assert found[0].fixed is None

    def test_a_fix_version_that_is_not_a_version_is_ignored(self) -> None:
        """Advisory data is written by hand; "TBD" in a fix list must not crash the audit."""
        found = audit.findings(report("httpx", "0.28.1", [vuln("GHSA-1", ["TBD", "0.29.0"])]))
        assert found[0].fixed == "0.29.0"

    def test_several_advisories_against_one_package_are_separate_findings(self) -> None:
        found = audit.findings(report("httpx", "0.28.1", [vuln("GHSA-1"), vuln("GHSA-2")]))
        assert [item.id for item in found] == ["GHSA-1", "GHSA-2"]


class TestRunningTheScanner:
    def test_the_scanner_output_is_returned_as_a_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.dumps(report("httpx", "0.28.1", [vuln("GHSA-1")]))
        monkeypatch.setattr(
            subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=payload, stderr="")
        )
        assert audit.findings(audit.scan())[0].id == "GHSA-1"

    def test_the_lock_is_audited_rather_than_a_fresh_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []

        def record(command: list[str], **_: Any) -> SimpleNamespace:
            seen.append(command)
            return SimpleNamespace(stdout="{}", stderr="")

        monkeypatch.setattr(subprocess, "run", record)
        audit.scan()
        assert seen[0][:3] == ["uv", "run", "pip-audit"]

    def test_a_scanner_that_cannot_start_is_an_error_not_a_clean_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scan that did not run is not a scan that found nothing."""

        def missing(*_: Any, **__: Any) -> SimpleNamespace:
            raise OSError("No such file")

        monkeypatch.setattr(subprocess, "run", missing)
        with pytest.raises(audit.AuditError, match="could not be started"):
            audit.scan()

    def test_unreadable_output_is_an_error_and_carries_what_the_scanner_said(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_, **__: SimpleNamespace(stdout="", stderr="no such option"),
        )
        with pytest.raises(audit.AuditError, match="no such option"):
            audit.scan()


class TestSeverity:
    def test_the_rating_is_read_from_the_advisory_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *_, **__: _response({"database_specific": {"severity": "HIGH"}}),
        )
        assert audit.severity_of("GHSA-1") == "high"

    def test_an_advisory_the_database_does_not_rate_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_, **__: _response({}))
        assert audit.severity_of("GHSA-1") is None

    def test_an_unreachable_database_rates_nothing_rather_than_rating_it_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Which leaves the finding unknown, and unknown blocks."""

        def unreachable(*_: Any, **__: Any) -> None:
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(urllib.request, "urlopen", unreachable)
        assert audit.severity_of("GHSA-1") is None

    def test_the_severity_is_read_from_the_advisory_database(self) -> None:
        found = audit.rate(
            [audit.Finding("httpx", "0.28.1", "GHSA-1", "0.29.0")], severity_of=lambda _: "critical"
        )
        assert found[0].severity == "critical"

    def test_an_advisory_the_database_cannot_rate_is_unknown(self) -> None:
        found = audit.rate(
            [audit.Finding("httpx", "0.28.1", "GHSA-1", "0.29.0")], severity_of=lambda _: None
        )
        assert found[0].severity == "unknown"


class TestVerdict:
    def _finding(self, package: str = "httpx", severity: str = "critical") -> audit.Finding:
        return audit.Finding(package, "0.28.1", "GHSA-1", "0.29.0", severity=severity)

    def test_a_critical_finding_in_the_base_set_blocks(self) -> None:
        verdict = audit.assess([self._finding()], graph=GRAPH, policy=EMPTY, today=TODAY)
        assert verdict.blocking

    def test_a_low_finding_is_tracked_but_does_not_block(self) -> None:
        verdict = audit.assess(
            [self._finding(severity="low")], graph=GRAPH, policy=EMPTY, today=TODAY
        )
        assert not verdict.blocking
        assert verdict.tracked

    def test_a_finding_reachable_only_through_an_extra_still_blocks(self) -> None:
        """Reduced blast radius is not an excuse; it is a sentence in the report."""
        verdict = audit.assess(
            [self._finding(package="redis")], graph=GRAPH, policy=EMPTY, today=TODAY
        )
        assert verdict.blocking
        assert "redis extra" in verdict.report

    def test_a_finding_in_the_development_set_does_not_block_a_consumer_release(self) -> None:
        """It is still reported — it can compromise the build — but it ships to nobody."""
        verdict = audit.assess(
            [self._finding(package="pytest")], graph=GRAPH, policy=EMPTY, today=TODAY
        )
        assert not verdict.blocking
        assert "development only" in verdict.report

    def test_a_suppressed_finding_does_not_block(self) -> None:
        suppressed = policy.Policy(
            suppressions=(
                policy.Suppression(
                    id="GHSA-1",
                    kind="advisory",
                    owner="@sam123ben",
                    reason="Not reachable: the affected code path is never called.",
                    expires=dt.date(2026, 9, 1),
                ),
            )
        )
        verdict = audit.assess([self._finding()], graph=GRAPH, policy=suppressed, today=TODAY)
        assert not verdict.blocking
        assert "@sam123ben" in verdict.report

    def test_an_expired_suppression_fails_the_build_on_its_own(self) -> None:
        """The suppression is the finding at that point: nobody reviewed it in time."""
        stale = policy.Policy(
            suppressions=(
                policy.Suppression(
                    id="GHSA-OLD",
                    kind="advisory",
                    owner="@sam123ben",
                    reason="Waiting on the upstream fix that has since been released.",
                    expires=dt.date(2026, 1, 1),
                ),
            )
        )
        verdict = audit.assess([], graph=GRAPH, policy=stale, today=TODAY)
        assert verdict.blocking
        assert "expired" in verdict.report

    def test_a_clean_scan_says_so(self) -> None:
        verdict = audit.assess([], graph=GRAPH, policy=EMPTY, today=TODAY)
        assert not verdict.blocking
        assert "no advisories" in verdict.report.lower()


class TestReport:
    def test_a_finding_names_the_package_advisory_and_first_fixed_version(self) -> None:
        verdict = audit.assess(
            [audit.Finding("httpx", "0.28.1", "GHSA-1", "0.29.0", severity="high")],
            graph=GRAPH,
            policy=EMPTY,
            today=TODAY,
        )
        for expected in ("httpx", "0.28.1", "GHSA-1", "0.29.0", "high", "every consumer"):
            assert expected in verdict.report

    def test_an_advisory_with_no_fix_is_reported_as_needing_a_mitigation(self) -> None:
        verdict = audit.assess(
            [audit.Finding("httpx", "0.28.1", "GHSA-1", None, severity="high")],
            graph=GRAPH,
            policy=EMPTY,
            today=TODAY,
        )
        assert "no fixed version" in verdict.report


class TestCommandLine:
    def test_a_blocking_finding_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(audit, "scan", lambda: report("httpx", "0.28.1", [vuln("GHSA-1")]))
        monkeypatch.setattr(audit, "severity_of", lambda _: "critical")
        assert audit.main([]) == 1
        assert "GHSA-1" in capsys.readouterr().out

    def test_a_clean_scan_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(audit, "scan", lambda: {"dependencies": []})
        assert audit.main([]) == 0
        assert "no advisories" in capsys.readouterr().out.lower()

    def test_a_scanner_that_will_not_run_fails_the_job_rather_than_passing_it(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A scan that did not happen is not a scan that found nothing."""

        def broken() -> dict[str, Any]:
            raise audit.AuditError("pip-audit is not installed")

        monkeypatch.setattr(audit, "scan", broken)
        assert audit.main([]) == 1
        assert "pip-audit" in capsys.readouterr().err
