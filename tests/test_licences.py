"""Licence exposure inherited through the kit, decided rather than guessed.

A consuming product inherits every obligation in the graph, and inherits it silently: the
extra someone added on a Tuesday is in the legal review two years later. So an unknown
licence blocks, and a dual licence is not resolved by picking the convenient half — it is
a decision, with a name against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tools import licences

if TYPE_CHECKING:
    from pathlib import Path

POLICY = """
allowed = ["MIT", "BSD-3-Clause", "Apache-2.0", "MPL-2.0"]

[[decision]]
package = "chardet"
licence = "MPL-2.0"
owner = "@sam123ben"
reason = "Dual LGPL/MPL; we take MPL-2.0, which is on the allow list."
"""


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "licences.toml"
    path.write_text(body, encoding="utf-8")
    return path


def policy(tmp_path: Path, body: str = POLICY) -> licences.Policy:
    return licences.load(write(tmp_path, body))


class TestLoading:
    def test_the_allow_list_is_read(self, tmp_path: Path) -> None:
        assert "MIT" in policy(tmp_path).allowed

    def test_a_policy_with_no_allow_list_is_an_error(self, tmp_path: Path) -> None:
        """An empty allow list would pass nothing; an absent one must not pass everything."""
        with pytest.raises(licences.LicenceError, match="allowed"):
            policy(tmp_path, "")

    def test_a_missing_policy_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(licences.LicenceError, match=r"licences\.toml"):
            licences.load(tmp_path / "licences.toml")

    def test_a_decision_without_an_owner_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(licences.LicenceError, match="owner"):
            policy(tmp_path, 'allowed = ["MIT"]\n[[decision]]\npackage = "x"\nlicence = "MIT"\n')

    def test_a_policy_that_is_not_valid_toml_is_an_error(self, tmp_path: Path) -> None:
        """A policy the loader silently ignored would be a gate that allows everything."""
        with pytest.raises(licences.LicenceError, match="TOML"):
            policy(tmp_path, 'allowed = ["MIT"\n')

    def test_a_decision_with_a_field_nobody_recognises_is_rejected(self, tmp_path: Path) -> None:
        """A misspelt `owner` would otherwise read as a decision with no owner at all."""
        body = POLICY.replace('owner = "@sam123ben"', 'owner = "@sam123ben"\nexpiry = "never"')
        with pytest.raises(licences.LicenceError, match="expiry"):
            policy(tmp_path, body)

    def test_a_decision_choosing_a_licence_the_policy_forbids_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Recording a decision is not a way to widen the allow list quietly."""
        body = POLICY.replace('licence = "MPL-2.0"', 'licence = "GPL-3.0-only"')
        with pytest.raises(licences.LicenceError, match=r"GPL-3\.0-only"):
            policy(tmp_path, body)


class TestVerdicts:
    def test_an_allowed_licence_passes(self, tmp_path: Path) -> None:
        assert licences.check("httpx", "BSD-3-Clause", policy=policy(tmp_path)) is None

    def test_a_licence_the_policy_does_not_allow_is_named_in_the_violation(
        self, tmp_path: Path
    ) -> None:
        violation = licences.check("copyleft-thing", "GPL-3.0-only", policy=policy(tmp_path))
        assert violation is not None
        assert "copyleft-thing" in violation
        assert "GPL-3.0-only" in violation

    def test_an_unknown_licence_blocks(self, tmp_path: Path) -> None:
        """A component whose licence nobody could determine is the one to look at."""
        violation = licences.check("mystery", None, policy=policy(tmp_path))
        assert violation is not None
        assert "no licence" in violation

    def test_a_licence_is_matched_regardless_of_case(self, tmp_path: Path) -> None:
        assert licences.check("httpx", "bsd-3-clause", policy=policy(tmp_path)) is None

    def test_every_half_of_an_and_expression_must_be_allowed(self, tmp_path: Path) -> None:
        """`AND` means both sets of obligations arrive, so both have to be acceptable."""
        assert licences.check("dual", "MIT AND Apache-2.0", policy=policy(tmp_path)) is None
        assert licences.check("dual", "MIT AND GPL-3.0-only", policy=policy(tmp_path)) is not None


class TestAmbiguity:
    def test_an_or_expression_needs_a_recorded_decision(self, tmp_path: Path) -> None:
        """Both halves being allowed does not say which obligations we actually took on."""
        violation = licences.check("either", "MIT OR Apache-2.0", policy=policy(tmp_path))
        assert violation is not None
        assert "decision" in violation

    def test_a_recorded_decision_resolves_the_ambiguity(self, tmp_path: Path) -> None:
        assert licences.check("chardet", "LGPL-2.1 OR MPL-2.0", policy=policy(tmp_path)) is None

    def test_a_decision_must_name_one_of_the_offered_licences(self, tmp_path: Path) -> None:
        """Otherwise the decision records a licence the package was never offered under."""
        violation = licences.check("chardet", "MIT OR Apache-2.0", policy=policy(tmp_path))
        assert violation is not None
        assert "MPL-2.0" in violation

    def test_the_decision_carries_its_owner(self, tmp_path: Path) -> None:
        taken = policy(tmp_path).decisions["chardet"]
        assert (taken.licence, taken.owner) == ("MPL-2.0", "@sam123ben")


class TestAcceptingOneOffObligations:
    """The allow list is a blanket permission; some licences are acceptable only here."""

    ACCEPTED = (
        POLICY
        + """
[[acceptance]]
package = "psycopg"
licence = "LGPL-3.0-only"
owner = "@sam123ben"
reason = "Imported unmodified behind the postgres extra; obligation is documented."
"""
    )

    def test_a_licence_off_the_allow_list_is_accepted_for_the_named_package(
        self, tmp_path: Path
    ) -> None:
        loaded = policy(tmp_path, self.ACCEPTED)
        assert licences.check("psycopg", "LGPL-3.0-only", policy=loaded) is None

    def test_the_acceptance_does_not_extend_to_any_other_package(self, tmp_path: Path) -> None:
        """A copyleft obligation accepted once must not become a blanket permission."""
        loaded = policy(tmp_path, self.ACCEPTED)
        assert licences.check("something-else", "LGPL-3.0-only", policy=loaded) is not None

    def test_the_acceptance_does_not_extend_to_another_licence(self, tmp_path: Path) -> None:
        loaded = policy(tmp_path, self.ACCEPTED)
        assert licences.check("psycopg", "GPL-3.0-only", policy=loaded) is not None

    def test_an_acceptance_without_a_reason_is_rejected(self, tmp_path: Path) -> None:
        body = self.ACCEPTED.replace(
            'reason = "Imported unmodified behind the postgres extra; obligation is documented."',
            "",
        )
        with pytest.raises(licences.LicenceError, match="reason"):
            policy(tmp_path, body)


class TestLegacyMetadata:
    """Distributions predating SPDX expressions write prose in the licence field."""

    def test_a_recognised_legacy_name_is_normalised(self, tmp_path: Path) -> None:
        assert licences.check("protobuf", "3-Clause BSD License", policy=policy(tmp_path)) is None

    def test_an_unrecognised_legacy_name_still_blocks(self, tmp_path: Path) -> None:
        """Guessing at prose is how a copyleft licence gets read as permissive."""
        assert licences.check("odd", "Free for research use", policy=policy(tmp_path)) is not None


class TestReadingInstalledMetadata:
    def test_a_declared_expression_is_used_as_is(self) -> None:
        assert licences.declared("pydantic") == "MIT"

    def test_a_legacy_licence_field_is_used_when_there_is_no_expression(self) -> None:
        assert licences.declared("httpx") == "BSD-3-Clause"

    def test_a_package_that_is_not_installed_has_no_declared_licence(self) -> None:
        assert licences.declared("not-a-real-distribution") is None


class TestTheRepositoryItself:
    def test_every_installed_dependency_satisfies_the_policy(self) -> None:
        """The gate itself, so a licence change fails locally before it fails a release."""
        loaded = licences.load()
        violations = [
            violation
            for name in licences.installed()
            if (violation := licences.check(name, licences.declared(name), policy=loaded))
        ]
        assert violations == []


class TestCommandLine:
    def test_a_clean_graph_passes(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert licences.main([]) == 0
        assert "no licence violations" in capsys.readouterr().out

    def test_a_violation_fails_the_job(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(licences, "installed", lambda: ["copyleft-thing"])
        monkeypatch.setattr(licences, "declared", lambda _: "GPL-3.0-only")
        assert licences.main([]) == 1
        assert "copyleft-thing" in capsys.readouterr().out
