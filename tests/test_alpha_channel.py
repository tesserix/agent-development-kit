"""Every merge to main builds an alpha; configured channels publish it for early consumers.

An early consumer is the most valuable source of design feedback there is, and it is
only available before the API freezes. The cost of that is a channel nobody must land in
by accident: a consumer asking for a stable version never resolves a pre-release, and
the ones that accumulate are cleaned up on a stated rule rather than a judgement call.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from tools import alpha

RELEASED = ("0.1.0", "0.2.0a1", "0.2.0a2", "0.2.0rc1")

ROOT = Path(__file__).resolve().parents[1]
STABILITY = ROOT / "docs" / "stability.md"
PACKAGE = ROOT / "src" / "tesserix_adk"


class TestNextAlpha:
    def test_the_first_alpha_of_a_version_is_a1(self) -> None:
        assert alpha.next_alpha("0.3.0", released=RELEASED) == "0.3.0a1"

    def test_the_next_alpha_follows_the_highest_already_published(self) -> None:
        assert alpha.next_alpha("0.2.0", released=RELEASED) == "0.2.0a3"

    def test_a_release_candidate_does_not_hold_back_the_alpha_numbering(self) -> None:
        """The two are separate series; an rc1 does not make the next alpha an a2."""
        assert alpha.next_alpha("0.2.0", released=("0.2.0rc1",)) == "0.2.0a1"

    def test_alphas_of_other_versions_are_not_counted(self) -> None:
        assert alpha.next_alpha("0.4.0", released=("0.3.0a9",)) == "0.4.0a1"

    def test_a_version_the_index_holds_that_is_not_a_version_is_ignored(self) -> None:
        """The index is a foreign input; one unparseable entry must not stop a release."""
        assert alpha.next_alpha("0.2.0", released=("0.2.0a1", "nightly-2019")) == "0.2.0a2"

    def test_an_alpha_for_an_already_released_version_is_refused(self) -> None:
        """0.1.0a3 after 0.1.0 shipped describes work that is already stable."""
        with pytest.raises(alpha.AlphaError, match="already released"):
            alpha.next_alpha("0.1.0", released=RELEASED)


class TestNextBase:
    def test_the_base_is_the_next_minor_after_the_last_stable_release(self) -> None:
        """Pre-1.0 the minor is the breaking channel, so main is always heading there."""
        assert alpha.next_base(released=RELEASED) == "0.2.0"

    def test_the_first_base_before_any_release_is_the_first_minor(self) -> None:
        assert alpha.next_base(released=()) == "0.1.0"

    def test_pre_releases_do_not_move_the_base(self) -> None:
        assert alpha.next_base(released=("0.1.0", "0.9.0a1")) == "0.2.0"

    def test_after_1_0_the_base_is_the_next_minor_too(self) -> None:
        """A breaking change waits for a major, and it will not be discovered by an alpha."""
        assert alpha.next_base(released=("1.4.2",)) == "1.5.0"


class TestOptIn:
    """The guarantee is PEP 440's, but the kit depends on it, so it is asserted here."""

    @pytest.mark.parametrize("specifier", ["", ">=0.1", ">=0.1,<1.0", "<1.0"])
    def test_a_consumer_asking_for_a_stable_version_never_gets_an_alpha(
        self, specifier: str
    ) -> None:
        resolved = list(SpecifierSet(specifier).filter([Version(v) for v in RELEASED]))
        assert all(not version.is_prerelease for version in resolved)

    def test_a_specifier_only_pre_releases_satisfy_does_resolve_them(self) -> None:
        """PEP 440's fallback: `==0.2.*` before 0.2.0 ships has only pre-releases to pick
        from, so it takes one. Documented in docs/stability.md rather than worked around,
        because the alternative is an unsatisfiable pin with no explanation."""
        resolved = list(SpecifierSet("==0.2.*").filter([Version(v) for v in RELEASED]))
        assert resolved == [Version("0.2.0a1"), Version("0.2.0a2"), Version("0.2.0rc1")]

    def test_the_fallback_stops_as_soon_as_a_stable_version_exists(self) -> None:
        released = ("0.2.0a1", "0.2.0rc1", "0.2.0")
        resolved = list(SpecifierSet("==0.2.*").filter([Version(v) for v in released]))
        assert resolved == [Version("0.2.0")]

    def test_an_exact_pin_on_an_alpha_resolves_it(self) -> None:
        """A reproducible build pins the alpha it was tested against, not the newest one."""
        resolved = list(SpecifierSet("==0.2.0a1").filter([Version(v) for v in RELEASED]))
        assert resolved == [Version("0.2.0a1")]

    def test_asking_for_pre_releases_resolves_the_newest_alpha(self) -> None:
        resolved = SpecifierSet(">=0.2.0a1").filter(
            [Version(v) for v in RELEASED], prereleases=True
        )
        assert max(resolved) == Version("0.2.0rc1")


class TestRetention:
    def test_alphas_of_a_version_that_has_since_shipped_are_obsolete(self) -> None:
        """Nobody should be building against a pre-release of an already-stable version."""
        released = ("0.1.0a1", "0.1.0a2", "0.1.0", "0.2.0a1")
        assert alpha.stale(released=released, keep=5) == ["0.1.0a1", "0.1.0a2"]

    def test_only_the_most_recent_alphas_of_the_open_version_are_kept(self) -> None:
        released = ("0.2.0a1", "0.2.0a2", "0.2.0a3", "0.2.0a4")
        assert alpha.stale(released=released, keep=2) == ["0.2.0a1", "0.2.0a2"]

    def test_stable_releases_are_never_stale(self) -> None:
        """The retention rule touches the alpha channel only; a yank of a stable is an incident."""
        assert alpha.stale(released=("0.1.0", "0.2.0", "0.3.0"), keep=1) == []

    def test_release_candidates_are_kept(self) -> None:
        """An rc is a release under evaluation, not a dev build."""
        assert alpha.stale(released=("0.2.0rc1", "0.2.0rc2"), keep=1) == []

    def test_nothing_is_stale_before_the_retention_count_is_reached(self) -> None:
        assert alpha.stale(released=("0.2.0a1",), keep=3) == []


class TestCommandLine:
    @pytest.fixture(autouse=True)
    def no_repository_versions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alpha, "tagged_versions", lambda: ())

    def test_a_repository_release_advances_the_alpha_when_pypi_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(alpha, "released_versions", lambda: ())
        monkeypatch.setattr(alpha, "tagged_versions", lambda: ("0.52.0",))

        assert alpha.main([]) == 0
        assert capsys.readouterr().out.strip() == "0.53.0a1"

    def test_the_next_alpha_is_printed_for_the_workflow_to_tag(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(alpha, "released_versions", lambda: RELEASED)
        assert alpha.main([]) == 0
        assert capsys.readouterr().out.strip() == "0.2.0a3"

    def test_the_retention_report_names_what_to_yank(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            alpha, "released_versions", lambda: ("0.2.0a1", "0.2.0a2", "0.2.0a3", "0.2.0a4")
        )
        assert alpha.main(["--retention", "--keep", "2"]) == 0
        out = capsys.readouterr().out
        assert "0.2.0a1" in out
        assert "0.2.0a4" not in out

    def test_a_clean_retention_report_says_so(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(alpha, "released_versions", lambda: ("0.2.0a1",))
        assert alpha.main(["--retention"]) == 0
        assert "nothing" in capsys.readouterr().out.lower()

    def test_an_alpha_that_cannot_be_numbered_fails_rather_than_guessing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(alpha, "released_versions", lambda: ("0.2.0",))
        monkeypatch.setattr(alpha, "next_base", lambda released: "0.2.0")  # noqa: ARG005
        assert alpha.main([]) == 1
        assert "already released" in capsys.readouterr().err


class TestStabilityMatrix:
    """A subpackage with no stated stability is a promise nobody made and consumers assume."""

    def _matrix(self) -> list[tuple[str, str]]:
        row = r"^\| `([a-z0-9_]+)` \| `([a-z]+)` \|"
        return re.findall(row, STABILITY.read_text("utf-8"), re.M)

    def test_every_subpackage_states_its_stability(self) -> None:
        documented = {name for name, _ in self._matrix()}
        shipped = {
            path.name
            for path in PACKAGE.iterdir()
            if path.is_dir() and not path.name.startswith(("_", "."))
        }
        assert shipped <= documented

    def test_every_stated_stability_is_one_of_the_documented_levels(self) -> None:
        stated = {level for _, level in self._matrix()}
        assert stated
        assert stated <= alpha.STABILITY_LEVELS

    def test_the_page_says_the_alpha_channel_carries_no_stability_promise(self) -> None:
        assert "alpha" in STABILITY.read_text("utf-8").lower()
