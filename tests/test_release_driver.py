"""Choosing the version, rather than remembering which digit to move.

The policy in docs/versioning.md is unambiguous and still easy to get wrong at the moment
it is applied, which is the moment nobody can take back: an index artefact is that version
forever. So the release number is derived from what is actually pending — the change
fragments and the public surface diff — and the releaser confirms an answer instead of
inventing one.

The bias throughout is upward. Shipping breaking work as a patch is the failure that
reaches consumers as a broken build; shipping a fix as a minor costs a version number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tools.release as driver
from tools.release import Bump, ReleaseError, next_version, plan, why
from tools.release_notes import Fragment

if TYPE_CHECKING:
    from pathlib import Path

FIRST = "0.1.0"


def _fragment(kind: str, issue: str = "1") -> Fragment:
    migration = "do the thing" if kind in {"breaking", "removed"} else None
    return Fragment(id=issue, kind=kind, text="something changed", migration=migration)


def _diff(**kinds: list[str]) -> dict[str, list[str]]:
    return {"added": [], "removed": [], "changed": [], **kinds}


class TestTheFirstRelease:
    def test_with_no_tag_the_first_release_is_named(self) -> None:
        assert next_version(current=None, fragments=(), diff=_diff()) == FIRST

    def test_what_is_pending_does_not_change_which_release_is_first(self) -> None:
        """0.1.0 is the first number the policy allows; there is nothing to bump from."""
        assert next_version(current=None, fragments=(_fragment("breaking"),), diff=_diff()) == FIRST


class TestBeforeOneWhereTheMinorIsTheBreakingChannel:
    def test_a_breaking_fragment_takes_the_minor(self) -> None:
        assert next_version(current="0.4.2", fragments=(_fragment("breaking"),), diff=_diff()) == (
            "0.5.0"
        )

    def test_a_removal_is_breaking_whatever_it_is_called(self) -> None:
        assert next_version(current="0.4.2", fragments=(_fragment("removed"),), diff=_diff()) == (
            "0.5.0"
        )

    def test_new_surface_takes_the_minor_too(self) -> None:
        """The policy is explicit: adding surface is at least a minor, never a patch."""
        assert next_version(current="0.4.2", fragments=(_fragment("added"),), diff=_diff()) == (
            "0.5.0"
        )

    def test_a_fix_alone_takes_the_patch(self) -> None:
        assert next_version(current="0.4.2", fragments=(_fragment("fixed"),), diff=_diff()) == (
            "0.4.3"
        )

    def test_nothing_pending_still_moves_the_number(self) -> None:
        """Re-releasing a version with different content is the one thing never allowed."""
        assert next_version(current="0.4.2", fragments=(), diff=_diff()) == "0.4.3"


class TestFromOneOnwardWhereTheMajorIsTheBreakingChannel:
    def test_a_breaking_change_takes_the_major(self) -> None:
        assert next_version(current="1.4.2", fragments=(_fragment("breaking"),), diff=_diff()) == (
            "2.0.0"
        )

    def test_new_surface_still_takes_the_minor(self) -> None:
        assert next_version(current="1.4.2", fragments=(_fragment("added"),), diff=_diff()) == (
            "1.5.0"
        )

    def test_a_fix_still_takes_the_patch(self) -> None:
        assert next_version(current="1.4.2", fragments=(_fragment("fixed"),), diff=_diff()) == (
            "1.4.3"
        )


class TestTheSurfaceSnapshotOutranksWhatWasWrittenDown:
    def test_a_symbol_that_disappeared_is_breaking_even_if_no_fragment_says_so(self) -> None:
        """The snapshot is evidence; a fragment is a claim, and consumers meet the evidence."""
        pending = (_fragment("fixed"),)
        assert next_version(current="0.4.2", fragments=pending, diff=_diff(removed=["a.b"])) == (
            "0.5.0"
        )

    def test_a_symbol_that_changed_shape_is_breaking_too(self) -> None:
        assert next_version(current="0.4.2", fragments=(), diff=_diff(changed=["a.b"])) == "0.5.0"

    def test_a_new_symbol_lifts_a_patch_to_a_minor(self) -> None:
        assert next_version(current="0.4.2", fragments=(), diff=_diff(added=["a.b"])) == "0.5.0"

    def test_the_largest_reason_wins_rather_than_the_last_one_read(self) -> None:
        pending = (_fragment("fixed", "1"), _fragment("breaking", "2"), _fragment("added", "3"))
        assert next_version(current="0.4.2", fragments=pending, diff=_diff()) == "0.5.0"


class TestSayingWhy:
    def test_the_reason_names_what_forced_the_bump(self) -> None:
        """A number nobody can argue with is a number nobody checks."""
        reasons = why(fragments=(_fragment("breaking"),), diff=_diff())
        assert any("breaking" in reason for reason in reasons[Bump.BREAKING])

    def test_a_removed_symbol_is_named_so_it_can_be_looked_at(self) -> None:
        reasons = why(fragments=(), diff=_diff(removed=["tesserix_adk.core.Gone"]))
        assert any("tesserix_adk.core.Gone" in reason for reason in reasons[Bump.BREAKING])

    def test_nothing_pending_says_so_rather_than_saying_nothing(self) -> None:
        assert why(fragments=(), diff=_diff()) == {}


class TestThePlanTheReleaserConfirms:
    def test_it_names_the_version_that_is_going_out(self) -> None:
        rendered = plan(current="0.4.2", fragments=(_fragment("added"),), diff=_diff())
        assert "0.4.2 -> 0.5.0" in rendered

    def test_it_names_the_first_release_readably(self) -> None:
        rendered = plan(current=None, fragments=(), diff=_diff())
        assert "first release -> 0.1.0" in rendered

    def test_it_carries_the_reasons_into_what_the_releaser_reads(self) -> None:
        rendered = plan(current="0.4.2", fragments=(_fragment("breaking"),), diff=_diff())
        assert "breaking" in rendered

    def test_a_release_with_nothing_pending_says_so(self) -> None:
        assert "nothing pending" in plan(current="0.4.2", fragments=(), diff=_diff())

    def test_a_long_tail_of_ordinary_reasons_is_counted_rather_than_listed(self) -> None:
        """A hundred lines of `adds surface` is a page nobody reads."""
        many = tuple(_fragment("added", str(n)) for n in range(20))
        assert "and 15 more" in plan(current="0.4.2", fragments=many, diff=_diff())

    def test_every_breaking_reason_is_listed_however_many_there_are(self) -> None:
        """The one kind worth reading in full is the one that breaks a consumer."""
        many = tuple(_fragment("breaking", str(n)) for n in range(20))
        rendered = plan(current="0.4.2", fragments=many, diff=_diff())
        assert rendered.count("is breaking") == 20


class TestWhatIsRefused:
    def test_a_current_version_that_names_no_release_is_refused(self) -> None:
        with pytest.raises(ReleaseError):
            next_version(current="not-a-version", fragments=(), diff=_diff())

    def test_a_breaking_fragment_with_no_migration_note_is_refused(self) -> None:
        """The note is the whole reason the breaking channel exists."""
        undocumented = Fragment(id="1", kind="breaking", text="gone")
        with pytest.raises(ReleaseError):
            next_version(current="0.4.2", fragments=(undocumented,), diff=_diff())


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """A repository at 0.4.2 with one fix pending, and a notes tool that records its call."""
    folded: list[list[str]] = []

    def record(argv: list[str]) -> int:
        folded.append(argv)
        return 0

    monkeypatch.setattr(driver, "latest_tag", lambda: "v0.4.2")
    monkeypatch.setattr(driver, "read_fragments", lambda _: (_fragment("fixed"),))
    monkeypatch.setattr(driver, "surface_diff", _diff)
    monkeypatch.setattr(driver, "assemble", record)
    return folded


@pytest.mark.usefixtures("repo")
class TestTheCommandTheReleaserRuns:
    def test_it_shows_the_plan_without_touching_anything(
        self, capsys: pytest.CaptureFixture[str], repo: list[list[str]]
    ) -> None:
        assert driver.main([]) == 0
        assert "0.4.2 -> 0.4.3" in capsys.readouterr().out
        assert repo == []

    def test_it_names_the_command_that_cuts_the_release_it_just_described(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        driver.main([])
        assert "make release VERSION=0.4.3" in capsys.readouterr().out

    def test_applying_folds_the_notes_and_consumes_the_fragments(
        self, repo: list[list[str]]
    ) -> None:
        assert driver.main(["--apply"]) == 0
        assert repo == [["--version", "0.4.3", "--release"]]

    def test_applying_writes_the_release_body_where_it_was_asked_to(
        self, tmp_path: Path, repo: list[list[str]]
    ) -> None:
        driver.main(["--apply", "--notes", str(tmp_path / "notes.md")])
        assert "--output" in repo[0]

    def test_it_stops_short_of_the_commit_and_the_tag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Pushing the tag is the publish, so it stays a decision somebody makes."""
        driver.main(["--apply"])
        printed = capsys.readouterr().out
        assert "git tag -a v0.4.3" in printed
        assert "git push origin v0.4.3" in printed

    def test_an_overridden_version_is_used_and_said_out_loud(
        self, capsys: pytest.CaptureFixture[str], repo: list[list[str]]
    ) -> None:
        driver.main(["--apply", "--version", "1.0.0"])
        assert "overridden" in capsys.readouterr().out
        assert repo == [["--version", "1.0.0", "--release"]]

    def test_a_version_that_is_not_a_version_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`make release` defaults VERSION to a word; releasing it would tag `vnext`."""
        assert driver.main(["--version", "next"]) == 1
        assert "next" in capsys.readouterr().err

    def test_a_failure_in_the_notes_tool_stops_the_release(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An undocumented change must block the release, not be discovered after it."""
        monkeypatch.setattr(driver, "assemble", lambda _: 2)
        assert driver.main(["--apply"]) == 2


def test_a_tag_that_names_no_release_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(driver, "latest_tag", lambda: "nightly-2026-08-13")
    assert driver.main([]) == 1
    assert "nightly-2026-08-13" in capsys.readouterr().err


def test_before_the_first_tag_the_plan_still_works(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(driver, "latest_tag", lambda: None)
    monkeypatch.setattr(driver, "read_fragments", lambda _: ())
    monkeypatch.setattr(driver, "surface_diff", _diff)
    assert driver.main([]) == 0
    assert f"first release -> {FIRST}" in capsys.readouterr().out
