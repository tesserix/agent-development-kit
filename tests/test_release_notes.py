"""Release notes are derived from the repository, not typed by hand.

The entries that get left out of hand-written notes are the breaking ones, and a
consumer then meets the change as a failing test. So the assembly is mechanical, and a
change nobody documented fails the release rather than shipping silently.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from tools import release_notes
from tools.release_check import ReleaseCheckError
from tools.release_notes import Commit, Fragment, NoteError

if TYPE_CHECKING:
    from pathlib import Path

BREAKING_FRAGMENT = Fragment(
    id="42",
    kind="removed",
    surface="tesserix_adk.core.load_config",
    migration="Call `resolve_config` and read `.config`.",
    text="`load_config` is gone.",
)


def write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


class TestFragments:
    def test_the_kind_comes_from_the_file_name(self, tmp_path: Path) -> None:
        write(tmp_path, "11.added.md", "A new thing.\n")
        (fragment,) = release_notes.read_fragments(tmp_path)
        assert (fragment.id, fragment.kind, fragment.text) == ("11", "added", "A new thing.")

    def test_a_header_carries_the_surface_and_the_migration(self, tmp_path: Path) -> None:
        write(
            tmp_path,
            "42.removed.md",
            "---\nsurface: tesserix_adk.core.load_config\nmigration: Use `resolve_config`.\n---\n"
            "`load_config` is gone.\n",
        )
        (fragment,) = release_notes.read_fragments(tmp_path)
        assert fragment.surface == "tesserix_adk.core.load_config"
        assert fragment.migration == "Use `resolve_config`."

    def test_a_kind_nobody_recognises_is_refused(self, tmp_path: Path) -> None:
        write(tmp_path, "11.improved.md", "Better.\n")
        with pytest.raises(NoteError, match="improved"):
            release_notes.read_fragments(tmp_path)

    def test_a_file_that_does_not_name_an_issue_and_a_kind_is_refused(self, tmp_path: Path) -> None:
        write(tmp_path, "notes.md", "Better.\n")
        with pytest.raises(NoteError, match=r"notes\.md"):
            release_notes.read_fragments(tmp_path)

    def test_an_empty_fragment_is_refused(self, tmp_path: Path) -> None:
        """A fragment with a header and nothing to say documents nothing."""
        write(tmp_path, "11.added.md", "---\nsurface: tesserix_adk.core\n---\n\n")
        with pytest.raises(NoteError, match="no text"):
            release_notes.read_fragments(tmp_path)

    def test_a_breaking_fragment_without_a_migration_is_refused(self, tmp_path: Path) -> None:
        write(tmp_path, "42.removed.md", "`load_config` is gone.\n")
        with pytest.raises(NoteError, match="migration"):
            release_notes.read_fragments(tmp_path)

    def test_a_missing_directory_is_no_fragments_rather_than_an_error(self, tmp_path: Path) -> None:
        assert release_notes.read_fragments(tmp_path / "absent") == ()

    def test_fragments_are_read_in_issue_order(self, tmp_path: Path) -> None:
        """So the notes do not reorder themselves between two runs on the same tree."""
        write(tmp_path, "100.added.md", "Later.\n")
        write(tmp_path, "9.added.md", "Earlier.\n")
        assert [f.id for f in release_notes.read_fragments(tmp_path)] == ["9", "100"]


class TestCommitSubjects:
    @pytest.mark.parametrize(
        ("subject", "kind"),
        [
            ("feat(core): add resolve_config", "added"),
            ("fix(runtime): stop dropping the last chunk", "fixed"),
            ("refactor(core): fold the frozen base class in", "changed"),
            ("perf(rag): batch the embedding calls", "changed"),
            ("docs: describe the release path", "internal"),
            ("chore(deps): bump httpx", "internal"),
            ("revert: feat(core): add resolve_config", "reverted"),
        ],
    )
    def test_a_conventional_subject_names_its_section(self, subject: str, kind: str) -> None:
        parsed = release_notes.parse_subject(subject)
        assert parsed is not None
        assert parsed.kind == kind

    def test_a_bang_marks_the_change_breaking(self) -> None:
        parsed = release_notes.parse_subject("feat(core)!: replace load_config")
        assert parsed is not None
        assert parsed.kind == "breaking"

    @pytest.mark.parametrize("subject", ["fixed a thing", "WIP", "core: tidy up"])
    def test_a_subject_that_follows_no_convention_is_not_parsed(self, subject: str) -> None:
        assert release_notes.parse_subject(subject) is None

    def test_the_scope_is_kept_for_attribution(self) -> None:
        parsed = release_notes.parse_subject("feat(rag): hybrid retrieval")
        assert parsed is not None
        assert parsed.scope == "rag"
        assert parsed.text == "hybrid retrieval"


class TestUndocumentedChanges:
    def test_a_commit_with_neither_a_fragment_nor_a_readable_subject_blocks_the_release(
        self,
    ) -> None:
        commits = (Commit(sha="abc1234", subject="fixed a thing"),)
        problems = release_notes.undocumented(commits, ())
        assert any("abc1234" in problem for problem in problems)

    def test_a_fragment_documents_the_commits_that_reference_its_issue(self) -> None:
        commits = (Commit(sha="abc1234", subject="fixed a thing (#42)"),)
        assert release_notes.undocumented(commits, (BREAKING_FRAGMENT,)) == []

    def test_a_breaking_commit_with_no_migration_note_blocks_the_release(self) -> None:
        """The migration note is the whole point of documenting a breaking change."""
        commits = (Commit(sha="abc1234", subject="feat(core)!: replace load_config"),)
        problems = release_notes.undocumented(commits, ())
        assert any("migration" in problem for problem in problems)

    def test_a_breaking_commit_is_satisfied_by_a_fragment_carrying_the_migration(self) -> None:
        commits = (Commit(sha="abc1234", subject="feat(core)!: replace load_config (#42)"),)
        assert release_notes.undocumented(commits, (BREAKING_FRAGMENT,)) == []

    def test_a_trailer_links_a_commit_to_its_fragment(self) -> None:
        """`Closes #42` is where the convention puts the link, and a pushed subject cannot
        be edited to move it."""
        commits = (
            Commit(sha="abc1234", subject="feat(core)!: replace load_config", body="Closes #42"),
        )
        assert release_notes.undocumented(commits, (BREAKING_FRAGMENT,)) == []

    def test_every_undocumented_change_is_reported_at_once(self) -> None:
        commits = (
            Commit(sha="abc1234", subject="fixed a thing"),
            Commit(sha="def5678", subject="another thing"),
        )
        assert len(release_notes.undocumented(commits, ())) == 2


class TestAssembly:
    def test_each_kind_lands_in_its_own_section(self) -> None:
        sections = release_notes.assemble(
            fragments=(
                Fragment(id="1", kind="added", text="A thing."),
                Fragment(id="2", kind="fixed", text="A fix."),
            ),
            commits=(),
        )
        assert sections["Added"] == ["A thing."]
        assert sections["Fixed"] == ["A fix."]

    def test_a_breaking_entry_carries_its_migration(self) -> None:
        sections = release_notes.assemble(fragments=(BREAKING_FRAGMENT,), commits=())
        (entry,) = sections["Breaking changes"]
        assert "Call `resolve_config`" in entry
        assert "tesserix_adk.core.load_config" in entry

    def test_a_change_spanning_several_commits_appears_once(self) -> None:
        """Attributed to the surface it affects, not repeated per subpackage."""
        commits = (
            Commit(sha="a", subject="feat(core): part one (#42)"),
            Commit(sha="b", subject="feat(rag): part two (#42)"),
        )
        sections = release_notes.assemble(fragments=(BREAKING_FRAGMENT,), commits=commits)
        assert sum(len(entries) for entries in sections.values()) == 1

    def test_experimental_changes_are_kept_apart_from_stable_surface(self) -> None:
        """Both sections exist so a reader can tell which promises apply."""
        sections = release_notes.assemble(
            fragments=(
                Fragment(id="1", kind="added", surface="tesserix_adk.experimental.x", text="New."),
                Fragment(id="2", kind="added", surface="tesserix_adk.core", text="Also new."),
            ),
            commits=(),
        )
        assert "New." in sections["Experimental"][0]
        assert "Also new." in sections["Added"][0]

    def test_a_revert_is_reported_because_it_is_a_change_too(self) -> None:
        """Silently reverting a feature breaks whoever adopted it."""
        commits = (Commit(sha="a", subject="revert: feat(core): add resolve_config"),)
        sections = release_notes.assemble(fragments=(), commits=commits)
        assert sections["Reverted"] == ["feat(core): add resolve_config"]

    def test_housekeeping_commits_produce_no_consumer_facing_entry(self) -> None:
        commits = (Commit(sha="a", subject="chore(deps): bump httpx"),)
        assert release_notes.assemble(fragments=(), commits=commits) == {}

    def test_a_commit_entry_names_the_scope_it_affected(self) -> None:
        commits = (Commit(sha="a", subject="feat(rag): hybrid retrieval"),)
        sections = release_notes.assemble(fragments=(), commits=commits)
        assert sections["Added"] == ["**rag**: hybrid retrieval"]


class TestSurfaceDiff:
    def test_the_diff_reports_what_the_snapshot_says_changed(self) -> None:
        diff = release_notes.surface_diff(
            baseline={"a": "def a()", "b": "def b()", "c": "def c()"},
            current={"a": "def a()", "b": "def b(x)", "d": "def d()"},
        )
        assert diff == {"added": ["d"], "removed": ["c"], "changed": ["b"]}

    def test_an_unchanged_surface_produces_an_empty_diff(self) -> None:
        assert release_notes.surface_diff(baseline={"a": "def a()"}, current={"a": "def a()"}) == {
            "added": [],
            "removed": [],
            "changed": [],
        }


class TestRendering:
    def test_the_notes_state_the_version_and_every_section_with_entries(self) -> None:
        rendered = release_notes.render(
            version="0.3.0",
            sections={"Added": ["A thing."], "Fixed": ["A fix."]},
            diff={"added": [], "removed": [], "changed": []},
            deprecations=(),
        )
        assert "## 0.3.0" in rendered
        assert "### Added" in rendered
        assert "- A thing." in rendered

    def test_an_empty_section_is_not_rendered(self) -> None:
        rendered = release_notes.render(
            version="0.3.0",
            sections={"Added": ["A thing."]},
            diff={"added": [], "removed": [], "changed": []},
            deprecations=(),
        )
        assert "### Fixed" not in rendered

    def test_the_surface_diff_is_attached(self) -> None:
        rendered = release_notes.render(
            version="0.3.0",
            sections={},
            diff={"added": ["tesserix_adk.core.resolve_config"], "removed": [], "changed": []},
            deprecations=(),
        )
        assert "tesserix_adk.core.resolve_config" in rendered

    def test_live_deprecations_are_listed_with_their_removal_version(self) -> None:
        from tesserix_adk.core.deprecation import Deprecation

        record = Deprecation(
            name="tesserix_adk.core.old", since="0.3.0", removal="0.5.0", alternative="new"
        )
        rendered = release_notes.render(
            version="0.3.0",
            sections={},
            diff={"added": [], "removed": [], "changed": []},
            deprecations=(record,),
        )
        assert "tesserix_adk.core.old" in rendered
        assert "0.5.0" in rendered

    def test_notes_with_nothing_in_them_say_so_rather_than_rendering_blank(self) -> None:
        rendered = release_notes.render(
            version="0.3.0",
            sections={},
            diff={"added": [], "removed": [], "changed": []},
            deprecations=(),
        )
        assert "No consumer-visible changes" in rendered


class TestChangelog:
    def test_the_release_section_replaces_the_unreleased_heading(self) -> None:
        changelog = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Old entry.\n"
        updated = release_notes.update_changelog(changelog, notes="## 0.3.0\n\n### Added\n\n- A.\n")
        assert "## [Unreleased]" in updated
        assert updated.index("## 0.3.0") > updated.index("## [Unreleased]")

    def test_the_hand_written_unreleased_entries_do_not_ship_twice(self) -> None:
        """They describe the same work the fragments do, written by hand as each merged."""
        changelog = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Old entry.\n"
        updated = release_notes.update_changelog(changelog, notes="## 0.3.0\n\n- A.\n")
        assert "Old entry" not in updated

    def test_the_releases_already_published_are_left_alone(self) -> None:
        changelog = "# Changelog\n\n## [Unreleased]\n\n- Pending.\n\n## 0.2.0\n\n- Shipped.\n"
        updated = release_notes.update_changelog(changelog, notes="## 0.3.0\n\n- A.\n")
        assert "## 0.2.0\n\n- Shipped." in updated
        assert updated.index("## 0.3.0") < updated.index("## 0.2.0")

    def test_the_link_definitions_at_the_foot_survive(self) -> None:
        """Dropping them turns every reference in the file into literal brackets."""
        changelog = (
            "# Changelog\n\n## [Unreleased]\n\n- Pending.\n\n[Keep a Changelog]: https://x\n"
        )
        updated = release_notes.update_changelog(changelog, notes="## 0.3.0\n\n- A.\n")
        assert updated.endswith("[Keep a Changelog]: https://x\n")

    def test_a_changelog_with_no_unreleased_heading_is_refused(self) -> None:
        with pytest.raises(NoteError, match="Unreleased"):
            release_notes.update_changelog("# Changelog\n", notes="## 0.3.0\n")


class TestCommandLine:
    def test_a_dry_run_renders_the_notes_without_touching_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write(tmp_path, "11.added.md", "A new thing.\n")
        monkeypatch.setattr(release_notes, "FRAGMENTS", tmp_path)
        monkeypatch.setattr(release_notes, "_history", lambda: ())
        monkeypatch.setattr(release_notes, "_surface_diff_against_last_release", lambda: _NO_DIFF)

        assert release_notes.main(["--version", "0.3.0", "--dry-run"]) == 0
        assert "A new thing." in capsys.readouterr().out

    def test_an_undocumented_change_fails_and_names_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(release_notes, "FRAGMENTS", tmp_path)
        monkeypatch.setattr(
            release_notes, "_history", lambda: (Commit(sha="abc1234", subject="a thing"),)
        )
        monkeypatch.setattr(release_notes, "_surface_diff_against_last_release", lambda: _NO_DIFF)

        assert release_notes.main(["--version", "0.3.0", "--dry-run"]) == 1
        assert "abc1234" in capsys.readouterr().err

    def test_a_bad_fragment_fails_with_the_file_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write(tmp_path, "11.improved.md", "Better.\n")
        monkeypatch.setattr(release_notes, "FRAGMENTS", tmp_path)

        assert release_notes.main(["--version", "0.3.0", "--dry-run"]) == 1
        assert "improved" in capsys.readouterr().err

    def test_the_notes_can_be_written_to_a_file_for_the_release_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path / "changes", "11.added.md", "A new thing.\n")
        monkeypatch.setattr(release_notes, "FRAGMENTS", tmp_path / "changes")
        monkeypatch.setattr(release_notes, "_history", lambda: ())
        monkeypatch.setattr(release_notes, "_surface_diff_against_last_release", lambda: _NO_DIFF)
        output = tmp_path / "notes.md"

        assert release_notes.main(["--version", "0.3.0", "--output", str(output)]) == 0
        assert "A new thing." in output.read_text(encoding="utf-8")

    def test_a_tag_release_body_reuses_the_reviewed_changelog_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [Unreleased]\n\n## 0.3.0\n\n### Added\n\n"
            "- The reviewed release note.\n\n## 0.2.0\n\n- Earlier.\n",
            encoding="utf-8",
        )
        output = tmp_path / "notes.md"
        monkeypatch.setattr(release_notes, "CHANGELOG", changelog)
        monkeypatch.setattr(
            release_notes,
            "_history",
            lambda: pytest.fail("released notes must not be rebuilt after fragments are consumed"),
        )

        assert release_notes.main(["--version", "0.3.0", "--output", str(output)]) == 0
        assert output.read_text(encoding="utf-8") == (
            "## 0.3.0\n\n### Added\n\n- The reviewed release note.\n"
        )

    def test_releasing_folds_the_fragments_into_the_changelog_and_clears_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fragment left behind would be announced again in the next release."""
        fragments = tmp_path / "changes"
        fragment = write(fragments, "11.added.md", "A new thing.\n")
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8")
        monkeypatch.setattr(release_notes, "FRAGMENTS", fragments)
        monkeypatch.setattr(release_notes, "CHANGELOG", changelog)
        monkeypatch.setattr(release_notes, "_history", lambda: ())
        monkeypatch.setattr(release_notes, "_surface_diff_against_last_release", lambda: _NO_DIFF)

        assert release_notes.main(["--version", "0.3.0", "--release"]) == 0
        assert "A new thing." in changelog.read_text(encoding="utf-8")
        assert not fragment.exists()

    def test_releasing_rebases_fragment_links_for_the_root_changelog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fragments = tmp_path / "changes"
        write(
            fragments,
            "11.added.md",
            "See [dead letters](../docs/dead-letters.md), "
            "the [project](https://example.com), and [details](#details).\n",
        )
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8")
        monkeypatch.setattr(release_notes, "FRAGMENTS", fragments)
        monkeypatch.setattr(release_notes, "CHANGELOG", changelog)
        monkeypatch.setattr(release_notes, "_history", lambda: ())
        monkeypatch.setattr(release_notes, "_surface_diff_against_last_release", lambda: _NO_DIFF)

        assert release_notes.main(["--version", "0.3.0", "--release"]) == 0
        released = changelog.read_text(encoding="utf-8")
        assert "[dead letters](docs/dead-letters.md)" in released
        assert "[dead letters](../docs/dead-letters.md)" not in released
        assert "[project](https://example.com)" in released
        assert "[details](#details)" in released


class TestRepositoryFragments:
    def test_every_fragment_in_the_repository_is_valid(self) -> None:
        """The dry-run job proves this on a pull request; this proves it on every run."""
        assert release_notes.read_fragments(release_notes.FRAGMENTS) is not None

    def test_the_history_is_read_from_git(self) -> None:
        """Not asserting a value: the range depends on where the last tag is."""
        assert isinstance(release_notes._history(), tuple)


_NO_DIFF: dict[str, list[str]] = {"added": [], "removed": [], "changed": []}


class TestUncheckedPaths:
    def test_a_documented_breaking_commit_still_needs_the_migration_on_its_fragment(self) -> None:
        """The fragment exists, so nothing else complains — this is the only check left."""
        fragment = Fragment(id="42", kind="added", text="A thing.")
        commits = (Commit(sha="abc1234", subject="feat(core)!: replace it (#42)"),)
        problems = release_notes.undocumented(commits, (fragment,))
        assert any("migration" in problem for problem in problems)

    def test_an_ordinary_conventional_commit_needs_no_fragment(self) -> None:
        commits = (Commit(sha="abc1234", subject="feat(core): add a thing"),)
        assert release_notes.undocumented(commits, ()) == []

    def test_a_history_git_cannot_read_is_no_commits_rather_than_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before the first commit, or outside a work tree, the notes are simply empty."""
        monkeypatch.setattr(release_notes, "latest_tag", lambda: None)

        def failed(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args, 128, "", "not a git repository")

        monkeypatch.setattr("tools.release_notes.subprocess.run", failed)
        assert release_notes._history() == ()

    def test_the_first_release_has_no_surface_to_diff_against(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(release_notes, "latest_tag", lambda: None)
        assert release_notes._surface_diff_against_last_release() == _NO_DIFF

    def test_a_release_whose_snapshot_cannot_be_read_still_produces_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The snapshot postdates the tag on the first release that introduced it."""

        def unreadable(ref: str, path: str) -> str:
            raise ReleaseCheckError(f"{path} is not in {ref}")

        monkeypatch.setattr(release_notes, "latest_tag", lambda: "v0.1.0")
        monkeypatch.setattr(release_notes, "read_at", unreadable)
        assert release_notes._surface_diff_against_last_release() == _NO_DIFF

    def test_the_diff_is_taken_against_the_snapshot_published_at_the_last_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(release_notes, "latest_tag", lambda: "v0.1.0")
        monkeypatch.setattr(release_notes, "read_at", lambda ref, path: "gone :: def gone()\n")  # noqa: ARG005
        monkeypatch.setattr(release_notes, "collect_surface", dict)
        assert release_notes._surface_diff_against_last_release()["removed"] == ["gone"]

    def test_clearing_the_fragments_leaves_anything_that_is_not_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The directory's README is not a change and must survive every release."""
        readme = write(tmp_path, "README.md", "How to write one.\n")
        consumed = write(tmp_path, "11.added.md", "A thing.\n")
        monkeypatch.setattr(release_notes, "FRAGMENTS", tmp_path)

        release_notes._consumed([Fragment(id="11", kind="added", text="A thing.")])
        assert readme.exists()
        assert not consumed.exists()
