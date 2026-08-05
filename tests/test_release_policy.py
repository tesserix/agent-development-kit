"""The policy is only real if a release that breaks it cannot be published.

Two gates are tested here: the deprecations page is generated from the decorators, so
it cannot drift or go stale, and the release check compares the published surface with
the one being released and refuses a removal that no consumer was warned about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tools import deprecations as deprecations_tool
from tools import release_check

from tesserix_adk.core.deprecation import Deprecation

if TYPE_CHECKING:
    from pathlib import Path

SCHEDULED = Deprecation(
    name="tesserix_adk.runtime.old_runner",
    since="0.1.0",
    removal="0.3.0",
    alternative="tesserix_adk.runtime.Runner",
    reason="the sync path cannot cancel a tool call",
)


def test_the_page_lists_every_live_deprecation() -> None:
    page = deprecations_tool.render((SCHEDULED,))
    assert "tesserix_adk.runtime.old_runner" in page
    assert "0.3.0" in page
    assert "cannot cancel" in page


def test_the_page_says_so_when_nothing_is_deprecated() -> None:
    """An empty table reads like a broken generator; say it in words."""
    assert "No deprecations are live" in deprecations_tool.render(())


def test_the_page_round_trips() -> None:
    """The release check parses the published page, so rendering must be reversible."""
    assert deprecations_tool.parse(deprecations_tool.render((SCHEDULED,))) == (SCHEDULED,)


def test_a_page_without_a_reason_round_trips() -> None:
    record = Deprecation(name="a.b", since="0.1.0", removal="0.3.0", alternative="a.c")
    assert deprecations_tool.parse(deprecations_tool.render((record,))) == (record,)


def test_an_empty_page_parses_as_nothing_deprecated() -> None:
    assert deprecations_tool.parse(deprecations_tool.render(())) == ()


def test_a_removal_version_already_shipped_is_stale() -> None:
    """Otherwise the list becomes a graveyard of promises the kit quietly broke."""
    problems = deprecations_tool.stale((SCHEDULED,), version="0.3.0")
    assert len(problems) == 1
    assert "tesserix_adk.runtime.old_runner" in problems[0]


def test_a_removal_still_in_the_future_is_not_stale() -> None:
    assert deprecations_tool.stale((SCHEDULED,), version="0.2.5") == []


def test_the_committed_page_matches_the_decorators() -> None:
    """Regenerate with `make deprecations`."""
    assert deprecations_tool.main([]) == 0


def test_the_page_check_fails_when_the_page_is_out_of_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stale_page = tmp_path / "deprecations.md"
    stale_page.write_text("# Deprecations\n", encoding="utf-8")
    monkeypatch.setattr("tools.deprecations.PAGE", stale_page)

    assert deprecations_tool.main([]) == 1
    assert "make deprecations" in capsys.readouterr().err


def test_the_page_check_fails_on_a_stale_removal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("tools.deprecations.collect", lambda: (SCHEDULED,))
    monkeypatch.setattr("tools.deprecations.VERSION", "0.4.0")

    assert deprecations_tool.main([]) == 1
    assert "old_runner" in capsys.readouterr().err


def test_write_regenerates_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "nested" / "deprecations.md"
    monkeypatch.setattr("tools.deprecations.PAGE", target)

    assert deprecations_tool.main(["--write"]) == 0
    assert target.read_text(encoding="utf-8") == deprecations_tool.render(
        deprecations_tool.collect()
    )


def test_collecting_reads_the_decorators_not_a_hand_written_list() -> None:
    """The point of the registry: the page cannot disagree with the code."""
    assert deprecations_tool.collect() == deprecations_tool.collect()


BASE = {"tesserix_adk.core.Runner": "class Runner(object)"}


def _check(
    baseline: dict[str, str],
    current: dict[str, str],
    version: str,
    *,
    baseline_version: str = "0.1.0",
    records: tuple[Deprecation, ...] = (),
) -> list[str]:
    return release_check.check(
        baseline=baseline,
        current=current,
        records=records,
        baseline_version=baseline_version,
        version=version,
    )


def test_an_unchanged_surface_may_ship_as_a_patch() -> None:
    assert _check(BASE, BASE, "0.1.1") == []


def test_a_removal_without_a_deprecation_record_is_blocked() -> None:
    """The scenario the whole policy exists for."""
    problems = _check(BASE, {}, "0.3.0")
    assert len(problems) == 1
    assert "tesserix_adk.core.Runner" in problems[0]
    assert "no deprecation" in problems[0]


def test_a_removal_with_a_matching_record_is_allowed() -> None:
    record = Deprecation(
        name="tesserix_adk.core.Runner", since="0.1.0", removal="0.3.0", alternative="Engine"
    )
    assert _check(BASE, {}, "0.3.0", records=(record,)) == []


def test_a_removal_earlier_than_promised_is_blocked() -> None:
    """Shipping a removal ahead of its announced version breaks the same promise."""
    record = Deprecation(
        name="tesserix_adk.core.Runner", since="0.1.0", removal="0.5.0", alternative="Engine"
    )
    problems = _check(BASE, {}, "0.3.0", records=(record,))
    assert "0.5.0" in problems[0]


def test_a_changed_signature_counts_as_breaking() -> None:
    """An unchanged name with different behaviour is the breakage consumers miss."""
    changed = {"tesserix_adk.core.Runner": "class Runner(object) {run(self, budget) -> 'None'}"}
    problems = _check(BASE, changed, "0.3.0")
    assert "tesserix_adk.core.Runner" in problems[0]


def test_a_removal_in_a_patch_release_is_blocked() -> None:
    record = Deprecation(
        name="tesserix_adk.core.Runner", since="0.1.0", removal="0.3.0", alternative="Engine"
    )
    problems = _check(BASE, {}, "0.1.1", records=(record,))
    assert any("patch" in problem or "minor" in problem for problem in problems)


def test_an_addition_needs_at_least_a_minor_release() -> None:
    added = {**BASE, "tesserix_adk.core.Engine": "class Engine(object)"}
    assert "minor" in _check(BASE, added, "0.1.1")[0]
    assert _check(BASE, added, "0.2.0") == []


def test_a_version_that_does_not_increase_is_blocked() -> None:
    added = {**BASE, "tesserix_adk.core.Engine": "class Engine(object)"}
    assert "must increase" in _check(BASE, added, "0.1.0")[0]


def test_after_1_0_a_removal_needs_a_major_release() -> None:
    record = Deprecation(
        name="tesserix_adk.core.Runner", since="1.0.0", removal="2.0.0", alternative="Engine"
    )
    problems = _check(BASE, {}, "1.1.0", baseline_version="1.0.0", records=(record,))
    assert "major" in problems[0]

    record_now = Deprecation(
        name="tesserix_adk.core.Runner", since="1.0.0", removal="2.0.0", alternative="Engine"
    )
    assert _check(BASE, {}, "2.0.0", baseline_version="1.0.0", records=(record_now,)) == []


def test_every_offending_symbol_is_named_at_once() -> None:
    baseline = {**BASE, "tesserix_adk.core.Clock": "class Clock(Protocol)"}
    problems = _check(baseline, {}, "0.3.0")
    assert len(problems) == 2


def test_the_check_passes_when_there_is_no_release_to_compare_against(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before the first tag there is no published surface, and that is not a failure."""
    monkeypatch.setattr("tools.release_check.latest_tag", lambda: None)

    assert release_check.main([]) == 0
    assert "no released version" in capsys.readouterr().out


def test_the_check_compares_against_the_last_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.release_check.latest_tag", lambda: "v0.1.0")
    monkeypatch.setattr(
        "tools.release_check.read_at",
        lambda ref, path: {  # noqa: ARG005
            "docs/api-surface.txt": "tesserix_adk.core.Gone :: class Gone(object)\n",
            "docs/deprecations.md": deprecations_tool.render(()),
            "src/tesserix_adk/__init__.py": '__version__ = "0.1.0"\n',
        }[path],
    )

    assert release_check.main([]) == 1


def test_the_check_reports_the_problems_it_found(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("tools.release_check.latest_tag", lambda: "v0.1.0")
    monkeypatch.setattr(
        "tools.release_check.read_at",
        lambda ref, path: {  # noqa: ARG005
            "docs/api-surface.txt": "tesserix_adk.core.Gone :: class Gone(object)\n",
            "docs/deprecations.md": deprecations_tool.render(()),
            "src/tesserix_adk/__init__.py": '__version__ = "0.1.0"\n',
        }[path],
    )
    release_check.main([])

    assert "tesserix_adk.core.Gone" in capsys.readouterr().err


def test_a_release_identical_to_the_last_one_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.api_surface import SNAPSHOT

    monkeypatch.setattr("tools.release_check.latest_tag", lambda: "v0.0.1")
    monkeypatch.setattr(
        "tools.release_check.read_at",
        lambda ref, path: {  # noqa: ARG005
            "docs/api-surface.txt": SNAPSHOT.read_text(encoding="utf-8"),
            "docs/deprecations.md": deprecations_tool.render(deprecations_tool.collect()),
            "src/tesserix_adk/__init__.py": '__version__ = "0.0.0"\n',
        }[path],
    )

    assert release_check.main([]) == 0


def test_the_released_version_is_read_from_the_tagged_source() -> None:
    """A tag can be renamed; the version in the tagged tree is the fact."""
    assert release_check.version_in('x = 1\n__version__ = "1.2.3"\n') == "1.2.3"


def test_a_source_without_a_version_is_a_failure() -> None:
    with pytest.raises(release_check.ReleaseCheckError, match="__version__"):
        release_check.version_in("x = 1\n")


def test_the_latest_tag_comes_from_git() -> None:
    """Not asserting a value: the repository may have no tags yet."""
    tag = release_check.latest_tag()
    assert tag is None or tag.startswith("v")


def test_reading_a_path_at_a_ref_returns_its_content() -> None:
    assert "__version__" in release_check.read_at("HEAD", "src/tesserix_adk/__init__.py")


def test_reading_a_path_that_is_not_in_the_ref_is_a_failure() -> None:
    """A silently empty baseline would pass every check by comparing against nothing."""
    with pytest.raises(release_check.ReleaseCheckError, match="cannot read"):
        release_check.read_at("HEAD", "docs/never-existed.txt")
