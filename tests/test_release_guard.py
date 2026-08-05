"""A tag is the release. Anything the guard lets through is what consumers install.

Index artefacts are immutable, so every one of these checks has to happen before the
build rather than after the upload: a wrong version cannot be corrected, only yanked and
superseded.
"""

from __future__ import annotations

import io
import json
import urllib.error
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from tools import release_guard
from tools.versions import ReleaseVersionError, parts, release_segment

if TYPE_CHECKING:
    from collections.abc import Iterator

RELEASED = ("0.1.0", "0.2.0")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.2.3", (1, 2, 3)),
        ("0.0.1", (0, 0, 1)),
        ("0.2.0rc1", (0, 2, 0)),
        ("0.1.dev7+g1a2b3c4", (0, 1, 0)),
        ("1.2.3.post1", (1, 2, 3)),
    ],
)
def test_a_version_is_compared_on_its_release_numbers(
    version: str, expected: tuple[int, ...]
) -> None:
    """A development build still has to answer "which release is this before"."""
    assert parts(version) == expected


@pytest.mark.parametrize("version", ["", "x.y.z", "v1.2.3", "dev"])
def test_a_version_that_names_no_release_is_refused(version: str) -> None:
    with pytest.raises(ReleaseVersionError):
        parts(version)


def test_the_release_segment_drops_the_pre_release_suffix() -> None:
    assert release_segment("0.2.0rc1") == "0.2.0"


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("v1.2.3", "1.2.3"),
        ("v0.2.0rc1", "0.2.0rc1"),
        ("v0.2.0a1", "0.2.0a1"),
        ("v0.2.0b2", "0.2.0b2"),
    ],
)
def test_the_version_is_the_tag_without_its_prefix(tag: str, version: str) -> None:
    """The tag is the single source of truth, so the artefact cannot disagree with it."""
    assert release_guard.version_of(tag) == version


@pytest.mark.parametrize(
    "tag", ["1.2.3", "v1.2", "v1.2.3.4", "release-1.2.3", "v1.2.3-rc1", "vX.Y.Z"]
)
def test_a_tag_that_is_not_the_documented_format_is_refused(tag: str) -> None:
    problems = release_guard.check(tag=tag, on_main=True, released=RELEASED)
    assert any("format" in problem for problem in problems)


def test_a_tag_on_main_with_an_unreleased_version_is_allowed() -> None:
    assert release_guard.check(tag="v0.3.0", on_main=True, released=RELEASED) == []


def test_a_pre_release_takes_the_same_path_as_a_stable_release() -> None:
    """A separate route for pre-releases is a release path nobody has tested."""
    assert release_guard.check(tag="v0.3.0rc1", on_main=True, released=RELEASED) == []


def test_a_tag_off_main_is_refused() -> None:
    """A release built from an unreviewed commit is a supply-chain problem, not a mistake."""
    problems = release_guard.check(tag="v0.3.0", on_main=False, released=RELEASED)
    assert any("main" in problem for problem in problems)


def test_a_version_already_on_the_index_is_refused() -> None:
    problems = release_guard.check(tag="v0.2.0", on_main=True, released=RELEASED)
    assert any("immutable" in problem for problem in problems)


def test_a_pre_release_of_an_already_released_version_is_refused() -> None:
    """0.2.0rc1 after 0.2.0 shipped describes a release that already happened."""
    problems = release_guard.check(tag="v0.2.0rc1", on_main=True, released=RELEASED)
    assert problems != []


def test_a_first_release_with_nothing_on_the_index_is_allowed() -> None:
    assert release_guard.check(tag="v0.1.0", on_main=True, released=()) == []


def test_every_reason_is_reported_at_once() -> None:
    problems = release_guard.check(tag="v0.2.0", on_main=False, released=RELEASED)
    assert len(problems) == 2


def test_the_guard_passes_a_releasable_tag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("tools.release_guard.on_main", lambda tag: True)  # noqa: ARG005
    monkeypatch.setattr("tools.release_guard.released_versions", lambda: RELEASED)

    assert release_guard.main(["--tag", "v0.3.0"]) == 0
    assert "0.3.0" in capsys.readouterr().out


def test_the_guard_blocks_and_names_the_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("tools.release_guard.on_main", lambda tag: True)  # noqa: ARG005
    monkeypatch.setattr("tools.release_guard.released_versions", lambda: RELEASED)

    assert release_guard.main(["--tag", "v0.2.0"]) == 1
    assert "immutable" in capsys.readouterr().err


def test_an_index_that_does_not_know_the_package_yet_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first release queries an index that has never heard of the project."""

    def not_found(url: str, timeout: float) -> object:  # noqa: ARG001
        raise release_guard.IndexUnavailableError(url)

    monkeypatch.setattr("tools.release_guard._fetch", not_found)
    assert release_guard.released_versions() == ()


def test_the_index_versions_are_read_from_its_json_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.release_guard._fetch",
        lambda url, timeout: {"releases": {"0.1.0": [], "0.2.0": []}},  # noqa: ARG005
    )
    assert release_guard.released_versions() == ("0.1.0", "0.2.0")


def test_the_index_is_read_over_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stubbed at the transport, so the guard's own request shape is still exercised."""

    @contextmanager
    def urlopen(url: str, timeout: float) -> Iterator[io.BytesIO]:  # noqa: ARG001
        yield io.BytesIO(b'{"releases": {"0.1.0": []}}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert release_guard._fetch(release_guard.INDEX_JSON, 1.0) == {"releases": {"0.1.0": []}}


@pytest.mark.parametrize(
    "failure", [urllib.error.URLError("down"), TimeoutError, json.JSONDecodeError("bad", "", 0)]
)
def test_an_unreachable_or_unreadable_index_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """An index that cannot be read must not be mistaken for an empty one."""

    def fail(url: str, timeout: float) -> None:  # noqa: ARG001
        raise failure

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(release_guard.IndexUnavailableError):
        release_guard._fetch(release_guard.INDEX_JSON, 1.0)


def test_whether_a_tag_is_on_main_comes_from_git() -> None:
    """Not asserting a value: this repository may have no tags at all."""
    assert release_guard.on_main("v0.0.0-absent") is False
