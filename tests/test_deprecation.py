"""A deprecation is a promise with a date on it, so the mechanism has to keep both.

These tests pin the two halves that make an upgrade plannable: a consumer is told what
to move to and when it disappears, and the kit cannot record a promise it is allowed to
break — a window shorter than the policy fails at import, not at review.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core.deprecation import (
    WARNINGS_AS_ERRORS_ENV,
    AdkDeprecationWarning,
    Deprecation,
    DeprecationPolicyError,
    deprecate,
    deprecations,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets an empty registry: a leaked entry would show up in the docs page."""
    monkeypatch.setattr("tesserix_adk.core.deprecation._REGISTRY", {})
    monkeypatch.setattr("tesserix_adk.core.deprecation._ANNOUNCED", set())


def _deprecated_function() -> Callable[..., int]:
    @deprecate(since="0.1.0", removal="0.3.0", alternative="new_thing")
    def old_thing(value: int = 1) -> int:
        """Do the old thing."""
        return value * 2

    return old_thing


def test_a_deprecated_function_still_works() -> None:
    """A deprecation is a warning, not a removal; breaking now defeats the window."""
    old_thing = _deprecated_function()
    with pytest.warns(AdkDeprecationWarning):
        assert old_thing(21) == 42


def test_the_warning_names_the_alternative_and_the_removal_version() -> None:
    old_thing = _deprecated_function()
    with pytest.warns(AdkDeprecationWarning) as caught:
        old_thing()
    message = str(caught[0].message)
    assert "new_thing" in message
    assert "0.3.0" in message
    assert "old_thing" in message


def test_the_reason_is_included_when_given() -> None:
    @deprecate(
        since="0.1.0",
        removal="0.3.0",
        alternative="Runner.run",
        reason="the sync path cannot cancel a tool call",
    )
    def run_blocking() -> None:
        """Run without cancellation."""

    with pytest.warns(AdkDeprecationWarning, match="cannot cancel"):
        run_blocking()


def test_the_warning_is_attributed_to_the_caller_not_the_kit() -> None:
    """A warning pointing at the kit's own frame tells the consumer nothing to change."""
    old_thing = _deprecated_function()
    with pytest.warns(AdkDeprecationWarning) as caught:
        old_thing()

    assert caught[0].filename == __file__


def test_a_call_site_in_a_loop_warns_once() -> None:
    """Once per site, not once per call: a hot path would otherwise drown its own logs."""
    old_thing = _deprecated_function()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            old_thing()

    assert len(caught) == 1


def test_each_distinct_call_site_warns() -> None:
    """Deduplication is per site, so a second unmigrated caller is still told."""
    old_thing = _deprecated_function()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_thing()
        old_thing()

    assert len(caught) == 2


def test_a_deprecated_class_warns_on_construction_and_still_works() -> None:
    @deprecate(since="0.1.0", removal="0.3.0", alternative="MemoryStore")
    class OldStore:
        """Store things the old way."""

        def __init__(self, name: str) -> None:
            self.name = name

    with pytest.warns(AdkDeprecationWarning, match="MemoryStore"):
        store = OldStore("s")

    assert store.name == "s"


def test_a_deprecated_class_keeps_its_identity() -> None:
    """Decorating must not swap the class for a function, or isinstance checks break."""

    @deprecate(since="0.1.0", removal="0.3.0", alternative="MemoryStore")
    class OldStore:
        """Store things the old way."""

    with pytest.warns(AdkDeprecationWarning):
        assert isinstance(OldStore(), OldStore)


def test_the_wrapper_keeps_the_name_and_signature() -> None:
    old_thing = _deprecated_function()
    assert old_thing.__name__ == "old_thing"
    assert "value" in str(__import__("inspect").signature(old_thing))


def test_the_docstring_records_the_deprecation() -> None:
    """`help()` is where a consumer looks first; the notice has to survive to there."""
    old_thing = _deprecated_function()
    doc = old_thing.__doc__ or ""
    assert "Deprecated since 0.1.0" in doc
    assert "removed in 0.3.0" in doc
    assert "new_thing" in doc


def test_the_registry_records_the_deprecation() -> None:
    _deprecated_function()
    (record,) = deprecations()
    assert record.since == "0.1.0"
    assert record.removal == "0.3.0"
    assert record.alternative == "new_thing"
    assert record.name.endswith("old_thing")


def test_the_registry_is_sorted_and_immutable() -> None:
    """The docs page is generated from this, so ordering cannot depend on import order."""

    @deprecate(since="0.1.0", removal="0.3.0", alternative="b")
    def zebra() -> None:
        """Z."""

    @deprecate(since="0.1.0", removal="0.3.0", alternative="a")
    def antelope() -> None:
        """A."""

    names = [record.name.rsplit(".", 1)[-1] for record in deprecations()]
    assert names == ["antelope", "zebra"]
    assert isinstance(deprecations(), tuple)


def test_registering_the_same_name_twice_with_different_terms_is_refused() -> None:
    """Two live promises about one symbol means one of them is a lie."""

    def make(removal: str) -> None:
        @deprecate(since="0.1.0", removal=removal, alternative="new_thing")
        def duplicated() -> None:
            """D."""

    make("0.3.0")
    with pytest.raises(DeprecationPolicyError, match="already deprecated"):
        make("0.4.0")


def test_a_module_reimported_does_not_trip_the_duplicate_check() -> None:
    """Identical re-registration is a reload, not a conflict."""

    def make() -> None:
        @deprecate(since="0.1.0", removal="0.3.0", alternative="new_thing")
        def repeated() -> None:
            """R."""

    make()
    make()
    assert len(deprecations()) == 1


@pytest.mark.parametrize(
    ("since", "removal", "complaint"),
    [
        ("0.1.0", "0.2.0", "minor releases of notice"),
        ("0.1.0", "0.1.0", "after"),
        ("0.3.0", "0.1.0", "after"),
        ("1.2.0", "1.4.0", "major release"),
        ("1.2.0", "2.1.0", "major release"),
        ("1.2.0", "1.2.1", "major release"),
    ],
)
def test_a_window_shorter_than_the_policy_is_refused(
    since: str, removal: str, complaint: str
) -> None:
    """The policy is enforced at import so it cannot be argued about per pull request."""
    with pytest.raises(DeprecationPolicyError, match=complaint):

        @deprecate(since=since, removal=removal, alternative="new_thing")
        def too_soon() -> None:
            """T."""


@pytest.mark.parametrize(("since", "removal"), [("0.1.0", "0.3.0"), ("1.2.0", "2.0.0")])
def test_a_window_meeting_the_policy_is_accepted(since: str, removal: str) -> None:
    """Pre-1.0 the minor is the breaking channel; from 1.0 removals wait for a major."""

    @deprecate(since=since, removal=removal, alternative="new_thing")
    def scheduled() -> None:
        """S."""

    assert deprecations()[0].removal == removal


@pytest.mark.parametrize("version", ["1", "1.2.3.4", "1.x.0", "", "v1.2.0"])
def test_a_version_that_is_not_three_numbers_is_refused(version: str) -> None:
    with pytest.raises(DeprecationPolicyError, match=r"major\.minor\.patch"):

        @deprecate(since=version, removal="9.0.0", alternative="new_thing")
        def unparseable() -> None:
            """U."""


def test_an_empty_alternative_is_refused() -> None:
    """ "Use something else" is not a migration path."""
    with pytest.raises(DeprecationPolicyError, match="alternative"):

        @deprecate(since="0.1.0", removal="0.3.0", alternative="  ")
        def unhelpful() -> None:
            """U."""


def test_the_environment_switch_turns_the_warning_into_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer's CI can fail on the deprecation months before the removal lands."""
    monkeypatch.setenv(WARNINGS_AS_ERRORS_ENV, "1")
    old_thing = _deprecated_function()

    with pytest.raises(AdkDeprecationWarning, match="new_thing"):
        old_thing()


def test_the_switch_off_leaves_the_call_working(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WARNINGS_AS_ERRORS_ENV, "0")
    old_thing = _deprecated_function()

    with pytest.warns(AdkDeprecationWarning):
        assert old_thing(2) == 4


def test_errors_mode_reports_every_call_site_not_just_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deduplication must not hide the second site from a consumer preparing an upgrade."""
    monkeypatch.setenv(WARNINGS_AS_ERRORS_ENV, "1")
    old_thing = _deprecated_function()

    for _ in range(2):
        with pytest.raises(AdkDeprecationWarning):
            old_thing()


def test_the_record_renders_the_promise_as_one_line() -> None:
    record = Deprecation(
        name="tesserix_adk.runtime.old_thing",
        since="0.1.0",
        removal="0.3.0",
        alternative="new_thing",
    )
    assert "tesserix_adk.runtime.old_thing" in record.message
    assert "0.3.0" in record.message


def test_a_deprecation_warning_is_a_deprecation_warning() -> None:
    """Consumers filter on the stdlib category; ours must be caught by that filter."""
    assert issubclass(AdkDeprecationWarning, DeprecationWarning)
