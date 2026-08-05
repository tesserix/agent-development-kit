"""Deprecate a public name with a named alternative and a removal version.

A deprecation is a promise to consumers, so the window is enforced here rather than
argued about per pull request: a removal is scheduled at least two minor releases out,
and only in a release allowed to break (a major from 1.0 onward, a minor before it).
An unmeetable window fails at import, which is the last moment it costs nothing.

Consumers set `TESSERIX_ADK_DEPRECATIONS_AS_ERRORS=1` in their own CI to fail on a
deprecation months before the removal ships. See docs/versioning.md.
"""

from __future__ import annotations

import functools
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from tesserix_adk.core.errors import AdkError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "WARNINGS_AS_ERRORS_ENV",
    "AdkDeprecationWarning",
    "Deprecation",
    "DeprecationPolicyError",
    "deprecate",
    "deprecations",
]

WARNINGS_AS_ERRORS_ENV = "TESSERIX_ADK_DEPRECATIONS_AS_ERRORS"

# The minimum notice, in minor releases, between announcing a removal and making it.
MINIMUM_WINDOW_MINORS = 2

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_VERSION = re.compile(r"\d+\.\d+\.\d+")

_T = TypeVar("_T", bound="Callable[..., Any] | type")

_REGISTRY: dict[str, Deprecation] = {}
_ANNOUNCED: set[tuple[str, int]] = set()


class AdkDeprecationWarning(DeprecationWarning):
    """Warns that a public name of the kit is scheduled for removal.

    A `DeprecationWarning` subclass, so a consumer's existing filters catch it, while
    the distinct category lets them fail on this kit's deprecations alone.
    """


class DeprecationPolicyError(AdkError):
    """Raised when a deprecation would break the published versioning policy."""


@dataclass(frozen=True)
class Deprecation:
    """A scheduled removal, as promised to consumers.

    Args:
        name: Dotted name of the deprecated symbol.
        since: Version in which the deprecation was announced.
        removal: Version in which the symbol disappears.
        alternative: What to use instead.
        reason: Why it is going, when the alternative does not make that obvious.
    """

    name: str
    since: str
    removal: str
    alternative: str
    reason: str | None = None

    @property
    def message(self) -> str:
        """The one line a consumer sees, naming the alternative and the removal."""
        why = f" ({self.reason})" if self.reason else ""
        return (
            f"{self.name} is deprecated since {self.since} and will be removed in "
            f"{self.removal}; use {self.alternative} instead.{why}"
        )


def deprecations() -> tuple[Deprecation, ...]:
    """Every live deprecation, sorted by name.

    Generated documentation reads this, so the order cannot depend on import order.
    """
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def _parse(label: str, version: str) -> tuple[int, int, int]:
    if not _VERSION.fullmatch(version):
        raise DeprecationPolicyError(f"{label} version {version!r} must be major.minor.patch")
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def _check_window(since: str, removal: str) -> None:
    """Enforce the published window, so a promise cannot be made that policy forbids."""
    announced = _parse("since", since)
    scheduled = _parse("removal", removal)
    if scheduled <= announced:
        raise DeprecationPolicyError(
            f"removal {removal} must come after the deprecation in {since}"
        )

    breaking = scheduled[1:] == (0, 0) if announced[0] else scheduled[2] == 0
    if not breaking:
        channel = "a minor release" if announced[0] == 0 else "a major release"
        raise DeprecationPolicyError(
            f"removal {removal} is not {channel}: before 1.0 removals happen in a minor "
            f"release and from 1.0 onward only in a major release"
        )

    # From 1.0 the breaking check already forces the next major or later, which is more
    # notice than the window asks for. Before it, the minor carries the same meaning.
    if announced[0] == 0 and scheduled[1] - announced[1] < MINIMUM_WINDOW_MINORS:
        raise DeprecationPolicyError(
            f"removing in {removal} gives less than {MINIMUM_WINDOW_MINORS} minor releases "
            f"of notice from {since}"
        )


def _register(record: Deprecation) -> None:
    existing = _REGISTRY.get(record.name)
    if existing is not None and existing != record:
        raise DeprecationPolicyError(
            f"{record.name} is already deprecated on different terms ({existing.message}); "
            f"two live promises about one symbol means one of them is a lie"
        )
    _REGISTRY[record.name] = record


def _announce(record: Deprecation) -> None:
    """Warn the caller once per call site, or raise when a consumer asked for errors."""
    if os.environ.get(WARNINGS_AS_ERRORS_ENV, "").strip().lower() in _TRUTHY:
        raise AdkDeprecationWarning(record.message)

    caller = sys._getframe(2)
    site = (caller.f_code.co_filename, caller.f_lineno)
    if site in _ANNOUNCED:
        return
    _ANNOUNCED.add(site)
    warnings.warn(record.message, AdkDeprecationWarning, stacklevel=3)


def _documented(doc: str | None, record: Deprecation) -> str:
    notice = (
        f"Deprecated since {record.since}, removed in {record.removal}; "
        f"use {record.alternative} instead."
    )
    return f"{doc.rstrip()}\n\n{notice}" if doc else notice


def deprecate(
    *, since: str, removal: str, alternative: str, reason: str | None = None
) -> Callable[[_T], _T]:
    """Mark a public function or class as deprecated on a fixed schedule.

    The decorated object keeps working: calling it warns once per call site with
    `AdkDeprecationWarning`, attributed to the caller's frame rather than the kit's.
    Decorating a class warns on construction and leaves the class itself intact.

    Args:
        since: Version announcing the deprecation, as `major.minor.patch`.
        removal: Version in which it disappears. Must satisfy the published window.
        alternative: What the consumer should use instead. Required — "use something
            else" is not a migration path.
        reason: Why it is going, when the alternative does not make that obvious.

    Raises:
        DeprecationPolicyError: If either version is malformed, the alternative is
            empty, the window is shorter than policy, or the same name is already
            deprecated on different terms.
    """
    if not alternative.strip():
        raise DeprecationPolicyError("a deprecation must name the alternative to migrate to")
    _check_window(since, removal)

    def decorate(target: _T) -> _T:
        record = Deprecation(
            name=f"{target.__module__}.{target.__qualname__}",
            since=since,
            removal=removal,
            alternative=alternative,
            reason=reason,
        )
        _register(record)

        if isinstance(target, type):
            original = target.__init__  # type: ignore[misc]

            @functools.wraps(original)
            def __init__(self: object, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, N807
                _announce(record)
                original(self, *args, **kwargs)

            target.__init__ = __init__  # type: ignore[misc]
            target.__doc__ = _documented(target.__doc__, record)
            return target

        function = cast("Callable[..., Any]", target)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            _announce(record)
            return function(*args, **kwargs)

        wrapper.__doc__ = _documented(function.__doc__, record)
        return cast("_T", wrapper)

    return decorate
