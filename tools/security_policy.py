"""What blocks a build, and what may be suppressed, by whom, until when.

Every scanner's failure mode is the same: an inconvenient finding gets silenced and the
silence outlives everyone who understood it. So a suppression carries an owner, a reason
and an end date, and an expired one fails the build exactly as the finding would.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["POLICY", "Policy", "PolicyError", "Suppression", "blocks", "load"]

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "security" / "policy.toml"

KINDS = frozenset({"advisory", "secret"})
BLOCKING = frozenset({"critical", "high", "unknown"})

REQUIRED = ("id", "kind", "owner", "reason", "expires")
OPTIONAL = ("mitigation",)

# A quarter is long enough to schedule an upgrade and short enough that the person who
# accepted the risk is still the person who reviews it.
MAX_DAYS = 90
MIN_REASON = 30


class PolicyError(Exception):
    """Raised when the policy file cannot be read as a set of reviewable decisions."""


@dataclass(frozen=True)
class Suppression:
    """One finding, accepted by one person, until one date."""

    id: str
    kind: str
    owner: str
    reason: str
    expires: dt.date
    mitigation: str | None = None

    def live(self, today: dt.date) -> bool:
        """Suppressions run to the end of their last day."""
        return today <= self.expires


@dataclass(frozen=True)
class Policy:
    """The suppressions in force, and the questions the scanners ask of them."""

    suppressions: tuple[Suppression, ...]

    def suppresses(self, finding: str, *, kind: str, today: dt.date) -> bool:
        """Is this finding accepted today? A suppression covers one kind only."""
        return any(
            entry.id == finding and entry.kind == kind and entry.live(today)
            for entry in self.suppressions
        )

    def expired(self, today: dt.date) -> list[Suppression]:
        """Suppressions whose date has passed, which the build fails on."""
        return [entry for entry in self.suppressions if not entry.live(today)]


def blocks(severity: str) -> bool:
    """Does a finding at this severity block a merge?

    An unrated advisory blocks: nobody having scored it yet is not evidence of safety.
    """
    return severity.lower() in BLOCKING


def _date(raw: object, field: str) -> dt.date:
    if isinstance(raw, dt.date):
        return raw
    try:
        return dt.date.fromisoformat(str(raw))
    except ValueError as err:
        raise PolicyError(f"{field} is not a date: {raw!r}") from err


def _suppression(entry: dict[str, Any], today: dt.date) -> Suppression:
    unknown = set(entry) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise PolicyError(f"unknown field in suppression: {', '.join(sorted(unknown))}")

    missing = [field for field in REQUIRED if field not in entry]
    if missing:
        raise PolicyError(f"suppression is missing {', '.join(missing)}")

    if entry["kind"] not in KINDS:
        raise PolicyError(f"unknown suppression kind {entry['kind']!r}")

    if len(str(entry["reason"]).strip()) < MIN_REASON:
        raise PolicyError(
            f"the reason for {entry['id']} is too short to review later: {entry['reason']!r}"
        )

    expires = _date(entry["expires"], "expires")
    if expires > today + dt.timedelta(days=MAX_DAYS):
        raise PolicyError(
            f"{entry['id']} is suppressed past the {MAX_DAYS} days a suppression may run; "
            f"an open-ended suppression with a date on it is still open-ended"
        )

    return Suppression(
        id=str(entry["id"]),
        kind=str(entry["kind"]),
        owner=str(entry["owner"]),
        reason=str(entry["reason"]),
        expires=expires,
        mitigation=entry.get("mitigation"),
    )


def load(path: Path = POLICY, *, today: dt.date | None = None) -> Policy:
    """Read and validate the policy.

    Raises:
        PolicyError: If the file is absent, or any suppression is incomplete, unreviewable
            or open-ended.
    """
    today = today or dt.date.today()
    try:
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except OSError as err:
        raise PolicyError(f"{path} cannot be read") from err
    except tomllib.TOMLDecodeError as err:
        raise PolicyError(f"{path} is not valid TOML") from err

    suppressions = tuple(_suppression(entry, today) for entry in parsed.get("suppression", []))

    seen = [(entry.id, entry.kind) for entry in suppressions]
    duplicated = {pair for pair in seen if seen.count(pair) > 1}
    if duplicated:
        listed = ", ".join(sorted(finding for finding, _ in duplicated))
        raise PolicyError(f"suppressed twice: {listed}")

    return Policy(suppressions=suppressions)
