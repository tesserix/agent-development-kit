"""Check the coordinated disclosure process, and regenerate the tables in `SECURITY.md`.

The published policy states a private channel and a response target per severity. Those
targets live in `security/disclosure.toml`, the page is generated from them by
`make disclosure`, and each report is a record under `security/advisories/`. CI runs the
same check, so a target the process no longer meets fails the build rather than sitting
in prose a reporter is relying on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "Advisory",
    "Channel",
    "DisclosureError",
    "Target",
    "advisories",
    "channel",
    "render",
    "target",
    "targets",
    "violations",
]

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "security" / "disclosure.toml"
ADVISORIES = ROOT / "security" / "advisories"
PAGE = ROOT / "SECURITY.md"

REQUIRED = ("id", "title", "severity", "reported", "acknowledged", "profiles", "affected", "fixed")
OPTIONAL = ("published", "notified", "mitigation", "credit", "disclosed_publicly")

TARGETS_MARKER = "response-targets"
ADVISORIES_MARKER = "advisories"
_BEGIN = "<!-- generated: {} -->"
_END = "<!-- end generated: {} -->"


class DisclosureError(Exception):
    """Raised when the policy or an advisory record cannot be read or is incomplete."""


@dataclass(frozen=True)
class Channel:
    """Where a report goes, and who is on the rota to pick it up."""

    private: str
    rota: tuple[str, ...]


@dataclass(frozen=True)
class Target:
    """The response commitment for one severity, in days."""

    severity: str
    acknowledge_days: int
    fix_days: int


@dataclass(frozen=True)
class Advisory:
    """One report and everything the process promised about it."""

    id: str
    title: str
    severity: str
    reported: dt.date
    acknowledged: dt.date
    profiles: tuple[str, ...]
    affected: tuple[str, ...]
    fixed: tuple[str, ...]
    published: dt.date | None = None
    notified: dt.date | None = None
    mitigation: str = ""
    credit: str = ""
    disclosed_publicly: bool = False


def _policy(path: Path = POLICY) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as err:
        raise DisclosureError(f"{path} cannot be read") from err
    except tomllib.TOMLDecodeError as err:
        raise DisclosureError(f"{path} is not valid TOML") from err


def channel(path: Path = POLICY) -> Channel:
    """The private reporting channel and the maintainer rota behind it."""
    entry = _policy(path).get("channel", {})
    missing = [field for field in ("private", "rota") if not entry.get(field)]
    if missing:
        raise DisclosureError(f"the channel is missing {', '.join(missing)}")
    return Channel(private=str(entry["private"]), rota=tuple(entry["rota"]))


def targets(path: Path = POLICY) -> dict[str, Target]:
    """The response commitment per severity, keyed by severity and nothing else."""
    found: dict[str, Target] = {}
    for entry in _policy(path).get("target", []):
        missing = [f for f in ("severity", "acknowledge_days", "fix_days") if not entry.get(f)]
        if missing:
            raise DisclosureError(f"a response target is missing {', '.join(missing)}")
        found[str(entry["severity"])] = Target(
            severity=str(entry["severity"]),
            acknowledge_days=int(entry["acknowledge_days"]),
            fix_days=int(entry["fix_days"]),
        )
    if not found:
        raise DisclosureError(f"{path} declares no response targets")
    return found


def target(severity: str, *, profiles: Sequence[str] = (), path: Path = POLICY) -> Target:
    """The commitment for a severity.

    `profiles` is accepted and deliberately ignored: a flaw reached only through an
    optional extra exposes the products that install it exactly as much as any other, so
    it cannot be deprioritised for living outside the base install.
    """
    del profiles
    known = targets(path)
    if severity not in known:
        raise DisclosureError(f"{severity} is not a severity the policy declares")
    return known[severity]


def advisories(directory: Path = ADVISORIES) -> tuple[Advisory, ...]:
    """Every advisory record, oldest report first.

    Raises:
        DisclosureError: If a record is unreadable, is not valid TOML, is missing a
            required field, carries a field nothing recognises, or names a severity the
            policy does not declare.
    """
    records = [_advisory(path) for path in sorted(directory.glob("*.toml"))]
    return tuple(sorted(records, key=lambda record: (record.reported, record.id)))


def _advisory(path: Path) -> Advisory:
    try:
        with path.open("rb") as handle:
            entry = tomllib.load(handle)
    except OSError as err:  # pragma: no cover — glob only yields readable paths
        raise DisclosureError(f"{path} cannot be read") from err
    except tomllib.TOMLDecodeError as err:
        raise DisclosureError(f"{path.name} is not valid TOML") from err

    missing = [field for field in REQUIRED if entry.get(field) is None]
    if missing:
        raise DisclosureError(f"{path.name} is missing {', '.join(missing)}")
    unknown = set(entry) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise DisclosureError(f"{path.name} has unknown fields: {', '.join(sorted(unknown))}")
    if entry["severity"] not in targets():
        raise DisclosureError(f"{path.name} names a severity the policy does not declare")

    return Advisory(
        id=str(entry["id"]),
        title=str(entry["title"]),
        severity=str(entry["severity"]),
        reported=_date(entry["reported"], path.name, "reported"),
        acknowledged=_date(entry["acknowledged"], path.name, "acknowledged"),
        profiles=tuple(entry["profiles"]),
        affected=tuple(entry["affected"]),
        fixed=tuple(entry["fixed"]),
        published=_optional_date(entry.get("published"), path.name, "published"),
        notified=_optional_date(entry.get("notified"), path.name, "notified"),
        mitigation=str(entry.get("mitigation", "")),
        credit=str(entry.get("credit", "")),
        disclosed_publicly=bool(entry.get("disclosed_publicly", False)),
    )


def _date(raw: object, where: str, field: str) -> dt.date:
    if not isinstance(raw, dt.date):
        raise DisclosureError(f"{where}: {field} is not a date")
    return raw


def _optional_date(raw: object, where: str, field: str) -> dt.date | None:
    return None if raw is None else _date(raw, where, field)


def violations(
    records: Iterable[Advisory],
    *,
    targets: dict[str, Target],
    today: dt.date | None = None,
) -> list[str]:
    """Every way the recorded advisories disagree with the published process."""
    now = today or dt.date.today()
    found: list[str] = []
    for record in records:
        commitment = targets[record.severity]
        found += _check_acknowledgement(record, commitment)
        found += _check_coverage(record)
        found += _check_publication(record, commitment, now)
    return found


def _check_acknowledgement(record: Advisory, commitment: Target) -> list[str]:
    if record.acknowledged < record.reported:
        return [f"{record.id}: acknowledged before it was reported"]
    late = (record.acknowledged - record.reported).days - commitment.acknowledge_days
    if late > 0:
        return [
            f"{record.id}: acknowledged {late} day(s) after the "
            f"{commitment.acknowledge_days}-day target for {record.severity}"
        ]
    return []


def _check_coverage(record: Advisory) -> list[str]:
    if not record.fixed:
        return []
    fixed_minors = {_minor(version) for version in record.fixed}
    affected = set(record.affected)
    missing = sorted(affected - fixed_minors)
    if missing:
        return [
            f"{record.id}: no patched release for the supported minor(s) "
            f"{', '.join(missing)}, which the advisory says are affected"
        ]
    stray = sorted(fixed_minors - affected)
    if stray:
        return [f"{record.id}: fixes {', '.join(stray)}, which is not in the affected list"]
    return []


def _check_publication(record: Advisory, commitment: Target, now: dt.date) -> list[str]:
    found: list[str] = []
    if not record.fixed:
        overdue = (now - record.reported).days - commitment.fix_days
        if overdue > 0:
            found.append(
                f"{record.id}: unfixed {overdue} day(s) past the "
                f"{commitment.fix_days}-day target for {record.severity}"
            )
        if record.disclosed_publicly and not record.mitigation:
            found.append(f"{record.id}: disclosed with no fix and no interim mitigation")
        return found

    if record.published is None:
        found.append(f"{record.id}: has a fix but no published advisory")
        return found
    if record.published < record.acknowledged:
        found.append(f"{record.id}: published before it was acknowledged")
    if record.notified is None:
        found.append(f"{record.id}: no record of notifying the embedding products")
    elif record.notified > record.published:
        found.append(f"{record.id}: consumers notified after publication, not with it")
    return found


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def render(page: str, *, path: Path = POLICY, directory: Path = ADVISORIES) -> str:
    """Replace the generated blocks in the page, leaving the prose around them alone."""
    page = _replace(page, TARGETS_MARKER, _targets_table(targets(path)))
    return _replace(page, ADVISORIES_MARKER, _advisories_table(advisories(directory)))


def _replace(page: str, marker: str, body: str) -> str:
    begin, end = _BEGIN.format(marker), _END.format(marker)
    if begin not in page or end not in page:
        raise DisclosureError(f"{PAGE.name} has no generated block for {marker}")
    head, rest = page.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n\n{body}\n{end}{tail}"


def _targets_table(known: dict[str, Target]) -> str:
    rows = "".join(
        f"| {t.severity} | {t.acknowledge_days} day(s) | {t.fix_days} day(s) |\n"
        for t in known.values()
    )
    return (
        "| Severity | Acknowledged within | Fix released within |\n| --- | --- | --- |\n" + rows
    ).rstrip("\n")


def _advisories_table(records: Sequence[Advisory]) -> str:
    if not records:
        return "No advisories have been published."
    rows = "".join(
        f"| {r.id} | {r.title} | {r.severity} | {', '.join(r.affected)} | "
        f"{', '.join(r.fixed) or '—'} | {r.published or 'embargoed'} |\n"
        for r in records
    )
    return (
        "| ID | Summary | Severity | Affected | Fixed in | Published |\n"
        "| --- | --- | --- | --- | --- | --- |\n" + rows
    ).rstrip("\n")


def main(argv: list[str] | None = None) -> int:
    """Check the process, or regenerate the tables in the published policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate SECURITY.md in place")
    args = parser.parse_args(argv)

    current = PAGE.read_text(encoding="utf-8")
    regenerated = render(current)
    if args.write:
        PAGE.write_text(regenerated, encoding="utf-8")

    found = violations(advisories(), targets=targets())
    if found:
        sys.stdout.write("the disclosure process has been missed:\n\n")
        sys.stdout.write("".join(f"  {violation}\n" for violation in found))
        return 1
    if not args.write and regenerated != current:
        sys.stdout.write(f"{PAGE.name} is out of date. Run `make disclosure`.\n")
        return 1

    sys.stdout.write("disclosure process on target.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
