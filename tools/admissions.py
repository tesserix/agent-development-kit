"""The admission gate for third-party dependencies.

A package added here is inherited by every product that installs the kit, and none of
those teams reviewed it. So each runtime dependency carries a decision record naming the
alternative that was rejected, and the resolved graph is committed to
`security/inventory.toml` — otherwise the base install grows through a version bump that
reads like a chore. `make admissions` regenerates the inventory; CI runs the check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import Requirement

from tools.dependency_policy import published
from tools.lockfile import Graph, project_graph

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "Admission",
    "AdmissionError",
    "graph",
    "inventory",
    "load_inventory",
    "records",
    "violations",
]

ROOT = Path(__file__).resolve().parents[1]
ADMISSIONS = ROOT / "security" / "admissions"
INVENTORY = ROOT / "security" / "inventory.toml"

REQUIRED = (
    "package",
    "profile",
    "decided",
    "owner",
    "need",
    "alternatives",
    "maintenance",
    "licence",
    "transitive",
    "security_history",
    "review_by",
)
OPTIONAL = ("native_build", "notes")

# Provider and store SDKs. Behind an extra and behind a protocol, never in the base
# install: a consumer who wants none of them should install none of them.
INTEGRATIONS = frozenset({"mcp", "temporalio", "graphiti-core", "redis", "psycopg"})

PREAMBLE = """# The resolved runtime graph, committed so it can be reviewed.
# Regenerate with `make admissions`. Every line a change adds is a package every
# consuming product inherits; the decision behind a direct one is in security/admissions/.

"""


class AdmissionError(Exception):
    """Raised when a decision record cannot be read or does not decide anything."""


@dataclass(frozen=True)
class Admission:
    """One recorded decision to depend on a third-party package."""

    package: str
    profile: str
    decided: dt.date
    owner: str
    need: str
    alternatives: str
    maintenance: str
    licence: str
    transitive: int
    security_history: str
    review_by: dt.date
    native_build: bool = False
    notes: str = ""


def records(directory: Path = ADMISSIONS) -> tuple[Admission, ...]:
    """Every decision record, by package.

    Raises:
        AdmissionError: If a record is not valid TOML, is missing a required field,
            carries a field nothing recognises, or names a profile that does not exist.
    """
    return tuple(_record(path) for path in sorted(directory.glob("*.toml")))


def _record(path: Path) -> Admission:
    try:
        with path.open("rb") as handle:
            entry = tomllib.load(handle)
    except tomllib.TOMLDecodeError as err:
        raise AdmissionError(f"{path.name} is not valid TOML") from err

    missing = [field for field in REQUIRED if entry.get(field) is None]
    if missing:
        raise AdmissionError(f"{path.name} is missing {', '.join(missing)}")
    unknown = set(entry) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise AdmissionError(f"{path.name} has unknown fields: {', '.join(sorted(unknown))}")
    _profile(str(entry["profile"]), path.name)

    return Admission(
        package=str(entry["package"]),
        profile=str(entry["profile"]),
        decided=_date(entry["decided"], path.name, "decided"),
        owner=str(entry["owner"]),
        need=str(entry["need"]),
        alternatives=str(entry["alternatives"]),
        maintenance=str(entry["maintenance"]),
        licence=str(entry["licence"]),
        transitive=int(entry["transitive"]),
        security_history=str(entry["security_history"]),
        review_by=_date(entry["review_by"], path.name, "review_by"),
        native_build=bool(entry.get("native_build", False)),
        notes=str(entry.get("notes", "")),
    )


def _profile(profile: str, where: str) -> str:
    if profile in {"base", "vendored"} or profile.startswith("extra:"):
        return profile
    raise AdmissionError(f"{where} names the profile {profile}, which is not one that exists")


def _date(raw: object, where: str, field: str) -> dt.date:
    if not isinstance(raw, dt.date):
        raise AdmissionError(f"{where}: {field} is not a date")
    return raw


def graph() -> Graph:
    """The graph this repository's lock resolves to."""
    return project_graph()


def inventory(resolved: Graph) -> dict[str, str]:
    """Every package a consumer can end up with, and the profiles that reach it.

    Development-only packages are absent: they are never in a consumer's resolution, so
    they carry a lighter bar and do not belong in the surface being reviewed.
    """
    found = {}
    for name, labels in resolved.reach.items():
        shipped = sorted(label for label in labels if not label.startswith("group:"))
        if shipped:
            found[name] = ", ".join(shipped)
    return found


def load_inventory(path: Path = INVENTORY) -> dict[str, str]:
    """The committed inventory, as it was at the last review."""
    try:
        with path.open("rb") as handle:
            return {name: str(profile) for name, profile in tomllib.load(handle).items()}
    except OSError as err:
        raise AdmissionError(f"{path} cannot be read; run `make admissions`") from err
    except tomllib.TOMLDecodeError as err:
        raise AdmissionError(f"{path} is not valid TOML") from err


def violations(
    resolved: Graph,
    *,
    records: Iterable[Admission],
    recorded: dict[str, str],
    direct: dict[str, list[str]],
    today: dt.date | None = None,
) -> list[str]:
    """Every way the dependency surface disagrees with what was approved."""
    now = today or dt.date.today()
    decided = {record.package: record for record in records}
    current = inventory(resolved)

    required = {Requirement(text).name for texts in direct.values() for text in texts}
    found = _check_direct(direct, resolved, decided)
    found += _check_inventory(current, recorded)
    found += [
        f"{package}: approved, but no longer a requirement of this kit. Delete the "
        f"record, so what is left reads as the surface as it is."
        for package, record in sorted(decided.items())
        if package not in required and record.profile != "vendored"
    ]
    found += [
        f"{record.package}: approved on {record.decided}, due for re-review on "
        f"{record.review_by}. An approval nothing revisits outlives the maintenance "
        f"that justified it."
        for record in decided.values()
        if record.review_by < now
    ]
    return found


def _check_direct(
    direct: dict[str, list[str]], resolved: Graph, decided: dict[str, Admission]
) -> list[str]:
    found = []
    for profile, requirements in sorted(direct.items()):
        for text in requirements:
            name = Requirement(text).name
            if profile == "" and name in INTEGRATIONS:
                found.append(
                    f"{name}: a provider or store SDK in the base install. It belongs "
                    f"behind an extra and behind a protocol."
                )
            if name not in decided:
                found.append(
                    f"{name}: no decision record in {ADMISSIONS.name}/. It reaches "
                    f"{resolved.blast_radius(name)}."
                )
    return found


def _check_inventory(current: dict[str, str], recorded: dict[str, str]) -> list[str]:
    found = []
    for name in sorted(set(current) - set(recorded)):
        found.append(
            f"{name}: in the lock but not in the committed inventory. It reaches "
            f"{current[name]}; run `make admissions` and review what arrived."
        )
    found += [
        f"{name}: in the committed inventory but no longer in the lock; run `make admissions`."
        for name in sorted(set(recorded) - set(current))
    ]
    found += [
        f"{name}: reached {current[name]}, but the inventory recorded {recorded[name]}. "
        f"A package moving into the base install needs a fresh decision."
        for name in sorted(set(current) & set(recorded))
        if current[name] != recorded[name]
    ]
    return found


def render(current: dict[str, str]) -> str:
    """The committed inventory, in a form whose diff reads as a list of arrivals."""
    lines = (f'"{name}" = "{profile}"\n' for name, profile in sorted(current.items()))
    return PREAMBLE + "".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Check the dependency surface, or regenerate the committed inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the inventory")
    args = parser.parse_args(argv)

    resolved = graph()
    if args.write:
        INVENTORY.write_text(render(inventory(resolved)), encoding="utf-8")
        sys.stdout.write(f"wrote {INVENTORY.name}.\n")
        return 0

    found = violations(
        resolved,
        records=records(),
        recorded=load_inventory(),
        direct=_direct(),
    )
    if found:
        sys.stdout.write("the dependency surface was not reviewed:\n\n")
        sys.stdout.write("".join(f"  {violation}\n" for violation in found))
        sys.stdout.write("\nRecord the decision, then run `make admissions`.\n")
        return 1

    sys.stdout.write("every dependency a consumer inherits is recorded.\n")
    return 0


def _direct() -> dict[str, list[str]]:
    """The published requirements, with the union extra dropped: it decides nothing."""
    return {profile: text for profile, text in published().items() if text}


if __name__ == "__main__":
    raise SystemExit(main())
