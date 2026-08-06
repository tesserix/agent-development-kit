"""What the kit's published requirements may say to a consumer.

Reproducibility is the lockfile's job. These are the requirements that travel with the
distribution, and the constraint that does the damage there is the speculative upper
bound: it turns an upgrade the consuming product chose into a resolution error they
cannot fix without forking the kit. So a cap is an exception with an owner and a trigger
for removing it, and a floor is a claim the `lowest-direct` CI leg has to keep true.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.requirements import InvalidRequirement, Requirement

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["Cap", "Floor", "Policy", "PolicyError", "load", "published", "violations"]

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "security" / "dependencies.toml"
PYPROJECT = ROOT / "pyproject.toml"

PROJECT = "tesserix-adk"
FLOOR_FIELDS = ("package", "floor", "reason")
CAP_FIELDS = ("package", "cap", "incompatibility", "trigger", "owner")


class PolicyError(Exception):
    """Raised when the dependency policy itself cannot be read or is incomplete."""


@dataclass(frozen=True)
class Floor:
    """The oldest version the kit claims to work with, and why that one."""

    package: str
    floor: str
    reason: str


@dataclass(frozen=True)
class Cap:
    """An upper bound, the incompatibility that earned it, and what removes it."""

    package: str
    cap: str
    incompatibility: str
    trigger: str
    owner: str


@dataclass(frozen=True)
class Policy:
    """The justified floors and the recorded upper-bound exceptions."""

    floors: dict[str, Floor]
    caps: dict[str, Cap]


def load(path: Path = POLICY) -> Policy:
    """Read the dependency policy.

    Raises:
        PolicyError: If the file is absent, is not valid TOML, or holds a record
            missing any of its required fields.
    """
    try:
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except OSError as err:
        raise PolicyError(f"{path} cannot be read") from err
    except tomllib.TOMLDecodeError as err:
        raise PolicyError(f"{path} is not valid TOML") from err

    floors = [Floor(**_record(entry, FLOOR_FIELDS)) for entry in parsed.get("floor", [])]
    caps = [Cap(**_record(entry, CAP_FIELDS)) for entry in parsed.get("cap", [])]
    return Policy(
        floors={floor.package: floor for floor in floors},
        caps={cap.package: cap for cap in caps},
    )


def _record(entry: dict[str, Any], fields: tuple[str, ...]) -> dict[str, str]:
    missing = [field for field in fields if not entry.get(field)]
    if missing:
        named = entry.get("package", "a record")
        raise PolicyError(f"{named} is missing {', '.join(missing)}")
    unknown = set(entry) - set(fields)
    if unknown:
        raise PolicyError(f"{entry['package']} has unknown fields: {', '.join(sorted(unknown))}")
    return {field: str(entry[field]) for field in fields}


def published(path: Path = PYPROJECT) -> dict[str, list[str]]:
    """Every requirement a consumer inherits, by the profile that carries it.

    The base install is keyed `""`; each extra is keyed by its own name. Development
    groups are absent: a cap there is never in a consumer's resolution.
    """
    with path.open("rb") as handle:
        project = tomllib.load(handle)["project"]

    profiles = {"": list(project.get("dependencies", []))}
    for extra, requirements in project.get("optional-dependencies", {}).items():
        # A union extra names the project itself, which constrains nothing on its own.
        profiles[extra] = [
            requirement
            for requirement in requirements
            if Requirement(requirement).name.replace("_", "-") != PROJECT
        ]
    return profiles


def violations(profiles: dict[str, list[str]], *, policy: Policy) -> list[str]:
    """Every way the published requirements disagree with the policy."""
    required: set[str] = set()
    found: list[str] = []

    for profile, requirements in sorted(profiles.items()):
        where = f" (extra: {profile})" if profile else ""
        for text in requirements:
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                found.append(f"{text}{where}: is not a requirement this can read")
                continue
            required.add(requirement.name)
            found += _check(requirement, where=where, policy=policy)

    found += [
        f"{package}: the policy records a cap for a package nothing depends on any more"
        for package in sorted(policy.caps)
        if package not in required
    ]
    return found


def _check(requirement: Requirement, *, where: str, policy: Policy) -> list[str]:
    name = requirement.name
    lower = [spec for spec in requirement.specifier if spec.operator in {">=", "==", "~="}]
    upper = [spec for spec in requirement.specifier if spec.operator in {"<", "<="}]

    found = []
    if not lower:
        found.append(f"{name}{where}: declares no floor, so no resolution proves its oldest")
    else:
        found += _check_floor(name, lower[0].version, where=where, policy=policy)
    if upper:
        found += _check_cap(name, upper[0].version, where=where, policy=policy)
    return found


def _check_floor(name: str, declared: str, *, where: str, policy: Policy) -> list[str]:
    recorded = policy.floors.get(name)
    if recorded is None:
        return [f"{name}{where}: the floor {declared} is not justified in {POLICY.name}"]
    if recorded.floor != declared:
        return [
            f"{name}{where}: declares floor {declared}, but the policy records {recorded.floor}"
        ]
    return []


def _check_cap(name: str, declared: str, *, where: str, policy: Policy) -> list[str]:
    recorded = policy.caps.get(name)
    if recorded is None:
        return [
            f"{name}{where}: caps at <{declared} with no recorded incompatibility. A cap "
            f"nobody proved breaks a consumer's upgrade for a break nobody has seen."
        ]
    if recorded.cap != declared:
        return [f"{name}{where}: caps at <{declared}, but the policy records <{recorded.cap}"]
    return []


def render(found: Iterable[str]) -> str:
    return "".join(f"  {violation}\n" for violation in found)


def main(argv: list[str] | None = None) -> int:
    """Check the published requirements against the recorded policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY, help="path to dependencies.toml")
    args = parser.parse_args(argv)

    found = violations(published(), policy=load(args.policy))
    if not found:
        sys.stdout.write("no dependency policy violations.\n")
        return 0

    sys.stdout.write(
        f"dependency policy violations. Record or remove them in {args.policy.name}:\n\n"
    )
    sys.stdout.write(render(found))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
