"""The licence obligations a consuming product inherits by importing the kit.

Read from installed distribution metadata, checked against an allow list, and blocking on
anything unknown. A dual licence is not resolved by taking the convenient half: it is a
recorded decision with an owner, because that is what a legal review asks for two years
later.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distributions, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.lockfile import RUNTIME, project_graph

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "Decision",
    "LicenceError",
    "Policy",
    "check",
    "declared",
    "installed",
    "load",
    "spdx",
]

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "security" / "licences.toml"

REQUIRED = ("package", "licence", "owner", "reason")

# SPDX expressions this reads: a disjunction offers a choice, a conjunction imposes both.
OR = re.compile(r"\s+OR\s+", re.IGNORECASE)
AND = re.compile(r"\s+AND\s+", re.IGNORECASE)

# What a pre-SPDX distribution writes in its licence field. Unmatched means unknown, which
# blocks, so the failure mode of both tables is a question rather than a silent pass.
ALIASES = {
    "3-clause bsd license": "BSD-3-Clause",
    "bsd-3-clause license": "BSD-3-Clause",
    "2-clause bsd license": "BSD-2-Clause",
    "mit license": "MIT",
    "apache license 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "the unlicense (unlicense)": "Unlicense",
}

CLASSIFIERS = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
}


class LicenceError(Exception):
    """Raised when the licence policy itself cannot be read or is self-contradictory."""


@dataclass(frozen=True)
class Decision:
    """Which licence was taken where a package offered a choice, and who took it."""

    package: str
    licence: str
    owner: str
    reason: str


@dataclass(frozen=True)
class Policy:
    """The allowed licences, the decisions on ambiguous ones, and the one-off acceptances."""

    allowed: frozenset[str]
    decisions: dict[str, Decision]
    acceptances: dict[str, Decision]

    def permits(self, licence: str, *, package: str = "") -> bool:
        """Whether this licence is allowed, blanket or accepted for this package alone."""
        wanted = spdx(licence).casefold()
        if wanted in {entry.casefold() for entry in self.allowed}:
            return True
        accepted = self.acceptances.get(package)
        return accepted is not None and accepted.licence.casefold() == wanted


def spdx(licence: str) -> str:
    """A licence identifier, normalising the prose older distributions declare."""
    return ALIASES.get(licence.strip().casefold(), licence.strip())


def load(path: Path = POLICY) -> Policy:
    """Read the licence policy.

    Raises:
        LicenceError: If the file is absent, holds no allow list, or records a decision
            that is incomplete or that chooses a licence the allow list forbids.
    """
    try:
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except OSError as err:
        raise LicenceError(f"{path} cannot be read") from err
    except tomllib.TOMLDecodeError as err:
        raise LicenceError(f"{path} is not valid TOML") from err

    if not parsed.get("allowed"):
        raise LicenceError(f"{path} lists no allowed licences")

    policy = Policy(allowed=frozenset(parsed["allowed"]), decisions={}, acceptances={})
    # Acceptances first: a decision may take a licence that is only acceptable here.
    for entry in parsed.get("acceptance", []):
        accepted = _decision(entry)
        policy.acceptances[accepted.package] = accepted
    for entry in parsed.get("decision", []):
        taken = _decision(entry)
        if not policy.permits(taken.licence, package=taken.package):
            raise LicenceError(
                f"{taken.package}: the decision takes {taken.licence}, which is not allowed"
            )
        policy.decisions[taken.package] = taken
    return policy


def _decision(entry: dict[str, Any]) -> Decision:
    missing = [field for field in REQUIRED if not entry.get(field)]
    if missing:
        raise LicenceError(f"a licence decision is missing {', '.join(missing)}")
    unknown = set(entry) - set(REQUIRED)
    if unknown:
        raise LicenceError(f"a licence decision has unknown fields: {', '.join(sorted(unknown))}")
    return Decision(**{field: str(entry[field]) for field in REQUIRED})


def check(package: str, licence: str | None, *, policy: Policy) -> str | None:
    """The violation this package's licence represents, or `None` if it is acceptable."""
    if not licence:
        return f"{package}: no licence declared"

    offered = [part.strip(" ()") for part in OR.split(licence)]
    if len(offered) > 1:
        return _resolve(package, offered, policy=policy)

    forbidden = [
        part
        for part in (piece.strip(" ()") for piece in AND.split(licence))
        if not policy.permits(part, package=package)
    ]
    if forbidden:
        return f"{package}: {licence} is not an allowed licence"
    return None


def _resolve(package: str, offered: list[str], *, policy: Policy) -> str | None:
    """A choice of licences says nothing about which obligations were accepted."""
    taken = policy.decisions.get(package)
    if taken is None:
        return (
            f"{package}: {' OR '.join(offered)} offers a choice and needs a recorded "
            f"decision in {POLICY.name}"
        )
    if not any(taken.licence.casefold() == part.casefold() for part in offered):
        return (
            f"{package}: the recorded decision takes {taken.licence}, which the package "
            f"does not offer ({' OR '.join(offered)})"
        )
    return None


def declared(package: str) -> str | None:
    """The licence a distribution declares, or `None` if it declares none this can read."""
    try:
        fields = metadata(package)
    except PackageNotFoundError:
        return None

    expression = fields.get("License-Expression")
    if expression:
        return str(expression).strip()

    for classifier in fields.get_all("Classifier") or []:
        if str(classifier) in CLASSIFIERS:
            return CLASSIFIERS[str(classifier)]

    legacy = str(fields.get("License") or "").strip()
    # A full licence text in the legacy field is not an identifier; only a short one is.
    return legacy if legacy and "\n" not in legacy and len(legacy) < 40 else None


def installed() -> list[str]:
    """The distributions a consumer receives: everything reachable outside a dev group."""
    graph = project_graph()
    shipped = {
        name
        for name, labels in graph.reach.items()
        if RUNTIME in labels or any(label.startswith("extra:") for label in labels)
    }
    present = {
        str(distribution.metadata["Name"]).casefold()
        for distribution in distributions()
        if distribution.metadata["Name"]
    }
    return sorted(name for name in shipped if name.casefold() in present)


def violations(names: Iterable[str], *, policy: Policy) -> list[str]:
    """Every licence violation across the given packages."""
    return [
        violation
        for name in names
        if (violation := check(name, declared(name), policy=policy)) is not None
    ]


def main(argv: list[str] | None = None) -> int:
    """Check every shipped dependency against the licence policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY, help="path to licences.toml")
    args = parser.parse_args(argv)

    found = violations(installed(), policy=load(args.policy))
    if not found:
        sys.stdout.write("no licence violations.\n")
        return 0

    sys.stdout.write(
        "licence policy violations. Record a decision in "
        f"{args.policy.name} or remove the dependency:\n\n"
    )
    sys.stdout.write("".join(f"  {violation}\n" for violation in found))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
