"""Scan the resolved lock for advisories and apply the security policy.

Reported with the package, the advisory, the first fixed version and who receives it, so
a consuming team can act without repeating the reachability analysis themselves.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

from tools.lockfile import project_graph
from tools.security_policy import Policy, blocks, load

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tools.lockfile import Graph

__all__ = ["AuditError", "Finding", "Verdict", "assess", "findings", "rate", "scan"]

OSV = "https://api.osv.dev/v1/vulns/"
TIMEOUT_SECONDS = 15.0
UNKNOWN = "unknown"
NOT_SHIPPED = "development only, not shipped to consumers"


class AuditError(Exception):
    """Raised when the scan cannot be run. Never treated as a clean result."""


@dataclass(frozen=True)
class Finding:
    """One advisory against one locked package."""

    package: str
    version: str
    id: str
    fixed: str | None
    severity: str = UNKNOWN


@dataclass(frozen=True)
class Verdict:
    """Whether the scan blocks, what it tracked, and the report either way."""

    blocking: bool
    tracked: tuple[Finding, ...]
    report: str


def scan() -> dict[str, Any]:
    """Run pip-audit over the environment resolved from the lock.

    Raises:
        AuditError: If the scanner cannot run or does not return readable JSON.
    """
    command = ["uv", "run", "pip-audit", "--format=json", "--progress-spinner=off"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    except OSError as err:
        raise AuditError(f"pip-audit could not be started: {err}") from err

    try:
        parsed: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        raise AuditError(f"pip-audit returned no readable JSON: {result.stderr.strip()}") from err
    return parsed


def findings(report: dict[str, Any]) -> list[Finding]:
    """Every advisory in a pip-audit report, one Finding each."""
    return [
        Finding(
            package=dependency["name"],
            version=dependency.get("version", "?"),
            id=vulnerability["id"],
            fixed=_first_fix(vulnerability.get("fix_versions", [])),
        )
        for dependency in report.get("dependencies", [])
        for vulnerability in dependency.get("vulns", [])
    ]


def _first_fix(fixes: Sequence[str]) -> str | None:
    """The lowest version that fixes it: the smallest upgrade a consumer can take."""
    parsed = []
    for fix in fixes:
        try:
            parsed.append(Version(fix))
        except InvalidVersion:
            continue
    return str(min(parsed)) if parsed else None


def severity_of(advisory: str) -> str | None:
    """The advisory's rating, from OSV. `None` when it has not been rated."""
    try:
        with urllib.request.urlopen(f"{OSV}{advisory}", timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    rated = payload.get("database_specific", {}).get("severity")
    return str(rated).lower() if rated else None


def rate(found: Sequence[Finding], *, severity_of: Callable[[str], str | None]) -> list[Finding]:
    """Attach a severity to each finding, defaulting to unknown, which blocks."""
    return [replace(item, severity=severity_of(item.id) or UNKNOWN) for item in found]


def assess(found: Sequence[Finding], *, graph: Graph, policy: Policy, today: dt.date) -> Verdict:
    """Apply the policy to the findings and render the report."""
    lines: list[str] = []
    tracked: list[Finding] = []
    blocking = False

    for item in sorted(found, key=lambda finding: (finding.package, finding.id)):
        radius = graph.blast_radius(item.package)
        fix = f"fixed in {item.fixed}" if item.fixed else "no fixed version yet"

        if policy.suppresses(item.id, kind="advisory", today=today):
            owner = next(entry.owner for entry in policy.suppressions if entry.id == item.id)
            state = f"suppressed by {owner}"
        elif radius != NOT_SHIPPED and blocks(item.severity):
            state = "blocks"
            blocking = True
        else:
            state = "tracked"
            tracked.append(item)

        lines.append(
            f"{item.package} {item.version}: {item.id} ({item.severity}, {fix}) "
            f"— {radius} — {state}"
        )

    for stale in policy.expired(today):
        blocking = True
        lines.append(
            f"{stale.id}: suppression expired {stale.expires}, owner {stale.owner} — "
            f"re-review it or let the finding block"
        )

    if not lines:
        lines.append("no advisories against the locked dependency set.")

    return Verdict(blocking=blocking, tracked=tuple(tracked), report="\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    """Scan, apply the policy, and fail on anything blocking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    try:
        report = scan()
    except AuditError as err:
        sys.stderr.write(f"{err}\n")
        return 1

    rated = rate(findings(report), severity_of=severity_of)
    verdict = assess(rated, graph=project_graph(), policy=load(), today=dt.date.today())
    sys.stdout.write(verdict.report + "\n")
    return 1 if verdict.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
