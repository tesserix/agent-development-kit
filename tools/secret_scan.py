"""Find credential-shaped values in anything this repository ships.

The kit ships recorded provider traffic as fixtures, so a real key committed once is
distributed to every consumer with the next sdist. A fixture that deliberately looks like
a credential is legitimate and must be declared in `security/policy.toml`: inferring which
keys are fake is exactly how the real one gets through.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tools.security_policy import Policy, load

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "Finding",
    "Match",
    "matches",
    "recorded_traffic",
    "render",
    "scan",
    "tracked_files",
]

ROOT = Path(__file__).resolve().parents[1]

# Shapes only: length and alphabet, not provider lookups. A rule that needs the network to
# decide is a rule that fails open when the network is down.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic-key", re.compile(r"sk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_\-]{32,}")),
    ("openai-key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{40,}")),
    ("aws-access-key-id", re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("google-api-key", re.compile(r"AIza[A-Za-z0-9_\-]{35}")),
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{20,}")),
)

# Recorded traffic was recorded from a live exchange, so it can carry someone's details
# even when it carries no credential. These rules apply only there: a maintainer address in
# CODEOWNERS or a support number in the docs is the point of those files.
PERSONAL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email-address", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("phone-number", re.compile(r"\+\d{1,3}[\s\-]?(?:\(?\d{2,4}\)?[\s\-]?){2,4}\d{2,4}")),
)

# A value that has already been redacted, or a name being talked about rather than used.
INNOCENT = re.compile(r"REDACTED|EXAMPLE|YOUR[-_]|XXXX|\*{3,}|\bos\.environ\b|\bgetenv\b")

SKIP_DIRECTORIES = frozenset({".git", ".venv", "node_modules", "__pycache__", "dist", "build"})
TRAFFIC_DIRECTORIES = ("cassettes", "recordings")
EVIDENCE_CHARACTERS = 8


@dataclass(frozen=True)
class Match:
    """A rule that fired, and just enough of the value to recognise it."""

    rule: str
    evidence: str


@dataclass(frozen=True)
class Finding:
    """A credential shape at a place in the tree."""

    path: Path
    line: int
    rule: str
    evidence: str

    @property
    def id(self) -> str:
        """How the finding is named in `security/policy.toml`."""
        return f"{self.rule}:{self.path.name}"


def matches(text: str, *, personal: bool = False) -> list[Match]:
    """Every credential shape in a line, with the value truncated.

    A scanner that prints the credential in full has published it to the build log, where
    it is readable by anyone who can see the run.
    """
    if INNOCENT.search(text):
        return []
    rules = (*RULES, *PERSONAL_RULES) if personal else RULES
    return [
        Match(rule=rule, evidence=found.group(0)[:EVIDENCE_CHARACTERS] + "…")
        for rule, pattern in rules
        if (found := pattern.search(text))
    ]


def scan(
    paths: Iterable[Path], *, policy: Policy, today: dt.date, personal: bool = False
) -> list[Finding]:
    """Findings in the given files, minus the ones declared in the policy."""
    found: list[Finding] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for number, text in enumerate(content.splitlines(), start=1):
            found.extend(
                Finding(path=path, line=number, rule=match.rule, evidence=match.evidence)
                for match in matches(text, personal=personal)
            )

    return [
        finding
        for finding in found
        if not policy.suppresses(finding.id, kind="secret", today=today)
    ]


def tracked_files() -> list[Path]:
    """Everything git tracks, which is everything an sdist can carry."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    return [
        ROOT / name
        for name in result.stdout.split("\0")
        if name and not SKIP_DIRECTORIES.intersection(Path(name).parts)
    ]


def recorded_traffic() -> list[Path]:
    """Tracked files under a cassette or recording directory."""
    return [
        path for path in tracked_files() if any(part in TRAFFIC_DIRECTORIES for part in path.parts)
    ]


def render(found: Sequence[Finding]) -> str:
    """The report, leading with rotation because that is the only urgent step."""
    if not found:
        return "no credential-shaped values found."

    lines = [
        "credential-shaped values found. Rotate the credential first — it is already",
        "compromised, and rewriting history does not un-publish it. Then remove it, and",
        "declare the value in security/policy.toml only if it is a deliberate fixture.",
        "",
    ]
    lines.extend(
        f"  {finding.path}:{finding.line}: {finding.rule} ({finding.evidence})" for finding in found
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Scan the named paths, or everything git tracks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files to scan; defaults to every tracked file")
    args = parser.parse_args(argv)

    policy, today = load(), dt.date.today()
    if args.paths:
        found = scan([Path(name) for name in args.paths], policy=policy, today=today)
    else:
        # Cassettes are in both passes, so a credential there would be reported twice.
        found = list(
            dict.fromkeys(
                scan(tracked_files(), policy=policy, today=today)
                + scan(recorded_traffic(), policy=policy, today=today, personal=True)
            )
        )
    sys.stdout.write(render(found) + "\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
