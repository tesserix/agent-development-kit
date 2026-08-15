"""A check that keeps business rules out of prompt text.

Prompts accumulate pricing, refund thresholds, cancellation windows and eligibility
conditions written in English. Those are business rules with none of the properties a
business rule needs: untested, unversioned against the code that depends on them, and
advisory — the model may simply not apply one, and nothing fails when it does not.

The split this enforces is the kit's own: agents reason, code transacts. Role, task framing,
style and output contracts belong in a prompt. Monetary thresholds, eligibility conditions,
time windows, authorisation decisions and irreversible actions belong in a tool the prompt
asks, so the rule is enforced by something that can be tested and refuses when it should.

Findings can be suppressed, with a reason. A suppression carrying none is itself a finding,
and every report counts its suppressions, so a project that silenced the check rather than
acted on it says so out loud.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "PROMPT_SUFFIXES",
    "RULES",
    "LintFinding",
    "LintReport",
    "LintRule",
    "Severity",
    "lint_directory",
    "lint_prompt",
]

Severity = Literal["error", "warning"]

PROMPT_SUFFIXES = (".toml", ".txt", ".md", ".prompt")

# An em dash, an en dash or a hyphen may introduce the reason; people type all three.
_SUPPRESSION = re.compile(
    "adk-lint:\\s*allow\\s+(ADK-P\\d{3})\\s*(?:[\\u2014\\u2013-]\\s*(?P<reason>.*))?$"
)
_MINIMUM_REASON = 8


class LintRule(AdkModel):
    """One thing a prompt may not say, and what to do instead.

    Args:
        code: The stable identifier, quoted in suppressions and in CI output.
        summary: What was found, in the words a reviewer would use.
        severity: `error` fails the check; `warning` is reported and does not.
        remedy: The refactor. Every rule has one, because a check that only says no is a
            check projects turn off.
        patterns: The expressions that match it.
    """

    code: str
    summary: str
    severity: Severity
    remedy: str
    patterns: tuple[str, ...]


RULES = (
    LintRule(
        code="ADK-P000",
        summary="a suppression with no recorded justification",
        severity="error",
        remedy="write the reason after the code, e.g. `adk-lint: allow ADK-P001 — legacy "
        "tariff, ticket PLAT-91`",
        patterns=(),
    ),
    LintRule(
        code="ADK-P001",
        summary="a monetary threshold or a share of an amount",
        severity="error",
        remedy="move the threshold into a tool the prompt asks, so it is versioned with the "
        "code that charges and can be tested",
        patterns=(
            r"[$€£¥]\s?\d",
            r"\b\d[\d,.]*\s?(usd|eur|aud|gbp|inr)\b",
            r"\b\d+\s?(%|per\s?cent|percent)\s+(of|off)\b",
        ),
    ),
    LintRule(
        code="ADK-P002",
        summary="a conditional rule chain",
        severity="error",
        remedy="ask a tool for the decision and let the prompt use its answer",
        patterns=(
            r"^\s*(if|unless|when|whenever)\b[^.]*(,|\bthen\b)",
            r"\bonly if\b",
            r"\bis (not )?eligible\b",
        ),
    ),
    LintRule(
        code="ADK-P003",
        summary="a time window governing an outcome",
        severity="error",
        remedy="move the window into the tool that evaluates it; a date comparison in prose "
        "is a rule nobody can test",
        patterns=(
            r"\b(more|less|older|newer|within|after|before)\s+(than\s+)?\d+\s*"
            r"(minute|hour|day|week|month|year)s?\b",
        ),
    ),
    LintRule(
        code="ADK-P004",
        summary="language granting the model authority to decide",
        severity="error",
        remedy="an authorisation decision belongs to code; have the prompt request one",
        patterns=(
            r"\byou (may|can|are allowed to|are authorised to|are authorized to)\s+"
            r"(approve|authorise|authorize|grant|waive|override|issue|refund)\b",
            r"\bat your (own )?discretion\b",
            r"\bgrant (access|permission)\b",
        ),
    ),
    LintRule(
        code="ADK-P005",
        summary="an instruction to perform an irreversible action directly",
        severity="error",
        remedy="call an approval-gated, idempotent tool instead — a prompt is not an "
        "enforcement point, and a retry of one is a second refund",
        patterns=(
            r"\b(issue|process|charge|delete|cancel|reverse|transfer|refund)\s+"
            r"(the\s+|a\s+|their\s+)?(refund|payment|charge|booking|order|account|card|"
            r"subscription|invoice)\b",
        ),
    ),
    LintRule(
        code="ADK-P006",
        summary="an embedded endpoint, URL or service name",
        severity="warning",
        remedy="declare the tool and let the kit describe it; an endpoint in prose outlives "
        "the endpoint",
        patterns=(
            r"\b(GET|POST|PUT|PATCH|DELETE)\s+/\S+",
            r"https?://\S+",
            r"\b\w+_(api|endpoint|service)\b",
        ),
    ),
)

_BY_CODE = {rule.code: rule for rule in RULES}
_COMPILED = tuple(
    (rule, tuple(re.compile(pattern, re.IGNORECASE) for pattern in rule.patterns))
    for rule in RULES
    if rule.patterns
)


class LintFinding(AdkModel):
    """One thing a prompt says that a tool should decide instead.

    Args:
        code: Which rule matched.
        severity: The rule's severity.
        summary: What was found.
        remedy: What to do about it.
        source: The file it came from, empty for a prompt linted inline.
        line: The one-based line number.
        text: The offending line, stripped.
    """

    code: str
    severity: Severity
    summary: str
    remedy: str
    source: str = ""
    line: int = 0
    text: str = ""

    def __str__(self) -> str:
        """The one-line form a CI log wants."""
        where = f"{self.source}:{self.line}" if self.source else f"line {self.line}"
        return f"{where}: {self.code} {self.summary} — {self.remedy}"


class LintReport(AdkModel):
    """What one run of the check found, over one prompt or a whole directory.

    Args:
        findings: Everything matched, ordered by file, line and code.
        files: How many prompt files were read.
        suppressed: How many findings a suppression silenced. Counted and reported, so a
            project suppressing everything is visible rather than green.
    """

    findings: tuple[LintFinding, ...] = ()
    files: int = 0
    suppressed: int = 0

    @property
    def ok(self) -> bool:
        """Whether the check passes. Warnings are reported and do not fail it."""
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def exit_code(self) -> int:
        """`0` where the check passes, so a project can `sys.exit(report.exit_code)`."""
        return 0 if self.ok else 1

    def summary(self) -> str:
        """One line naming the counts, including suppressions."""
        if not self.findings and not self.suppressed:
            return f"prompt lint clean over {self.files} file(s)"
        counts = dict.fromkeys(_BY_CODE, 0)
        for finding in self.findings:
            counts[finding.code] += 1
        found = ", ".join(f"{code} x{count}" for code, count in counts.items() if count)
        return f"{found or 'no findings'} over {self.files} file(s), {self.suppressed} suppressed"


def lint_prompt(text: str, *, source: str = "") -> tuple[LintFinding, ...]:
    """Lint one prompt body, wherever it is kept.

    Args:
        text: The prompt, registry-held or an inline string.
        source: What to call it in the findings.

    Returns:
        The findings, ordered by line then code. Empty means it passed.

    Example:
        >>> lint_prompt("Refund 50% of the fare.")[0].code
        'ADK-P001'
    """
    return _linted(text, source=source)[0]


def lint_directory(directory: Path, *, suffixes: tuple[str, ...] = PROMPT_SUFFIXES) -> LintReport:
    """Lint every prompt file under `directory`, for a project to run in CI.

    Args:
        directory: The project's prompt directory. Searched recursively.
        suffixes: Which files are prompts. Anything else is left alone.

    Returns:
        A report whose `exit_code` a CI step can exit on.
    """
    findings: list[LintFinding] = []
    files = 0
    suppressed = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        files += 1
        found, silenced = _linted(path.read_text(encoding="utf-8"), source=str(path))
        findings.extend(found)
        suppressed += silenced
    return LintReport(findings=tuple(findings), files=files, suppressed=suppressed)


def _linted(text: str, *, source: str) -> tuple[tuple[LintFinding, ...], int]:
    """Every finding in `text`, and how many a suppression silenced."""
    lines = text.splitlines()
    allowed = _suppressions(lines)
    findings: list[LintFinding] = []
    silenced = 0
    for number, line in enumerate(lines, start=1):
        matched = sorted(
            {rule.code for rule, patterns in _COMPILED if any(p.search(line) for p in patterns)}
        )
        for code in matched:
            if code in allowed.get(number, {}):
                silenced += 1
                if not allowed[number][code]:
                    findings.append(_finding("ADK-P000", source, number, line))
                continue
            findings.append(_finding(code, source, number, line))
    return tuple(findings), silenced


def _suppressions(lines: list[str]) -> dict[int, dict[str, bool]]:
    """Which codes are allowed on which lines, and whether each carries a justification.

    A suppression covers its own line and the one below it, so it can sit on either.
    """
    allowed: dict[int, dict[str, bool]] = {}
    for number, line in enumerate(lines, start=1):
        found = _SUPPRESSION.search(line.strip())
        if not found:
            continue
        justified = len((found.group("reason") or "").strip()) >= _MINIMUM_REASON
        for covered in (number, number + 1):
            allowed.setdefault(covered, {})[found.group(1)] = justified
    return allowed


def _finding(code: str, source: str, line: int, text: str) -> LintFinding:
    """One finding, carrying its rule's words rather than restating them."""
    rule = _BY_CODE[code]
    return LintFinding(
        code=rule.code,
        severity=rule.severity,
        summary=rule.summary,
        remedy=rule.remedy,
        source=source,
        line=line,
        text=text.strip(),
    )
