"""Catching non-determinism before a worker runs it, and proving a replay still holds.

A model call is non-deterministic by definition, so the moment one happens on the workflow
path rather than inside an activity the replay diverges: the run wedges, or worse, it quietly
re-decides. `uuid4`, `random`, `time.time()`, `datetime.now()` and a registry iterated in load
order do the same thing. None of it fails in development. All of it fails on the first replay
in production, on a run that is already in flight.

Two things live here. `guard` reads workflow-marked modules and refuses the calls that cannot
replay, naming file, line and remedy, so the build fails before a worker sees the code.
`assert_replays` takes a recorded history and re-drives the current code against it, raising
`NonDeterminismError` at the diverging command rather than letting a divergence read as a
pass.

A module opts in by declaring `__adk_workflow__ = True` at module level. Activity modules and
ordinary in-process runtime code are not scanned: a guard that fires on code it does not
govern is a guard consumers turn off.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import ast
from pathlib import Path  # noqa: TC003 — pydantic needs the runtime type
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import NonDeterminismError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "RULES",
    "WORKFLOW_MARKER",
    "RecordedHistory",
    "ReplayFinding",
    "ReplayReport",
    "ReplayRule",
    "assert_replays",
    "guard",
    "guard_source",
]

WORKFLOW_MARKER = "__adk_workflow__"
"""What a module sets to `True` to say its code runs on the workflow path."""


class ReplayRule(AdkModel):
    """One thing that cannot happen on a workflow path.

    Args:
        code: Its identifier, which is what a suppression or an exception names.
        summary: What was found, in the words a build log should carry.
        remedy: Where to do it instead. A rule with no remedy gets disabled.
        calls: The dotted call names it matches, as they are written in source.
    """

    code: str
    summary: str
    remedy: str
    calls: tuple[str, ...] = ()


RULES = (
    ReplayRule(
        code="ADK-W001",
        summary="a model provider is called on the workflow path",
        remedy="call it through model_call_activity, which records its result in history",
        calls=("complete", "stream", "generate", "chat"),
    ),
    ReplayRule(
        code="ADK-W002",
        summary="an id comes from randomness rather than from the run",
        remedy="use DeterministicIds, or take the id from an activity result",
        calls=("uuid.uuid4", "uuid4", "random.random", "random.choice", "random.randint"),
    ),
    ReplayRule(
        code="ADK-W003",
        summary="the wall clock is read on the workflow path",
        remedy="use WorkflowClock, whose instants come from the run's own state",
        calls=(
            "time.time",
            "time.monotonic",
            "time.sleep",
            "datetime.now",
            "datetime.utcnow",
            "datetime.today",
        ),
    ),
    ReplayRule(
        code="ADK-W004",
        summary="network I/O happens on the workflow path",
        remedy="move it into an activity, where its result is recorded once",
        calls=(
            "requests.get",
            "requests.post",
            "httpx.get",
            "httpx.post",
            "urllib.request.urlopen",
            "socket.socket",
        ),
    ),
    ReplayRule(
        code="ADK-W005",
        summary="a helper that is itself unsafe is called on the workflow path",
        remedy="move the helper behind an activity; two frames away is still the same replay",
    ),
)
"""The ruleset, in the order a report lists it."""

_BY_CODE = {rule.code: rule for rule in RULES}


class ReplayFinding(AdkModel):
    """One refusal, where it is, and what to do about it.

    Args:
        code: Which rule.
        summary: What was found.
        remedy: Where the call belongs instead.
        source: The file.
        line: The line.
        call: The call as it was written.
    """

    code: str
    summary: str
    remedy: str
    source: str = ""
    line: int = 0
    call: str = ""

    def __str__(self) -> str:
        """The line a build log prints."""
        return (
            f"{self.source}:{self.line}: {self.code} {self.summary} ({self.call})\n"
            f"    {self.remedy}"
        )


class ReplayReport(AdkModel):
    """What the guard found across everything it read.

    Args:
        findings: One per refused call, in file then line order.
        scanned: How many workflow-marked modules were read.
    """

    findings: tuple[ReplayFinding, ...] = ()
    scanned: int = 0

    @property
    def ok(self) -> bool:
        """Whether the build may proceed."""
        return not self.findings

    @property
    def exit_code(self) -> int:
        """`0` where nothing was found, `1` otherwise."""
        return 0 if self.ok else 1

    def summary(self) -> str:
        """What CI prints: the count, then every finding."""
        headline = (
            f"{len(self.findings)} replay-safety problem(s) in {self.scanned} workflow module(s)"
        )
        return headline + "".join(f"\n{finding}" for finding in self.findings)


class RecordedHistory(AdkModel):
    """A run's command sequence, committed as a fixture and replayed in CI.

    Args:
        run_id: The run it was recorded from.
        commands: The activity step ids, in the order the run issued them.
        patches: The patch names the run knew about, so a replay takes the paths it took.
    """

    run_id: str
    commands: tuple[str, ...] = ()
    patches: tuple[str, ...] = ()


def guard_source(text: str, *, source: str = "") -> tuple[ReplayFinding, ...]:
    """Read one module and return what cannot replay.

    A module without the `__adk_workflow__` marker returns nothing: activity code and
    ordinary runtime code are allowed everything this refuses.

    Args:
        text: The module's source.
        source: Its path, for the findings.

    Returns:
        The findings, in line order.

    Raises:
        SyntaxError: If the source does not parse. A file the guard cannot read is not a
            file the guard passes.
    """
    tree = ast.parse(text)
    return _scanned(tree, source=source) if _marked(tree) else ()


def _scanned(tree: ast.Module, *, source: str) -> tuple[ReplayFinding, ...]:
    """Every refused call in a module already known to be workflow code."""
    unsafe = _unsafe_helpers(tree)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _called(node.func)
        rule = _matching(called, unsafe)
        if rule is not None:
            findings.append(
                ReplayFinding(
                    code=rule.code,
                    summary=rule.summary,
                    remedy=rule.remedy,
                    source=source,
                    line=node.lineno,
                    call=called,
                )
            )
    return tuple(sorted(findings, key=lambda finding: finding.line))


def guard(paths: Iterable[Path]) -> ReplayReport:
    """Read every Python file under `paths` and report what cannot replay.

    Args:
        paths: Files or directories. Directories are walked for `.py` files.

    Returns:
        The report. `exit_code` is what a CI step returns.
    """
    findings: list[ReplayFinding] = []
    scanned = 0
    for path in sorted(_files(paths)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _marked(tree):
            continue
        scanned += 1
        findings.extend(_scanned(tree, source=str(path)))
    return ReplayReport(findings=tuple(findings), scanned=scanned)


def assert_replays(history: RecordedHistory, commands: Sequence[str]) -> None:
    """Check that the current code issues the commands the history recorded.

    Args:
        history: What the run did, committed as a fixture.
        commands: What the code asks for now, in order.

    Raises:
        NonDeterminismError: At the first index where they differ, naming both sides. A
            run that issues fewer or more commands than the history has diverged too: a
            replay is not allowed to be a prefix of the truth.
    """
    for index, expected in enumerate(history.commands):
        actual = commands[index] if index < len(commands) else ""
        if actual != expected:
            raise NonDeterminismError(
                f"replaying {history.run_id} diverged at command {index}: the history "
                f"records {expected!r} and the code asked for {actual or 'nothing'!r}",
                run_id=history.run_id,
                command=index,
                expected=expected,
                actual=actual,
            )
    if len(commands) > len(history.commands):
        index = len(history.commands)
        raise NonDeterminismError(
            f"replaying {history.run_id} diverged at command {index}: the history ends "
            f"and the code asked for {commands[index]!r}",
            run_id=history.run_id,
            command=index,
            expected="",
            actual=commands[index],
        )


def _files(paths: Iterable[Path]) -> list[Path]:
    """Every Python file under the given paths."""
    found: list[Path] = []
    for path in paths:
        found.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return found


def _marked(tree: ast.Module) -> bool:
    """Whether the module declares itself workflow code."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == WORKFLOW_MARKER for target in node.targets
        ):
            return isinstance(node.value, ast.Constant) and node.value.value is True
    return False


def _unsafe_helpers(tree: ast.Module) -> frozenset[str]:
    """Module-level functions that themselves do something unreplayable.

    A consumer's helper two frames deep is the same divergence as the call written inline,
    so the caller is refused as well.
    """
    unsafe = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and _matching(_called(inner.func), frozenset()):
                unsafe.add(node.name)
                break
    return frozenset(unsafe)


def _called(func: ast.expr) -> str:
    """The call's dotted name, as it was written."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return f"{_called(func.value)}.{func.attr}".removeprefix(".")
    return ""


def _matching(called: str, unsafe: frozenset[str]) -> ReplayRule | None:
    """Which rule refuses this call, where one does."""
    if called in unsafe:
        return _BY_CODE["ADK-W005"]
    tail = called.rsplit(".", 1)[-1]
    for rule in RULES:
        named = any(called == call or called.endswith(f".{call}") for call in rule.calls)
        if named or (rule.code == "ADK-W001" and tail in rule.calls):
            return rule
    return None
