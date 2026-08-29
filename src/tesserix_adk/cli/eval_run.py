"""Discover and run one evaluation suite deterministically, with explicit live spend."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from tesserix_adk import __version__
from tesserix_adk.cli.artifacts import ARTIFACT_VERSION, ArtifactHeader, ArtifactWriter
from tesserix_adk.core import ConfigurationError, RunState
from tesserix_adk.evals import CaseResult, CaseStatus, EvalCase, EvalSuite, SuiteResult, SuiteRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import TextIO

    from tesserix_adk.evals import CaseExecutor

__all__ = [
    "CASE_ERROR",
    "CONFIGURATION_ERROR",
    "GATE_FAILED",
    "EvalTarget",
    "Resolve",
    "load_target",
    "main",
]

OK = 0
GATE_FAILED = 1
CONFIGURATION_ERROR = 2
CASE_ERROR = 3
INTERRUPTED = 130
DEFAULT_TARGET = "eval_target:target"
DEFAULT_OUTPUT = Path(".adk/eval-artifacts")
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")


class LiveEstimate(Protocol):
    """Estimate total suite spend before the first live case is scheduled."""

    def __call__(self, suite: EvalSuite) -> Decimal:
        """Return the high-case estimate in USD for every selected case."""
        ...


@dataclass(frozen=True)
class EvalTarget:
    """Deterministic and deliberately separate live executors for one agent.

    Args:
        deterministic: Cassette-backed or otherwise network-free case executor.
        live: Provider-backed executor, unavailable unless explicitly supplied.
        estimate_live: Whole-suite high-case estimator required before live execution.
        live_ceiling: Maximum permitted estimated spend for one invocation, in USD.
        max_parallel: Provider or application concurrency ceiling.
    """

    deterministic: CaseExecutor
    live: CaseExecutor | None = None
    estimate_live: LiveEstimate | None = None
    live_ceiling: Decimal = Decimal("0")
    max_parallel: int = 4

    def __post_init__(self) -> None:
        """Refuse a target whose own throttle permits no work."""
        if self.max_parallel < 1:
            raise ConfigurationError("an eval target's max_parallel must be at least one")
        if self.live_ceiling < 0:
            raise ConfigurationError("an eval target's live_ceiling cannot be negative")


type Resolve = Callable[[str], EvalTarget]
"""Resolve the evaluator target named by ``--target``."""


@dataclass(frozen=True)
class _Outcome:
    """One case result plus built-in metric verdicts."""

    result: CaseResult
    metrics: Mapping[str, int]
    passed: bool
    regressions: tuple[str, ...] = ()


async def main(
    argv: Sequence[str],
    *,
    resolve: Resolve,
    out: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Execute one suite and emit a table, JSON or JUnit report.

    Args:
        argv: Arguments after ``eval``.
        resolve: Project wiring for the selected target.
        out: Report destination. Defaults to stdout.
        stdin: Confirmation source for interactive live runs. Defaults to stdin.

    Returns:
        ``0`` for a passing gate, ``1`` for measured gate failures, ``2`` for
        configuration/refusal, ``3`` where cases errored, and ``130`` after interruption.
    """
    writer = out if out is not None else sys.stdout
    reader = stdin if stdin is not None else sys.stdin
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return CONFIGURATION_ERROR
    if parsed.parallel < 1:
        writer.write("--parallel must be at least one\n")
        return CONFIGURATION_ERROR
    if parsed.live_ceiling is not None and parsed.live_ceiling < 0:
        writer.write("--live-ceiling cannot be negative\n")
        return CONFIGURATION_ERROR
    path = Path(parsed.suite)
    if not path.is_file():
        writer.write(f"suite path not found: {path}\n")
        return CONFIGURATION_ERROR
    try:
        suite = EvalSuite.from_jsonl(path)
    except (ConfigurationError, ValueError, OSError) as error:
        writer.write(f"suite cannot be used: {error}\n")
        return CONFIGURATION_ERROR
    if not suite.cases:
        writer.write(f"suite {path} exists but contains no cases\n")
        return CONFIGURATION_ERROR
    if parsed.filter:
        suite = suite.model_copy(update={"cases": suite.tagged(parsed.filter)})
        if not suite.cases:
            writer.write(f"filter {parsed.filter!r} selected no cases\n")
            return CONFIGURATION_ERROR
    try:
        target = resolve(parsed.target)
    except (ConfigurationError, ImportError, AttributeError, TypeError, ValueError) as error:
        writer.write(f"eval target could not be loaded: {error}\n")
        return CONFIGURATION_ERROR

    output = Path(parsed.output)
    try:
        _prepare_outputs(output, suite)
    except OSError as error:
        writer.write(f"eval artefact directory is not writable: {error}\n")
        return CONFIGURATION_ERROR
    requested = parsed.parallel
    concurrency = min(requested, target.max_parallel)
    executor = target.deterministic
    if parsed.live:
        live = _live_executor(target, suite, parsed, reader, writer)
        if live is None:
            return CONFIGURATION_ERROR
        executor = live

    try:
        results = await SuiteRunner(executor, concurrency=concurrency).run(suite)
    except (KeyboardInterrupt, asyncio.CancelledError):
        task = asyncio.current_task()
        if task is not None and hasattr(task, "uncancel"):
            task.uncancel()
        writer.write("eval interrupted; unfinished case status files remain not_run\n")
        return INTERRUPTED

    try:
        artefacts = _record_results(output, suite, results, target=parsed.target)
        baseline = _baseline(Path(parsed.baseline)) if parsed.baseline else {}
    except (ConfigurationError, OSError, ValueError) as error:
        writer.write(f"eval result could not be recorded: {error}\n")
        return CONFIGURATION_ERROR
    outcomes = tuple(
        _score(case, result, baseline)
        for case, result in zip(suite.cases, results.results, strict=True)
    )
    report = _report(suite, outcomes, artefacts, concurrency=concurrency, requested=requested)
    rendered = _render(report, parsed.report)
    if parsed.report_path:
        try:
            _atomic_write(Path(parsed.report_path), rendered)
        except OSError as error:
            writer.write(f"report could not be written: {error}\n")
            return CONFIGURATION_ERROR
        if parsed.report == "table":
            writer.write(f"report written to {parsed.report_path}\n")
    else:
        writer.write(rendered)
    if any(outcome.result.status is not CaseStatus.COMPLETED for outcome in outcomes):
        return CASE_ERROR
    if any(not outcome.passed for outcome in outcomes):
        return GATE_FAILED
    return OK


def load_target(reference: str) -> EvalTarget:
    """Import ``module:attribute`` as an :class:`EvalTarget` or no-argument factory.

    Raises:
        ConfigurationError: The reference is malformed or resolves to another type.
        ImportError: The module or one of its dependencies is unavailable.
        AttributeError: The named attribute does not exist.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ConfigurationError("eval target must look like package.module:target")
    loaded: object = getattr(importlib.import_module(module_name), attribute)
    if callable(loaded) and not isinstance(loaded, EvalTarget):
        if inspect.signature(loaded).parameters:
            raise ConfigurationError("an eval target factory must take no arguments")
        loaded = loaded()
    if not isinstance(loaded, EvalTarget):
        raise ConfigurationError("eval target must be EvalTarget or a factory returning one")
    return loaded


def _live_executor(
    target: EvalTarget,
    suite: EvalSuite,
    parsed: argparse.Namespace,
    reader: TextIO,
    writer: TextIO,
) -> CaseExecutor | None:
    """Authorize the whole live suite before returning anything callable."""
    if target.live is None or target.estimate_live is None:
        writer.write("live eval requires both a live executor and whole-suite estimator\n")
        return None
    estimate = target.estimate_live(suite)
    ceiling = parsed.live_ceiling if parsed.live_ceiling is not None else target.live_ceiling
    if estimate < 0:
        writer.write("live estimator returned a negative cost; no case was scheduled\n")
        return None
    writer.write(f"live estimate=USD {estimate} ceiling=USD {ceiling}\n")
    if estimate > ceiling:
        writer.write("live eval refused before scheduling any case: estimate exceeds ceiling\n")
        return None
    if not parsed.yes:
        if not bool(getattr(reader, "isatty", lambda: False)()):
            writer.write("live eval requires --yes when no interactive terminal is available\n")
            return None
        writer.write("run the whole live suite at this estimate? [y/N] ")
        writer.flush()
        if reader.readline().strip().casefold() not in {"y", "yes"}:
            writer.write("live eval declined before scheduling any case\n")
            return None
    return target.live


def _prepare_outputs(output: Path, suite: EvalSuite) -> None:
    """Pre-create honest not-run status for every case before any provider can be called."""
    home = output / _safe(suite.name)
    home.mkdir(parents=True, exist_ok=True)
    for case in suite.cases:
        _atomic_write(
            home / f"{_safe(case.id)}.status.json",
            json.dumps({"case_id": case.id, "status": "not_run"}, sort_keys=True) + "\n",
        )


def _record_results(
    output: Path, suite: EvalSuite, result: SuiteResult, *, target: str
) -> dict[str, str]:
    """Commit each completed run in the shared inspectable format and update status."""
    home = output / _safe(suite.name)
    paths: dict[str, str] = {}
    for case, one in zip(suite.cases, result.results, strict=True):
        stem = _safe(case.id)
        status_path = home / f"{stem}.status.json"
        _atomic_write(
            status_path,
            json.dumps(
                {"case_id": case.id, "status": one.status.value, "reason": one.reason},
                sort_keys=True,
            )
            + "\n",
        )
        if one.run is None:
            paths[case.id] = str(status_path)
            continue
        artifact_path = home / f"{stem}.jsonl"
        artifact_path.unlink(missing_ok=True)
        artifact = ArtifactWriter(
            artifact_path,
            ArtifactHeader(
                version=ARTIFACT_VERSION,
                kit_version=__version__,
                target=target,
                input=case.input,
                tenant=case.tenant,
                user=case.user,
                agent=one.run.agent_name,
            ),
        )
        artifact.finish(one.run)
        paths[case.id] = str(artifact_path)
    return paths


def _score(
    case: EvalCase, result: CaseResult, baseline: Mapping[str, Mapping[str, int]]
) -> _Outcome:
    """Apply deterministic built-in metrics and optional no-regression checks."""
    if result.run is None:
        return _Outcome(result=result, metrics={}, passed=False)
    run = result.run
    metrics: dict[str, int] = {"state_completed": int(run.state is RunState.COMPLETED)}
    if case.expected is not None:
        answer = run.text
        if not answer and run.output is not None:
            answer = json.dumps(run.output.model_dump(mode="json"), sort_keys=True)
        metrics["answer_match"] = int(case.expected in answer)
    if case.expected_tools:
        metrics["tool_sequence"] = int(
            tuple(call.name for call in run.tool_calls) == case.expected_tools
        )
    regressions = tuple(
        metric
        for metric, before in baseline.get(case.id, {}).items()
        if metric in metrics and metrics[metric] < before
    )
    return _Outcome(
        result=result,
        metrics=metrics,
        passed=all(value == 1 for value in metrics.values()) and not regressions,
        regressions=regressions,
    )


def _baseline(path: Path) -> dict[str, dict[str, int]]:
    """Read a previous JSON report as a fail-closed no-regression baseline."""
    if not path.is_file():
        raise ConfigurationError(f"baseline path not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != "tesserix-adk-eval-result":
        raise ConfigurationError("baseline is not a tesserix-adk-eval-result JSON report")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ConfigurationError("baseline has no cases")
    measured: dict[str, dict[str, int]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ConfigurationError("baseline contains a case without an id")
        metrics = case.get("metrics")
        if not isinstance(metrics, dict):
            raise ConfigurationError(f"baseline case {case['id']} has no metrics")
        measured[case["id"]] = {str(key): int(value) for key, value in metrics.items()}
    return measured


def _report(
    suite: EvalSuite,
    outcomes: tuple[_Outcome, ...],
    artifacts: Mapping[str, str],
    *,
    concurrency: int,
    requested: int,
) -> dict[str, Any]:
    """Build the single ordered report all renderers consume."""
    cases = []
    for outcome in outcomes:
        result = outcome.result
        cases.append(
            {
                "id": result.case_id,
                "status": result.status.value,
                "passed": outcome.passed,
                "reason": result.reason,
                "metrics": dict(outcome.metrics),
                "regressions": list(outcome.regressions),
                "artifact": artifacts[result.case_id],
            }
        )
    return {
        "format": "tesserix-adk-eval-result",
        "version": 1,
        "suite": suite.name,
        "suite_version": suite.version,
        "parallel": concurrency,
        "requested_parallel": requested,
        "cases": cases,
    }


def _render(report: Mapping[str, Any], kind: str) -> str:
    """Render the report in the selected stable format."""
    if kind == "json":
        return json.dumps(report, indent=2, sort_keys=True) + "\n"
    cases = report["cases"]
    if not isinstance(cases, list):  # pragma: no cover — built above
        raise TypeError("report cases are not a list")
    if kind == "junit":
        return _junit(str(report["suite"]), cases)
    requested = int(report["requested_parallel"])
    parallel = int(report["parallel"])
    throttle = f" (requested {requested}, throttled by target)" if requested != parallel else ""
    lines = [
        f"suite={report['suite']} parallel={parallel}{throttle}",
        "case  status  verdict  metrics",
    ]
    for case in cases:
        metrics = ",".join(f"{key}={value}" for key, value in case["metrics"].items()) or "-"
        lines.append(
            f"{case['id']}  {case['status']}  {'pass' if case['passed'] else 'fail'}  {metrics}"
        )
    return "\n".join(lines) + "\n"


def _junit(suite: str, cases: list[dict[str, Any]]) -> str:
    """Render CI-portable JUnit with metric values on every failure."""
    failures = sum(1 for case in cases if case["status"] == "completed" and not case["passed"])
    errors = sum(1 for case in cases if case["status"] != "completed")
    root = ET.Element(
        "testsuite",
        {"name": suite, "tests": str(len(cases)), "failures": str(failures), "errors": str(errors)},
    )
    for case in cases:
        node = ET.SubElement(root, "testcase", {"name": str(case["id"]), "classname": suite})
        metrics = " ".join(f"{key}={value}" for key, value in case["metrics"].items())
        if case["status"] != "completed":
            ET.SubElement(node, "error", {"message": str(case["reason"])}).text = str(
                case["reason"]
            )
        elif not case["passed"]:
            reason = metrics or "gate failed"
            ET.SubElement(
                node, "failure", {"message": reason}
            ).text = f"metrics: {reason}; artifact: {case['artifact']}"
        ET.SubElement(node, "system-out").text = f"artifact={case['artifact']} metrics={metrics}"
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n"


def _safe(value: str) -> str:
    """Keep simple ids readable and make every other id path-safe and collision-resistant."""
    if _SAFE_ID.fullmatch(value) and value not in {".", ".."}:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"case-{digest}"


def _atomic_write(path: Path, content: str) -> None:
    """Replace one report/status only after its whole new content is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    """Build the ``tesserix-adk eval`` command line."""
    parser = argparse.ArgumentParser(prog="tesserix-adk eval", description=__doc__)
    parser.add_argument("suite", help="versioned EvalSuite JSONL path")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="evaluator module:attribute")
    parser.add_argument("--filter", default="", help="only cases carrying this tag")
    parser.add_argument("--parallel", type=int, default=4, help="requested case concurrency")
    parser.add_argument("--baseline", help="previous JSON result for no-regression comparison")
    parser.add_argument("--report", choices=("table", "json", "junit"), default="table")
    parser.add_argument("--report-path", help="write the report to this path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="per-case artefact directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--deterministic", action="store_true", help="network-free mode (default)")
    mode.add_argument("--live", action="store_true", help="explicitly use the live executor")
    parser.add_argument("--live-ceiling", type=Decimal, help="per-invocation estimate ceiling USD")
    parser.add_argument("--yes", action="store_true", help="confirm an allowed live estimate")
    return parser
