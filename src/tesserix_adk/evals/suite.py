"""Replaying a dataset the same way twice, and saying plainly which cases did not run.

A suite that runs its cases in whatever order they finish, against a live model, produces a
different number every time, and a number that moves on its own cannot gate anything. So
the runner fixes what it can: results come back in dataset order, each case's run id is
derived from the suite and the case rather than generated, and the digest of a result set
covers the answers and not how long they took.

The other half is honesty about failure. A case whose recording is missing, whose executor
raised, or whose run never reached a terminal state is not a pass and is not a fail — it is
a case that did not run, and it is reported as one. A harness that quietly scores those is
worse than no harness, because it reports green on the day the recordings went stale.

Determinism is enforced here, not provided here: the executor decides whether it replays a
cassette or calls a provider. `evals` cannot import `testing`, by design — the layering
keeps test doubles out of the shipped judgement path.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.tenancy import TenantContext, tenant_scope

if TYPE_CHECKING:
    from pathlib import Path

    from tesserix_adk.core.run import Run
    from tesserix_adk.evals.dataset import EvalCase, EvalSuite

__all__ = [
    "CaseExecutor",
    "CaseResult",
    "CaseStatus",
    "SuiteResult",
    "SuiteRunner",
]

_ANSWER_FIELDS = ("state", "messages", "tool_calls", "output", "usage")


class CaseStatus(StrEnum):
    """What happened to one case.

    `COMPLETED` says the agent answered, whatever the answer was — a failed run is a result
    to measure. `ERRORED` and `INCOMPLETE` both mean nothing was measured, and both keep
    the suite from reporting green.
    """

    COMPLETED = "completed"
    ERRORED = "errored"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One case's outcome, and the run behind it where there is one.

    Args:
        case_id: The case this answers for.
        run_id: Derived, so two machines replaying the same suite agree on it.
        status: See `CaseStatus`.
        reason: Why a case errored or stopped short, in the words the executor used.
        run: The run, present only when the case completed.
        seconds: Wall-clock time, reported but deliberately kept out of `digest`.
    """

    case_id: str
    run_id: str
    status: CaseStatus
    reason: str = ""
    run: Run[Any] | None = None
    seconds: float = 0.0

    def answer(self) -> dict[str, Any]:
        """The part of this result that is compared between runs, timings excluded."""
        answered: dict[str, Any] = {}
        if self.run is not None:
            answered = self.run.model_dump(mode="json", include=set(_ANSWER_FIELDS))
        return {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "status": str(self.status),
            "reason": self.reason,
            "run": answered,
        }


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """Every case's outcome, in dataset order.

    Args:
        suite_name: The suite that produced this.
        suite_version: The dataset version measured, since results only compare within one.
        results: One per case, in the order the cases appear in the dataset.
    """

    suite_name: str
    suite_version: str
    results: tuple[CaseResult, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether every case actually ran. A case that could not run is never a pass."""
        return all(result.status is CaseStatus.COMPLETED for result in self.results)

    @property
    def exit_code(self) -> int:
        """`0` when every case ran, `1` otherwise — what CI reads."""
        return 0 if self.ok else 1

    def errored(self) -> tuple[CaseResult, ...]:
        """The cases whose executor raised, each carrying its reason."""
        return tuple(result for result in self.results if result.status is CaseStatus.ERRORED)

    def incomplete(self) -> tuple[CaseResult, ...]:
        """The cases whose run never reached a terminal state."""
        return tuple(result for result in self.results if result.status is CaseStatus.INCOMPLETE)

    def digest(self) -> str:
        """A hash over the answers, stable across two replays of the same dataset.

        Timings are excluded: wall-clock time is not a result, and a digest that included
        it would never match twice.
        """
        payload = json.dumps(
            {
                "suite": self.suite_name,
                "version": self.suite_version,
                "results": [result.answer() for result in self.results],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CaseExecutor(Protocol):
    """How one case is turned into a run.

    The implementation owns determinism: a cassette-backed executor replays, a live one
    does not. It is passed the run id rather than choosing one, so the id is reproducible.
    """

    async def __call__(self, case: EvalCase, *, run_id: str) -> Run[Any]:
        """Answer `case`, as the run identified by `run_id`."""
        ...


class SuiteRunner:
    """Runs a dataset through an executor and reports what happened to each case.

    Args:
        execute: What answers a case. See `CaseExecutor`.
        concurrency: How many cases may be in flight at once. Bounded because a suite that
            fans out without a ceiling is rate-limited into flakiness.
        seed: Mixed into each derived run id, so one dataset can be replayed under
            distinguishable ids without the ids becoming random.
        artefacts: A directory to write per-case evidence into, or `None` to keep the run
            in memory.

    Raises:
        ConfigurationError: `concurrency` is not positive.

    Example:
        >>> runner = SuiteRunner(lambda case, *, run_id: None, concurrency=2)
        >>> runner.concurrency
        2
    """

    def __init__(
        self,
        execute: CaseExecutor,
        *,
        concurrency: int = 4,
        seed: str = "",
        artefacts: Path | None = None,
    ) -> None:
        if concurrency < 1:
            raise ConfigurationError("concurrency must be at least one case at a time")
        self._execute = execute
        self.concurrency = concurrency
        self._seed = seed
        self._artefacts = artefacts

    async def run(self, suite: EvalSuite) -> SuiteResult:
        """Run every case in `suite` and return the outcomes in dataset order.

        Args:
            suite: The dataset to replay.

        Returns:
            One `CaseResult` per case, in the order the cases appear in the dataset.
        """
        gate = asyncio.Semaphore(self.concurrency)

        async def bounded(case: EvalCase) -> CaseResult:
            async with gate:
                return await self._one(suite, case)

        results = await asyncio.gather(*(bounded(case) for case in suite.cases))
        return SuiteResult(
            suite_name=suite.name, suite_version=suite.version, results=tuple(results)
        )

    def run_id_for(self, suite: EvalSuite, case: EvalCase) -> str:
        """Derive this case's run id from the suite, the version, the case and the seed."""
        material = "\n".join((self._seed, suite.name, suite.version, case.id))
        return f"eval-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"

    async def _one(self, suite: EvalSuite, case: EvalCase) -> CaseResult:
        """Run one case under its own tenant, recording what happened either way."""
        run_id = self.run_id_for(suite, case)
        home = self._case_dir(suite, case)
        pending = CaseResult(
            case_id=case.id,
            run_id=run_id,
            status=CaseStatus.INCOMPLETE,
            reason="the case had not finished when the suite stopped",
        )
        self._record(home, case, pending)
        started = time.monotonic()
        try:
            with tenant_scope(TenantContext(tenant=case.tenant, user=case.user)):
                run = await self._execute(case, run_id=run_id)
        except asyncio.CancelledError:
            raise
        # Any executor failure belongs to that case, not to the suite.
        except Exception as failed:
            result = CaseResult(
                case_id=case.id,
                run_id=run_id,
                status=CaseStatus.ERRORED,
                reason=str(failed) or type(failed).__name__,
                seconds=time.monotonic() - started,
            )
        else:
            result = self._judge(case, run_id, run, time.monotonic() - started)
        self._record(home, case, result)
        return result

    @staticmethod
    def _judge(case: EvalCase, run_id: str, run: Run[Any], seconds: float) -> CaseResult:
        """Decide what a returned run means, refusing one answered for another tenant."""
        if run.tenant != case.tenant:
            return CaseResult(
                case_id=case.id,
                run_id=run_id,
                status=CaseStatus.ERRORED,
                reason=f"the run came back for tenant {run.tenant!r}, not {case.tenant!r}",
                seconds=seconds,
            )
        if not run.state.is_terminal:
            return CaseResult(
                case_id=case.id,
                run_id=run_id,
                status=CaseStatus.INCOMPLETE,
                reason=f"the run stopped in {run.state}, which is not an outcome",
                run=run,
                seconds=seconds,
            )
        return CaseResult(
            case_id=case.id, run_id=run_id, status=CaseStatus.COMPLETED, run=run, seconds=seconds
        )

    def _case_dir(self, suite: EvalSuite, case: EvalCase) -> Path | None:
        """Where this case's evidence goes, or `None` when none was asked for."""
        if self._artefacts is None:
            return None
        home = self._artefacts / suite.name / case.id
        home.mkdir(parents=True, exist_ok=True)
        return home

    @staticmethod
    def _record(home: Path | None, case: EvalCase, result: CaseResult) -> None:
        """Write the case, its run and its outcome, timings kept in their own file."""
        if home is None:
            return
        _write(home / "case.json", case.model_dump(mode="json"))
        _write(
            home / "result.json",
            {
                "case_id": result.case_id,
                "run_id": result.run_id,
                "status": str(result.status),
                "reason": result.reason,
            },
        )
        _write(home / "timings.json", {"seconds": result.seconds})
        if result.run is not None:
            _write(home / "run.json", result.run.model_dump(mode="json"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Write one artefact, sorted so a diff between two runs reads as a diff of answers."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
