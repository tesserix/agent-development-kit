"""What a guard may say about content, and the order guards are asked in.

Safety checks written inline in application code are invisible in an agent's definition:
the same agent is guarded in one product and unguarded in another, and nobody can tell by
reading it. Worse, an inline check that raises is usually swallowed, so the run continues
with nothing checking it — an unavailable guard silently becomes a permissive one.

So a guard returns a typed `GuardResult` rather than a truth value, and a
`GuardrailPipeline` asks them in a declared, stable order. A guard may allow, may hand
back redacted content for the guards after it to look at, or may block; a guard that
raises, times out or answers with something unreadable blocks the content instead of
being read as consent.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/guardrails.md`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from tesserix_adk.core.errors import (
    ConfigurationError,
    GuardrailEvaluationError,
    GuardrailViolationError,
)
from tesserix_adk.core.protocols import Guardrail, verify_conformance

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tesserix_adk.core.protocols import Tracer

__all__ = ["GuardResult", "GuardStage", "GuardVerdict", "GuardrailPipeline"]


class GuardStage(StrEnum):
    """Where in a run a check happened. Carried on every verdict and every refusal."""

    INPUT = "input"
    OUTPUT = "output"


class GuardVerdict(StrEnum):
    """What a guard decided.

    Ordered by restrictiveness: where two guards disagree the more restrictive one is what
    the run acts on, so a pipeline's answer does not depend on which guard happened to be
    asked last.
    """

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class GuardResult:
    """One guard's answer about one piece of content.

    Args:
        verdict: What it decided.
        content: What the guards after it should see, for a redaction. `None` otherwise.
        code: A machine-readable reason. Required for a block, so a caller can match on
            why rather than parsing a sentence that will be reworded.
        detail: A short, redacted explanation. Never the offending content.

    Example:
        >>> GuardResult.blocked(code="pii_detected").verdict
        <GuardVerdict.BLOCK: 'block'>
    """

    verdict: GuardVerdict
    content: str | None = None
    code: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        """Refuse a verdict nobody could act on, where it is written."""
        if self.verdict is GuardVerdict.BLOCK and not self.code:
            raise ConfigurationError("a block needs a code a caller can match on")
        if self.verdict is GuardVerdict.REDACT and self.content is None:
            raise ConfigurationError("a redaction needs the content to use instead")
        if self.verdict is GuardVerdict.ALLOW and self.content is not None:
            raise ConfigurationError("an allow replaces nothing, so it carries no content")

    @classmethod
    def allow(cls, *, code: str = "", detail: str = "") -> GuardResult:
        """Nothing to object to."""
        return cls(verdict=GuardVerdict.ALLOW, code=code, detail=detail)

    @classmethod
    def redacted(cls, content: str, *, code: str, detail: str = "") -> GuardResult:
        """Continue, but on `content` rather than what was shown."""
        return cls(verdict=GuardVerdict.REDACT, content=content, code=code, detail=detail)

    @classmethod
    def blocked(cls, *, code: str, detail: str = "") -> GuardResult:
        """Stop. The content does not continue in any form."""
        return cls(verdict=GuardVerdict.BLOCK, code=code, detail=detail)


class GuardrailPipeline:
    """An ordered, fail-closed chain of guards over one stage of a run.

    Args:
        guards: The guards, in the order they are asked. Order is what it says it is:
            never dependent on registration timing. An empty pipeline is legal, but it is
            written as one — `GuardrailPipeline(())` — so a review can tell a deliberate
            opt-out from an omission.
        timeout_seconds: How long one guard has to answer. A guard that does not is given
            up on and blocks, since a check nobody waited for is not a check.
        tracer: Where verdicts are recorded. Guard name, stage, verdict and duration, and
            never the content that was looked at.

    Raises:
        ConfigurationError: If two guards answer to one name, or the timeout could never
            permit a check.
        ProtocolConformanceError: If something in `guards` is not a guard.

    Example:
        >>> import asyncio
        >>> asyncio.run(GuardrailPipeline(()).check_input("ask"))
        'ask'
    """

    __slots__ = ("_guards", "_timeout", "_tracer")

    def __init__(
        self,
        guards: Sequence[Guardrail],
        *,
        timeout_seconds: float | None = 5.0,
        tracer: Tracer | None = None,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ConfigurationError("a timeout of zero would refuse every check")
        for guard in guards:
            verify_conformance(guard, Guardrail)
        names = [guard.name for guard in guards]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ConfigurationError(
                f"two guards answer to one name, so a verdict could mean either: "
                f"{', '.join(duplicated)}"
            )
        self._guards = tuple(guards)
        self._timeout = timeout_seconds
        self._tracer = tracer

    @property
    def guards(self) -> tuple[str, ...]:
        """The guard names, in the order they are asked."""
        return tuple(guard.name for guard in self._guards)

    async def check_input(self, content: str) -> str:
        """Check what is about to reach the model.

        Args:
            content: What was asked.

        Returns:
            The content the run should continue on, redactions applied in order.

        Raises:
            GuardrailViolationError: If a guard blocked it.
            GuardrailEvaluationError: If a guard could not decide. The content does not
                continue: a guard that is down is not a guard that consented.
        """
        return await self._through(GuardStage.INPUT, content)

    async def check_output(self, content: str) -> str:
        """Check what is about to leave the run, the same way and with the same refusals."""
        return await self._through(GuardStage.OUTPUT, content)

    async def check_stream(self, chunks: AsyncIterator[str]) -> AsyncIterator[str]:
        """Check a streamed answer, handing on nothing until the verdict is in.

        Args:
            chunks: The answer as it arrives.

        Yields:
            The checked answer, as one piece. Output guards decide on whole content, so
            streaming through them buys latency at the cost of emitting what a guard was
            about to block — which is the whole failure this exists to prevent.

        Raises:
            GuardrailViolationError: If a guard blocked it. Nothing has been yielded.
            GuardrailEvaluationError: If a guard could not decide.
        """
        buffered = "".join([chunk async for chunk in chunks])
        yield await self._through(GuardStage.OUTPUT, buffered)

    async def _through(self, stage: GuardStage, content: str) -> str:
        """Ask each guard in turn, carrying redactions forward and stopping at a refusal."""
        for guard in self._guards:
            result = await self._verdict_of(guard, stage, content)
            if result.verdict is GuardVerdict.BLOCK:
                raise GuardrailViolationError(
                    f"{guard.name} blocked the {stage} of this run: {result.code}",
                    code=result.code,
                    stage=stage,
                    guard=guard.name,
                    detail=result.detail,
                )
            if result.verdict is GuardVerdict.REDACT:
                content = result.content or ""
        return content

    async def _verdict_of(self, guard: Guardrail, stage: GuardStage, content: str) -> GuardResult:
        """One guard's answer, or a refusal standing in for the answer it did not give."""
        check = guard.check_input if stage is GuardStage.INPUT else guard.check_output
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout):
                # Typed as a verdict; a guard the type checker never saw may answer otherwise.
                answered: object = await check(content)
        except TimeoutError as expired:
            self._unevaluated(guard.name, stage, started)
            raise GuardrailEvaluationError(
                f"{guard.name} did not answer within {self._timeout}s",
                guard=guard.name,
                stage=stage,
                reason="timeout",
            ) from expired
        except Exception as failure:
            self._unevaluated(guard.name, stage, started)
            raise GuardrailEvaluationError(
                f"{guard.name} could not evaluate the {stage} of this run",
                guard=guard.name,
                stage=stage,
                reason="raised",
            ) from failure
        if not isinstance(answered, GuardResult):
            self._unevaluated(guard.name, stage, started)
            raise GuardrailEvaluationError(
                f"{guard.name} answered with something that is not a verdict",
                guard=guard.name,
                stage=stage,
                reason="unreadable",
            )
        self._record(guard.name, stage, str(answered.verdict), answered.code, started)
        return answered

    def _unevaluated(self, guard: str, stage: GuardStage, started: float) -> None:
        """Record that a guard was asked and did not answer, before the refusal is raised."""
        self._record(guard, stage, "unevaluated", "", started)

    def _record(
        self, guard: str, stage: GuardStage, verdict: str, code: str, started: float
    ) -> None:
        """Emit what was decided. Tracing fails open: a collector outage decides nothing."""
        if self._tracer is None:
            return
        with contextlib.suppress(Exception):
            self._tracer.event(
                "guardrail",
                guard=guard,
                stage=str(stage),
                verdict=verdict,
                code=code,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
