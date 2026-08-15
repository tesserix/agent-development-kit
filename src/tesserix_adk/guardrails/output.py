"""Validating the answer a caller is about to be handed, and bounded repair before failing.

A caller that parses free text is writing string handling against a model's mood. Here the
terminal result is either a validated instance of the declared type or a typed error — there
is no third outcome, and in particular no partially parsed object with the missing fields
filled in with defaults to make it validate.

Repair is opt-in and bounded. Re-asking is a real model call against the run's budget, so
the cap is declared up front and the run fails closed when it is spent, rather than looping
until something happens to validate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from tesserix_adk.core.errors import OutputValidationError, SchemaViolationError
from tesserix_adk.core.guards import GuardResult
from tesserix_adk.core.output_policy import Abstention, PolicyReport, evaluate
from tesserix_adk.guardrails.base import Guard
from tesserix_adk.runtime.structured import unwrap_fenced

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from tesserix_adk.core.output_policy import Policy
    from tesserix_adk.runtime.structured import OutputContract

__all__ = ["PolicyGuard", "SchemaGuard", "validated"]

Reask = Callable[[str], Awaitable[str]]
"""Asks the model again with the correction text, and returns the next answer."""


class SchemaGuard(Guard):
    """Turns what a provider returned into the declared type, or into a typed error.

    Args:
        contract: What the model was asked for.
        abstention: Whether "I do not know" is an acceptable answer here. Where it is, an
            `Abstention` payload is returned as itself rather than failing the schema —
            without it a model that does not know invents something that validates.

    Example:
        >>> SchemaGuard.__name__
        'SchemaGuard'
    """

    name = "schema"

    def __init__(self, contract: OutputContract, *, abstention: bool = False) -> None:
        self.contract = contract
        self.abstention = abstention

    def parse(self, content: str) -> BaseModel:
        """The validated answer.

        Raises:
            OutputValidationError: Where the content is not the declared type. Carries the
                raw payload for a debugger and every failing path.
        """
        unwrapped, _ = unwrap_fenced(content)
        abstained = self._abstention(unwrapped)
        if abstained is not None:
            return abstained
        try:
            return self.contract.parse(unwrapped)
        except SchemaViolationError as refused:
            raise self._rejected(refused, attempts=1) from refused

    async def check_output(self, content: str) -> GuardResult:
        """Block an answer that is not the declared type, naming the paths and not the value."""
        try:
            self.parse(content)
        except OutputValidationError as refused:
            return GuardResult.blocked(
                code="schema_violation",
                detail=f"{refused.model}: {', '.join(refused.paths) or 'did not parse'}",
            )
        return GuardResult.allow()

    def _abstention(self, content: str) -> Abstention | None:
        if not self.abstention:
            return None
        try:
            return Abstention.model_validate_json(content)
        except ValidationError:
            return None

    def _rejected(self, refused: SchemaViolationError, *, attempts: int) -> OutputValidationError:
        return OutputValidationError(
            str(refused),
            model=refused.model or self.contract.output_type.__name__,
            paths=refused.paths,
            problems=refused.problems,
            payload=refused.payload,
            attempts=attempts,
            guard=self.name,
        )


class PolicyGuard(Guard):
    """Applies this tenant's rules to an answer that has already parsed.

    Args:
        policies: The rules, in declaration order. Evaluated in code — a rule enforced by
            asking the model that wrote the answer is a suggestion.

    Example:
        >>> PolicyGuard(()).name
        'output_policy'
    """

    name = "output_policy"

    def __init__(self, policies: Sequence[Policy]) -> None:
        self.policies = tuple(policies)

    def check(self, result: BaseModel) -> PolicyReport:
        """What this answer violates.

        Raises:
            GuardrailEvaluationError: Where a rule could not run, which is not a pass.
        """
        return evaluate(result, self.policies)

    def raise_for(self, result: BaseModel, *, attempts: int = 1) -> None:
        """Reject the answer, naming the rules and never the values.

        Raises:
            OutputValidationError: Where any rule was violated.
        """
        report = self.check(result)
        if not report.rejected:
            return
        raise OutputValidationError(
            f"answer violated {', '.join(report.policies)}",
            model=type(result).__name__,
            policies=report.policies,
            paths=_paths(report),
            problems={violation.path: violation.detail for violation in report.violations},
            attempts=attempts,
            guard=self.name,
        )

    def correction(self, report: PolicyReport) -> str:
        """The text sent back on a repair attempt, built only from what was violated.

        It says which rule failed and what the rule is about. It never supplies a value:
        an answer the kit dictated is the kit's answer with the model's name on it.
        """
        listed = "\n".join(
            f"- {violation.path or 'the answer'}: {violation.detail} ({violation.policy})"
            for violation in report.violations
        )
        return (
            f"That answer broke rules it has to satisfy:\n{listed}\n\n"
            "Answer again with one JSON object and nothing else, correcting only what is "
            "listed above and inventing nothing. If you cannot answer within those rules, "
            "say so rather than adjusting the answer to fit."
        )


async def validated(
    content: str,
    *,
    schema: SchemaGuard,
    policy: PolicyGuard | None = None,
    reask: Reask | None = None,
    attempts: int = 1,
) -> BaseModel:
    """The answer a caller may be handed, after at most `attempts` tries.

    Args:
        content: What the provider returned first.
        schema: The declared type this must validate against.
        policy: This tenant's rules, applied once it has parsed.
        reask: How to ask again with a correction. Without one there is no repair, which is
            the default: a re-ask is a model call and nobody asked for the spend.
        attempts: How many answers may be asked for in total, including the first. The cap
            is what stops a validation loop becoming unbounded spend.

    Returns:
        The validated answer, or an `Abstention` where the guard allows one.

    Raises:
        OutputValidationError: Where the last attempt still did not satisfy the type or the
            rules. Carries the failing paths, the violated rule identifiers and the attempt
            count, and the raw payload for a debugger.
        GuardrailEvaluationError: Where a rule could not run.

    Example:
        >>> validated.__name__
        'validated'
    """
    answer = content
    cap = max(attempts, 1)
    attempt = 0
    while True:
        attempt += 1
        judged = _judge(answer, schema=schema, policy=policy, attempt=attempt)
        if not isinstance(judged, _Rejected):
            return judged
        if reask is None or attempt >= cap:
            raise judged.error
        answer = await reask(judged.correction)


@dataclass(frozen=True, slots=True)
class _Rejected:
    """Why this attempt failed, and what to send back if there is another one."""

    error: OutputValidationError
    correction: str


def _judge(
    content: str,
    *,
    schema: SchemaGuard,
    policy: PolicyGuard | None,
    attempt: int,
) -> BaseModel | _Rejected:
    try:
        parsed = schema.parse(content)
    except OutputValidationError as refused:
        return _Rejected(
            error=_at(refused, attempt),
            correction=schema.contract.repair_prompt(_as_violation(refused)),
        )
    if policy is None or isinstance(parsed, Abstention):
        return parsed
    report = policy.check(parsed)
    if not report.rejected:
        return parsed
    return _Rejected(
        error=_at(_broke(report, parsed), attempt),
        correction=policy.correction(report),
    )


def _at(refused: OutputValidationError, attempt: int) -> OutputValidationError:
    refused.attempts = attempt
    refused.details["attempts"] = str(attempt)
    return refused


def _broke(report: PolicyReport, result: BaseModel) -> OutputValidationError:
    return OutputValidationError(
        f"answer violated {', '.join(report.policies)}",
        model=type(result).__name__,
        policies=report.policies,
        paths=_paths(report),
        problems={violation.path: violation.detail for violation in report.violations},
        guard="output_policy",
    )


def _paths(report: PolicyReport) -> tuple[str, ...]:
    return tuple(sorted({violation.path for violation in report.violations if violation.path}))


def _as_violation(refused: OutputValidationError) -> SchemaViolationError:
    return SchemaViolationError(
        str(refused),
        model=refused.model,
        paths=refused.paths,
        problems=refused.problems,
        payload=refused.payload,
    )
