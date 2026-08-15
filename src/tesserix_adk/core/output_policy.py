"""Business rules about an answer, decided in code rather than by the model that wrote it.

A schema says a quote has a total. It does not say the total is inside the band this tenant
sells at, and asking the model to check its own arithmetic is asking the thing that got it
wrong to mark its own work. A policy is a deterministic function of the parsed result, so
the same answer is judged the same way on every provider and every replay.

A violation names the policy and the path and never the value: a refusal that quotes the
out-of-band total has published the out-of-band total.

Abstention is a first-class outcome. Where "I do not know" is not expressible, a model that
does not know invents something that validates, which is the worst of the failure modes
because it is the one nothing downstream can detect.
"""

from __future__ import annotations

import unicodedata
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from tesserix_adk.core.errors import GuardrailEvaluationError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "Abstention",
    "Bounded",
    "Invariant",
    "OneOf",
    "Policy",
    "PolicyReport",
    "PolicyViolation",
    "RequiresCitation",
    "evaluate",
    "reach",
]


class Abstention(AdkModel):
    """An answer of "I do not know", typed rather than a plausible invented alternative.

    Args:
        abstained: Always true, and required: it is what tells an abstention apart from an
            answer that happens to carry a reason.
        reason: Why, in the model's words. Shown to a caller, never parsed by the kit.

    Example:
        >>> Abstention(abstained=True, reason="no fare data for that route").abstained
        True
    """

    abstained: Literal[True]
    reason: str = ""


class PolicyViolation(AdkModel):
    """One rule an answer broke.

    Args:
        policy: Which rule, so a caller can match on the identifier rather than the prose.
        path: Where in the answer, dotted. Empty where the rule is about the whole answer.
        detail: What was wrong, in terms of the rule. Never the offending value.

    Example:
        >>> PolicyViolation(policy="total_in_band", path="total").detail
        ''
    """

    policy: str
    path: str = ""
    detail: str = ""


@runtime_checkable
class Policy(Protocol):
    """A deterministic rule about a parsed answer.

    A consumer's own rule — a cross-field invariant, a lookup against a rate card — plugs in
    beside the built-ins. It must not call a model: a rule enforced by a model is a
    suggestion.
    """

    name: str

    def check(self, result: BaseModel) -> tuple[PolicyViolation, ...]:
        """Every violation in this answer. Empty where the rule is satisfied."""
        ...


def reach(result: BaseModel, path: str) -> object:
    """The value at a dotted `path` inside `result`.

    Args:
        result: The parsed answer.
        path: A dotted field path, `total` or `quote.total`.

    Returns:
        The value, or `None` where the path runs through a field that is itself `None`.

    Raises:
        GuardrailEvaluationError: Where the path does not exist on the type. A policy
            pointed at a field that was renamed silently passes otherwise, which is a rule
            that has stopped running and nobody was told.

    Example:
        >>> class Quote(BaseModel):
        ...     total: int
        >>> reach(Quote(total=90), "total")
        90
    """
    found: object = result
    for part in path.split("."):
        if found is None:
            return None
        if not hasattr(found, part):
            raise GuardrailEvaluationError(
                f"no field {path!r} on {type(result).__name__}",
                guard="output_policy",
                reason="unknown_path",
            )
        found = getattr(found, part)
    return found


class Bounded:
    """A numeric field that has to stay inside a band.

    Comparison is decimal, so a provider returning `90.10` as a string and another returning
    it as a float are judged the same way.

    Args:
        path: The dotted field.
        minimum: The lowest acceptable value, inclusive. None for no floor.
        maximum: The highest acceptable value, inclusive. None for no ceiling.
        required: Whether a missing value is itself a violation. False by default, so a
            legitimately empty result is expressible.
        name: The identifier a caller matches on.

    Example:
        >>> class Quote(BaseModel):
        ...     total: int
        >>> Bounded("total", maximum=100).check(Quote(total=140))[0].policy
        'total_within_band'
    """

    def __init__(
        self,
        path: str,
        *,
        minimum: Decimal | int | float | None = None,
        maximum: Decimal | int | float | None = None,
        required: bool = False,
        name: str = "",
    ) -> None:
        self.path = path
        self.minimum = None if minimum is None else Decimal(str(minimum))
        self.maximum = None if maximum is None else Decimal(str(maximum))
        self.required = required
        self.name = name or f"{path.replace('.', '_')}_within_band"

    def check(self, result: BaseModel) -> tuple[PolicyViolation, ...]:
        """A violation where the value is outside the band, or missing and required."""
        value = reach(result, self.path)
        if value is None:
            return self._missing()
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return (self._violation("is not a number"),)
        if self.minimum is not None and number < self.minimum:
            return (self._violation(f"is below the floor of {self.minimum}"),)
        if self.maximum is not None and number > self.maximum:
            return (self._violation(f"is above the ceiling of {self.maximum}"),)
        return ()

    def _missing(self) -> tuple[PolicyViolation, ...]:
        return (self._violation("is required and was not answered"),) if self.required else ()

    def _violation(self, detail: str) -> PolicyViolation:
        return PolicyViolation(policy=self.name, path=self.path, detail=detail)


class OneOf:
    """A field that has to be one of a known set.

    Text is compared in NFC, so a provider that returns a composed character and one that
    returns the decomposed form do not disagree about membership.

    Args:
        path: The dotted field.
        allowed: What it may be.
        required: Whether a missing value is itself a violation.
        name: The identifier a caller matches on.

    Example:
        >>> class Answer(BaseModel):
        ...     status: str
        >>> OneOf("status", ("open", "closed")).check(Answer(status="pending"))[0].path
        'status'
    """

    def __init__(
        self,
        path: str,
        allowed: Sequence[str],
        *,
        required: bool = False,
        name: str = "",
    ) -> None:
        self.path = path
        self.allowed = tuple(_folded(value) for value in allowed)
        self.required = required
        self.name = name or f"{path.replace('.', '_')}_is_known"

    def check(self, result: BaseModel) -> tuple[PolicyViolation, ...]:
        """A violation where the value is outside the set, or missing and required."""
        value = reach(result, self.path)
        if value is None:
            if self.required:
                return (self._violation("is required and was not answered"),)
            return ()
        if _folded(str(value)) not in self.allowed:
            return (self._violation(f"is not one of the {len(self.allowed)} allowed values"),)
        return ()

    def _violation(self, detail: str) -> PolicyViolation:
        return PolicyViolation(policy=self.name, path=self.path, detail=detail)


class RequiresCitation:
    """An assertion drawn from retrieved content has to say where it came from.

    Without it a run that retrieves and a run that guesses produce the same-shaped answer,
    and the difference between them is exactly what a reader needs.

    Args:
        path: The field carrying the assertion.
        citations: The field carrying its sources.
        name: The identifier a caller matches on.

    Example:
        >>> class Answer(BaseModel):
        ...     claim: str
        ...     sources: tuple[str, ...] = ()
        >>> RequiresCitation("claim", "sources").check(Answer(claim="it closes at six"))[0].policy
        'claim_is_cited'
    """

    def __init__(self, path: str, citations: str = "citations", *, name: str = "") -> None:
        self.path = path
        self.citations = citations
        self.name = name or f"{path.replace('.', '_')}_is_cited"

    def check(self, result: BaseModel) -> tuple[PolicyViolation, ...]:
        """A violation where something was asserted with nothing behind it."""
        asserted = reach(result, self.path)
        if asserted is None or asserted == "":
            return ()
        cited = reach(result, self.citations)
        if cited:
            return ()
        return (
            PolicyViolation(
                policy=self.name,
                path=self.path,
                detail=f"was asserted with nothing in {self.citations!r} behind it",
            ),
        )


class Invariant[ModelT: BaseModel]:
    """A rule about the answer as a whole, for the ones that are not about one field.

    Args:
        name: The identifier a caller matches on.
        holds: The rule. True where the answer is acceptable.
        detail: What it means when it does not hold. Never the value.
        path: The field a reader should look at first, where there is one.

    Example:
        >>> class Quote(BaseModel):
        ...     net: int
        ...     gross: int
        >>> rule = Invariant[Quote]("covers_net", lambda q: q.gross >= q.net, "gross below net")
        >>> rule.check(Quote(net=100, gross=90))[0].detail
        'gross below net'
    """

    def __init__(
        self,
        name: str,
        holds: Callable[[ModelT], bool],
        detail: str,
        *,
        path: str = "",
    ) -> None:
        self.name = name
        self.holds = holds
        self.detail = detail
        self.path = path

    def check(self, result: BaseModel) -> tuple[PolicyViolation, ...]:
        """A violation where the rule does not hold."""
        if self.holds(cast("ModelT", result)):
            return ()
        return (PolicyViolation(policy=self.name, path=self.path, detail=self.detail),)


class PolicyReport(AdkModel):
    """Kept for the caller that asked what was violated, without what was answered.

    Args:
        violations: Every rule the answer broke, in the order the policies were declared.

    Example:
        >>> PolicyReport().rejected
        False
    """

    violations: tuple[PolicyViolation, ...] = Field(default_factory=tuple)

    @property
    def rejected(self) -> bool:
        """Whether anything was violated at all."""
        return bool(self.violations)

    @property
    def policies(self) -> tuple[str, ...]:
        """Which rules, sorted, for a caller matching on identifiers."""
        return tuple(sorted({violation.policy for violation in self.violations}))


def evaluate(result: BaseModel, policies: Sequence[Policy]) -> PolicyReport:
    """Every rule this answer breaks, in declaration order.

    An abstention is not judged: a model that said it does not know has not quoted a total
    outside the band, and rejecting the abstention teaches it to invent one.

    Args:
        result: The parsed answer.
        policies: The rules, in the order they were declared.

    Returns:
        What was violated, and nothing about what was answered.

    Raises:
        GuardrailEvaluationError: Where a policy raises. A rule that could not run is not a
            rule that passed.

    Example:
        >>> evaluate(Abstention(abstained=True), ()).rejected
        False
    """
    if isinstance(result, Abstention):
        return PolicyReport()
    found: list[PolicyViolation] = []
    for policy in policies:
        found.extend(_asked(policy, result))
    return PolicyReport(violations=tuple(found))


def _asked(policy: Policy, result: BaseModel) -> tuple[PolicyViolation, ...]:
    try:
        return policy.check(result)
    except GuardrailEvaluationError:
        raise
    except Exception as broke:
        raise GuardrailEvaluationError(
            f"policy {policy.name!r} could not be evaluated",
            guard="output_policy",
            reason="raised",
        ) from broke


def _folded(value: str) -> str:
    return unicodedata.normalize("NFC", value)
