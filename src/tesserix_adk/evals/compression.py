"""Measuring what compression saves and what it costs, in the same run.

A compression ratio on its own is not a result. Any ratio is achievable by deleting more,
and the failure mode of every compression system is a change that improves the headline
number while quietly costing accuracy on the cases nobody put in the suite.

So savings and task accuracy are measured together, per content type, against floors
checked into the repository. An aggregate improvement never masks a per-type regression:
the verdict is the conjunction of the kinds, so JSON getting better cannot pay for code
getting worse. Retrieval is an outcome in its own right — a case whose answer lives in the
elided detail has to go and fetch it, and answering from memory is a failure even when the
answer is right, because next time it will not be.

The fixtures are synthetic, deterministic and offline. Nothing here calls a model or a
network; the solver is consumer code, and a solver that raises fails its case rather than
scoring it.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.errors import ConfigurationError, EvalIncompleteError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.memory.compression import ContentKind, PassThrough, estimate_tokens

if TYPE_CHECKING:
    from tesserix_adk.memory.reversible import ReversibleRouter

__all__ = [
    "DEFAULT_FIXTURES",
    "DEFAULT_FLOORS",
    "Answer",
    "CaseMeasurement",
    "CompressionCase",
    "CompressionFixtures",
    "CompressionReport",
    "Floor",
    "FloorPolicy",
    "KindReport",
    "Solver",
    "measure_compression",
]

Outcome = Literal["kept", "recovered", "lost", "unmeasured"]


class CompressionCase(AdkModel):
    """One fixture: content, a question about it, and what a right answer contains.

    Args:
        id: Stable across edits, since it is what a failing report names.
        kind: What the content is, so the report can be read per content type.
        content: The content admitted to the prompt. Synthetic — a fixture carrying real
            customer data turns a test suite into a data store.
        question: What is asked about it.
        expected: What a right answer contains.
        needs_detail: Whether the answer lives in detail a compressor may elide. Such a
            case is expected to trigger retrieval, and not retrieving is a failure even
            where the answer happens to be right.
        tags: Labels a run can select on.
    """

    id: str = Field(min_length=1)
    kind: ContentKind
    content: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    needs_detail: bool = False
    tags: tuple[str, ...] = ()


class CompressionFixtures(AdkModel):
    """The fixture set, versioned so a result names the corpus it was measured on.

    Args:
        name: What the set is called.
        version: Its version. Numbers only compare within one.
        cases: The fixtures, in a fixed order.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    cases: tuple[CompressionCase, ...] = ()

    @model_validator(mode="after")
    def _one_case_per_id(self) -> CompressionFixtures:
        """Refuse a duplicate id, which would make a failing case ambiguous."""
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("a fixture set holds one case per id")
        return self


class Answer(AdkModel):
    """What a solver produced, and what it had to fetch to produce it.

    Args:
        text: The answer.
        expanded: The handles it expanded on the way, which is how retrieval is measured.
    """

    text: str = ""
    expanded: tuple[str, ...] = ()


@runtime_checkable
class Solver(Protocol):
    """Whatever answers a fixture's question from the content in front of it.

    It is consumer code — a real agent, a scripted provider, or a reader — because the
    kit cannot know what will read the compressed content in production.
    """

    async def answer(self, case: CompressionCase, content: str, *, handle: str) -> Answer:
        """Answer `case` from `content`, expanding `handle` where the detail is not there."""
        ...


class Floor(AdkModel):
    """The least a content type may do before the build fails.

    Args:
        kind: The content type.
        accuracy: The share of its cases that must come out right, retrieval included.
        savings: The share of tokens compression must remove. A floor above zero is what
            makes a compressor silently becoming a pass-through visible.
    """

    kind: ContentKind
    accuracy: float = Field(ge=0.0, le=1.0)
    savings: float = Field(default=0.0, ge=0.0, le=1.0)


class FloorPolicy(AdkModel):
    """The floors this project holds itself to, versioned so lowering one is reviewable.

    Args:
        version: The floors' version, reported alongside every result.
        floors: One per content type judged. A type with no floor is measured and
            reported, and decides nothing.
    """

    version: str = Field(min_length=1)
    floors: tuple[Floor, ...] = ()

    @model_validator(mode="after")
    def _one_floor_per_kind(self) -> FloorPolicy:
        """Refuse two floors for one kind, which would decide by ordering."""
        kinds = [floor.kind for floor in self.floors]
        if len(kinds) != len(set(kinds)):
            raise ConfigurationError("a policy declares one floor per content type")
        return self

    def floor(self, kind: ContentKind) -> Floor | None:
        """The floor for `kind`, or `None` where this project declared none."""
        return next((each for each in self.floors if each.kind == kind), None)


class CaseMeasurement(AdkModel):
    """One fixture, measured on both axes at once.

    Args:
        case_id: The fixture.
        kind: Its content type.
        compressor: What compressed it, or `pass-through`.
        tokens_before: Its size admitted whole.
        tokens_after: Its size as admitted.
        correct: Whether the answer from the compressed content was right.
        correct_whole: Whether the answer from the uncompressed content was right, so a
            case the solver could never answer is not blamed on compression.
        retrieved: Whether the solver expanded the handle.
        needs_detail: Whether it had to.
        outcome: `kept` where compression cost nothing, `recovered` where retrieval put it
            back, `lost` where the answer or the retrieval went missing, `unmeasured`
            where the solver raised.
        reason: Why, where the outcome was not `kept`.
    """

    case_id: str
    kind: ContentKind
    compressor: str = PassThrough.name
    tokens_before: int = Field(default=0, ge=0)
    tokens_after: int = Field(default=0, ge=0)
    correct: bool = False
    correct_whole: bool = False
    retrieved: bool = False
    needs_detail: bool = False
    outcome: Outcome = "kept"
    reason: str = ""

    @property
    def removed_tokens(self) -> int:
        """How many tokens compression took out."""
        return self.tokens_before - self.tokens_after

    @property
    def savings(self) -> float:
        """The share of tokens removed. `0.0` where nothing was."""
        if not self.tokens_before:
            return 0.0
        return self.removed_tokens / self.tokens_before

    @property
    def passed(self) -> bool:
        """Whether this case survived compression, retrieval included."""
        return self.outcome in {"kept", "recovered"}


class KindReport(AdkModel):
    """One content type: what it saved, what it cost, and whether that clears its floor.

    Args:
        kind: The content type.
        n: How many cases it holds.
        savings: The mean share of tokens removed across them.
        accuracy: The share that survived, retrieval included.
        pass_through: How many were admitted whole, which is how a compressor that stopped
            compressing shows up.
        floor: The floor judged against, `None` where the project declared none.
        verdict: `pass`, `warn` where nothing was declared, `fail` otherwise.
        cases: The case ids that did not survive.
        reason: What a reviewer needs to know, empty where the type passed cleanly.
    """

    kind: ContentKind
    n: int = Field(default=0, ge=0)
    savings: float = 0.0
    accuracy: float = 0.0
    pass_through: int = Field(default=0, ge=0)
    floor: Floor | None = None
    verdict: Literal["pass", "warn", "fail"] = "pass"
    cases: tuple[str, ...] = ()
    reason: str = ""

    def line(self) -> str:
        """The row the report's table prints."""
        floor = "-" if self.floor is None else f"{self.floor.accuracy:g}/{self.floor.savings:g}"
        return (
            f"| {self.kind.value} | {self.n} | {self.savings:.0%} | {self.accuracy:.0%} | "
            f"{self.pass_through} | {floor} | {self.verdict} | {self.reason} |"
        )


class CompressionReport(AdkModel):
    """Savings and accuracy per content type, and the verdict that gates the merge.

    Args:
        fixtures: The fixture set measured.
        version: Its version.
        floors: The floor policy's version, so a result names what it was judged against.
        kinds: One report per content type present, in `ContentKind` order.
        cases: Every case, so a reviewer can see what a failing type actually did.
    """

    fixtures: str
    version: str
    floors: str
    kinds: tuple[KindReport, ...] = ()
    cases: tuple[CaseMeasurement, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every content type cleared its floor."""
        return not any(measured.verdict == "fail" for measured in self.kinds)

    @property
    def exit_code(self) -> int:
        """`0` where the merge may proceed, `1` where it may not."""
        return 0 if self.ok else 1

    @property
    def savings(self) -> float:
        """The overall share of tokens removed, reported but never gating on its own."""
        before = sum(case.tokens_before for case in self.cases)
        removed = sum(case.removed_tokens for case in self.cases)
        return removed / before if before else 0.0

    def kind(self, kind: ContentKind) -> KindReport:
        """One content type's report.

        Raises:
            KeyError: Nothing in the fixture set was of that kind.
        """
        for measured in self.kinds:
            if measured.kind == kind:
                return measured
        raise KeyError(f"no {kind.value} content in {self.fixtures}")

    def case(self, case_id: str) -> CaseMeasurement:
        """One case's measurement.

        Raises:
            KeyError: No such case in the fixture set.
        """
        for measured in self.cases:
            if measured.case_id == case_id:
                return measured
        raise KeyError(f"no case {case_id!r} in {self.fixtures}")

    def failing(self) -> tuple[KindReport, ...]:
        """Every content type that fell below its floor."""
        return tuple(measured for measured in self.kinds if measured.verdict == "fail")

    def summary(self) -> str:
        """One line for a log: the total, then the types that failed by name."""
        failed = ", ".join(
            f"{measured.kind.value} {measured.accuracy:.0%}" for measured in self.failing()
        )
        verdict = f"below floor: {failed}" if failed else "every type at or above its floor"
        return (
            f"compression {self.fixtures} {self.version} on floors {self.floors}: "
            f"{self.savings:.0%} saved, {verdict}"
        )

    def table(self) -> str:
        """The per-type table, zero-savings types included."""
        head = (
            "| kind | cases | saved | accuracy | pass-through | floor acc/save | verdict | note |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        )
        return "\n".join([*head, *(measured.line() for measured in self.kinds)])

    def as_dict(self) -> dict[str, object]:
        """The machine-readable report the CI gate reads."""
        return {
            "fixtures": self.fixtures,
            "version": self.version,
            "floors": self.floors,
            "ok": self.ok,
            "savings": self.savings,
            "kinds": [measured.model_dump(mode="json") for measured in self.kinds],
            "cases": [measured.model_dump(mode="json") for measured in self.cases],
        }


async def measure_compression(
    fixtures: CompressionFixtures,
    router: ReversibleRouter,
    solver: Solver,
    *,
    floors: FloorPolicy,
    tenant: str = "evals",
    run_id: str = "compression-gate",
) -> CompressionReport:
    """Measure savings and accuracy over the fixture set and judge every content type.

    Args:
        fixtures: The corpus, spanning the content types the router routes.
        router: What compresses, and what hands back the original when a case needs it.
        solver: What answers the questions. Consumer code, so it is allowed to raise.
        floors: What each content type must clear.
        tenant: The isolation boundary the admissions happen in.
        run_id: The run they belong to, which scopes the retained originals.

    Returns:
        The report, whose `exit_code` is `1` where any content type fell below its floor.

    Raises:
        EvalIncompleteError: A floor names a content type the fixture set does not cover,
            so the gate would report a pass on a type it never measured.
    """
    covered = {case.kind for case in fixtures.cases}
    missing = sorted(floor.kind.value for floor in floors.floors if floor.kind not in covered)
    if missing:
        raise EvalIncompleteError(
            f"{fixtures.name} has no {', '.join(missing)} cases, so their floors cannot be judged",
            coverage=len(covered) / (len(covered) + len(missing)),
        )
    measured = [
        await _measure(case, router, solver, tenant=tenant, run_id=run_id)
        for case in fixtures.cases
    ]
    kinds = tuple(
        _kind(kind, [case for case in measured if case.kind == kind], floors.floor(kind))
        for kind in ContentKind
        if any(case.kind == kind for case in measured)
    )
    return CompressionReport(
        fixtures=fixtures.name,
        version=fixtures.version,
        floors=floors.version,
        kinds=kinds,
        cases=tuple(measured),
    )


async def _measure(
    case: CompressionCase,
    router: ReversibleRouter,
    solver: Solver,
    *,
    tenant: str,
    run_id: str,
) -> CaseMeasurement:
    """Admit one case compressed and whole, and compare what came back."""
    admitted = await router.admit(
        case.content, budget_tokens=_budget(case), tenant=tenant, run_id=run_id
    )
    try:
        whole = await solver.answer(case, case.content, handle="")
        compressed = await solver.answer(case, admitted.content, handle=admitted.handle)
    # A solver is consumer code: its failure is a case nobody measured, not a case that failed.
    except Exception as broken:
        return CaseMeasurement(
            case_id=case.id,
            kind=case.kind,
            compressor=admitted.compressor,
            tokens_before=admitted.original_tokens,
            tokens_after=admitted.compressed_tokens,
            needs_detail=case.needs_detail,
            outcome="unmeasured",
            reason=f"the solver raised {type(broken).__name__}",
        )
    correct = _right(case, compressed)
    retrieved = admitted.handle != "" and admitted.handle in compressed.expanded
    outcome, reason = _outcome(case, correct=correct, retrieved=retrieved)
    return CaseMeasurement(
        case_id=case.id,
        kind=case.kind,
        compressor=admitted.compressor,
        tokens_before=admitted.original_tokens,
        tokens_after=admitted.compressed_tokens,
        correct=correct,
        correct_whole=_right(case, whole),
        retrieved=retrieved,
        needs_detail=case.needs_detail,
        outcome=outcome,
        reason=reason,
    )


def _budget(case: CompressionCase) -> int:
    """What a compressor is asked to fit the case into.

    A budget the content already fits measures nothing, so the gate squeezes hard: a
    thirty-second of what the content costs whole, and never fewer than eight tokens.
    """
    return max(estimate_tokens(case.content) // 32, 8)


def _right(case: CompressionCase, answer: Answer) -> bool:
    """Whether the answer contains what the case says a right one contains."""
    return case.expected.casefold() in answer.text.casefold()


def _outcome(case: CompressionCase, *, correct: bool, retrieved: bool) -> tuple[Outcome, str]:
    """What the case did, with retrieval treated as an outcome rather than a detail."""
    if case.needs_detail and not retrieved:
        return "lost", "the answer needed elided detail and nothing was retrieved"
    if not correct:
        return "lost", "the answer no longer contains what the case asks for"
    return ("recovered", "") if retrieved else ("kept", "")


def _kind(kind: ContentKind, cases: list[CaseMeasurement], floor: Floor | None) -> KindReport:
    """One content type's numbers, judged against its floor where it has one."""
    savings = sum(case.savings for case in cases) / len(cases)
    accuracy = sum(1 for case in cases if case.passed) / len(cases)
    failing = tuple(case.case_id for case in cases if not case.passed)
    report = KindReport(
        kind=kind,
        n=len(cases),
        savings=savings,
        accuracy=accuracy,
        pass_through=sum(1 for case in cases if case.compressor == PassThrough.name),
        floor=floor,
        cases=failing,
    )
    if floor is None:
        return report.model_copy(
            update={"verdict": "warn", "reason": "no floor declared, so it decides nothing"}
        )
    reasons = []
    if accuracy < floor.accuracy:
        reasons.append(f"accuracy {accuracy:.0%} is below the {floor.accuracy:.0%} floor")
    if savings < floor.savings:
        reasons.append(f"saved {savings:.0%}, below the {floor.savings:.0%} floor")
    if not reasons:
        return report
    return report.model_copy(update={"verdict": "fail", "reason": "; ".join(reasons)})


def _rows(count: int) -> str:
    """A synthetic result set: repeated keys, a shared field, one distinguishing row."""
    rows = [
        {"order": f"A-{index:04d}", "status": "shipped", "region": "north", "items": 2}
        for index in range(count)
    ]
    rows[-1] = {"order": "A-9999", "status": "held", "region": "north", "items": 7}
    return json.dumps(rows, indent=2)


def _table(count: int) -> str:
    """A synthetic report, aligned into columns nobody reads."""
    head = "order      status     region     items"
    body = [f"A-{index:04d}    shipped    north      2" for index in range(count)]
    body[-1] = "A-9999    held       north      7"
    return "\n".join([head, *body])


_CODE = (
    '''"""A module of the kind a tool returns whole."""

import json


def refund(order: str, amount: int) -> dict[str, object]:
    """Refund an order and return the receipt."""
    receipt = {"order": order, "amount": amount, "state": "refunded"}
    return json.loads(json.dumps(receipt))


class Ledger:
    """Where refunds are recorded."""

    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def record(self, receipt: dict[str, object]) -> None:
        """Append one receipt."""
        self.entries.append(receipt)

    def total(self) -> int:
        """The sum of every refund recorded."""
        return sum(int(entry["amount"]) for entry in self.entries)
'''
    + "\n\n"
    + "\n\n".join(
        f'def helper_{index}(value: int) -> int:\n    """Double it."""\n    return value * 2'
        for index in range(20)
    )
)

_PROSE = (
    "Refunds are issued to the original payment method within five working days. "
    "A courier manifest is checked before any order leaves the depot. "
) * 20

DEFAULT_FIXTURES = CompressionFixtures(
    name="compression",
    version="2026-08-20",
    cases=(
        CompressionCase(
            id="json-orders",
            kind=ContentKind.JSON,
            content=_rows(60),
            question="which order is held?",
            expected="A-9999",
        ),
        CompressionCase(
            id="json-held-detail",
            kind=ContentKind.JSON,
            content=_rows(120),
            question="how many items does the held order have?",
            expected='"items": 7',
            needs_detail=True,
        ),
        CompressionCase(
            id="tabular-report",
            kind=ContentKind.TABULAR,
            content=_table(60),
            question="which order is held?",
            expected="A-9999",
        ),
        CompressionCase(
            id="code-ledger",
            kind=ContentKind.CODE,
            content=_CODE,
            question="what does the ledger class expose?",
            expected="def total",
        ),
        CompressionCase(
            id="prose-policy",
            kind=ContentKind.PROSE,
            content="Refunds take five working days. " + _PROSE,
            question="how long does a refund take?",
            expected="five working days",
        ),
        CompressionCase(
            id="mixed-unknown",
            kind=ContentKind.UNKNOWN,
            content="\x00\x01 " + "".join(f"{index:x}" for index in range(600)),
            question="what is in it?",
            expected="0",
        ),
    ),
)
"""A synthetic corpus spanning every content type the router routes, offline and stable."""

DEFAULT_FLOORS = FloorPolicy(
    version="2026-08-20",
    floors=(
        Floor(kind=ContentKind.JSON, accuracy=1.0, savings=0.2),
        Floor(kind=ContentKind.TABULAR, accuracy=1.0, savings=0.2),
        Floor(kind=ContentKind.CODE, accuracy=1.0, savings=0.1),
        Floor(kind=ContentKind.PROSE, accuracy=1.0, savings=0.1),
    ),
)
"""What this kit holds its own compressors to. Lowering one is a diff, not a config tweak."""
