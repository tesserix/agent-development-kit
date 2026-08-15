"""Compressing admitted content by what it is, rather than by how long it is.

Tool output is the largest and least dense thing in a prompt, and the prompt is rebuilt
every turn, so a hundred-row result is paid for a hundred times. Most of what it costs is
repeated keys, alignment whitespace and identical rows — none of which the model reads.

A generic summariser cannot remove those safely: applied to JSON it eats the field names
needed to write the next query, applied to code it eats the identifiers. So content is
classified first and dispatched to a compressor that understands it, and anything that
cannot be classified with confidence is admitted whole. Paying full price for content is
worse than paying nothing for content a wrong compressor destroyed.

Nothing here runs the content, calls out, or unwraps an untrusted marker: compression is
not sanitisation, and a compressed payload is exactly as untrusted as what it came from.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field

from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic import JsonValue

__all__ = [
    "CodeCompressor",
    "Compressed",
    "Compressor",
    "ContentKind",
    "ContentRouter",
    "PassThrough",
    "ProseCompressor",
    "SizeCounter",
    "StructuredCompressor",
    "TabularCompressor",
    "classify",
    "estimate_tokens",
]

type SizeCounter = Callable[[str], int]
"""How many tokens a string occupies, answered by whatever will read it."""

_COMPACT = (",", ":")
_COLUMNS = re.compile(r"\t|\s{2,}|\s*\|\s*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")


def estimate_tokens(content: str) -> int:
    """Approximate the tokens in `content` without a tokeniser.

    Four characters to a token is the usual English rule of thumb. It is used only to
    compare a before against an after, so its bias cancels; pass a real counter to
    `ContentRouter` where the ratio is being reported to someone paying for it.

    Args:
        content: The text to size.

    Returns:
        A stable estimate. Zero only for the empty string.

    Example:
        >>> estimate_tokens("")
        0
        >>> estimate_tokens("a token is about four characters")
        8
    """
    return -(-len(content) // 4)


class ContentKind(StrEnum):
    """What admitted content turned out to be."""

    JSON = "json"
    """Parsed as JSON, so its structure is known rather than guessed."""

    TABULAR = "tabular"
    """Rows of consistently delimited columns: a query result, a listing, a report."""

    CODE = "code"
    """Parsed as a Python syntax tree carrying definitions or imports."""

    PROSE = "prose"
    """Sentences of natural language."""

    UNKNOWN = "unknown"
    """Nothing could be established. It is admitted untouched."""


def classify(content: str) -> ContentKind:
    """Decide what `content` is, or decline to.

    The content is parsed, never executed, and references inside it are never followed.
    Anything that half-looks like a type it does not parse as is `UNKNOWN`, because that
    is exactly the content a compressor would mangle.

    Args:
        content: The admitted text.

    Returns:
        The kind, or `ContentKind.UNKNOWN` where confidence was not available.

    Example:
        >>> classify('[{"id": 1}]')
        <ContentKind.JSON: 'json'>
        >>> classify('{"id": 1, "region": ')
        <ContentKind.UNKNOWN: 'unknown'>
    """
    text = content.strip()
    if not text:
        return ContentKind.UNKNOWN
    if text[0] in "{[":
        return ContentKind.JSON if _json(text) is not None else ContentKind.UNKNOWN
    if _is_code(text):
        return ContentKind.CODE
    if _is_tabular(text):
        return ContentKind.TABULAR
    return ContentKind.PROSE if _is_prose(text) else ContentKind.UNKNOWN


def _json(text: str) -> JsonValue | None:
    """The parsed value, or None where it is not JSON. Parsing is not evaluation."""
    try:
        parsed: JsonValue = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return parsed


def _is_code(text: str) -> bool:
    """Whether this parses as Python and declares something. An expression is not code."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError):
        return False
    declarations = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    return any(isinstance(node, declarations) for node in ast.walk(tree))


def _rows(text: str) -> list[list[str]]:
    """Lines split into columns on tabs, pipes or run-on spaces."""
    return [_COLUMNS.split(line.strip()) for line in text.splitlines() if line.strip()]


def _is_tabular(text: str) -> bool:
    """Whether every line has the same two-or-more columns under one delimiter."""
    rows = _rows(text)
    widths = {len(row) for row in rows}
    return len(rows) >= 2 and widths == {len(rows[0])} and len(rows[0]) >= 2


def _is_prose(text: str) -> bool:
    """Whether this reads as sentences: mostly words, and ended like a sentence."""
    words = _WORD.findall(text)
    if not words:
        return False
    return sum(len(word) for word in words) / len(text) > 0.5


@runtime_checkable
class Compressor(Protocol):
    """Something that makes one kind of content smaller without losing what it carries.

    Implementations are pure and offline: the same bytes in produce the same bytes out,
    with no network call and no state, or the provider's prefix cache cannot hold across
    turns. Raising is allowed — the router falls back rather than propagating.
    """

    name: str
    """What the admission record calls this compressor."""

    def compress(self, content: str, *, budget_tokens: int) -> str:
        """Return `content` made smaller.

        Args:
            content: The admitted text, of the kind this compressor handles.
            budget_tokens: What the caller would like it to fit in. Advisory: a
                compressor that cannot reach it without losing content returns what it
                has and lets the caller decide.

        Returns:
            The compressed text.

        Raises:
            Exception: Where the content was not what it appeared to be.
        """
        ...


class StructuredCompressor:
    """JSON, factored: shared fields hoisted, rows deduplicated, whitespace gone.

    A homogeneous array of objects becomes a field list and a list of value rows, which
    removes one copy of every key per row. Fields identical across every row are written
    once under `shared`, and identical rows are folded with a repeat count. Every distinct
    field name and every distinct value remains present — this is a re-encoding, not a
    summary, so the model can still write the next query from it.
    """

    name = "structured"

    def compress(self, content: str, *, budget_tokens: int) -> str:
        """Re-encode JSON content. Anything that is not row-shaped loses its whitespace."""
        del budget_tokens
        parsed = json.loads(content)
        table = _table(parsed)
        if table is None:
            return json.dumps(parsed, separators=_COMPACT)
        return json.dumps(table, separators=_COMPACT)


def _table(parsed: JsonValue) -> JsonValue | None:
    """The row-factored form of a homogeneous array of objects, where it is one."""
    if not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if not all(isinstance(row, dict) for row in parsed):
        return None
    rows: list[dict[str, JsonValue]] = [row for row in parsed if isinstance(row, dict)]
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        return None
    shared = {
        field: rows[0][field]
        for field in fields
        if all(row[field] == rows[0][field] for row in rows)
    }
    varying = [field for field in fields if field not in shared]
    folded, repeats = _folded([[row[field] for field in varying] for row in rows])
    table: dict[str, JsonValue] = {"fields": list(varying), "rows": folded}
    if shared:
        table["shared"] = shared
    if any(count > 1 for count in repeats):
        table["repeats"] = list(repeats)
    return table


def _folded(rows: list[list[JsonValue]]) -> tuple[list[JsonValue], list[int]]:
    """Consecutive identical rows collapsed to one, with how many times each stood."""
    kept: list[JsonValue] = []
    repeats: list[int] = []
    for row in rows:
        if kept and kept[-1] == row:
            repeats[-1] += 1
            continue
        kept.append(list(row))
        repeats.append(1)
    return kept, repeats


class TabularCompressor:
    """Aligned columns, unaligned: single-space delimiters and no repeated rows.

    Alignment whitespace is presentation for a human terminal and is charged for as
    tokens. The columns and their order are kept, so what the model reads is the same
    table with the padding taken out.
    """

    name = "tabular"

    def compress(self, content: str, *, budget_tokens: int) -> str:
        """Normalise the delimiters and fold consecutive identical rows."""
        del budget_tokens
        rows = _rows(content)
        if len(rows) < 2:
            return content
        lines: list[str] = []
        repeats = 0
        for row in rows:
            line = " ".join(cell for cell in row if cell)
            if lines and line == lines[-1].split(" x")[0]:
                repeats += 1
                lines[-1] = f"{line} x{repeats + 1}"
                continue
            repeats = 0
            lines.append(line)
        return "\n".join(lines)


class CodeCompressor:
    """Signatures, types and errors kept; bodies elided.

    Driven by the syntax tree rather than by a regex, because a regex over source finds
    the word `def` inside a string literal and stops finding it inside a decorator. What
    a reader needs from code they are calling is the shape of the call and what it can
    raise; the loop in the middle is not read.
    """

    name = "code"

    def compress(self, content: str, *, budget_tokens: int) -> str:
        """Return the module with function bodies replaced by their docstring and raises.

        Raises:
            SyntaxError: If the content does not parse. The router falls back.
        """
        del budget_tokens
        tree = ast.parse(content)
        return ast.unparse(_Elided().visit(tree))


class _Elided(ast.NodeTransformer):
    """Replaces a function body with the parts of it a caller reads."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Keep the signature, the docstring and every raise; drop the rest."""
        node.body = _kept(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """The same, for a coroutine."""
        node.body = _kept(node.body)
        return node


def _kept(body: Sequence[ast.stmt]) -> list[ast.stmt]:
    """A docstring, every raise anywhere inside, and an ellipsis standing for the rest."""
    kept: list[ast.stmt] = []
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        kept.append(first)
    kept.append(ast.Expr(value=ast.Constant(value=...)))
    kept.extend(
        node for statement in body for node in ast.walk(statement) if isinstance(node, ast.Raise)
    )
    return kept


class ProseCompressor:
    """Whitespace collapsed and repeated sentences dropped, in the order first seen.

    Deliberately conservative: no model call, no paraphrase, nothing removed that was not
    already said. Where that still does not reach the budget, what is left is truncated at
    a sentence boundary and marked, rather than cut mid-word.
    """

    name = "prose"

    def compress(self, content: str, *, budget_tokens: int) -> str:
        """Drop repeated sentences, then trim to the budget at a sentence boundary."""
        seen: dict[str, None] = {}
        for sentence in _SENTENCE.split(" ".join(content.split())):
            if sentence:
                seen.setdefault(sentence, None)
        kept: list[str] = []
        for sentence in seen:
            kept.append(sentence)
            if estimate_tokens(" ".join([*kept, "…"])) > budget_tokens:
                kept[-1] = "…"
                break
        return " ".join(kept)


class PassThrough:
    """The terminal fallback: content admitted exactly as it arrived."""

    name = "pass-through"

    def compress(self, content: str, *, budget_tokens: int) -> str:
        """Return the content unchanged."""
        del budget_tokens
        return content


DEFAULT_COMPRESSORS: Mapping[ContentKind, Compressor] = {
    ContentKind.JSON: StructuredCompressor(),
    ContentKind.TABULAR: TabularCompressor(),
    ContentKind.CODE: CodeCompressor(),
    ContentKind.PROSE: ProseCompressor(),
}
"""What each kind is compressed by unless the caller says otherwise."""


class Compressed(AdkModel):
    """One admission: what went in, what came out, and by what.

    Args:
        content: What is admitted to the prompt.
        kind: What the content was classified as.
        compressor: Which compressor produced it, or `pass-through`.
        original_tokens: The size before, in whatever the router counts in.
        compressed_tokens: The size after.
        reason: Why nothing was compressed, where nothing was. Empty on the happy path.
        untrusted: Whether this content is still untrusted. Compression never changes it.
    """

    content: str = ""
    kind: ContentKind = ContentKind.UNKNOWN
    compressor: str = PassThrough.name
    original_tokens: int = Field(default=0, ge=0)
    compressed_tokens: int = Field(default=0, ge=0)
    reason: str = ""
    untrusted: bool = False

    @property
    def ratio(self) -> float:
        """What fraction of the original this costs. 1.0 where nothing was compressed."""
        if not self.original_tokens:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def saved_tokens(self) -> int:
        """How many tokens every future turn no longer pays for. Never negative."""
        return max(0, self.original_tokens - self.compressed_tokens)


class ContentRouter:
    """Classifies admitted content and hands it to the compressor that understands it.

    Args:
        compressors: What handles each kind. A kind with nothing registered passes
            through, so removing a compressor is safe rather than fatal.
        count: How tokens are counted. The estimate by default; pass the provider's
            counter where the ratio is being reported as money.
        threshold_tokens: Content at or below this size skips the router entirely —
            classifying a two-line result costs more than compressing it saves.
    """

    def __init__(
        self,
        *,
        compressors: Mapping[ContentKind, Compressor] | None = None,
        count: SizeCounter = estimate_tokens,
        threshold_tokens: int = 256,
    ) -> None:
        self._compressors = DEFAULT_COMPRESSORS if compressors is None else compressors
        self._count = count
        self._threshold = threshold_tokens

    def admit(self, content: str, *, budget_tokens: int, untrusted: bool = False) -> Compressed:
        """Compress `content` by what it is, and record what happened either way.

        Nothing raises out of here. Every way compression can fail to help ends in the
        original content being admitted with the reason recorded against it.

        Args:
            content: The text being admitted to context.
            budget_tokens: What the caller would like it to fit in.
            untrusted: Whether the content came from outside. Carried onto the result
                unchanged — compressing a payload does not make it safe.

        Returns:
            What to admit, with the compressor, the sizes and the ratio on it.
        """
        before = self._count(content)
        if before <= self._threshold:
            return self._left(content, before, untrusted, "below the compression threshold")
        kind = classify(content)
        compressor = self._compressors.get(kind)
        if compressor is None:
            return self._left(content, before, untrusted, f"no compressor for {kind}", kind=kind)
        try:
            compressed = compressor.compress(content, budget_tokens=budget_tokens)
        # A compressor's failure is the router's problem, never the caller's.
        except Exception as failure:
            reason = f"{compressor.name} failed: {type(failure).__name__}"
            return self._left(content, before, untrusted, reason, kind=kind)
        after = self._count(compressed)
        if after >= before:
            reason = f"{compressor.name} expanded its input"
            return self._left(content, before, untrusted, reason, kind=kind)
        return Compressed(
            content=compressed,
            kind=kind,
            compressor=compressor.name,
            original_tokens=before,
            compressed_tokens=after,
            untrusted=untrusted,
        )

    def _left(
        self,
        content: str,
        tokens: int,
        untrusted: bool,
        reason: str,
        *,
        kind: ContentKind = ContentKind.UNKNOWN,
    ) -> Compressed:
        """The content as it arrived, with why it was left that way."""
        return Compressed(
            content=content,
            kind=kind,
            compressor=PassThrough.name,
            original_tokens=tokens,
            compressed_tokens=tokens,
            reason=reason,
            untrusted=untrusted,
        )
