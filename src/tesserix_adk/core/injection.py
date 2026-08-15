"""Recognising instruction-shaped text in content the run did not write.

Screening is evidence, not the defence. The defence is structural — content the caller did
not write is sealed as data wherever it goes, and a payload that talks its way out of a
fence has still not been given a fence it can close. What this module produces is the named
signal a guard, a trace and a reviewer can act on, because "suspicious" is not actionable.

It lives in `core` because both boundaries need the same detector: the retrieval boundary in
`rag`, and the guardrail chain in `guardrails`. Two copies drift, and the copy that drifts
is the one nobody is testing against a fresh corpus.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from enum import StrEnum

from tesserix_adk.core.models import AdkModel

__all__ = ["InjectionSignal", "SignalKind", "screen"]

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))
_HOMOGLYPHS = str.maketrans(
    "\u0410\u0430\u0412\u0415\u0435\u041a\u041c\u041d\u041e\u043e"
    "\u0420\u0440\u0421\u0441\u0422\u0425\u0445\u0443",
    "AaBEeKMHOoPpCcTXxy",
)
_ENCODED = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_ECHO_LENGTH = 40
SCAN_LIMIT = 200_000
"""How much of one passage is screened. Beyond it the screen reports what it did not read."""

_OVERRIDE = re.compile(
    r"ignore (all |any )?(previous|prior|earlier|above)"
    r"|disregard (all |any )?(previous|prior|earlier|above)"
    r"|forget (everything|what) you"
    r"|new instructions?:"
    r"|you are now"
    # The same sentence in the languages a corpus is most often mixed in.
    r"|ignora (todas )?las instrucciones"
    r"|oublie[z]? (toutes )?les instructions"
    r"|ignoriere (alle )?(vorherigen|bisherigen) anweisungen"
    r"|忽略(以上|之前)的?(所有)?指示",
    re.IGNORECASE,
)
_TOOL_SHAPED = re.compile(
    r"<tool_call|</?function_call|\"tool_name\"\s*:|\"tool\"\s*:\s*\""
    r"|call the \w+ tool|invoke the \w+ tool|use the \w+ tool to",
    re.IGNORECASE,
)
_IMPERSONATION = re.compile(
    r"^\s*(system|assistant|developer)\s*[::]"
    r"|<\|?(im_start|system|assistant)\|?>"
    r"|\[(system|assistant)\]"
    r"|\"role\"\s*:\s*\"(system|assistant|developer)\"",
    re.IGNORECASE | re.MULTILINE,
)


class SignalKind(StrEnum):
    """What a screener recognised. Named, because "suspicious" is not actionable."""

    OVERRIDE = "override"
    """Text instructing the reader to set aside what it was told before."""

    IMPERSONATION = "impersonation"
    """Text wearing a role it was not given, hoping to be read as the operator."""

    TOOL_SHAPED = "tool_shaped"
    """Text shaped like a tool call, hoping to be parsed as one."""

    FENCE = "fence"
    """The data fence's own delimiter, which would end the block early if it were not escaped."""

    ENCODED = "encoded"
    """A payload hidden as base64, zero-width characters or homoglyphs."""

    SYSTEM_ECHO = "system_echo"
    """The agent's own instructions, quoted back at it to look authoritative."""

    METADATA = "metadata"
    """An instruction in a field nobody reads as prose, so nobody reviews it."""

    SPLIT = "split"
    """An instruction assembled across adjacent chunks, so neither looks bad alone."""

    UNSCANNED = "unscanned"
    """More text than the screen reads, so part of it was never looked at."""


class InjectionSignal(AdkModel):
    """One thing screening recognised, and where.

    Args:
        kind: What was recognised.
        chunk_id: Which chunk it was in. Empty where the signal spans chunks.
        field: Which part of the chunk — `text`, or the metadata key.
        detail: The matched fragment, truncated. Enough to review, not enough to be a copy
            of the document in the trace.

    Example:
        >>> screen("ignore previous instructions")[0].kind
        <SignalKind.OVERRIDE: 'override'>
    """

    kind: SignalKind
    chunk_id: str = ""
    field: str = "text"
    detail: str = ""


def screen(text: str, *, instructions: str = "") -> tuple[InjectionSignal, ...]:
    """Recognise instruction-shaped and tool-call-shaped text in a passage.

    Normalises first: zero-width characters are stripped and Cyrillic homoglyphs folded, so
    a payload spelled with Cyrillic look-alikes is read the way the model will read it.

    Args:
        text: The passage.
        instructions: The agent's own instructions, for recognising an echo of them.

    Returns:
        A signal per pattern recognised, and none for ordinary prose. This is evidence for
        the guardrail chain, not a filter: the seal is what makes the passage safe.
    """
    signals = [InjectionSignal(kind=kind, detail=detail) for kind, detail in matched_kinds(text)]
    echo = instructions.strip()[:_ECHO_LENGTH]
    if echo and echo in normalised(text[:SCAN_LIMIT]):
        signals.append(InjectionSignal(kind=SignalKind.SYSTEM_ECHO, detail=echo))
    return tuple(signals)


def matched_kinds(text: str) -> list[tuple[SignalKind, str]]:
    """Every pattern the normalised text matches, with the fragment that matched.

    Screening is bounded at `SCAN_LIMIT` characters and reports the tail it did not read as
    a signal of its own, so an oversized payload fails closed rather than passing unread.
    """
    folded = normalised(text[:SCAN_LIMIT])
    found: list[tuple[SignalKind, str]] = []
    if len(text) > SCAN_LIMIT:
        found.append((SignalKind.UNSCANNED, f"{len(text) - SCAN_LIMIT} characters unread"))
    patterns = (
        (SignalKind.OVERRIDE, _OVERRIDE),
        (SignalKind.IMPERSONATION, _IMPERSONATION),
        (SignalKind.TOOL_SHAPED, _TOOL_SHAPED),
    )
    for kind, pattern in patterns:
        match = pattern.search(folded)
        if match:
            found.append((kind, match.group(0)))
    if "<untrusted-data" in folded or "</untrusted-data" in folded:
        found.append((SignalKind.FENCE, "untrusted-data"))
    hidden = _hidden(text[:SCAN_LIMIT], folded)
    if hidden:
        found.append((SignalKind.ENCODED, hidden))
    return found


def normalised(text: str) -> str:
    """The text as the model will read it: composed, zero-width stripped, homoglyphs folded."""
    return unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).translate(_HOMOGLYPHS)


def _hidden(text: str, folded: str) -> str:
    """A payload the reader was not meant to see: zero width, homoglyph, or base64."""
    if folded != text.translate(_ZERO_WIDTH):
        return "homoglyph"
    if text != folded:
        return "zero-width"
    for candidate in _ENCODED.findall(folded):
        if _OVERRIDE.search(_decoded(candidate)):
            return "base64"
    return ""


def _decoded(candidate: str) -> str:
    """What a base64-looking run decodes to, or nothing where it is not base64 at all."""
    try:
        return base64.b64decode(candidate + "=" * (-len(candidate) % 4)).decode("utf-8", "ignore")
    except (binascii.Error, ValueError):
        return ""
