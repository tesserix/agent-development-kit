"""A peer's answer, treated as retrieved content rather than as a colleague's word.

Delegation is where trust quietly becomes transitive. The peer was chosen by the operator,
so its answer feels like the operator's — but the peer read a web page, or a document a
customer uploaded, and whatever was in there is now arriving inside a field the caller is
about to paste into a prompt. Every argument for fencing a tool result applies here, and
one more: a peer can be persuaded, so it can be made to ask on the attacker's behalf.

So an answer that has already passed the card's output schema is still sealed as untrusted
data, screened for instruction shape field by field, stripped of the characters that make a
payload look like structure, and redacted before anything records it. What the peer says
can inform the run. It cannot redirect it, and `permitted` is where that stops being a
convention and becomes a call the caller has to make deliberately.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tesserix_adk.a2a.invocation import PeerInvocationError, PeerInvocationReason
from tesserix_adk.core.errors import AdkError
from tesserix_adk.core.injection import SignalKind, screen
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.pii import PIIKind, redact
from tesserix_adk.core.provenance import ContentSource, Origin, sealed

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.a2a.invocation import PeerResult

__all__ = [
    "DEFAULT_KEPT_BYTES",
    "DEFAULT_MAX_CONTENT_BYTES",
    "PeerActionError",
    "PeerBoundary",
    "PeerContent",
    "PeerSuspicionError",
    "PeerTrustPolicy",
    "ProposedAction",
    "TrustDecision",
]

DEFAULT_MAX_CONTENT_BYTES = 32_768
"""The most rendered answer a caller will read. Beyond it the answer is refused, not cut."""

DEFAULT_KEPT_BYTES = 2_048
"""How much of a suspicious answer survives truncation, where that is the policy."""

_KEPT_CONTROL = (0x09, 0x0A)
_INVISIBLE = dict.fromkeys(
    codepoint
    for codepoint in (*range(0x20), 0x7F, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)
    if codepoint not in _KEPT_CONTROL
)
_ROLE_LINE = re.compile(
    r"^\s*(?:[#>*\-]+\s*)*(?:system|assistant|developer|user)\s*[::]", re.IGNORECASE
)


class TrustDecision(StrEnum):
    """What the caller does with an answer that screened as instructions."""

    ANNOTATE = "annotate"
    """Deliver it, sealed and flagged. The default: a false positive must not lose the answer."""

    TRUNCATE = "truncate"
    """Deliver the head of it, on the reading that the payload is usually appended."""

    REFUSE = "refuse"
    """Fail the call. For peers whose answers reach somewhere a payload would be expensive."""


class PeerTrustPolicy(AdkModel):
    """How much of another agent's word this run will take, per peer.

    Args:
        on_suspicion: What to do with an answer that screens as instructions.
        per_peer: Peers held to a different rule than the rest. One flaky peer is not a
            reason to fail closed on all of them, and the reverse is more often the case.
        max_content_bytes: The most rendered answer the caller will read.
        kept_bytes: How much survives truncation.

    Example:
        >>> PeerTrustPolicy(per_peer={"booker": TrustDecision.REFUSE}).for_peer("booker")
        <TrustDecision.REFUSE: 'refuse'>
    """

    on_suspicion: TrustDecision = TrustDecision.ANNOTATE
    per_peer: dict[str, TrustDecision] = Field(default_factory=dict)
    max_content_bytes: int = Field(default=DEFAULT_MAX_CONTENT_BYTES, ge=1)
    kept_bytes: int = Field(default=DEFAULT_KEPT_BYTES, ge=1)

    def for_peer(self, peer: str) -> TrustDecision:
        """The rule this peer is held to."""
        return self.per_peer.get(peer, self.on_suspicion)


class PeerSuspicionError(AdkError):
    """An answer screened as instructions and this peer's policy is to fail closed.

    Args:
        message: What was refused, in terms of the call.
        peer: Which agent.
        skill: Which skill.
        codes: What screening recognised. The text itself is deliberately absent: a
            refusal that quotes the payload has moved the payload into the logs.
    """

    def __init__(
        self, message: str, *, peer: str, skill: str, codes: tuple[SignalKind, ...]
    ) -> None:
        self.peer = peer
        self.skill = skill
        self.codes = codes
        super().__init__(
            message,
            details={"peer": peer, "skill": skill, "codes": ",".join(codes)},
        )


class PeerActionError(AdkError):
    """A gated action was reached for with nothing behind it but a peer's answer.

    Args:
        message: Which action, and which gate.
        peer: Which agent's answer was the reason.
        skill: Which skill.
        tool: The action that was refused.
        gate: Why it is gated.
    """

    def __init__(self, message: str, *, peer: str, skill: str, tool: str, gate: str) -> None:
        self.peer = peer
        self.skill = skill
        self.tool = tool
        self.gate = gate
        super().__init__(
            message, details={"peer": peer, "skill": skill, "tool": tool, "gate": gate}
        )


@dataclass(frozen=True, slots=True)
class ProposedAction:
    """Something the run is about to do with a peer's answer in hand.

    Args:
        tool: What would run.
        approval_required: Whether a human gates it.
        moves_money: Whether it spends, refunds or transfers.
        calls_a_peer: Whether it delegates further, which is how one poisoned answer
            becomes a chain of them.
    """

    tool: str
    approval_required: bool = False
    moves_money: bool = False
    calls_a_peer: bool = False

    def gates(self) -> tuple[str, ...]:
        """Every reason this needs a decision of the run's own."""
        reasons = (
            ("approval", self.approval_required),
            ("money", self.moves_money),
            ("delegation", self.calls_a_peer),
        )
        return tuple(name for name, gated in reasons if gated)


class PeerContent(AdkModel):
    """One peer's answer, in the only form the caller passes to a model.

    Args:
        peer: Which agent answered.
        skill: Which skill.
        card: The fingerprint of the card the answer was held to, where one was pinned.
        text: The sealed block. Instruction-shaped structure in it is already inert.
        codes: What screening recognised, if anything.
        fields: Where in the answer it was recognised, by path. The path is recordable;
            the text at it is not.
        decision: What the policy did about it.
        truncated: Whether part of the answer was dropped.
        redactions: Which kinds of identifier were replaced before this was delivered.
    """

    peer: str
    skill: str
    card: str = ""
    text: str
    codes: tuple[SignalKind, ...] = ()
    fields: tuple[str, ...] = ()
    decision: TrustDecision = TrustDecision.ANNOTATE
    truncated: bool = False
    redactions: tuple[PIIKind, ...] = ()

    def source(self) -> ContentSource:
        """Where this came from, in the form every other boundary in the kit takes."""
        return ContentSource(origin=Origin.PEER_AGENT, name=f"{self.peer}/{self.skill}")

    def attributes(self) -> dict[str, str]:
        """What a span records: the decision and where it came from, never the answer."""
        return {
            "a2a.peer": self.peer,
            "a2a.skill": self.skill,
            "a2a.card": self.card,
            "a2a.trust.decision": self.decision.value,
            "a2a.trust.codes": ",".join(self.codes),
            "a2a.trust.fields": ",".join(self.fields),
            "a2a.trust.truncated": str(self.truncated).lower(),
            "a2a.redactions": ",".join(self.redactions),
        }


@dataclass(frozen=True, slots=True)
class PeerBoundary:
    """The one place a peer's answer becomes something the caller may read.

    Args:
        policy: How much of another agent's word this run takes.
        instructions: The caller's own instructions, so an answer quoting them back is
            recognised as the impersonation it is.
    """

    policy: PeerTrustPolicy = dataclass_field(default_factory=PeerTrustPolicy)
    instructions: str = ""

    def contain(self, result: PeerResult, *, tenant: str, card: str = "") -> PeerContent:
        """Turn a validated answer into sealed, screened, redacted content.

        Args:
            result: The answer, already held to the peer's published output schema.
            tenant: Whose data this is, which decides the redaction pseudonyms.
            card: The fingerprint of the card it was held to, where the caller pinned one.

        Returns:
            The content, and everything a reviewer needs to say why it looks like that.

        Raises:
            PeerInvocationError: The answer is larger than the caller will read.
            PeerSuspicionError: It screened as instructions and this peer's policy is
                to fail closed.
        """
        self._within_the_ceiling(result)
        found = _found(result.output, self.instructions)
        codes = tuple(sorted({kind for _, kind in found}))
        fields = tuple(sorted({path for path, _ in found}))
        decision = self.policy.for_peer(result.peer) if codes else TrustDecision.ANNOTATE
        if decision is TrustDecision.REFUSE:
            raise PeerSuspicionError(
                f"{result.peer}/{result.skill} answered with content screening as "
                f"{', '.join(codes)}, and this peer is held to fail closed",
                peer=result.peer,
                skill=result.skill,
                codes=codes,
            )
        scrubbed = redact(_rendered(result.output), tenant=tenant)
        body, truncated = self._shortened(scrubbed.text, decision)
        return PeerContent(
            peer=result.peer,
            skill=result.skill,
            card=card,
            text=sealed(
                body,
                source=ContentSource(
                    origin=Origin.PEER_AGENT, name=f"{result.peer}/{result.skill}"
                ),
            ),
            codes=codes,
            fields=fields,
            decision=decision,
            truncated=truncated,
            redactions=scrubbed.kinds,
        )

    def permitted(
        self, content: PeerContent, action: ProposedAction, *, decided_by: str = ""
    ) -> None:
        """Refuse a gated action whose only justification is what a peer said.

        A peer's answer is evidence. Approval, spend and further delegation are decisions,
        and a decision needs something deterministic behind it — a policy, a rule, a
        human — that the caller can name afterwards.

        Args:
            content: The answer in hand.
            action: What the run is about to do.
            decided_by: The deterministic decision that authorised it. Naming the policy
                is the whole point: "the model thought so" has no name.

        Raises:
            PeerActionError: The action is gated and nothing but the answer is behind it.
        """
        gates = action.gates()
        if not gates or decided_by:
            return
        raise PeerActionError(
            f"{action.tool} is gated on {gates[0]} and the only thing asking for it is "
            f"{content.peer}'s answer",
            peer=content.peer,
            skill=content.skill,
            tool=action.tool,
            gate=gates[0],
        )

    def _within_the_ceiling(self, result: PeerResult) -> None:
        """Refuse an answer too big to read, rather than reading part of it."""
        size = len(_rendered(result.output).encode())
        if size > self.policy.max_content_bytes:
            raise PeerInvocationError(
                f"{result.peer}/{result.skill} answered with {size} bytes, and the caller "
                f"reads at most {self.policy.max_content_bytes}",
                peer=result.peer,
                skill=result.skill,
                reason=PeerInvocationReason.TOO_LARGE,
            )

    def _shortened(self, body: str, decision: TrustDecision) -> tuple[str, bool]:
        """The head of a suspicious answer, where truncating is what this peer gets."""
        if decision is not TrustDecision.TRUNCATE or len(body) <= self.policy.kept_bytes:
            return body, False
        return f"{body[: self.policy.kept_bytes]}\n… (truncated)", True


def _found(output: Mapping[str, Any], instructions: str) -> list[tuple[str, SignalKind]]:
    """Every instruction-shaped string in the answer, by path.

    Walking rather than screening the rendered whole: a payload in the fourth element of a
    nested list is the same attack as one in the summary, and the reviewer needs to know
    which field it was in without the field's contents being written down anywhere.
    """
    found: list[tuple[str, SignalKind]] = []
    for path, text in _strings(output, ""):
        found.extend((path, signal.kind) for signal in screen(text, instructions=instructions))
    return found


def _strings(value: object, path: str) -> list[tuple[str, str]]:
    """Every string leaf in the answer, with the path a reviewer would go and look at."""
    if isinstance(value, str):
        return [(path or "output", value)]
    if isinstance(value, dict):
        return [
            pair
            for name, nested in value.items()
            for pair in _strings(nested, f"{path}.{name}" if path else str(name))
        ]
    if isinstance(value, list):
        return [
            pair
            for index, nested in enumerate(value)
            for pair in _strings(nested, f"{path}[{index}]")
        ]
    return []


def _rendered(output: Mapping[str, Any]) -> str:
    """The answer as text, with every string leaf made inert first.

    Deterministic, because a prompt prefix cached on its bytes must not change between two
    runs that were handed the same answer.
    """
    return json.dumps(_inert(output), ensure_ascii=False, indent=2, sort_keys=True)


def _inert(value: object) -> object:
    """The same answer with nothing in it that a reader could mistake for structure."""
    if isinstance(value, str):
        return _quoted(_folded(value))
    if isinstance(value, dict):
        return {name: _inert(nested) for name, nested in value.items()}
    if isinstance(value, list):
        return [_inert(nested) for nested in value]
    return value


def _folded(text: str) -> str:
    """The text as the model will read it, minus the characters it was not meant to see.

    Homoglyphs are left standing on purpose: folding them is right for matching and wrong
    for delivery, since a Cyrillic word in a real answer is a word, not an attack.
    """
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE).replace("<", "&lt;")


def _quoted(text: str) -> str:
    """A line wearing a role marker, marked as the quotation it is."""
    return (
        "\n".join(
            f"(quoted) {line}" if _ROLE_LINE.match(line) else line for line in text.splitlines()
        )
        or text
    )
