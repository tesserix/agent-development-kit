"""Where a persisted fact came from, kept on the fact itself.

Memory poisoning has a far longer half-life than prompt injection. An injected instruction
in a single prompt affects one turn; the same instruction written into long-term memory
re-influences every future run for that user, and arrives on later turns wearing the
costume of a trusted internal fact rather than untrusted retrieved content.

The defence is not storage, it is provenance. A record that cannot say which run wrote it,
from which source, and whether a person asserted it or a model inferred it, cannot be
judged later — and "it is already in the store" is not a reason to believe anything.
"""

from __future__ import annotations

from enum import StrEnum

from tesserix_adk.core.models import AdkModel

__all__ = ["Origin", "Provenance"]


class Origin(StrEnum):
    """Who or what produced a fact, which is what decides how far it is trusted."""

    USER_ASSERTED = "user_asserted"
    """The person whose memory it is said so, in their own turn."""

    OPERATOR = "operator"
    """A human operator or a migration a human signed off wrote it."""

    MODEL_INFERRED = "model_inferred"
    """The model concluded it. Plausible, unverified, and never promoted to an assertion."""

    TOOL_OUTPUT = "tool_output"
    """A tool returned it. Trusted no further than the tool's own inputs."""

    RETRIEVED_CONTENT = "retrieved_content"
    """It came out of a document. Data about a corpus, not a fact about a user."""

    UNPROVEN = "unproven"
    """Nothing recorded where it came from — a record written before any policy existed."""

    @property
    def asserted(self) -> bool:
        """Whether a fact from here may stand as an assertion rather than an inference."""
        return self in {Origin.USER_ASSERTED, Origin.OPERATOR}


class Provenance(AdkModel):
    """The write this fact came from, carried on the record for as long as it is kept.

    Args:
        origin: Who or what produced it.
        run_id: The run the write happened in.
        turn: Which turn of that run, counting from 1. Zero means it was not written in
            a turn at all — an import, or a migration.
        source: The tool, document or person behind it, as it would be named to an auditor.
        citations: The citation ids the content was derived from, where it was derived.
        policy: The admission policy that let it in, by name. A record admitted under a
            policy that has since been tightened can be found by this.
        admitted_at: When the policy admitted it, in epoch seconds.
    """

    origin: Origin = Origin.UNPROVEN
    run_id: str = ""
    turn: int = 0
    source: str = ""
    citations: tuple[str, ...] = ()
    policy: str = ""
    admitted_at: float | None = None

    @property
    def unproven(self) -> bool:
        """Whether this says nothing an auditor could act on."""
        return self.origin is Origin.UNPROVEN
