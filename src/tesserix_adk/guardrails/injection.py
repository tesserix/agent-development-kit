"""The guard over untrusted segments, and the three things they may never change.

Screening is evidence. Containment is the defence: whatever a retrieved page, a tool result
or a peer agent's answer says, it does not widen the tool allowlist, it does not change who
the run is acting as, and it does not add a directive. Those three are what an injection is
actually trying to reach — the prose in between is only how it asks.

The guard defaults to blocking. Annotate-and-continue is a choice a consumer makes with the
corpus in front of them, because a guard that blocks a whole corpus on one match is a guard
that gets turned off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.errors import InjectionSuspectedError
from tesserix_adk.core.guards import GuardResult
from tesserix_adk.core.injection import screen
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.provenance import ContentSource, Origin
from tesserix_adk.guardrails.base import Guard

if TYPE_CHECKING:
    from tesserix_adk.core.injection import InjectionSignal

__all__ = ["Containment", "InjectionGuard"]

_ASSUMED = ContentSource(origin=Origin.RETRIEVAL, name="unnamed")
"""What a bare string is taken to be. The chain hands content, not where it came from."""


class InjectionGuard(Guard):
    """Screens untrusted content for text shaped like an instruction to the agent.

    Args:
        block: Whether a match stops the step. `False` annotates and continues, which is
            what a consumer picks when their corpus is noisy and the seal is doing the
            work anyway.
        instructions: The agent's own instructions, so content quoting them back can be
            recognised. Never rendered anywhere.

    Example:
        >>> InjectionGuard().name
        'injection'
    """

    name = "injection"

    def __init__(self, *, block: bool = True, instructions: str = "") -> None:
        self.block = block
        self.instructions = instructions

    def inspect(
        self, content: str, *, source: ContentSource = _ASSUMED
    ) -> tuple[InjectionSignal, ...]:
        """What screening recognises in one segment, given where the segment came from.

        Content from the caller is not screened for disobedience: a user telling the agent
        to disregard what it was told is the caller exercising the run, not an attack on it.
        """
        if not source.untrusted:
            return ()
        return screen(content, instructions=self.instructions)

    async def check_input(self, content: str) -> GuardResult:
        """Screen content on its way in, blocking or annotating as configured."""
        signals = self.inspect(content)
        if not signals:
            return GuardResult.allow()
        codes = tuple(sorted({signal.kind.value for signal in signals}))
        detail = f"{len(signals)} signal(s): {', '.join(codes)}"
        if not self.block:
            return GuardResult.allow()
        return GuardResult.blocked(code="injection_suspected", detail=detail)

    def raise_for(self, content: str, *, source: ContentSource) -> None:
        """Refuse the segment with the typed violation, naming the source and the codes.

        Raises:
            InjectionSuspectedError: Where anything was recognised. Carries the source and
                the match codes; the matched span is recorded as a length, never as text.
        """
        signals = self.inspect(content, source=source)
        if not signals:
            return
        raise InjectionSuspectedError(
            f"instruction-shaped text in {source.origin.value} content",
            source=source.name,
            origin=source.origin.value,
            codes=tuple(sorted({signal.kind.value for signal in signals})),
            span=signals[0].detail,
            guard=self.name,
        )


class Containment(AdkModel):
    """What a run is, in the three terms untrusted content is trying to change.

    Args:
        allowlist: The tools reachable on this step.
        principal: Who the run acts as.
        tenant: Whose data it is acting on.
        directives: The instructions in force, beyond the agent's own.

    Example:
        >>> Containment(allowlist=("search",), principal="user-1", tenant="acme").tenant
        'acme'
    """

    allowlist: tuple[str, ...] = ()
    principal: str = ""
    tenant: str = ""
    directives: tuple[str, ...] = ()

    def hold(self, proposed: Containment, *, source: ContentSource) -> None:
        """Refuse a change that untrusted content is not allowed to make.

        Narrowing is always allowed: untrusted content taking capability away is not an
        escalation, and refusing it would make a suspicious page harder to contain rather
        than easier. Trusted origins pass through — containment is about where the change
        came from, not about the change.

        Args:
            proposed: What the step would become.
            source: What is asking for it.

        Raises:
            InjectionSuspectedError: Where untrusted content would widen the allowlist,
                change the principal or tenant, or add a directive.
        """
        if not source.untrusted:
            return
        for reason in self._breaches(proposed):
            raise InjectionSuspectedError(
                f"{source.origin.value} content may not {reason}",
                source=source.name,
                origin=source.origin.value,
                codes=("containment",),
                guard="containment",
            )

    def _breaches(self, proposed: Containment) -> tuple[str, ...]:
        """Every rule the proposal breaks, in the order an operator reads them."""
        broken: list[str] = []
        if set(proposed.allowlist) - set(self.allowlist):
            broken.append("widen the tool allowlist")
        if proposed.principal != self.principal:
            broken.append("change the principal the run acts as")
        if proposed.tenant != self.tenant:
            broken.append("move the run to another tenant")
        if set(proposed.directives) - set(self.directives):
            broken.append("introduce a system directive")
        return tuple(broken)
