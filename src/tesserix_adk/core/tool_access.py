"""What a run may call, resolved once from every source that has a say.

Three layers each hold an opinion about tools: the agent declares what it was built to
call, the tenant states what their plan entitles them to, and the caller arrives with
scopes. Enforced separately they disagree — a tenant setting checked at configuration time
and an agent declaration checked at dispatch means the effective permission at the moment
of a call is not written down anywhere.

Here they are intersected once, at the boundary, into a value that cannot be widened. The
narrowest source wins, and a source that states nothing narrows nothing — which is not the
same as a source that states an empty set, and is why these are `None` rather than empty.

Names are compared under one normal form, so a capitalised name, a full-width one and a
plain one cannot be three permissions where an operator wrote one. The spelling the caller
asked for is what the refusal quotes, because that is what they need to find in their code.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from tesserix_adk.core.errors import ConfigurationError, ToolNotPermittedError
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

__all__ = ["DenyReason", "ToolAllowlist", "ToolDecision", "canonical"]


class DenyReason(StrEnum):
    """Which layer refused. An operator fixes the setting they own, not the one below it."""

    NOT_DECLARED = "not_declared"
    TENANT = "tenant"
    CALLER = "caller"


class ToolDecision(AdkModel):
    """One dispatch decision, as it is recorded.

    Args:
        tool: What was asked for, spelled the way it was asked for.
        permitted: Whether it may be called.
        reason: Which layer refused, or None where nothing did.
    """

    tool: str
    permitted: bool
    reason: DenyReason | None = None


def canonical(name: str) -> str:
    """The key a tool name is compared under.

    Args:
        name: A tool name as declared, configured or asked for.

    Returns:
        The compatibility-composed, case-folded, stripped form.

    Example:
        >>> canonical("  Search ")
        'search'
    """
    return unicodedata.normalize("NFKC", name).strip().casefold()


@dataclass(frozen=True, slots=True)
class ToolAllowlist:
    """The tools a run may call, resolved once and immutable for its lifetime.

    Frozen on purpose: an allowlist that can be appended to mid-run is an allowlist whose
    value at the moment of a call nobody can state afterwards.

    An empty one is valid and refuses everything, unlike `ToolRegistry.view`, which treats
    an empty allowlist as a misconfiguration. Here emptiness is a result rather than an
    instruction — a tenant whose plan includes none of an agent's tools is a real state.

    Args:
        agent: Whose allowlist it is, named in refusals.
        names: What may be called, in declaration order, spelled as declared.
        denials: Why each declared name was cut, keyed by its canonical form.

    Example:
        >>> ToolAllowlist.resolve(("search", "refund"), tenant={"search"}).names
        ('search',)
    """

    agent: str = ""
    names: tuple[str, ...] = ()
    denials: Mapping[str, DenyReason] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        declared: Iterable[str],
        *,
        tenant: Collection[str] | None = None,
        caller: Collection[str] | None = None,
        agent: str = "",
    ) -> Self:
        """Intersect every source that has a say into the set that is actually callable.

        Args:
            declared: What the agent was built to call.
            tenant: What the tenant's plan permits, or None where it states nothing.
            caller: What the caller's scopes cover, or None where they state nothing.
            agent: Whose allowlist it is.

        Returns:
            The resolved allowlist.

        Raises:
            ConfigurationError: If two declared names share one canonical form. Which of
                them a call reached would depend on iteration order, so neither is safe.
        """
        permitted: list[str] = []
        denials: dict[str, DenyReason] = {}
        seen: dict[str, str] = {}
        for name in declared:
            key = canonical(name)
            if key in seen:
                raise ConfigurationError(
                    f"{seen[key]!r} and {name!r} are the same tool name once normalised, so "
                    f"which one a call reached would depend on iteration order"
                )
            seen[key] = name
            refused = _cut_by(key, tenant=tenant, caller=caller)
            if refused is None:
                permitted.append(name)
            else:
                denials[key] = refused
        return cls(agent=agent, names=tuple(permitted), denials=denials)

    def permits(self, name: str) -> bool:
        """Whether `name` may be called, under any spelling of it."""
        return canonical(name) in {canonical(known) for known in self.names}

    def decide(self, name: str) -> ToolDecision:
        """Whether `name` may be called, and which layer refused it if not.

        Args:
            name: What was asked for.

        Returns:
            The decision, quoting the spelling that was asked for.
        """
        if self.permits(name):
            return ToolDecision(tool=name, permitted=True)
        reason = self.denials.get(canonical(name), DenyReason.NOT_DECLARED)
        return ToolDecision(tool=name, permitted=False, reason=reason)

    def check(self, name: str) -> None:
        """Refuse `name` before it is dispatched, or return.

        Args:
            name: What was asked for.

        Raises:
            ToolNotPermittedError: If it may not be called. Raised instead of calling it:
                an allowlist enforced after dispatch is a side effect that has already
                landed by the time the decision is recorded.
        """
        decision = self.decide(name)
        if decision.permitted:
            return
        raise ToolNotPermittedError(
            name,
            agent=self.agent,
            permitted=self.names,
            reason=decision.reason or DenyReason.NOT_DECLARED,
        )

    def narrowed(self, declared: Iterable[str], *, agent: str = "") -> Self:
        """The allowlist a peer agent gets, which can only be smaller than this one.

        A peer that declares a tool this run lost cannot be asked to call it on the run's
        behalf, so delegation is not a way around the layer that refused.

        Args:
            declared: What the peer was built to call.
            agent: Whose allowlist the result is.

        Returns:
            The intersection, carrying this allowlist's reasons for what it already cut.
        """
        narrowed = type(self).resolve(declared, caller=self.names, agent=agent)
        denials = {**narrowed.denials}
        for key, reason in self.denials.items():
            if key in denials:
                denials[key] = reason
        return type(self)(agent=agent, names=narrowed.names, denials=denials)


def _cut_by(
    key: str, *, tenant: Collection[str] | None, caller: Collection[str] | None
) -> DenyReason | None:
    """Which layer refuses a declared tool, innermost first, or None where none does."""
    if tenant is not None and key not in {canonical(name) for name in tenant}:
        return DenyReason.TENANT
    if caller is not None and key not in {canonical(name) for name in caller}:
        return DenyReason.CALLER
    return None
