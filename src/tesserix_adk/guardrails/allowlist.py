"""Enforcing the resolved allowlist in the dispatch path, and counting what it refused.

The check sits after argument validation and before execution, so a refusal is a decision
recorded rather than a side effect already taken. It is deliberately not negotiable: the
model is told the call was refused, and no retry, reformulation or peer agent widens it.

A refused call still cost a model turn, so it counts against the run's iteration and budget
caps. A refusal that is free is a refusal a looping model will attempt indefinitely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from tesserix_adk.core.tool_access import ToolAllowlist

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from tesserix_adk.core.tool_access import ToolDecision

__all__ = ["ToolAllowlistGuard"]


class ToolAllowlistGuard:
    """The allowlist as it is enforced, holding the tally of what it turned away.

    Args:
        allowlist: What this run may call, already resolved.

    Example:
        >>> guard = ToolAllowlistGuard.resolving(("search", "refund"), tenant={"search"})
        >>> guard.decide("refund").reason
        <DenyReason.TENANT: 'tenant'>
    """

    name = "tool_allowlist"

    def __init__(self, allowlist: ToolAllowlist) -> None:
        self._allowlist = allowlist
        self._attempts = 0
        self._refusals = 0

    @classmethod
    def resolving(
        cls,
        declared: Iterable[str],
        *,
        tenant: Collection[str] | None = None,
        caller: Collection[str] | None = None,
        agent: str = "",
    ) -> Self:
        """Build a guard over the intersection of every source that has a say.

        Args:
            declared: What the agent was built to call.
            tenant: What the tenant's plan permits, or None where it states nothing.
            caller: What the caller's scopes cover, or None where they state nothing.
            agent: Whose allowlist it is, named in refusals.

        Returns:
            The guard, over an allowlist that is fixed for the run.
        """
        return cls(ToolAllowlist.resolve(declared, tenant=tenant, caller=caller, agent=agent))

    @property
    def allowlist(self) -> ToolAllowlist:
        """What this run may call. Read-only, and the same value for the whole run."""
        return self._allowlist

    @property
    def attempts(self) -> int:
        """How many calls were put to the guard, refused ones included."""
        return self._attempts

    @property
    def refusals(self) -> int:
        """How many of those it turned away."""
        return self._refusals

    def decide(self, name: str) -> ToolDecision:
        """Judge one call and count it.

        Args:
            name: What the model asked for.

        Returns:
            The decision, naming the layer that refused where one did.
        """
        decision = self._allowlist.decide(name)
        self._attempts += 1
        if not decision.permitted:
            self._refusals += 1
        return decision

    def check(self, name: str) -> None:
        """Refuse one call before it is dispatched, or return.

        Args:
            name: What the model asked for.

        Raises:
            ToolNotPermittedError: If it may not be called, naming the layer that refused.
        """
        if not self.decide(name).permitted:
            self._allowlist.check(name)

    def permitted(self, names: Iterable[str]) -> tuple[str, ...]:
        """Which of `names` the model is told about, in the order they were given.

        Declaring a tool the run may not call invites the model to ask for it and be
        refused, which spends a turn to learn something the prompt already knew.
        """
        return tuple(name for name in names if self._allowlist.permits(name))

    def delegating(self, declared: Iterable[str], *, agent: str = "") -> Self:
        """The guard a peer agent runs under, which can only be narrower than this one.

        Args:
            declared: What the peer was built to call.
            agent: Whose guard the result is.

        Returns:
            A fresh guard with its own tally over the intersection.
        """
        return type(self)(self._allowlist.narrowed(declared, agent=agent))
