"""Where a check is written.

A guard covers both stages of a run, so a pipeline can be told what it covers without
calling it. Most guards care about one: `Guard` answers allow for both, and a subclass
overrides the stage it is about. A guard that overrides neither is a guard that does
nothing, which is at least visible in the definition rather than in the absence of one.
"""

from __future__ import annotations

from tesserix_adk.core.guards import GuardResult

__all__ = ["Guard"]


class Guard:
    """A guard that allows everything, for a subclass to narrow one stage at a time.

    Example:
        >>> class NoCardNumbers(Guard):
        ...     name = "no_card_numbers"
        ...
        ...     async def check_input(self, content: str) -> GuardResult:
        ...         if "4111" in content:
        ...             return GuardResult.blocked(code="pii_detected")
        ...         return GuardResult.allow()
    """

    name: str = "guard"

    async def check_input(self, content: str) -> GuardResult:
        """Allow, unless a subclass says otherwise."""
        del content
        return GuardResult.allow()

    async def check_output(self, content: str) -> GuardResult:
        """Allow, unless a subclass says otherwise."""
        del content
        return GuardResult.allow()
