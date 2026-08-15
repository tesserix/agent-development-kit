"""Three layers narrowing one allowlist, and a peer agent that cannot proxy around it.

Run it with `uv run python examples/tool_allowlist.py`.
"""

from __future__ import annotations

from tesserix_adk.core import ToolNotPermittedError
from tesserix_adk.guardrails import ToolAllowlistGuard

DECLARED = ("search", "book", "refund")


def main() -> None:
    """Resolve the allowlist, refuse what each layer cut, and delegate without widening."""
    guard = ToolAllowlistGuard.resolving(
        DECLARED,
        tenant={"search", "book"},
        caller={"search", "book", "refund"},
        agent="concierge",
    )
    print(f"declared {DECLARED} -> callable {guard.allowlist.names}")  # noqa: T201

    for tool in ("SEARCH", "refund", "transfer_funds"):
        try:
            guard.check(tool)
        except ToolNotPermittedError as refused:
            print(f"{tool}: refused by {refused.details['reason']}")  # noqa: T201
        else:
            print(f"{tool}: permitted")  # noqa: T201

    print(f"\nattempts {guard.attempts}, of which refused {guard.refusals}")  # noqa: T201
    print(f"the model is told about {guard.permitted(DECLARED)}")  # noqa: T201

    peer = guard.delegating(("book", "refund"), agent="pricing")
    print(f"\na peer declaring ('book', 'refund') gets {peer.allowlist.names}")  # noqa: T201
    print(f"it cannot proxy refund: {not peer.allowlist.permits('refund')}")  # noqa: T201


if __name__ == "__main__":
    main()
