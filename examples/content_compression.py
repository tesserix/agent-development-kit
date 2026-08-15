"""Compressing tool output by what it is, and declining to where that is not known.

Four scenarios: a 500-row query result, a source file the model is reading, an aligned
table, and content nothing can classify.

Run it with `python examples/content_compression.py`.
"""

from __future__ import annotations

import json

from tesserix_adk.memory import Compressed, ContentRouter

ROWS = json.dumps(
    [
        {"id": index, "region": "apac", "status": "active", "host": f"node-{index:03d}"}
        for index in range(500)
    ]
)

SOURCE = '''
class Ledger:
    """Money in, money out."""

    def credit(self, account: str, amount: int) -> int:
        """Add to an account."""
        running = self._balances.get(account, 0)
        for entry in self._pending(account):
            running += entry.amount
        if amount < 0:
            raise ValueError("amount must be positive")
        return running + amount
'''

TABLE = "id    region   status\n" + "\n".join(f"{index:<5} apac     active" for index in range(120))

OPAQUE = "<<<<>>>> ||| ??? ###" * 60


def report(what: str, admitted: Compressed) -> None:
    """One line per admission: what it was, what handled it, what it saved."""
    print(  # noqa: T201
        f"{what}: {admitted.kind} -> {admitted.compressor} "
        f"{admitted.original_tokens}->{admitted.compressed_tokens} tokens "
        f"(ratio {admitted.ratio:.2f}){' — ' + admitted.reason if admitted.reason else ''}"
    )


def main() -> None:
    """Admit each kind of content through one router and report what happened."""
    router = ContentRouter(threshold_tokens=64)  # low, so the small fixtures below qualify

    rows = router.admit(ROWS, budget_tokens=4_000, untrusted=True)
    report("query result", rows)
    print(f"  still untrusted: {rows.untrusted}")  # noqa: T201
    print(f"  the last host survives: {'node-499' in rows.content}")  # noqa: T201

    report("source file", router.admit(SOURCE, budget_tokens=1_000))
    report("aligned table", router.admit(TABLE, budget_tokens=1_000))
    report("unclassifiable", router.admit(OPAQUE, budget_tokens=1_000))


if __name__ == "__main__":
    main()
