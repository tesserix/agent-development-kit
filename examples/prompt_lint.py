"""Linting a prompt directory for business rules that should be code.

Writes two prompts — one carrying a refund rule, one that only frames the task — runs the
check over them, and prints what CI would print.

Run it with `python examples/prompt_lint.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tesserix_adk.core import lint_directory

RULE_IN_PROSE = "If the booking was made more than 24 hours ago, refund 50% of the fare.\n"
REFACTORED = (
    "Ask refund_quote what the booking is entitled to.\n"
    "Tell the customer the amount and the reason it gives.\n"
)


def main() -> None:
    """Lint a directory holding one offending prompt and one clean one."""
    with tempfile.TemporaryDirectory() as directory:
        prompts = Path(directory)
        (prompts / "refunds.toml").write_text(RULE_IN_PROSE, encoding="utf-8")
        (prompts / "support.toml").write_text(REFACTORED, encoding="utf-8")

        report = lint_directory(prompts)

        for finding in report.findings:
            print(f"{finding.code} {finding.summary}")  # noqa: T201
            print(f"  {finding.text}")  # noqa: T201
            print(f"  remedy: {finding.remedy}")  # noqa: T201

        print(f"\n{report.summary()}")  # noqa: T201
        print(f"exit code: {report.exit_code}")  # noqa: T201


if __name__ == "__main__":
    main()
