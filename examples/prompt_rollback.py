"""Diffing two prompt versions and rolling an alias back, as an incident would.

Builds a registry with two versions, diffs them, tries a rollback the call sites cannot
render, then rolls back to one they can.

Run it with `python examples/prompt_rollback.py`.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk.cli import prompts_main
from tesserix_adk.core import FilePromptRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping

ITINERARY = """
owner = "platform"

[aliases]
current = "5"

[versions.4]
body = "Plan an itinerary for ${city}."
variables = ["city"]

[versions.5]
body = "Plan an itinerary for ${city} within ${budget}."
variables = ["city", "budget"]
"""


class Book:
    """A deployment's alias store, in memory for the length of this example."""

    def __init__(self) -> None:
        self.pointing = {"current": "5"}

    async def aliases(self, name: str, *, tenant: str = "") -> Mapping[str, str]:
        """Which version each alias points at."""
        del name, tenant
        return dict(self.pointing)

    async def evaluated(self, name: str, version: str, *, tenant: str = "") -> str:
        """The last eval result, which both versions here have."""
        del name, tenant
        return "pass" if version in {"4", "5"} else ""

    async def repoint(
        self,
        name: str,
        *,
        alias: str,
        to: str,
        expected: str,
        by: str,
        reason: str,
        tenant: str = "",
    ) -> None:
        """Move the alias, refusing where it moved under us."""
        del name, reason, tenant
        if self.pointing[alias] != expected:
            message = "alias moved under the rollback"
            raise RuntimeError(message)
        self.pointing[alias] = to
        print(f"  store recorded: {alias} {expected} -> {to} by {by}")  # noqa: T201


async def main() -> None:
    """Diff, refuse a rollback, then take one that is safe."""
    with tempfile.TemporaryDirectory() as directory:
        prompts = Path(directory)
        (prompts / "itinerary_system.toml").write_text(ITINERARY, encoding="utf-8")
        registry = FilePromptRegistry(prompts)
        book = Book()

        await prompts_main(["diff", "itinerary_system", "4", "5"], prompts=registry, aliases=book)

        print("\nrolling current back to 4:")  # noqa: T201
        code = await prompts_main(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada"],
            prompts=registry,
            aliases=book,
        )
        print(f"exit code {code}, current still on {book.pointing['current']}")  # noqa: T201

        (prompts / "itinerary_system.toml").write_text(
            ITINERARY.replace('variables = ["city", "budget"]', 'variables = ["city"]'),
            encoding="utf-8",
        )
        print("\nonce version 5 no longer declares budget:")  # noqa: T201
        await prompts_main(
            ["rollback", "itinerary_system", "--to", "4", "--by", "ada", "--reason", "PLAT-91"],
            prompts=FilePromptRegistry(prompts),
            aliases=book,
        )


if __name__ == "__main__":
    asyncio.run(main())
