"""Prompts as versioned artifacts, and what a run records about the one it ran on.

Resolves an alias to a concrete version, instructs an agent with it, and shows what reaches
telemetry — and what a repointed alias and an in-place edit each do.

Run it with `python examples/prompt_registry.py`.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from tesserix_adk.core import Agent, FilePromptRegistry, PromptRejectedError

GREETING = """
owner = "platform"
task_class = "conversation"

[aliases]
current = "3"
candidate = "4"

[versions.3]
body = "You greet the customer by name."
variables = ["customer"]

[versions.4]
body = "You greet the customer by name, once."
variables = ["customer"]
"""


async def main() -> None:
    """Resolve, instruct, and then move the file under a published version."""
    with tempfile.TemporaryDirectory() as directory:
        prompts = Path(directory)
        (prompts / "greeting.toml").write_text(GREETING, encoding="utf-8")
        registry = FilePromptRegistry(prompts)

        print(f"published: {await registry.list_versions('greeting')}")  # noqa: T201

        prompt = await registry.get("greeting", alias="current")
        print(f"current resolved to {prompt.ref.label}, digest {prompt.digest[:12]}…")  # noqa: T201
        print(f"telemetry sees: {prompt.ref.attributes()}")  # noqa: T201

        agent = prompt.instruct(
            Agent(name="greeter", instructions="placeholder", model="gpt-5", free_text=True)
        )
        print(f"agent runs on {agent.prompt.label if agent.prompt else 'nothing registered'}")  # noqa: T201
        print(f"and fits a 4k window: {prompt.fits(4_000)}")  # noqa: T201

        (prompts / "greeting.toml").write_text(
            GREETING.replace('current = "3"', 'current = "4"'), encoding="utf-8"
        )
        moved = await registry.get("greeting", alias="current")
        print(f"alias repointed: current is now {moved.ref.label}, {prompt.ref.label} unmoved")  # noqa: T201

        (prompts / "greeting.toml").write_text(
            GREETING.replace("by name.", "by title."), encoding="utf-8"
        )
        try:
            await registry.get("greeting", version="3")
        except PromptRejectedError as edited:
            print(f"refused: {edited}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
