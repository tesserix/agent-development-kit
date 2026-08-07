"""An agent as a reviewable artifact, and the revision a finished run names.

Four scenarios: what a definition insists on; an allowlist naming a tool nobody registered;
what an edit does to the revision; and the revision landing on the run and every span.

Run it with `python examples/agent_definition.py`. A scripted provider stands in for the
vendor, so nothing here reaches the network and no key is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tesserix_adk.core import Agent, AgentDefinition, ConfigurationError, ModelCapabilities, Owner
from tesserix_adk.observability import attributes_of, spend_of
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)
OWNER = Owner(team="search-platform", contact="search@example.gov", service="aequitas-search")
REGISTERED = ("search", "fetch_page")


def clerk(
    *, instructions: str = "Answer from the file. Cite the page.", tools: tuple[str, ...] = ()
) -> Agent[Any]:
    """The agent under review, before anything is agreed about it."""
    return Agent(
        name="clerk",
        instructions=instructions,
        model="llama-3.1-8b",
        free_text=True,
        tools=tools,
    )


def define(
    agent: Agent[Any] | None = None, *, instructions_ref: str | None = None
) -> AgentDefinition[Any]:
    """The same agent with an owner and a suite attached to it."""
    return AgentDefinition(
        agent=agent or clerk(),
        owner=OWNER,
        evaluation_suite="suites/clerk.yaml",
        instructions_ref=instructions_ref,
    )


def what_it_says() -> None:
    """One object holds what four call sites otherwise scatter."""
    definition = define(clerk(tools=("search",)), instructions_ref="prompts/clerk@3")
    print(  # noqa: T201
        f"{definition.key}: owned by {definition.owner.team},",
        f"checked by {definition.evaluation_suite}, tools {definition.agent.tools}",
    )


def a_tool_nobody_registered() -> None:
    """Caught at construction, not at the first production run that reaches for it."""
    try:
        AgentDefinition.declared(
            agent=clerk(tools=("search", "ledger")),
            owner=OWNER,
            evaluation_suite="suites/clerk.yaml",
            known_tools=REGISTERED,
        )
    except ConfigurationError as refused:
        print(f"refused: {refused}")  # noqa: T201


def an_edit_is_a_new_revision() -> None:
    """The digest is content-derived, so a rewritten prompt cannot pass as the old one."""
    before = define()
    after = define(clerk(instructions="Answer from the file. Say nothing else."))
    print(f"revision {before.revision} -> {after.revision} after an instruction edit")  # noqa: T201


async def what_the_run_names() -> None:
    """A past run points at the exact artifact, not a version that may have moved."""
    definition = define()
    run = await AgentRunner(
        provider=ScriptedProvider(
            ModelResponse(content="page 12."), name="scripted", capabilities=CAPABLE
        ),
        clock=FakeClock(),
    ).run(definition, "what does page 12 say?", tenant="acme")
    (record, *_) = spend_of(run)
    print(  # noqa: T201
        f"run {run.id} ran {run.agent_name}@{run.agent_version}",
        f"at revision {run.definition_revision};",
        f"span says adk.definition={attributes_of(record)['adk.definition']}",
    )


async def main() -> None:
    """Run every scenario in order."""
    what_it_says()
    a_tool_nobody_registered()
    an_edit_is_a_new_revision()
    await what_the_run_names()


if __name__ == "__main__":
    asyncio.run(main())
