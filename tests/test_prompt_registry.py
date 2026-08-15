"""Prompts as versioned artifacts, so a behaviour change names the edit that caused it."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    Agent,
    Cost,
    FilePromptRegistry,
    PromptDefinition,
    PromptNotFoundError,
    PromptRejectedError,
    Usage,
)
from tesserix_adk.observability import attributes_of, spend_of
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import CAPABLE, FakeClock, ScriptedProvider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

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


def written(directory: Path, name: str, text: str = GREETING) -> Path:
    """One prompt file, as a project would keep it in its own repository."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def registry(tmp_path: Path) -> FilePromptRegistry:
    """A registry over one prompt with two versions and two aliases."""
    written(tmp_path, "greeting")
    return FilePromptRegistry(tmp_path)


class TestWhatARunIsAttributableTo:
    """A run nobody can tie to prompt text is a behaviour change nobody can explain."""

    async def test_an_alias_resolves_to_the_concrete_version(
        self, registry: FilePromptRegistry
    ) -> None:
        prompt = await registry.get("greeting", alias="current")

        assert prompt.version == "3"
        assert prompt.body == "You greet the customer by name."

    async def test_a_version_can_be_asked_for_directly(self, registry: FilePromptRegistry) -> None:
        assert (await registry.get("greeting", version="4")).version == "4"

    async def test_the_metadata_the_project_declared_comes_with_it(
        self, registry: FilePromptRegistry
    ) -> None:
        prompt = await registry.get("greeting", alias="current")

        assert prompt.owner == "platform"
        assert prompt.task_class == "conversation"
        assert prompt.variables == ("customer",)

    async def test_every_version_is_listable(self, registry: FilePromptRegistry) -> None:
        assert await registry.list_versions("greeting") == ("3", "4")

    async def test_telemetry_gets_the_name_version_and_digest_and_not_the_body(
        self, registry: FilePromptRegistry
    ) -> None:
        prompt = await registry.get("greeting", alias="current")

        attributes = prompt.ref.attributes()

        assert attributes["adk.prompt"] == "greeting@3"
        assert attributes["adk.prompt_digest"] == prompt.digest
        assert not any(prompt.body in value for value in attributes.values())

    async def test_the_same_text_under_two_names_stays_distinct(self, tmp_path: Path) -> None:
        written(tmp_path, "greeting")
        written(tmp_path, "welcome")
        registry = FilePromptRegistry(tmp_path)

        one = await registry.get("greeting", alias="current")
        other = await registry.get("welcome", alias="current")

        assert one.digest == other.digest
        assert one.ref != other.ref


class TestWhatIsNeverGuessedAt:
    """The kit never falls back to an empty prompt, a nearest match, or a default."""

    async def test_an_unknown_name_says_so(self, registry: FilePromptRegistry) -> None:
        with pytest.raises(PromptNotFoundError):
            await registry.get("farewell", alias="current")

    async def test_an_unknown_version_lists_the_ones_that_exist(
        self, registry: FilePromptRegistry
    ) -> None:
        with pytest.raises(PromptNotFoundError) as missing:
            await registry.get("greeting", version="9")

        assert missing.value.available == ("3", "4")

    async def test_an_unknown_alias_says_so(self, registry: FilePromptRegistry) -> None:
        with pytest.raises(PromptNotFoundError):
            await registry.get("greeting", alias="retired")

    async def test_asking_for_nothing_in_particular_is_refused(
        self, registry: FilePromptRegistry
    ) -> None:
        with pytest.raises(PromptNotFoundError):
            await registry.get("greeting")

    async def test_listing_an_unknown_prompt_says_so(self, registry: FilePromptRegistry) -> None:
        with pytest.raises(PromptNotFoundError):
            await registry.list_versions("farewell")

    async def test_a_file_that_does_not_parse_is_surfaced(self, tmp_path: Path) -> None:
        written(tmp_path, "broken", "this is not toml =")

        with pytest.raises(PromptRejectedError):
            await FilePromptRegistry(tmp_path).get("broken", version="1")


class TestWhatAPromptMayNotContain:
    """Prompt bodies travel to providers and to reviewers. Neither is a place for a secret."""

    async def test_a_body_holding_something_shaped_like_a_key_is_refused(
        self, tmp_path: Path
    ) -> None:
        written(
            tmp_path,
            "leaky",
            '[versions.1]\nbody = "Authenticate with sk-01234567890abcdef"\n',
        )

        with pytest.raises(PromptRejectedError):
            await FilePromptRegistry(tmp_path).get("leaky", version="1")

    async def test_an_empty_body_is_refused_rather_than_served(self, tmp_path: Path) -> None:
        written(tmp_path, "hollow", '[versions.1]\nbody = ""\n')

        with pytest.raises(PromptRejectedError):
            await FilePromptRegistry(tmp_path).get("hollow", version="1")


class TestWhenAPublishedVersionMoves:
    """A published version is immutable. An edit is a new version, not a quiet swap."""

    async def test_an_edit_in_place_is_caught_by_its_digest(self, tmp_path: Path) -> None:
        written(tmp_path, "greeting")
        registry = FilePromptRegistry(tmp_path)
        await registry.get("greeting", version="3")

        written(tmp_path, "greeting", GREETING.replace("by name.", "by title."))

        with pytest.raises(PromptRejectedError) as edited:
            await registry.get("greeting", version="3")

        assert "3" in str(edited.value)

    async def test_development_can_opt_out_of_sealing(self, tmp_path: Path) -> None:
        written(tmp_path, "greeting")
        registry = FilePromptRegistry(tmp_path, sealed=False)
        await registry.get("greeting", version="3")

        written(tmp_path, "greeting", GREETING.replace("by name.", "by title."))

        assert (await registry.get("greeting", version="3")).body.endswith("by title.")

    async def test_a_repointed_alias_does_not_move_a_prompt_already_resolved(
        self, tmp_path: Path
    ) -> None:
        written(tmp_path, "greeting")
        registry = FilePromptRegistry(tmp_path)
        resolved = await registry.get("greeting", alias="current")

        written(tmp_path, "greeting", GREETING.replace('current = "3"', 'current = "4"'))

        assert resolved.version == "3"
        assert (await registry.get("greeting", alias="current")).version == "4"


class TestOnePerTenant:
    """A tenant override may not become a way to read another tenant's prompt."""

    async def test_a_tenant_override_is_preferred_over_the_shared_prompt(
        self, tmp_path: Path
    ) -> None:
        written(tmp_path, "greeting")
        written(tmp_path / "tenants" / "acme", "greeting", '[versions.1]\nbody = "Hi."\n')
        registry = FilePromptRegistry(tmp_path)

        assert (await registry.get("greeting", version="1", tenant="acme")).body == "Hi."

    async def test_a_tenant_without_an_override_reads_the_shared_prompt(
        self, tmp_path: Path
    ) -> None:
        written(tmp_path, "greeting")
        registry = FilePromptRegistry(tmp_path)

        assert (await registry.get("greeting", alias="current", tenant="acme")).version == "3"

    async def test_no_tenant_can_reach_another_tenants_directory(self, tmp_path: Path) -> None:
        written(tmp_path / "tenants" / "beta", "greeting", '[versions.1]\nbody = "Beta only."\n')
        registry = FilePromptRegistry(tmp_path)

        with pytest.raises(PromptNotFoundError):
            await registry.get("greeting", version="1", tenant="../beta")

    async def test_a_prompt_name_cannot_walk_out_of_the_directory(
        self, registry: FilePromptRegistry
    ) -> None:
        with pytest.raises(PromptNotFoundError):
            await registry.get("../greeting", version="3")


class TestWhatTheAgentAndTheRunCarry:
    """The version is on the run and on every model-call span, with no per-project wiring."""

    async def test_a_prompt_instructs_an_agent_and_is_recorded_on_it(
        self, registry: FilePromptRegistry
    ) -> None:
        prompt = await registry.get("greeting", alias="current")

        agent = prompt.instruct(
            Agent(name="greeter", instructions="placeholder", model="m", free_text=True)
        )

        assert agent.instructions == prompt.body
        assert agent.prompt == prompt.ref

    async def test_a_reference_reads_as_a_name_and_a_version(
        self, registry: FilePromptRegistry
    ) -> None:
        prompt = await registry.get("greeting", alias="current")

        assert prompt.ref.label == "greeting@3"

    async def test_an_agent_with_inline_instructions_carries_no_reference(self) -> None:
        agent = Agent(name="greeter", instructions="Greet them.", model="m", free_text=True)

        assert agent.prompt is None


class TestNothingToWire:
    """The version reaches the run record and the spans without the project doing anything."""

    async def test_a_run_carries_the_prompt_it_ran_on(self, registry: FilePromptRegistry) -> None:
        prompt = await registry.get("greeting", alias="current")
        agent = prompt.instruct(
            Agent(name="greeter", instructions="placeholder", model="scripted-1", free_text=True)
        )

        finished = await AgentRunner(
            provider=ScriptedProvider(
                ModelResponse(content="Hello, Ada.", usage=Usage(input_tokens=8, output_tokens=3)),
                name="scripted",
                capabilities=CAPABLE,
            ),
            clock=FakeClock(),
        ).run(agent, "say hello", tenant="acme")

        assert finished.prompt == prompt.ref

    async def test_every_model_call_span_says_which_prompt_ran(
        self, registry: FilePromptRegistry
    ) -> None:
        prompt = await registry.get("greeting", alias="current")
        agent = prompt.instruct(
            Agent(name="greeter", instructions="placeholder", model="scripted-1", free_text=True)
        )
        finished = await AgentRunner(
            provider=ScriptedProvider(
                ModelResponse(
                    content="Hello, Ada.",
                    usage=Usage(
                        input_tokens=8,
                        output_tokens=3,
                        cost=Cost(input=Decimal("0.01"), currency="USD"),
                    ),
                ),
                name="scripted",
                capabilities=CAPABLE,
            ),
            clock=FakeClock(),
        ).run(agent, "say hello", tenant="acme")

        (record,) = spend_of(finished)
        attributes = attributes_of(record)

        assert attributes["adk.prompt"] == "greeting@3"
        assert attributes["adk.prompt_digest"] == prompt.digest
        assert not any(prompt.body in value for value in attributes.values())


class TestWhetherItFitsAtAll:
    """A prompt that fills the window leaves nothing to answer with."""

    async def test_a_prompt_within_the_window_fits(self, registry: FilePromptRegistry) -> None:
        assert (await registry.get("greeting", version="3")).fits(1_000) is True

    async def test_a_prompt_that_would_dominate_the_window_does_not(self, tmp_path: Path) -> None:
        written(tmp_path, "vast", f'[versions.1]\nbody = "{"word " * 4_000}"\n')

        prompt = await FilePromptRegistry(tmp_path).get("vast", version="1")

        assert prompt.fits(1_000) is False
        assert prompt.estimated_tokens > 1_000


class TestAnyOtherRegistry:
    """The protocol is the public surface; a database-backed one is an implementation."""

    async def test_a_registry_of_one_prompt_satisfies_the_protocol(self) -> None:
        from tesserix_adk.core import PromptRegistry

        class OnePrompt:
            async def get(
                self,
                name: str,
                *,
                version: str = "",  # noqa: ARG002 — the protocol's shape; this one has one prompt
                alias: str = "",  # noqa: ARG002 — same
                tenant: str = "",  # noqa: ARG002 — same
            ) -> PromptDefinition:
                return PromptDefinition(name=name, version="1", body="Greet them.")

            async def list_versions(
                self,
                name: str,  # noqa: ARG002 — same
                *,
                tenant: str = "",  # noqa: ARG002 — same
            ) -> tuple[str, ...]:
                return ("1",)

        assert isinstance(OnePrompt(), PromptRegistry)
