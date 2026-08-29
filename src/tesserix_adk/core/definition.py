"""An agent as a reviewable artifact: what it may do, who answers for it, what checks it.

An `Agent` says what the job is. A definition says what was *agreed* — the same declaration
plus the owner, the evaluation suite it is checked against and the prompt it was written
for — and carries a revision derived from all of it. Without one, model policy, tool
allowlist, limits and owner end up scattered across the call sites that construct the
agent, and nothing can be versioned, diffed in review, or named by a run that has already
happened.

The revision is content-derived on purpose. A definition is frozen, so changing what an
agent does produces a new revision rather than quietly moving what an old run pointed at.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Generic, Self

from pydantic import Field, TypeAdapter, field_validator, model_validator

from tesserix_adk.core.agent import Agent, TypedAgent  # noqa: TC001 — pydantic needs it at runtime
from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel, InputT, OutputT

__all__ = ["AgentDefinition", "Owner", "TypedAgentDefinition"]

_REVISION_LENGTH = 12

# An address or a URL. Anything else is a name, and a name cannot be paged.
_REACHABLE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$|^https?://\S+$")


class Owner(AdkModel):
    """Who answers for an agent when it misbehaves.

    Args:
        team: The team that owns the behaviour, not the repository.
        contact: Where to send the page — an address or a URL. A name is not reachable.
        service: The service that runs it, for joining an agent to a deployment.

    Example:
        >>> Owner(team="search", contact="search@example.gov", service="aequitas-search").team
        'search'
    """

    team: str = Field(min_length=1)
    contact: str = Field(min_length=1)
    service: str = Field(min_length=1)

    @field_validator("contact")
    @classmethod
    def _is_reachable(cls, value: str) -> str:
        if not _REACHABLE.match(value):
            raise ValueError(
                f"{value!r} is not an address or a URL, so nobody can be reached at it; "
                f"an owner who cannot be paged is an owner in name only"
            )
        return value


class AgentDefinition(AdkModel, Generic[OutputT]):  # noqa: UP046 — the parameter carries a default
    """One agent, as the artifact that was reviewed.

    Args:
        agent: The declaration itself — model policy, tool allowlist, limits, answer shape.
        owner: Who answers for it.
        evaluation_suite: The suite this agent is checked against. Required: an agent
            nobody evaluates is an agent whose regressions are found by its users.
        instructions_ref: Optional source identifier for instructions managed outside the
            built-in prompt registry. `PromptDefinition.instruct` records a concrete
            `PromptRef` on the agent and run; this field lets another registry preserve its
            own locator without resolving content inside the definition.
        memory_policy: The named memory policy this agent runs under. `None` means no
            memory, which is the safe reading rather than an unstated default.
        output_schema: The JSON schema of the answer shape. Derived from
            `agent.output_type` when one is declared, and kept as data so a stored
            definition still says what it was reviewed to answer.
        metadata: Consumer-owned annotations. The kit reads nothing from it.

    Example:
        >>> from tesserix_adk.core import Agent
        >>> definition = AgentDefinition(
        ...     agent=Agent(name="clerk", instructions="Cite the page.",
        ...                 model="llama-3.1-8b", free_text=True),
        ...     owner=Owner(team="search", contact="search@example.gov", service="search"),
        ...     evaluation_suite="suites/clerk.yaml",
        ... )
        >>> definition.key
        'clerk@1.0.0'
    """

    agent: Agent[OutputT]
    owner: Owner
    evaluation_suite: str = Field(min_length=1)
    output_schema: dict[str, Any] | None = None
    instructions_ref: str | None = None
    memory_policy: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _record_the_answer_shape(self) -> Self:
        # The answer type is a Python class and is not serialised, so a stored definition
        # would otherwise lose the shape it was reviewed for — and digest as another one.
        if self.output_schema is None and self.agent.output_type is not None:
            object.__setattr__(self, "output_schema", self.agent.output_type.model_json_schema())
        return self

    @classmethod
    def declared(
        cls,
        *,
        agent: Agent[OutputT],
        owner: Owner,
        evaluation_suite: str,
        known_tools: frozenset[str] | tuple[str, ...] | None = None,
        output_schema: dict[str, Any] | None = None,
        instructions_ref: str | None = None,
        memory_policy: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Self:
        """Build a definition, refusing an allowlist naming a tool nobody registered.

        Args:
            agent: The declaration.
            owner: Who answers for it.
            evaluation_suite: What checks it.
            known_tools: The names a registry actually holds. `None` checks nothing, which
                is right where the definition is authored before the registry is built.
            output_schema: The agreed answer shape, where it is not derived from the type.
            instructions_ref: Where the instructions came from.
            memory_policy: The named memory policy.
            metadata: Consumer-owned annotations.

        Returns:
            The definition.

        Raises:
            ConfigurationError: If the allowlist names a tool that is not registered.
                Raised here rather than at the first execution that happens to call it.
        """
        definition = cls(
            agent=agent,
            owner=owner,
            evaluation_suite=evaluation_suite,
            output_schema=output_schema,
            instructions_ref=instructions_ref,
            memory_policy=memory_policy,
            metadata=metadata or {},
        )
        if known_tools is not None:
            stray = sorted(set(agent.tools) - set(known_tools))
            if stray:
                raise ConfigurationError(
                    f"agent.tools names tools that are not registered: {', '.join(stray)}; "
                    f"available: {', '.join(sorted(known_tools)) or 'none'}"
                )
        return definition

    @property
    def name(self) -> str:
        """The agent's name, which is not unique on its own — see `key`."""
        return self.agent.name

    @property
    def version(self) -> str:
        """The declared version. Two versions of one name coexist without collision."""
        return self.agent.version

    @property
    def key(self) -> str:
        """Name and version together, which is what identifies a definition to a human."""
        return f"{self.name}@{self.version}"

    @property
    def revision(self) -> str:
        """A digest of everything this definition says, short enough to read in a trace.

        Derived from the content rather than declared, so an edit cannot pass as the
        revision a past run recorded. It covers the serialised form only, which is why the
        answer shape is kept as a schema rather than left as an unserialised class.
        """
        rendered = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(rendered.encode()).hexdigest()[:_REVISION_LENGTH]


class TypedAgentDefinition(
    AgentDefinition[OutputT],
    Generic[InputT, OutputT],  # noqa: UP046
):
    """A reviewed definition for a `TypedAgent` input and output contract."""

    agent: TypedAgent[InputT, OutputT]
    input_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _record_the_input_shape(self) -> Self:
        if self.input_schema is None:
            object.__setattr__(
                self, "input_schema", TypeAdapter(self.agent.input_type).json_schema()
            )
        return self

    @classmethod
    def declared_typed(
        cls,
        *,
        agent: TypedAgent[InputT, OutputT],
        owner: Owner,
        evaluation_suite: str,
        known_tools: frozenset[str] | tuple[str, ...] | None = None,
        output_schema: dict[str, Any] | None = None,
        input_schema: dict[str, Any] | None = None,
        instructions_ref: str | None = None,
        memory_policy: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Self:
        """Build a typed definition and validate its tool allowlist."""
        definition = cls(
            agent=agent,
            owner=owner,
            evaluation_suite=evaluation_suite,
            output_schema=output_schema,
            input_schema=input_schema,
            instructions_ref=instructions_ref,
            memory_policy=memory_policy,
            metadata=metadata or {},
        )
        if known_tools is not None:
            stray = sorted(set(agent.tools) - set(known_tools))
            if stray:
                raise ConfigurationError(
                    f"agent.tools names tools that are not registered: {', '.join(stray)}; "
                    f"available: {', '.join(sorted(known_tools)) or 'none'}"
                )
        return definition
