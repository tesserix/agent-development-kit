"""A peer's skill offered to a model as an ordinary tool.

The alternative is a hand-written wrapper per peer, and each one is a place where the
contract drifts: the schema the model is shown stops being the schema the peer publishes,
the peer's "this needs a human" is dropped, and an effectful skill gets retried because
the wrapper never said it was effectful. Here the card is the only source of any of that.

Nothing about the call changes by arriving through a tool. Scope is still attenuated, the
budget still charged, the trace still continued, and the peer's answer still returns as
data — the tool is the model's way in, not a second path around `PeerClient`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.a2a import AgentLimits, checkable, conforms
from tesserix_adk.adapters.mcp_surface import namespaced
from tesserix_adk.core import (
    ApprovalPolicy,
    ConfigurationError,
    ContentSource,
    Idempotency,
    IdempotencyPolicy,
    Origin,
    ToolArgumentValidationError,
    screen,
    sealed,
)
from tesserix_adk.tools import Tool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tesserix_adk.a2a import AgentSkill, PeerClient

__all__ = ["PeerArguments", "peer_tool"]


@dataclass(frozen=True, slots=True)
class PeerArguments:
    """The peer's published input schema, made the contract for what a model may send.

    Args:
        schema: The skill's input schema, exactly as the card carries it.
        tool: What the model calls this, for the refusal.
        max_bytes: The largest payload the peer accepts, held before anything is parsed.
    """

    schema: Mapping[str, Any]
    tool: str
    max_bytes: int

    def arguments(self, arguments: object) -> dict[str, Any]:
        """The validated payload, or a refusal that names the field and not its value."""
        parsed = self._parsed(arguments)
        violation = conforms(parsed, self.schema)
        if violation:
            raise self._refusal(
                f"{self.tool} was called with arguments the peer would reject",
                arguments,
                violation,
            )
        return parsed

    def _parsed(self, arguments: object) -> dict[str, Any]:
        """Whatever the provider sent, as one JSON object within the peer's ceiling."""
        if isinstance(arguments, str | bytes):
            self._within_the_ceiling(len(arguments), arguments)
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as failure:
                raise self._refusal(
                    f"{self.tool} was called with arguments that are not valid JSON",
                    arguments,
                    "the arguments are not valid JSON",
                ) from failure
        else:
            try:
                encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError) as failure:
                raise self._refusal(
                    f"{self.tool} was called with arguments that are not JSON data",
                    arguments,
                    "the arguments are not JSON data",
                ) from failure
            self._within_the_ceiling(len(encoded.encode()), arguments)
            parsed = json.loads(encoded)
        if not isinstance(parsed, dict):
            raise self._refusal(
                f"{self.tool} was sent a JSON {type(parsed).__name__}, and a call's arguments "
                f"are a JSON object naming each one",
                arguments,
                "the arguments are not a JSON object",
            )
        return parsed

    def _within_the_ceiling(self, size: int, arguments: object) -> None:
        if size > self.max_bytes:
            raise self._refusal(
                f"{self.tool} was called with {size} bytes of arguments, and the peer accepts "
                f"at most {self.max_bytes}",
                arguments,
                f"the arguments are larger than the {self.max_bytes} bytes the peer accepts",
            )

    def _refusal(
        self, message: str, arguments: object, problem: str
    ) -> ToolArgumentValidationError:
        return ToolArgumentValidationError(
            message,
            tool=self.tool,
            paths=("arguments",),
            problems={"arguments": problem},
            payload=arguments,
        )


def peer_tool(client: PeerClient, skill: str, *, name: str = "") -> Tool[..., dict[str, Any]]:
    """Offer one of a peer's skills to a model as a tool.

    Args:
        client: The client for that peer, already holding its card, its credentials and
            the run's budget. Everything the call is held to comes from there.
        skill: The skill as the card publishes it.
        name: What the model calls it. Defaults to the skill namespaced under the peer,
            so two peers that both price a leg are two tools rather than a collision.

    Returns:
        A tool whose schemas, idempotency and approval gate are the card's.

    Raises:
        ConfigurationError: The card does not publish that skill.
        UnsupportedSchemaError: Its input schema uses something the kit cannot check.
            Refused while the tool is built, rather than on the first call.
    """
    declared = _published(client, skill)
    checkable(declared.input_schema)
    limits = client.card.limits or AgentLimits()
    called = name or namespaced(skill, prefix=client.card.agent)

    async def invoke(**arguments: Any) -> dict[str, Any]:  # noqa: ANN401 — the card's schema
        return (await client.invoke(skill, arguments)).output

    return Tool[..., dict[str, Any]](
        name=called,
        description=_described(declared, client.card.agent),
        parameters_schema=dict(declared.input_schema),
        returns_schema=dict(declared.output_schema) if declared.output_schema else None,
        is_async=True,
        function=invoke,
        validator=PeerArguments(
            schema=declared.input_schema,
            tool=called,
            max_bytes=limits.max_payload_bytes,
        ),
        context_parameter=None,
        context_required=False,
        timeout=client.timeout_seconds,
        parallel_safe=declared.idempotent,
        approval=_approval(declared, client.card.agent),
        idempotency=IdempotencyPolicy(
            kind=Idempotency.IDEMPOTENT if declared.idempotent else Idempotency.EFFECTFUL
        ),
        returns_type=dict,
    )


def _published(client: PeerClient, skill: str) -> AgentSkill:
    """The skill as the peer published it, or a refusal naming what was asked for."""
    for declared in client.card.skills:
        if declared.name == skill:
            return declared
    raise ConfigurationError(
        f"{client.card.agent} does not publish {skill}, so it cannot be offered as a tool"
    )


def _described(declared: AgentSkill, agent: str) -> str:
    """A description another agent wrote, fenced where it reads as an instruction.

    The tool list is the most privileged place a peer's text reaches. One that screens as
    instructions is kept — the operator chose the peer — but as data rather than as a line.
    """
    if not screen(declared.description):
        return declared.description
    return sealed(
        declared.description,
        source=ContentSource(origin=Origin.PEER_AGENT, name=f"{agent}/{declared.name}"),
    )


def _approval(declared: AgentSkill, agent: str) -> ApprovalPolicy:
    """The peer's own gate, held on this side too, so the human is asked before the call."""
    if not declared.requires_approval:
        return ApprovalPolicy()
    return ApprovalPolicy(
        required=True, reason=f"{agent} requires a human before {declared.name} runs"
    )
