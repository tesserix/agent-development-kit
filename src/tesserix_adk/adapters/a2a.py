"""Official A2A 1.x cards and clients behind the optional reference SDK.

The kit's :mod:`tesserix_adk.a2a` package is its richer typed peer protocol. This
adapter is deliberately separate: it emits and consumes the official A2A wire types,
so supporting one protocol never silently claims conformance to the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from tesserix_adk.core.errors import AdkError
from tesserix_adk.core.extras import require_extra
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

    from a2a.client import Client, ClientCallInterceptor, ClientFactory
    from a2a.client.client_factory import TransportProducer
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.types import AgentCard

    from tesserix_adk.core.definition import AgentDefinition
    from tesserix_adk.core.identity import Principal
    from tesserix_adk.runtime import AgentRunner

__all__ = [
    "A2ABearerSecurity",
    "A2ACardError",
    "A2AExecutionError",
    "A2AInterface",
    "A2APrincipalResolver",
    "A2ARegistry",
    "A2ARegistryError",
    "A2ASkill",
    "a2a_agent_executor",
    "a2a_card_for",
    "a2a_client_factory",
    "a2a_client_from_registry",
]

A2A_PROTOCOL_VERSION = "1.0"
_DEFAULT_MODES = ("text/plain",)


class A2ARegistryError(AdkError):
    """Raised when a registry answer cannot safely identify the requested agent."""


class A2ACardError(AdkError):
    """Raised when required official Agent Card metadata is incomplete or ambiguous."""


class A2AExecutionError(AdkError):
    """Raised when an official A2A request cannot be mapped to a Tesserix run."""


@runtime_checkable
class A2APrincipalResolver(Protocol):
    """Authenticate and authorise one A2A request at the server trust boundary.

    The resolver must derive the principal from verified server or gateway context. It
    must reject expired authority on the same clock domain used by the runner and never
    trust a tenant, subject, or scope copied from the A2A message body. The runner checks
    principal liveness again before executing model or tool work.
    """

    def __call__(self, context: RequestContext, /) -> Awaitable[Principal]:
        """Return the verified principal allowed to invoke this agent."""


class A2AInterface(AdkModel):
    """One official A2A endpoint, including custom gateway bindings.

    Args:
        url: Absolute endpoint URL advertised to clients.
        protocol_binding: `JSONRPC`, `HTTP+JSON`, `GRPC`, or a custom binding registered
            with the official client's factory.
        protocol_version: A2A protocol version spoken by this interface.
        tenant: Optional A2A tenant path value for a tenant-specific interface.
    """

    url: str = Field(min_length=1)
    protocol_binding: str = Field(min_length=1)
    protocol_version: str = A2A_PROTOCOL_VERSION
    tenant: str = ""

    @field_validator("url")
    @classmethod
    def _absolute_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("an A2A interface URL must be absolute")
        return value


class A2ASkill(AdkModel):
    """A public task capability in an official A2A Agent Card.

    An A2A skill describes what a caller may ask the agent to accomplish. It is not an
    internal tool export; publishing those implementation details would couple peers to
    how the agent happens to do its work.

    Args:
        id: Stable machine-readable identifier.
        name: Human-readable capability name.
        description: What outcome the capability provides.
        tags: Discovery terms. A2A 1.x requires at least one.
        examples: Representative requests.
        input_modes: Modes that override the card defaults for this skill.
        output_modes: Modes that override the card defaults for this skill.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = Field(min_length=1)
    examples: tuple[str, ...] = ()
    input_modes: tuple[str, ...] = ()
    output_modes: tuple[str, ...] = ()


class A2ABearerSecurity(AdkModel):
    """Bearer authentication metadata published in an official Agent Card.

    This declares the contract only. The gateway or A2A server must still authenticate
    every request and authorise each task against its verified principal.

    Args:
        name: Security-scheme key referenced by requirements.
        description: Public description of the credential.
        bearer_format: Token format hint, conventionally `JWT`.
        scopes: Scopes required by the card-level security requirement.
    """

    name: str = "bearer"
    description: str = "Bearer access token"
    bearer_format: str = "JWT"
    scopes: tuple[str, ...] = ()


@runtime_checkable
class A2ARegistry(Protocol):
    """A vendor-neutral registry that resolves a name to an official Agent Card."""

    async def resolve(self, name: str) -> AgentCard:
        """Return the signed or otherwise trusted card registered under `name`."""


def a2a_card_for[OutputT: BaseModel](
    definition: AgentDefinition[OutputT],
    *,
    description: str,
    provider_url: str,
    interfaces: Iterable[A2AInterface],
    skills: Iterable[A2ASkill],
    documentation_url: str = "",
    default_input_modes: Sequence[str] = _DEFAULT_MODES,
    default_output_modes: Sequence[str] = _DEFAULT_MODES,
    streaming: bool = False,
    push_notifications: bool = False,
    extended_agent_card: bool = False,
    security: A2ABearerSecurity | None = None,
) -> AgentCard:
    """Build an official A2A 1.x Agent Card from a reviewed definition.

    Only public metadata supplied to this function is copied. Instructions, model choice,
    evaluation suite, and on-call contact remain outside discovery metadata.

    Raises:
        A2ACardError: If a required repeated field is empty or identifiers collide.
        MissingExtraError: If ``tesserix-adk[a2a]`` is not installed.
    """
    require_extra("a2a", "a2a")
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentProvider,
        AgentSkill,
        HTTPAuthSecurityScheme,
        SecurityRequirement,
        SecurityScheme,
        StringList,
    )

    public_interfaces = tuple(interfaces)
    public_skills = tuple(skills)
    input_modes = tuple(default_input_modes)
    output_modes = tuple(default_output_modes)
    _required_card_fields(
        description=description,
        provider_url=provider_url,
        interfaces=public_interfaces,
        skills=public_skills,
        input_modes=input_modes,
        output_modes=output_modes,
    )

    security_schemes: dict[str, SecurityScheme] = {}
    security_requirements: list[SecurityRequirement] = []
    if security is not None:
        security_schemes[security.name] = SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                description=security.description,
                scheme="bearer",
                bearer_format=security.bearer_format,
            )
        )
        security_requirements.append(
            SecurityRequirement(schemes={security.name: StringList(list=list(security.scopes))})
        )

    capabilities = AgentCapabilities()
    if streaming:
        capabilities.streaming = True
    if push_notifications:
        capabilities.push_notifications = True
    if extended_agent_card:
        capabilities.extended_agent_card = True

    return AgentCard(
        name=definition.name,
        description=description.strip(),
        supported_interfaces=[
            AgentInterface(
                url=interface.url,
                protocol_binding=interface.protocol_binding,
                protocol_version=interface.protocol_version,
                tenant=interface.tenant,
            )
            for interface in public_interfaces
        ],
        provider=AgentProvider(
            url=provider_url,
            organization=definition.owner.team,
        ),
        version=definition.version,
        documentation_url=documentation_url,
        capabilities=capabilities,
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        default_input_modes=list(input_modes),
        default_output_modes=list(output_modes),
        skills=[
            AgentSkill(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                tags=list(skill.tags),
                examples=list(skill.examples),
                input_modes=list(skill.input_modes),
                output_modes=list(skill.output_modes),
            )
            for skill in public_skills
        ],
    )


def a2a_agent_executor[OutputT: BaseModel](
    runner: AgentRunner,
    definition: AgentDefinition[OutputT],
    *,
    resolve: A2APrincipalResolver,
    max_input_bytes: int = 64 * 1024,
    max_output_bytes: int = 1024 * 1024,
) -> AgentExecutor:
    """Expose one reviewed Tesserix agent through the official A2A server contract.

    The official request handler and ``TaskStore`` retain ownership of transport,
    persistence, duplicate delivery, and resubscription. This executor validates text
    input, resolves a verified principal, binds its tenant and scopes to the normal
    Tesserix runner, and maps the terminal run to an A2A task artifact and status.

    Args:
        runner: Fully configured runner whose budgets, guardrails, tools, and telemetry
            remain in force.
        definition: Reviewed agent definition to execute.
        resolve: Authentication and per-agent authorisation at the serving boundary.
        max_input_bytes: Maximum UTF-8 request text accepted after joining text parts.
        max_output_bytes: Maximum UTF-8 answer artifact returned to the peer.

    Returns:
        An official SDK ``AgentExecutor`` for ``DefaultRequestHandler`` or a custom
        gateway handler.

    Raises:
        A2AExecutionError: If either byte limit is not positive.
        MissingExtraError: If ``tesserix-adk[a2a]`` is not installed.
    """
    require_extra("a2a", "a2a")
    if max_input_bytes < 1 or max_output_bytes < 1:
        raise A2AExecutionError(
            "A2A input and output byte limits must be positive",
            details={
                "max_input_bytes": str(max_input_bytes),
                "max_output_bytes": str(max_output_bytes),
            },
        )

    from tesserix_adk.adapters._a2a_server import make_executor

    return make_executor(
        runner,
        definition,
        resolve=resolve,
        max_input_bytes=max_input_bytes,
        max_output_bytes=max_output_bytes,
    )


def a2a_client_factory(
    *,
    protocol_bindings: Iterable[str] = ("JSONRPC",),
    transports: Mapping[str, TransportProducer] | None = None,
    streaming: bool = True,
    polling: bool = False,
    accepted_output_modes: Iterable[str] = (),
    use_client_preference: bool = False,
) -> ClientFactory:
    """Create the official SDK factory with optional custom gateway transports.

    Custom transport labels are automatically added to the client's supported bindings,
    so a card advertising that label can be selected through the normal SDK negotiation.

    Raises:
        MissingExtraError: If ``tesserix-adk[a2a]`` is not installed.
    """
    require_extra("a2a", "a2a")
    from a2a.client import ClientConfig, ClientFactory

    custom = dict(transports or {})
    bindings = list(dict.fromkeys((*protocol_bindings, *custom)))
    factory = ClientFactory(
        ClientConfig(
            streaming=streaming,
            polling=polling,
            supported_protocol_bindings=bindings,
            accepted_output_modes=list(accepted_output_modes),
            use_client_preference=use_client_preference,
        )
    )
    for label, producer in custom.items():
        factory.register(label, producer)
    return factory


async def a2a_client_from_registry(
    registry: A2ARegistry,
    name: str,
    *,
    factory: ClientFactory,
    interceptors: Sequence[ClientCallInterceptor] = (),
    verify: Callable[[AgentCard], None] | None = None,
) -> Client:
    """Resolve `name`, verify its identity, and create an official A2A client.

    The caller supplies the factory so its HTTP client and connection lifecycle remain
    explicit. A verifier can enforce signed-card or registry-specific trust policy.

    Raises:
        A2ARegistryError: If the registry substitutes a differently named agent.
    """
    card = await registry.resolve(name)
    if card.name != name:
        raise A2ARegistryError(
            f"registry resolved {name!r} to {card.name!r}; refusing an agent substitution",
            details={"requested": name, "resolved": card.name},
        )
    if verify is not None:
        verify(card)
    return factory.create(card, list(interceptors))


def _required_card_fields(
    *,
    description: str,
    provider_url: str,
    interfaces: tuple[A2AInterface, ...],
    skills: tuple[A2ASkill, ...],
    input_modes: tuple[str, ...],
    output_modes: tuple[str, ...],
) -> None:
    """Hold the proto's required-field annotations before bytes reach a peer."""
    missing = [
        label
        for label, value in (
            ("description", description.strip()),
            ("provider_url", provider_url.strip()),
            ("interfaces", interfaces),
            ("skills", skills),
            ("default_input_modes", input_modes),
            ("default_output_modes", output_modes),
        )
        if not value
    ]
    if missing:
        raise A2ACardError(
            f"an official A2A Agent Card requires {', '.join(missing)}",
            details={"missing": ",".join(missing)},
        )
    ids = [skill.id for skill in skills]
    if len(ids) != len(set(ids)):
        raise A2ACardError(
            "an official A2A Agent Card requires unique skill ids",
            details={"skill_ids": ",".join(ids)},
        )
    parsed = urlsplit(provider_url)
    if not parsed.scheme or not parsed.netloc:
        raise A2ACardError(
            "an official A2A provider URL must be absolute",
            details={"provider_url": provider_url},
        )
