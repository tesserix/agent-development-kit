"""Explicit interoperability with Google's Agent Development Kit.

Google FunctionTools enter through the framework-neutral importer, so their schema,
approval, concurrency, idempotency, tenant context and provenance are kit policy rather
than a parallel implementation. Google agents enter through the generic sub-agent wrapper;
the application supplies the Google Runner invocation because sessions, credentials and
artifacts remain application-owned resources.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel

from tesserix_adk.adapters.foreign_agents import (
    ForeignAgentContext,
    WrappedAgentPolicy,
    WrappedSubagent,
    wrap_agent_as_subagent,
)
from tesserix_adk.adapters.interop import (
    ToolImportPolicy,
    ToolTranslationError,
    import_tool,
    import_toolset,
)
from tesserix_adk.core import INLINE_REFS, ConfigurationError, schema_for
from tesserix_adk.core.extras import require_extra

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping

    from a2a.types import AgentCard
    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

    from tesserix_adk.core.guards import GuardrailPipeline
    from tesserix_adk.tools import Tool, ToolContext

__all__ = [
    "GOOGLE_ADK_CONTEXT_KEY",
    "GoogleAdkAgentInvoker",
    "google_adk_remote_agent",
    "import_google_adk_tool",
    "import_google_adk_toolset",
    "wrap_google_adk_agent",
]

GOOGLE_ADK_CONTEXT_KEY = "tesserix_adk"
"""Ephemeral Google session-state key containing credential-free caller context."""

_APP_NAME = "tesserix-adk-interop"
_TRACE_KEYS = frozenset({"baggage", "traceparent", "tracestate"})


class GoogleAdkAgentInvoker[InputT: BaseModel](Protocol):
    """Application-owned invocation of one Google agent under delegated context.

    The callable may return a raw structured output or ``ForeignAgentReply`` with measured
    usage. It owns Google Runner, session, artifact and credential configuration; it must
    not place credentials in ``ForeignAgentContext`` or its returned payload.
    """

    def __call__(
        self,
        agent: object,
        request: InputT,
        context: ForeignAgentContext,
    ) -> object:
        """Invoke ``agent`` without widening the supplied caller context."""
        ...


class _SessionService(Protocol):
    def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Mapping[str, object],
        session_id: str,
    ) -> Awaitable[object]: ...

    def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> Awaitable[None]: ...


@dataclass(frozen=True, slots=True)
class _GoogleToolCall:
    session_service: _SessionService
    context: object
    user_id: str
    session_id: str


def google_adk_remote_agent(
    *,
    name: str,
    agent_card: AgentCard | str,
    description: str = "",
    timeout_seconds: float = 60.0,
) -> RemoteA2aAgent:
    """Create a Google ADK remote agent for a Tesserix official A2A endpoint.

    The helper selects Google ADK's current A2A 1.x implementation instead of its legacy
    compatibility path. Authentication still belongs in Google ADK's credential or A2A
    client configuration; this function never accepts or stores a token.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive.
        MissingExtraError: If ``tesserix-adk[google-adk]`` is not installed.
    """
    require_extra("google-adk", "google.adk")
    if timeout_seconds <= 0:
        raise ValueError("a Google ADK remote-agent timeout must be positive")

    from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

    return RemoteA2aAgent(
        name=name,
        description=description,
        agent_card=agent_card,
        timeout=timeout_seconds,
        use_legacy=False,
    )


def import_google_adk_tool(
    source: object,
    *,
    policy: ToolImportPolicy,
) -> Tool[..., object]:
    """Import one Google FunctionTool through the generic kit tool boundary.

    The official Google invocation path is retained, including its argument conversion and
    ToolContext. The Google context contains only credential-free caller metadata in
    ``state[GOOGLE_ADK_CONTEXT_KEY]`` and is deleted with its ephemeral session after the
    call. Kit approval is presented as Google confirmation only after the kit chose to run
    an approval-bearing tool.

    Raises:
        MissingExtraError: If ``tesserix-adk[google-adk]`` is not installed.
        ToolTranslationError: If ``source`` is not a Google FunctionTool, has no usable
            typed schema, or requires confirmation while kit policy does not.
    """
    tools_module = require_extra("google-adk", "google.adk.tools")
    function_tool_type = getattr(tools_module, "FunctionTool", None)
    if not isinstance(function_tool_type, type) or not isinstance(source, function_tool_type):
        raise ToolTranslationError(
            "Google Agent Development Kit interop accepts FunctionTool definitions only",
            source="google-adk",
            construct="tool-type",
        )
    body = getattr(source, "func", None)
    name = getattr(source, "name", "")
    description = getattr(source, "description", "")
    if not callable(body) or not isinstance(name, str) or not name:
        raise ToolTranslationError(
            "a Google FunctionTool needs a named callable body",
            source="google-adk",
            construct="implementation",
        )
    confirmation = getattr(source, "_require_confirmation", False)
    if confirmation and not policy.requires_approval:
        raise ToolTranslationError(
            f"Google FunctionTool {name!r} requires confirmation but kit policy does not",
            tool=name,
            source=f"google-adk:function-tool:{name}",
            construct="approval",
        )
    try:
        parameters = schema_for(
            body,
            dialect=INLINE_REFS,
            exclude=("input_stream", "tool_context"),
        )
    except Exception as failure:
        raise ToolTranslationError(
            f"Google FunctionTool {name!r} has no portable typed input schema",
            tool=name,
            source=f"google-adk:function-tool:{name}",
            construct="schema",
        ) from failure

    async def invoke(*, context: ToolContext, **arguments: object) -> object:
        return await _invoke_google_tool(source, arguments, context, policy=policy)

    descriptor: dict[str, object] = {
        "type": "function",
        "function": {
            "name": name,
            "description": description if isinstance(description, str) else "",
            "parameters": parameters,
        },
    }
    return import_tool(
        descriptor,
        policy=policy,
        implementation=invoke,
        context_parameter="context",
        provenance=f"google-adk:function-tool:{name}",
    )


def import_google_adk_toolset(
    sources: Iterable[object],
    *,
    policy: ToolImportPolicy,
    known: Collection[str] = (),
) -> tuple[Tool[..., object], ...]:
    """Import Google FunctionTools and apply generic duplicate-name admission.

    Raises:
        MissingExtraError: If ``tesserix-adk[google-adk]`` is not installed.
        ToolTranslationError: If any definition cannot be translated or a name collides.
    """
    translated = tuple(import_google_adk_tool(source, policy=policy) for source in sources)
    return import_toolset(translated, policy=policy, known=known)


def wrap_google_adk_agent[InputT: BaseModel, OutputT: BaseModel](
    source: object,
    *,
    invoke: GoogleAdkAgentInvoker[InputT],
    input_type: type[InputT],
    output_type: type[OutputT],
    policy: WrappedAgentPolicy,
    guardrails: GuardrailPipeline | None = None,
    name: str | None = None,
) -> WrappedSubagent[InputT, OutputT]:
    """Wrap a Google BaseAgent as a typed, budgeted kit sub-agent.

    ``invoke`` is explicit because Google Runner sessions, persistence, plugins, artifacts
    and credentials are application resources. The generic wrapper performs cost preflight,
    scope/tool intersection, timeout, cancellation, lineage, guardrails and strict output
    validation around that callback.

    Raises:
        MissingExtraError: If ``tesserix-adk[google-adk]`` is not installed.
        ConfigurationError: If ``source`` is not a Google BaseAgent or has no usable name.
    """
    agents_module = require_extra("google-adk", "google.adk.agents")
    base_agent_type = getattr(agents_module, "BaseAgent", None)
    if not isinstance(base_agent_type, type) or not isinstance(source, base_agent_type):
        raise ConfigurationError("Google agent interop requires a google.adk.agents.BaseAgent")
    resolved_name = name or getattr(source, "name", "")
    if not isinstance(resolved_name, str) or not resolved_name.strip():
        raise ConfigurationError("a wrapped Google agent needs a non-empty local name")

    async def foreign(request: InputT, context: ForeignAgentContext) -> object:
        produced = invoke(source, request, context)
        if inspect.isawaitable(produced):
            return await cast("Awaitable[object]", produced)
        return produced

    return wrap_agent_as_subagent(
        foreign,
        name=resolved_name,
        input_type=input_type,
        output_type=output_type,
        policy=policy,
        guardrails=guardrails,
        provenance=f"google-adk:agent:{resolved_name}",
    )


async def _invoke_google_tool(
    source: object,
    arguments: Mapping[str, object],
    context: ToolContext,
    *,
    policy: ToolImportPolicy,
) -> object:
    call = await _google_tool_context(context, getattr(source, "name", "tool"), policy)
    run_async = getattr(source, "run_async", None)
    if not callable(run_async):
        raise ToolTranslationError(
            "a Google FunctionTool has no async invocation surface",
            source="google-adk",
            construct="implementation",
        )
    try:
        result = run_async(args=dict(arguments), tool_context=call.context)
        if not inspect.isawaitable(result):
            raise ToolTranslationError(
                "a Google FunctionTool returned before its async invocation completed",
                source="google-adk",
                construct="implementation",
            )
        return await cast("Awaitable[object]", result)
    finally:
        await call.session_service.delete_session(
            app_name=_APP_NAME,
            user_id=call.user_id,
            session_id=call.session_id,
        )


async def _google_tool_context(
    context: ToolContext,
    name: object,
    policy: ToolImportPolicy,
) -> _GoogleToolCall:
    sessions_module = require_extra("google-adk", "google.adk.sessions")
    invocation_module = require_extra("google-adk", "google.adk.agents.invocation_context")
    events_module = require_extra("google-adk", "google.adk.events")
    tools_module = require_extra("google-adk", "google.adk.tools")
    confirmation_module = require_extra("google-adk", "google.adk.tools.tool_confirmation")

    service_factory = cast(
        "Callable[[], _SessionService]",
        sessions_module.InMemorySessionService,
    )
    invocation_factory = cast(
        "Callable[..., object]",
        invocation_module.InvocationContext,
    )
    actions_factory = cast("Callable[..., object]", events_module.EventActions)
    context_factory = cast("Callable[..., object]", tools_module.ToolContext)
    confirmation_factory = cast(
        "Callable[..., object]",
        confirmation_module.ToolConfirmation,
    )

    service = service_factory()
    user_id = context.user or "service"
    session_id = _identifier(context.tenant, context.run_id, str(name))
    state = {
        GOOGLE_ADK_CONTEXT_KEY: {
            "run_id": context.run_id,
            "tenant": context.tenant,
            "user": context.user,
            "scopes": list(context.scopes),
            "trace": {
                key.lower(): value
                for key, value in context.trace.items()
                if key.lower() in _TRACE_KEYS
            },
        }
    }
    session = await service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )
    invocation = invocation_factory(
        session_service=service,
        invocation_id=context.run_id,
        session=session,
    )
    confirmation = confirmation_factory(confirmed=True) if policy.requires_approval else None
    google_context = context_factory(
        invocation,
        event_actions=actions_factory(),
        function_call_id=_identifier(context.run_id, str(name)),
        tool_confirmation=confirmation,
        run_id=context.run_id,
    )
    return _GoogleToolCall(
        session_service=service,
        context=google_context,
        user_id=user_id,
        session_id=session_id,
    )


def _identifier(*parts: str) -> str:
    encoded = "\x1f".join(parts).encode()
    return f"tesserix-{hashlib.sha256(encoded).hexdigest()[:32]}"
