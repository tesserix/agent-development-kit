"""Assemble a Tesserix A2A server and Google ADK remote agent without network I/O.

Run with ``uv run --extra google-adk python examples/google_adk_a2a_bridge.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesserix_adk.adapters import (
    A2ABearerSecurity,
    A2AInterface,
    A2ASkill,
    a2a_agent_executor,
    a2a_card_for,
    google_adk_remote_agent,
)
from tesserix_adk.core import Agent, AgentDefinition, MissingExtraError, Owner, Principal
from tesserix_adk.core.provider import ModelResponse
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock, ScriptedProvider

if TYPE_CHECKING:
    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from starlette.requests import Request


def definition() -> AgentDefinition[Any]:
    """Return the reviewed definition used for the public card and runner."""
    return AgentDefinition(
        agent=Agent(
            name="trip-planner",
            version="1.0.0",
            instructions="Plan a concise itinerary.",
            model="provider-model",
            free_text=True,
            scopes=("trips:read",),
        ),
        owner=Owner(
            team="Travel Platform",
            contact="https://agents.example.test/support",
            service="trip-planner-api",
        ),
        evaluation_suite="evals/trip-planner.jsonl",
    )


def main() -> None:
    """Build both sides of the bridge and report the mounted surfaces."""
    reviewed = definition()
    try:
        card = a2a_card_for(
            reviewed,
            description="Plans itineraries from traveller constraints.",
            provider_url="https://agents.example.test",
            documentation_url="https://agents.example.test/docs/trip-planner",
            interfaces=(
                A2AInterface(
                    url="https://agents.example.test/a2a/trip-planner",
                    protocol_binding="JSONRPC",
                ),
            ),
            skills=(
                A2ASkill(
                    id="plan-trip",
                    name="Plan a trip",
                    description="Creates a day-by-day itinerary.",
                    tags=("travel", "planning"),
                ),
            ),
            security=A2ABearerSecurity(scopes=("trips:read",)),
        )
    except MissingExtraError:
        print("install with: uv sync --frozen --extra google-adk")  # noqa: T201
        return

    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import (
        DefaultServerCallContextBuilder,
        create_agent_card_routes,
        create_jsonrpc_routes,
    )
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.applications import Starlette

    class VerifiedContextBuilder(DefaultServerCallContextBuilder):
        """Copy only a principal already authenticated by server middleware."""

        def build(self, request: Request) -> ServerCallContext:
            call_context = super().build(request)
            principal = request.scope.get("verified_principal")
            if isinstance(principal, Principal):
                call_context.state["principal"] = principal
                call_context.tenant = principal.tenant
            return call_context

    async def resolve_principal(context: RequestContext) -> Principal:
        principal = context.call_context.state.get("principal")
        if not isinstance(principal, Principal):
            raise PermissionError("verified A2A principal is required")
        return principal

    runner = AgentRunner(
        provider=ScriptedProvider(ModelResponse(content="Explore Melbourne's laneways.")),
        clock=FakeClock(),
    )
    executor = a2a_agent_executor(runner, reviewed, resolve=resolve_principal)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(
                handler,
                rpc_url="/a2a/trip-planner",
                context_builder=VerifiedContextBuilder(),
            ),
        ]
    )

    try:
        remote = google_adk_remote_agent(
            name="tesserix_trip_planner",
            description="A Tesserix agent reached through official A2A.",
            agent_card=card,
            timeout_seconds=30.0,
        )
    except MissingExtraError:
        print("server ready; add --extra google-adk for the Google client")  # noqa: T201
        return

    print(f"server routes: {len(app.routes)}")  # noqa: T201
    print(f"google remote: {remote.name}")  # noqa: T201
    print("production must replace InMemoryTaskStore and authenticate middleware")  # noqa: T201


if __name__ == "__main__":
    main()
