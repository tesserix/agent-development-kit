"""Official A2A server execution through the real Tesserix runner."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import httpx
import pytest

pytest.importorskip("a2a")

from a2a.client import ClientConfig, ClientFactory
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import TaskNotFoundError
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Value
from starlette.applications import Starlette

from tesserix_adk.adapters import (
    A2AInterface,
    A2ASkill,
    a2a_agent_executor,
    a2a_card_for,
)
from tesserix_adk.core import (
    Agent,
    AgentDefinition,
    HookDecision,
    HookPoint,
    HookSubject,
    Owner,
    Principal,
)
from tesserix_adk.core.provider import ModelResponse
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock, ScriptedProvider, StallingProvider


class RecordingQueue(EventQueue):
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class AttributionRecorder:
    def __init__(self) -> None:
        self.seen: list[HookSubject] = []

    @property
    def name(self) -> str:
        return "a2a-attribution"

    @property
    def points(self) -> tuple[HookPoint, ...]:
        return (HookPoint.BEFORE_MODEL_CALL,)

    async def on(self, subject: HookSubject) -> HookDecision:
        self.seen.append(subject)
        return HookDecision.proceed()


def definition() -> AgentDefinition[Any]:
    return AgentDefinition(
        agent=Agent(
            name="trip-planner",
            version="1.0.0",
            instructions="Plan trips.",
            model="test-model",
            free_text=True,
            scopes=("trips:read",),
        ),
        owner=Owner(team="Travel", contact="travel@example.test", service="planner"),
        evaluation_suite="evals/travel.jsonl",
    )


def context(*parts: Part, text: str = "Plan three days in Melbourne.") -> RequestContext:
    content = list(parts) or [Part(text=text)]
    return RequestContext(
        ServerCallContext(),
        SendMessageRequest(
            message=Message(
                message_id="message-1",
                role=Role.ROLE_USER,
                parts=content,
            )
        ),
        task_id="task-1",
        context_id="context-1",
    )


async def verified_principal(_context: RequestContext) -> Principal:
    return Principal(
        subject="ada",
        tenant="acme",
        scopes=frozenset({"trips:read"}),
    )


def states(queue: RecordingQueue) -> list[int]:
    return [
        event.status.state for event in queue.events if isinstance(event, TaskStatusUpdateEvent)
    ]


class TestA2AExecution:
    async def test_a_verified_request_runs_with_attribution_and_returns_an_artifact(self) -> None:
        recorder = AttributionRecorder()
        provider = ScriptedProvider(ModelResponse(content="Day one: explore Melbourne."))
        runner = AgentRunner(provider=provider, clock=FakeClock(), hooks=(recorder,))
        queue = RecordingQueue()

        executor = a2a_agent_executor(runner, definition(), resolve=verified_principal)
        await executor.execute(context(), queue)

        assert isinstance(queue.events[0], Task)
        assert states(queue) == [
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_COMPLETED,
        ]
        artifacts = [event for event in queue.events if isinstance(event, TaskArtifactUpdateEvent)]
        assert len(artifacts) == 1
        assert artifacts[0].artifact.name == "answer"
        assert artifacts[0].artifact.parts[0].text == "Day one: explore Melbourne."
        assert artifacts[0].artifact.parts[0].media_type == "text/plain"
        assert artifacts[0].last_chunk
        assert len(recorder.seen) == 1
        subject = recorder.seen[0]
        assert (subject.run_id, subject.tenant, subject.user) == (
            "task-1",
            "acme",
            "ada",
        )

    async def test_identity_resolution_fails_closed_before_the_runner_is_called(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="must not run"))
        queue = RecordingQueue()

        async def denied(_context: RequestContext) -> Principal:
            raise PermissionError("not entitled")

        executor = a2a_agent_executor(
            AgentRunner(provider=provider, clock=FakeClock()),
            definition(),
            resolve=denied,
        )
        await executor.execute(context(), queue)

        assert provider.requests == []
        assert states(queue) == [TaskState.TASK_STATE_REJECTED]

    async def test_principal_expiry_uses_the_runner_clock_domain(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="Still authorised."))
        queue = RecordingQueue()

        async def unexpired_on_runner_clock(_context: RequestContext) -> Principal:
            return Principal(
                subject="ada",
                tenant="acme",
                scopes=frozenset({"trips:read"}),
                expires_at=10.0,
            )

        executor = a2a_agent_executor(
            AgentRunner(provider=provider, clock=FakeClock(start=5.0)),
            definition(),
            resolve=unexpired_on_runner_clock,
        )

        await executor.execute(context(), queue)

        assert len(provider.requests) == 1
        assert states(queue)[-1] == TaskState.TASK_STATE_COMPLETED

    @pytest.mark.parametrize(
        "part",
        [Part(raw=b"binary"), Part(data=Value(string_value="private"))],
        ids=("raw", "data"),
    )
    async def test_non_text_input_is_rejected_instead_of_silently_dropped(self, part: Part) -> None:
        provider = ScriptedProvider(ModelResponse(content="must not run"))
        queue = RecordingQueue()
        executor = a2a_agent_executor(
            AgentRunner(provider=provider, clock=FakeClock()),
            definition(),
            resolve=verified_principal,
        )

        await executor.execute(context(part), queue)

        assert provider.requests == []
        assert states(queue) == [TaskState.TASK_STATE_REJECTED]

    async def test_input_and_output_limits_fail_before_unbounded_data_crosses_the_boundary(
        self,
    ) -> None:
        input_provider = ScriptedProvider(ModelResponse(content="must not run"))
        input_queue = RecordingQueue()
        input_executor = a2a_agent_executor(
            AgentRunner(provider=input_provider, clock=FakeClock()),
            definition(),
            resolve=verified_principal,
            max_input_bytes=4,
        )
        await input_executor.execute(context(text="12345"), input_queue)
        assert input_provider.requests == []
        assert states(input_queue) == [TaskState.TASK_STATE_REJECTED]

        output_provider = ScriptedProvider(ModelResponse(content="12345"))
        output_queue = RecordingQueue()
        output_executor = a2a_agent_executor(
            AgentRunner(provider=output_provider, clock=FakeClock()),
            definition(),
            resolve=verified_principal,
            max_output_bytes=4,
        )
        await output_executor.execute(context(), output_queue)
        assert not any(isinstance(event, TaskArtifactUpdateEvent) for event in output_queue.events)
        assert states(output_queue)[-1] == TaskState.TASK_STATE_FAILED

    async def test_a2a_cancellation_stops_the_tesserix_run_and_emits_one_terminal_state(
        self,
    ) -> None:
        provider = StallingProvider()
        queue = RecordingQueue()
        request = context()
        executor = a2a_agent_executor(
            AgentRunner(provider=provider, clock=FakeClock()),
            definition(),
            resolve=verified_principal,
        )

        execution = asyncio.create_task(executor.execute(request, queue))
        await provider.entered.wait()
        await executor.cancel(request, queue)
        async with asyncio.timeout(1):
            await execution

        assert states(queue).count(TaskState.TASK_STATE_CANCELED) == 1

    async def test_a_different_principal_cannot_cancel_an_active_task(self) -> None:
        provider = StallingProvider()
        queue = RecordingQueue()
        request = context()
        resolutions = 0

        async def owner_then_attacker(_context: RequestContext) -> Principal:
            nonlocal resolutions
            resolutions += 1
            if resolutions == 1:
                return await verified_principal(_context)
            return Principal(
                subject="mallory",
                tenant="other-tenant",
                scopes=frozenset({"trips:read"}),
            )

        executor = a2a_agent_executor(
            AgentRunner(provider=provider, clock=FakeClock()),
            definition(),
            resolve=owner_then_attacker,
        )
        execution = asyncio.create_task(executor.execute(request, queue))
        await provider.entered.wait()

        try:
            with pytest.raises(TaskNotFoundError):
                await executor.cancel(request, queue)
            assert not execution.done()
            assert TaskState.TASK_STATE_CANCELED not in states(queue)
        finally:
            execution.cancel()
            with suppress(asyncio.CancelledError):
                await execution

    async def test_a_failed_run_does_not_return_internal_error_text_to_the_peer(self) -> None:
        provider = ScriptedProvider(RuntimeError("api_key=private"))
        queue = RecordingQueue()
        executor = a2a_agent_executor(
            AgentRunner(provider=provider, clock=FakeClock()),
            definition(),
            resolve=verified_principal,
        )

        await executor.execute(context(), queue)

        assert states(queue)[-1] == TaskState.TASK_STATE_FAILED
        failed = [event for event in queue.events if isinstance(event, TaskStatusUpdateEvent)][-1]
        assert MessageToDict(failed.status.message.metadata) == {
            "tesserix": {"code": "run_failed", "run_state": "failed"}
        }

    async def test_the_official_http_handler_runs_the_bridge_end_to_end(self) -> None:
        public_card = a2a_card_for(
            definition(),
            description="Plans trips.",
            provider_url="https://agents.example.test",
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
                    description="Creates an itinerary.",
                    tags=("travel",),
                ),
            ),
        )
        executor = a2a_agent_executor(
            AgentRunner(
                provider=ScriptedProvider(ModelResponse(content="Visit the laneways.")),
                clock=FakeClock(),
            ),
            definition(),
            resolve=verified_principal,
        )
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            agent_card=public_card,
        )
        app = Starlette(
            routes=[
                *create_agent_card_routes(public_card),
                *create_jsonrpc_routes(handler, rpc_url="/a2a/trip-planner"),
            ]
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://agents.example.test",
        ) as http:
            client = ClientFactory(
                ClientConfig(
                    streaming=False,
                    polling=False,
                    httpx_client=http,
                    supported_protocol_bindings=["JSONRPC"],
                )
            ).create(public_card)
            # a2aproject/a2a-python#1158 removes the protobuf descriptor warning.
            with pytest.warns(DeprecationWarning, match=r"label\(\) is deprecated"):
                responses = [
                    response
                    async for response in client.send_message(
                        SendMessageRequest(
                            message=Message(
                                message_id="message-http-1",
                                role=Role.ROLE_USER,
                                parts=[Part(text="Plan one day in Melbourne.")],
                            )
                        )
                    )
                ]
            await client.close()

        assert len(responses) == 1
        assert responses[0].task.status.state == TaskState.TASK_STATE_COMPLETED
        assert responses[0].task.artifacts[0].parts[0].text == "Visit the laneways."
