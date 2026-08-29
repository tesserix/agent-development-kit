"""Internal official A2A executor; construct it through :func:`a2a_agent_executor`."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tesserix_adk.core.identity import Principal, principal_scope
from tesserix_adk.core.run import Run, RunState
from tesserix_adk.runtime.cancellation import CancellationToken

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.tasks.task_updater import TaskUpdater
    from a2a.types import Message

    from tesserix_adk.core.definition import AgentDefinition
    from tesserix_adk.runtime import AgentRunner

logger = logging.getLogger(__name__)

_REJECTED = "The request was not authorised or accepted."
_FAILED = "The agent could not complete the task."
_CANCELLED = "The task was cancelled."


@dataclass(slots=True)
class _ActiveRun:
    token: CancellationToken
    tenant: str
    subject: str
    terminal: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def claim_terminal(self) -> bool:
        async with self.lock:
            if self.terminal:
                return False
            self.terminal = True
            return True


class _ExecutorCore:
    """Official SDK executor backed by one Tesserix runner and definition."""

    def __init__(
        self,
        runner: AgentRunner,
        definition: AgentDefinition[Any],
        *,
        resolve: Callable[[RequestContext], Awaitable[Principal]],
        max_input_bytes: int,
        max_output_bytes: int,
    ) -> None:
        self._runner = runner
        self._definition = definition
        self._resolve = resolve
        self._max_input_bytes = max_input_bytes
        self._max_output_bytes = max_output_bytes
        self._active: dict[str, _ActiveRun] = {}
        self._active_lock = asyncio.Lock()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.server.tasks.task_updater import TaskUpdater
        from a2a.types import Task, TaskState, TaskStatus

        task_id, context_id, message = self._request_identity(context)
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[message],
            )
        )
        updater = TaskUpdater(event_queue, task_id, context_id)

        principal = await self._principal(context, updater)
        if principal is None:
            return
        user_input = await self._input(context, updater)
        if user_input is None:
            return

        active = _ActiveRun(
            token=CancellationToken(),
            tenant=principal.tenant,
            subject=principal.subject,
        )
        if not await self._register(task_id, active):
            await updater.reject(self._message(updater, _REJECTED, code="task_already_active"))
            return

        try:
            await updater.start_work()
            with principal_scope(principal):
                run = await self._runner.run(
                    self._definition,
                    user_input,
                    tenant=principal.tenant,
                    user=principal.subject,
                    run_id=task_id,
                    cancellation=active.token,
                )
            await self._finish(run, updater, active)
        except asyncio.CancelledError:
            active.token.cancel("the A2A request was cancelled")
            if await active.claim_terminal():
                await updater.cancel(self._message(updater, _CANCELLED, code="cancelled"))
            raise
        except Exception as err:
            logger.error(
                "Tesserix A2A task failed outside the run loop",
                extra={"task_id": task_id, "error_type": type(err).__name__},
            )
            if await active.claim_terminal():
                await updater.failed(self._message(updater, _FAILED, code="execution_failed"))
        finally:
            await self._unregister(task_id, active)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.server.tasks.task_updater import TaskUpdater
        from a2a.utils.errors import TaskNotFoundError

        task_id = context.task_id
        context_id = context.context_id
        if not task_id or not context_id:
            return

        try:
            principal = await self._verified_principal(context)
        except Exception as err:
            logger.warning(
                "Tesserix A2A cancellation identity resolution rejected a task",
                extra={"task_id": task_id, "error_type": type(err).__name__},
            )
            raise TaskNotFoundError from err

        active = await self._active_run(task_id)
        if active is not None and (
            active.tenant != principal.tenant or active.subject != principal.subject
        ):
            logger.warning(
                "Tesserix A2A cancellation principal did not own the active task",
                extra={"task_id": task_id},
            )
            raise TaskNotFoundError

        updater = TaskUpdater(event_queue, task_id, context_id)
        if active is None:
            await updater.cancel(self._message(updater, _CANCELLED, code="cancelled"))
            return

        active.token.cancel("the A2A peer cancelled the task")
        if await active.claim_terminal():
            await updater.cancel(self._message(updater, _CANCELLED, code="cancelled"))

    def _request_identity(self, context: RequestContext) -> tuple[str, str, Message]:
        if not context.task_id or not context.context_id or context.message is None:
            from tesserix_adk.adapters.a2a import A2AExecutionError

            raise A2AExecutionError("an A2A execution requires task, context, and message IDs")
        return context.task_id, context.context_id, context.message

    async def _principal(self, context: RequestContext, updater: TaskUpdater) -> Principal | None:
        try:
            principal = await self._verified_principal(context)
        except Exception as err:
            logger.warning(
                "Tesserix A2A identity resolution rejected a task",
                extra={"task_id": context.task_id, "error_type": type(err).__name__},
            )
            await updater.reject(self._message(updater, _REJECTED, code="not_authorised"))
            return None
        return principal

    async def _verified_principal(self, context: RequestContext) -> Principal:
        principal = await self._resolve(context)
        if not isinstance(principal, Principal):
            raise PermissionError("invalid principal")
        return principal

    async def _input(self, context: RequestContext, updater: TaskUpdater) -> str | None:
        from a2a.types import Role

        message = context.message
        if message is None or message.role != Role.ROLE_USER or not message.parts:
            await updater.reject(self._message(updater, _REJECTED, code="invalid_input"))
            return None

        text: list[str] = []
        for part in message.parts:
            if part.WhichOneof("content") != "text":
                await updater.reject(
                    self._message(updater, _REJECTED, code="unsupported_input_mode")
                )
                return None
            text.append(part.text)
        joined = "\n".join(text)
        if not joined.strip() or len(joined.encode("utf-8")) > self._max_input_bytes:
            await updater.reject(self._message(updater, _REJECTED, code="invalid_input"))
            return None
        return joined

    async def _finish(self, run: Run[Any], updater: TaskUpdater, active: _ActiveRun) -> None:
        from a2a.types import Part

        if not await active.claim_terminal():
            return
        if run.state is RunState.COMPLETED:
            answer, media_type = self._answer(run)
            if len(answer.encode("utf-8")) > self._max_output_bytes:
                await updater.failed(self._message(updater, _FAILED, code="output_too_large"))
                return
            await updater.add_artifact(
                parts=[Part(text=answer, media_type=media_type)],
                artifact_id=f"{run.id}-answer",
                name="answer",
                last_chunk=True,
            )
            await updater.complete()
            return
        if run.state is RunState.CANCELLED:
            await updater.cancel(self._message(updater, _CANCELLED, code="cancelled"))
            return

        await updater.failed(
            self._message(
                updater,
                _FAILED,
                code="run_failed",
                state=run.state.value,
            )
        )

    @staticmethod
    def _answer(run: Run[Any]) -> tuple[str, str]:
        if isinstance(run.output, BaseModel):
            return run.output.model_dump_json(), "application/json"
        return run.text, "text/plain"

    @staticmethod
    def _message(
        updater: TaskUpdater,
        text: str,
        *,
        code: str,
        state: str = "",
    ) -> Message:
        from a2a.types import Part

        metadata = {"tesserix": {"code": code}}
        if state:
            metadata["tesserix"]["run_state"] = state
        return updater.new_agent_message(parts=[Part(text=text)], metadata=metadata)

    async def _register(self, task_id: str, active: _ActiveRun) -> bool:
        async with self._active_lock:
            if task_id in self._active:
                return False
            self._active[task_id] = active
            return True

    async def _active_run(self, task_id: str) -> _ActiveRun | None:
        async with self._active_lock:
            return self._active.get(task_id)

    async def _unregister(self, task_id: str, active: _ActiveRun) -> None:
        async with self._active_lock:
            if self._active.get(task_id) is active:
                del self._active[task_id]


def make_executor(
    runner: AgentRunner,
    definition: AgentDefinition[Any],
    *,
    resolve: Callable[[RequestContext], Awaitable[Principal]],
    max_input_bytes: int,
    max_output_bytes: int,
) -> AgentExecutor:
    """Create the nominal SDK executor only after the optional extra is requested."""
    from a2a.server.agent_execution import AgentExecutor

    class TesserixA2AExecutor(_ExecutorCore, AgentExecutor):
        pass

    return TesserixA2AExecutor(
        runner,
        definition,
        resolve=resolve,
        max_input_bytes=max_input_bytes,
        max_output_bytes=max_output_bytes,
    )
