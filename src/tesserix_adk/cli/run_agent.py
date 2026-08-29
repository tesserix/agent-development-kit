"""Run one consumer-supplied agent locally with a redacted streamed trace."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from tesserix_adk import Agent, AgentRunner, ToolRegistry, __version__
from tesserix_adk.cli.artifacts import (
    ARTIFACT_VERSION,
    ArtifactHeader,
    ArtifactWriter,
    redacted_json,
)
from tesserix_adk.core import (
    AdkError,
    ApprovalDecision,
    ApprovalDenial,
    ApprovalGate,
    ApprovalRecord,
    ConfigurationError,
    ModelProvider,
    RunState,
)
from tesserix_adk.runtime import (
    AnswerDelta,
    CancellationToken,
    ProgressEvent,
    StructuredDelta,
    UsageUpdated,
)
from tesserix_adk.testing import RecordingProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import TextIO

    from tesserix_adk.core import Guardrail

__all__ = ["LocalAgent", "Resolve", "TargetLoadError", "load_target", "main"]

OK = 0
FAILED = 1
MISUSED = 2
INTERRUPTED = 130
LOCAL_TENANT = "local-dev"
LOCAL_USER = "local-user"
MAX_RENDERED_CHARS = 2_000


class TargetLoadError(ConfigurationError):
    """A local import reference did not resolve to a complete runnable agent."""


class RunnerFactory(Protocol):
    """Build a runner while allowing recording and the terminal approval gate."""

    def __call__(self, provider: ModelProvider, approvals: ApprovalGate) -> AgentRunner:
        """Return a runner using exactly ``provider`` and ``approvals``."""
        ...


@dataclass(frozen=True)
class LocalAgent:
    """The declaration and collaborators needed by the local-only command.

    Args:
        agent: Declaration to execute.
        provider: Provider to call. ``--record`` wraps it without changing its behaviour.
        tools: Registry visible to the declaration.
        guardrails: Named guardrails enforced by the default runner.
        runner_factory: Optional custom runner construction. It must use the supplied
            provider and approval gate or recording/interactive approval would be bypassed.
    """

    agent: Agent[Any]
    provider: ModelProvider
    tools: ToolRegistry | None = None
    guardrails: Mapping[str, Guardrail] | None = None
    runner_factory: RunnerFactory | None = None

    def runner(self, provider: ModelProvider, approvals: ApprovalGate) -> AgentRunner:
        """Build the runtime with the CLI's provider wrapper and fail-closed gate."""
        if self.runner_factory is not None:
            return self.runner_factory(provider, approvals)
        return AgentRunner(
            provider=provider,
            tools=self.tools,
            guardrails=self.guardrails,
            approvals=approvals,
            approval_denial=ApprovalDenial.FAIL_RUN,
        )


type Resolve = Callable[[str, ApprovalGate], LocalAgent]
"""Resolve ``module:attribute`` or an application-specific target reference."""


async def main(
    argv: Sequence[str],
    *,
    resolve: Resolve,
    out: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    """Stream one scoped local run and return a stable process exit code.

    Args:
        argv: Arguments after ``run``.
        resolve: Application wiring from target reference to declaration and collaborators.
        out: Human or NDJSON output destination. Defaults to stdout.
        stdin: Input and interactive approval source. Defaults to stdin.

    Returns:
        ``0`` only for a completed run, ``1`` for a terminal failure, ``2`` for command or
        configuration misuse, and ``130`` when interrupted after cancellation is flushed.
    """
    writer = out if out is not None else sys.stdout
    reader = stdin if stdin is not None else sys.stdin
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    try:
        user_input = _input(parsed, reader)
    except OSError as error:
        writer.write(f"input could not be read: {error}\n")
        return MISUSED
    if not user_input.strip():
        writer.write("input is empty; use --input, --input-file or stdin\n")
        return MISUSED
    tenant = parsed.tenant or LOCAL_TENANT
    user = parsed.user or LOCAL_USER
    gate = _TerminalGate(
        reader,
        writer,
        interactive=not parsed.no_interactive,
        json_mode=parsed.json,
    )
    try:
        target = resolve(parsed.target, gate)
    except (AdkError, ImportError, AttributeError, TypeError, ValueError) as error:
        writer.write(f"agent target could not be loaded: {error}\n")
        return MISUSED

    artifact: ArtifactWriter | None = None
    recorder: RecordingProvider | None = None
    provider = target.provider
    if parsed.record:
        try:
            artifact = ArtifactWriter(
                Path(parsed.record),
                ArtifactHeader(
                    version=ARTIFACT_VERSION,
                    kit_version=__version__,
                    target=parsed.target,
                    input=user_input,
                    tenant=tenant,
                    user=user,
                    agent=target.agent.name,
                ),
            )
        except OSError as error:
            writer.write(f"recording could not start: {error}\n")
            return MISUSED
        recorder = RecordingProvider(provider, provider=provider.name, version=__version__)
        provider = recorder

    token = CancellationToken()
    runner = target.runner(provider, gate)
    _render_scope(
        writer,
        tenant=tenant,
        user=user,
        defaults=parsed.tenant is None and parsed.user is None,
        json_mode=parsed.json,
    )
    started = time.monotonic()
    stream = runner.stream(
        target.agent,
        user_input,
        tenant=tenant,
        user=user,
        cancellation=token,
    )
    try:
        async with stream:
            async for event in stream:
                if artifact is not None:
                    artifact.append(event)
                _render(event, writer=writer, json_mode=parsed.json, started=started)
    except (KeyboardInterrupt, asyncio.CancelledError):
        current = asyncio.current_task()
        if current is not None and hasattr(current, "uncancel"):
            current.uncancel()
        token.cancel("local operator interrupted the run")
        await asyncio.shield(stream.aclose())
        if artifact is not None:
            if stream.run.state.is_terminal:
                artifact.finish(
                    stream.run,
                    cassette=recorder.cassette if recorder is not None else None,
                )
            else:
                artifact.close()
        writer.write("run cancelled by interrupt after in-flight work was released\n")
        return INTERRUPTED
    except (AdkError, OSError) as error:
        if artifact is not None:
            artifact.close()
        writer.write(f"run failed: {error}\ndiagnose configuration with: tesserix-adk doctor\n")
        return FAILED

    completed = stream.run
    if artifact is not None:
        try:
            artifact.finish(
                completed,
                cassette=recorder.cassette if recorder is not None else None,
            )
        except OSError as error:
            artifact.close()
            writer.write(f"run completed but its artefact could not be committed: {error}\n")
            return FAILED
    if completed.state is not RunState.COMPLETED:
        if not parsed.json:
            writer.write("diagnose provider and configuration with: tesserix-adk doctor\n")
        return FAILED
    return OK


def load_target(reference: str, approvals: ApprovalGate) -> LocalAgent:
    """Import ``module:attribute`` and resolve a :class:`LocalAgent`.

    A callable target may accept a keyword named ``approvals`` and is then given the same
    gate the CLI will enforce. A no-argument factory is supported for agents with no
    approval-required tools.

    Raises:
        TargetLoadError: The reference is malformed or resolves to another type.
        ImportError: The module or one of its dependencies cannot be imported.
        AttributeError: The named attribute does not exist.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise TargetLoadError("target must be an import reference such as package.agent:local")
    loaded: object = getattr(importlib.import_module(module_name), attribute)
    if isinstance(loaded, LocalAgent):
        return loaded
    if callable(loaded):
        signature = inspect.signature(loaded)
        if "approvals" in signature.parameters:
            loaded = loaded(approvals=approvals)
        elif not signature.parameters:
            loaded = loaded()
        else:
            raise TargetLoadError(
                "target factory must take no arguments or one keyword named 'approvals'"
            )
    if not isinstance(loaded, LocalAgent):
        raise TargetLoadError("target must be LocalAgent or a factory returning LocalAgent")
    return loaded


class _TerminalGate:
    """Ask the local operator, or deny immediately when interaction is disabled."""

    def __init__(
        self,
        reader: TextIO,
        writer: TextIO,
        *,
        interactive: bool,
        json_mode: bool,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._interactive = interactive
        self._json = json_mode

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Return one payload-bound human decision; absence is always denial."""
        if self._json:
            self._writer.write(
                json.dumps(
                    {
                        "kind": "approval_prompt",
                        "record_id": record.id,
                        "tool": record.tool_name,
                        "summary": record.summary,
                        "reason": record.reason,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            self._writer.write(
                f"approval required: {record.tool_name} {record.summary} {record.reason}\n"
            )
        granted = False
        identity = "system:no-interactive"
        reason = "interactive approval disabled; denied fail-closed"
        if self._interactive:
            self._writer.write("approve this exact call? [y/N] ")
            self._writer.flush()
            answer = (await asyncio.to_thread(self._reader.readline)).strip().casefold()
            granted = answer in {"y", "yes"}
            identity = LOCAL_USER
            reason = "approved in local terminal" if granted else "denied in local terminal"
        decision_kind = "approval_granted" if granted else "approval_denied"
        if self._json:
            self._writer.write(
                json.dumps(
                    {
                        "kind": decision_kind,
                        "record_id": record.id,
                        "decided_by": identity,
                        "reason": reason,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            self._writer.write(f"{decision_kind} by={identity} reason={reason}\n")
        return ApprovalDecision(
            record_id=record.id,
            granted=granted,
            decided_by=identity,
            decided_at=time.time(),
            reason=reason,
        )


def _input(parsed: argparse.Namespace, reader: TextIO) -> str:
    """Read exactly the selected input source."""
    if parsed.input is not None:
        return str(parsed.input)
    if parsed.input_file is not None:
        return Path(parsed.input_file).read_text(encoding="utf-8")
    interactive = bool(getattr(reader, "isatty", lambda: False)())
    return reader.readline() if interactive else reader.read()


def _render_scope(
    writer: TextIO, *, tenant: str, user: str, defaults: bool, json_mode: bool
) -> None:
    """Name the real scoped context, including whether local defaults supplied it."""
    if json_mode:
        writer.write(
            json.dumps(
                {
                    "kind": "local_scope",
                    "tenant": tenant,
                    "user": user,
                    "default": defaults,
                },
                sort_keys=True,
            )
            + "\n"
        )
        return
    label = " (local development default)" if defaults else ""
    writer.write(f"scope tenant={tenant} user={user}{label}\n")


def _render(event: ProgressEvent, *, writer: TextIO, json_mode: bool, started: float) -> None:
    """Render one progress event after applying the telemetry redaction vocabulary."""
    payload = redacted_json(event.model_dump(mode="json"))
    if json_mode:
        writer.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return
    elapsed = max(time.monotonic() - started, 0.0)
    prefix = f"+{elapsed:07.3f}s #{event.sequence:04d} {event.kind}"
    if isinstance(event, AnswerDelta):
        detail = event.text
    elif isinstance(event, StructuredDelta):
        detail = event.fragment
    elif isinstance(event, UsageUpdated):
        tokens = event.usage.input_tokens + event.usage.output_tokens
        cost = (
            f"{event.usage.cost.currency} {event.usage.cost.total}"
            if event.usage.cost is not None
            else "unknown"
        )
        detail = f"tokens={tokens} running_cost={cost}"
    else:
        hidden = {"kind", "run_id", "sequence", "at"}
        detail = " ".join(f"{key}={value}" for key, value in payload.items() if key not in hidden)
    shown = str(redacted_json(detail))
    if len(shown) > MAX_RENDERED_CHARS:
        shown = shown[:MAX_RENDERED_CHARS] + "… [terminal truncated; artefact retains full value]"
    writer.write(f"{prefix}{' ' + shown if shown else ''}\n")


def _parser() -> argparse.ArgumentParser:
    """Build the ``tesserix-adk run`` command line."""
    parser = argparse.ArgumentParser(prog="tesserix-adk run", description=__doc__)
    parser.add_argument("target", help="local agent as module:attribute")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", help="one input string")
    inputs.add_argument("--input-file", help="UTF-8 file containing the input")
    parser.add_argument("--tenant", help=f"isolation boundary (default: {LOCAL_TENANT})")
    parser.add_argument("--user", help=f"acting principal (default: {LOCAL_USER})")
    parser.add_argument("--json", action="store_true", help="emit newline-delimited JSON")
    parser.add_argument("--record", help="new path for a complete inspectable artefact")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="deny approval-required calls fail-closed",
    )
    return parser
