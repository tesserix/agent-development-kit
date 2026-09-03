"""Every run becomes one OTLP trace, shaped for Langfuse, shipped through a collector.

Ids are derived from the run id, never random, so a replayed run upserts its trace and a
score can attach to exactly one trace. Message content never leaves the process: the
run's events carry names, states and usage, and that is what becomes spans. Export is
queued behind a batch processor and fails open — a collector outage or a wrong endpoint
costs traces, never answers.

Configuration comes from the environment under one prefix, `AGENT_TELEMETRY_`, so a
deployment turns tracing on by setting an endpoint and never touches agent code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from opentelemetry import trace as api_trace
from pydantic import Field, SecretStr

from tesserix_adk.core import AdkModel, RunEventKind, current_session, scrub
from tesserix_adk.core.extras import require_extra
from tesserix_adk.core.recording import install_default_recorder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.trace import SpanContext, Status

    from tesserix_adk.core import Run, RunEvent, RunRecorder

__all__ = [
    "ENV_PREFIX",
    "OtlpRecorder",
    "OtlpSettings",
    "build_spans",
    "install_from_env",
    "recorder_from_env",
    "resource_for",
]

logger = logging.getLogger(__name__)

ENV_PREFIX = "AGENT_TELEMETRY_"
_EXTRA = "otlp"
_SCOPE_NAME = "tesserix_adk.observability.otlp"
_NS = 1_000_000_000
_SAMPLED = api_trace.TraceFlags.SAMPLED

_GENERATION_END = {RunEventKind.MODEL_RESPONSE, RunEventKind.ATTEMPT_FAILED}
_TOOL_END = {
    RunEventKind.TOOL_RESULT,
    RunEventKind.TOOL_ERROR,
    RunEventKind.TOOL_REFUSED,
    RunEventKind.TOOL_INDETERMINATE,
}
_ERROR_KINDS = {
    RunEventKind.ATTEMPT_FAILED,
    RunEventKind.TOOL_ERROR,
    RunEventKind.BUDGET_EXCEEDED,
    RunEventKind.DEADLINE_EXCEEDED,
    RunEventKind.SCHEMA_VIOLATION,
    RunEventKind.REPAIR_ABANDONED,
}
_WARNING_KINDS = {
    RunEventKind.TOOL_REFUSED,
    RunEventKind.TOOL_INDETERMINATE,
    RunEventKind.TOOL_RESULT_FLAGGED,
    RunEventKind.TOOL_RESULT_TRUNCATED,
    RunEventKind.GUARDRAIL_REFUSAL,
    RunEventKind.GUARDRAIL_REDACTION,
    RunEventKind.MODEL_FELL_BACK,
    RunEventKind.CONTEXT_DEGRADED,
    RunEventKind.REPEAT_DETECTED,
    RunEventKind.SCOPE_REFUSED,
    RunEventKind.FAN_OUT_REFUSED,
    RunEventKind.DELEGATION_REFUSED,
    RunEventKind.COMPENSATION_REQUIRED,
}


class OtlpSettings(AdkModel):
    """Where AI traces go and how they are labelled.

    Args:
        endpoint: The OTLP/HTTP traces URL, normally the in-cluster collector. Empty
            means tracing is off.
        product: The product every trace of this process belongs to. Becomes
            `service.namespace`, which is what routes a trace to that product's project.
        service_name: The deployable emitting the traces. Defaults to the product.
        environment: The deployment environment label.
        release: The build identity, where the deployment knows it.
        public_key: With `secret_key`, a Basic-auth pair for sending straight to a
            Langfuse instance rather than through a collector.
        secret_key: See `public_key`.
        timeout_seconds: How long one export may take before it is abandoned.
        queue_size: How many spans wait for export before new ones are dropped.
        flush_interval_seconds: How often the queue is drained.
    """

    endpoint: str = ""
    product: str = Field(min_length=1, max_length=40)
    service_name: str = Field(default="", max_length=80)
    environment: str = Field(default="prod", min_length=1, max_length=40)
    release: str = Field(default="", max_length=80)
    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    queue_size: int = Field(default=4096, ge=64, le=65536)
    flush_interval_seconds: float = Field(default=2.0, gt=0, le=30)

    @property
    def enabled(self) -> bool:
        """Whether anything will be exported."""
        return bool(self.endpoint)

    @property
    def service(self) -> str:
        """The service name, falling back to the product."""
        return self.service_name or self.product

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OtlpSettings | None:
        """Read `AGENT_TELEMETRY_*`, or None when no endpoint is set.

        Raises:
            pydantic.ValidationError: An endpoint is set but the rest is not usable, which
                is a deployment mistake worth failing startup on rather than tracing
                nothing quietly.
        """
        source = os.environ if environ is None else environ
        endpoint = source.get(f"{ENV_PREFIX}ENDPOINT", "").strip()
        if not endpoint:
            return None
        values: dict[str, Any] = {"endpoint": endpoint}
        for name in (
            "product",
            "service_name",
            "environment",
            "release",
            "public_key",
            "secret_key",
            "timeout_seconds",
            "queue_size",
            "flush_interval_seconds",
        ):
            raw = source.get(f"{ENV_PREFIX}{name.upper()}")
            if raw is not None and raw.strip():
                values[name] = raw.strip()
        return cls.model_validate(values, strict=False)


def resource_for(settings: OtlpSettings) -> Resource:
    """The resource every span carries. Product routing keys on `service.namespace`."""
    resources = require_extra(_EXTRA, "opentelemetry.sdk.resources")
    attributes: dict[str, Any] = {
        "service.name": settings.service,
        "service.namespace": settings.product,
        "deployment.environment.name": settings.environment,
        "tesserix.product": settings.product,
        "tesserix.signal": "ai",
    }
    if settings.release:
        attributes["service.version"] = settings.release
    resource: Resource = resources.Resource.create(attributes)
    return resource


def build_spans(
    run: Run[Any],
    *,
    settings: OtlpSettings,
    session_id: str | None = None,
    resource: Resource | None = None,
) -> tuple[ReadableSpan, ...]:
    """One root agent span plus a child per generation, tool call and notable event."""
    sdk_trace = require_extra(_EXTRA, "opentelemetry.sdk.trace")
    scope = require_extra(_EXTRA, "opentelemetry.sdk.util.instrumentation")
    resource = resource if resource is not None else resource_for(settings)
    instrumentation = scope.InstrumentationScope(_SCOPE_NAME, "1")
    trace_id = _trace_id(run.id)
    root_context = _context(trace_id, _span_id(run.id, 0))
    start, end = _bounds(run)
    children: list[ReadableSpan] = []
    for index, (name, kind, attributes, started, ended) in enumerate(_observations(run), 1):
        level = attributes.get("langfuse.observation.level", "DEFAULT")
        children.append(
            sdk_trace.ReadableSpan(
                name=name,
                context=_context(trace_id, _span_id(run.id, index)),
                parent=root_context,
                resource=resource,
                attributes=attributes,
                kind=api_trace.SpanKind.CLIENT
                if kind == "generation"
                else api_trace.SpanKind.INTERNAL,
                status=_status(level, attributes.get("langfuse.observation.status_message")),
                start_time=started,
                end_time=ended,
                instrumentation_scope=instrumentation,
            )
        )
    root_level = "ERROR" if run.state.value != "completed" else "DEFAULT"
    root = sdk_trace.ReadableSpan(
        name=run.agent_name,
        context=root_context,
        parent=None,
        resource=resource,
        attributes=_root_attributes(run, settings, session_id),
        kind=api_trace.SpanKind.SERVER,
        status=_status(root_level, run.state.value),
        start_time=start,
        end_time=end,
        instrumentation_scope=instrumentation,
    )
    return (root, *children)


class OtlpRecorder:
    """Queue a run's spans behind a batch processor so export latency never reaches the caller.

    Args:
        settings: Where the spans go and how they are labelled.
        exporter: The transport. Defaults to OTLP/HTTP at `settings.endpoint`; a test
            passes its own.
    """

    def __init__(self, settings: OtlpSettings, exporter: SpanExporter | None = None) -> None:
        export = require_extra(_EXTRA, "opentelemetry.sdk.trace.export")
        self._settings = settings
        self._resource = resource_for(settings)
        self._processor: SpanProcessor = export.BatchSpanProcessor(
            exporter if exporter is not None else _exporter(settings),
            max_queue_size=settings.queue_size,
            schedule_delay_millis=int(settings.flush_interval_seconds * 1000),
            export_timeout_millis=int(settings.timeout_seconds * 1000),
        )
        self.failures = 0

    def record(self, run: Run[Any]) -> None:
        """Translate the run and hand it to the queue. Never raises."""
        try:
            for span in build_spans(
                run, settings=self._settings, session_id=current_session(), resource=self._resource
            ):
                self._processor.on_end(span)
        except Exception as error:
            self.failures += 1
            logger.warning(
                "trace export skipped for run %s: %s",
                getattr(run, "id", "?"),
                type(error).__name__,
            )

    def shutdown(self) -> None:
        """Flush the queue. Never raises."""
        try:
            self._processor.shutdown()
        except Exception as error:
            self.failures += 1
            logger.warning("trace flush failed: %s", type(error).__name__)


def recorder_from_env(environ: Mapping[str, str] | None = None) -> RunRecorder | None:
    """An `OtlpRecorder` for `AGENT_TELEMETRY_*`, or None when no endpoint is set."""
    settings = OtlpSettings.from_env(environ)
    if settings is None:
        return None
    return OtlpRecorder(settings)


def install_from_env(environ: Mapping[str, str] | None = None) -> RunRecorder | None:
    """Make the environment's recorder the process default, if the environment names one.

    Called when the kit is imported, so every `AgentRunner` in a deployment that sets
    `AGENT_TELEMETRY_ENDPOINT` records without any wiring. Failure to build the recorder
    is logged and leaves the default alone: a misconfigured exporter must not stop the
    service that would have used it.
    """
    try:
        recorder = recorder_from_env(environ)
    except Exception as error:
        logger.warning("run recording not installed: %s", error)
        return None
    if recorder is not None:
        install_default_recorder(recorder)
    return recorder


def _root_attributes(
    run: Run[Any], settings: OtlpSettings, session_id: str | None
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "langfuse.observation.type": "agent",
        "langfuse.trace.name": run.agent_name,
        "langfuse.tags": [settings.product, run.agent_name, settings.service],
        "langfuse.environment": settings.environment,
        "langfuse.version": run.agent_version,
        "langfuse.trace.metadata.tenant": run.tenant,
        "langfuse.trace.metadata.run_id": run.id,
        "langfuse.trace.metadata.product": settings.product,
        "langfuse.trace.metadata.service": settings.service,
        "langfuse.trace.metadata.state": run.state.value,
        "langfuse.trace.metadata.depth": run.depth,
        "langfuse.observation.model.name": run.model,
        "langfuse.observation.level": "ERROR" if run.state.value != "completed" else "DEFAULT",
        "gen_ai.request.model": run.model,
        "gen_ai.usage.input_tokens": run.usage.input_tokens,
        "gen_ai.usage.output_tokens": run.usage.output_tokens,
        "adk.tenant": run.tenant,
        "adk.run_id": run.id,
        "adk.agent": run.agent_name,
        "adk.agent_version": run.agent_version,
        "adk.state": run.state.value,
        "adk.model": run.model,
        "tesserix.product": settings.product,
        "tesserix.signal": "ai",
    }
    if settings.release:
        attributes["langfuse.release"] = settings.release
    if session_id:
        attributes["langfuse.session.id"] = session_id
    if run.user:
        attributes["langfuse.user.id"] = run.user
    for key, value in (
        ("definition_revision", run.definition_revision),
        ("prompt_version", run.prompt_version),
        ("task_class", run.task_class),
    ):
        if value:
            attributes[f"langfuse.trace.metadata.{key}"] = value
    if run.path:
        attributes["langfuse.trace.metadata.path"] = "/".join(run.path)
    if run.usage.cached_tokens:
        attributes["gen_ai.usage.cache_read_input_tokens"] = run.usage.cached_tokens
    attributes["langfuse.observation.usage_details"] = _usage_json(
        run.usage.input_tokens, run.usage.output_tokens, run.usage.cached_tokens
    )
    return attributes


def _observations(run: Run[Any]) -> Sequence[tuple[str, str, dict[str, Any], int, int]]:
    """Pair call/response events into spans and keep the rest as point events."""
    rows: list[tuple[str, str, dict[str, Any], int, int]] = []
    open_generation: RunEvent | None = None
    open_tools: dict[str, RunEvent] = {}
    fallback = _bounds(run)[0]
    for event in run.events:
        at = _ns(event.at, fallback)
        if event.kind is RunEventKind.MODEL_CALL:
            open_generation = event
        elif event.kind in _GENERATION_END:
            started = _ns(open_generation.at, at) if open_generation is not None else at
            rows.append(
                (event.name or run.model, "generation", _generation(run, event), started, at)
            )
            open_generation = None
        elif event.kind is RunEventKind.TOOL_CALL and event.name:
            open_tools[event.name] = event
        elif event.kind in _TOOL_END and event.name:
            call = open_tools.pop(event.name, None)
            started = _ns(call.at, at) if call is not None else at
            rows.append((event.name, "tool", _tool(event), started, at))
        elif event.kind in _ERROR_KINDS or event.kind in _WARNING_KINDS:
            rows.append((event.kind.value, "event", _point(event), at, at))
    return rows


def _generation(run: Run[Any], event: RunEvent) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "langfuse.observation.type": "generation",
        "langfuse.observation.level": _level(event),
        "langfuse.observation.model.name": event.name or run.model,
        "gen_ai.request.model": event.name or run.model,
        "adk.event": event.kind.value,
    }
    if event.usage is not None:
        attributes["gen_ai.usage.input_tokens"] = event.usage.input_tokens
        attributes["gen_ai.usage.output_tokens"] = event.usage.output_tokens
        if event.usage.cached_tokens:
            attributes["gen_ai.usage.cache_read_input_tokens"] = event.usage.cached_tokens
        attributes["langfuse.observation.usage_details"] = _usage_json(
            event.usage.input_tokens, event.usage.output_tokens, event.usage.cached_tokens
        )
    if event.detail:
        attributes["langfuse.observation.status_message"] = _detail(event.detail)
    return attributes


def _tool(event: RunEvent) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "langfuse.observation.type": "tool",
        "langfuse.observation.level": _level(event),
        "adk.event": event.kind.value,
        "adk.name": event.name or "",
    }
    if event.detail:
        attributes["langfuse.observation.status_message"] = _detail(event.detail)
    return attributes


def _point(event: RunEvent) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "langfuse.observation.type": "event",
        "langfuse.observation.level": _level(event),
        "adk.event": event.kind.value,
    }
    if event.name:
        attributes["adk.name"] = event.name
    if event.detail:
        attributes["langfuse.observation.status_message"] = _detail(event.detail)
    return attributes


def _detail(detail: str) -> str:
    return scrub(detail[:500])


def _usage_json(input_tokens: int, output_tokens: int, cached: int) -> str:
    usage: dict[str, int] = {"input": input_tokens, "output": output_tokens}
    if cached:
        usage["cache_read_input_tokens"] = cached
    return json.dumps(usage, separators=(",", ":"))


def _level(event: RunEvent) -> str:
    if event.kind in _ERROR_KINDS:
        return "ERROR"
    if event.kind in _WARNING_KINDS:
        return "WARNING"
    return "DEFAULT"


def _status(level: str, detail: str | None = None) -> Status:
    if level == "ERROR":
        return api_trace.Status(api_trace.StatusCode.ERROR, detail)
    return api_trace.Status(api_trace.StatusCode.UNSET)


def _bounds(run: Run[Any]) -> tuple[int, int]:
    stamps = [event.at for event in run.events if event.at is not None]
    started = run.started_at if run.started_at is not None else (min(stamps) if stamps else None)
    ended = run.ended_at if run.ended_at is not None else (max(stamps) if stamps else None)
    start = _ns(started, int(time.time() * _NS))
    return start, _ns(ended, start)


def _ns(seconds: float | None, fallback: int) -> int:
    return fallback if seconds is None else int(seconds * _NS)


def _trace_id(run_id: str) -> int:
    return int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:16], "big") or 1


def _span_id(run_id: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{run_id}:{index}".encode()).digest()[:8], "big") or 1


def _context(trace_id: int, span_id: int) -> SpanContext:
    return api_trace.SpanContext(
        trace_id,
        span_id,
        is_remote=False,
        trace_flags=api_trace.TraceFlags(api_trace.TraceFlags.SAMPLED),
    )


def _exporter(settings: OtlpSettings) -> SpanExporter:
    http = require_extra(_EXTRA, "opentelemetry.exporter.otlp.proto.http.trace_exporter")
    headers: dict[str, str] = {}
    if settings.public_key is not None and settings.secret_key is not None:
        pair = f"{settings.public_key.get_secret_value()}:{settings.secret_key.get_secret_value()}"
        headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
    exporter: SpanExporter = http.OTLPSpanExporter(
        endpoint=settings.endpoint, headers=headers or None, timeout=int(settings.timeout_seconds)
    )
    return exporter
