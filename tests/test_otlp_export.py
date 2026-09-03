"""Every run becomes one Langfuse-shaped OTLP trace, and export never touches the answer."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("opentelemetry.exporter.otlp.proto.http")

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import ValidationError

from tesserix_adk.core import (
    Agent,
    Run,
    RunEvent,
    RunEventKind,
    RunState,
    Usage,
    bound_session,
    default_recorder,
    install_default_recorder,
)
from tesserix_adk.observability.otlp import (
    OtlpRecorder,
    OtlpSettings,
    build_spans,
    install_from_env,
    recorder_from_env,
    resource_for,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from opentelemetry.sdk.trace import ReadableSpan


class MemoryExporter(SpanExporter):
    def __init__(self, *, fail: bool = False) -> None:
        self.spans: list[ReadableSpan] = []
        self._fail = fail

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._fail:
            raise RuntimeError("collector down")
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True


SETTINGS = OtlpSettings(
    endpoint="http://otel-gateway.observability.svc:4318/v1/traces",
    product="kora",
    service_name="kora-ai-agents",
    environment="prod",
    release="main-abc123",
)


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def event(kind: RunEventKind, at: float, **fields: Any) -> RunEvent:
    return RunEvent(kind=kind, at=at, **fields)


def finished(state: RunState = RunState.COMPLETED, events: Sequence[RunEvent] = ()) -> Run:
    run = Run(
        id="run_42",
        tenant="acme",
        user="u_7",
        agent_name="planner",
        agent_version="3",
        model="claude-sonnet-5",
        usage=Usage(input_tokens=30, output_tokens=12, cached_tokens=4),
        started_at=100.0,
        ended_at=104.0,
    )
    for item in events:
        run = run.record_event(item)
    return run.model_copy(update={"state": state})


CONVERSATION = (
    event(RunEventKind.MODEL_CALL, 100.5, name="claude-sonnet-5"),
    event(
        RunEventKind.MODEL_RESPONSE,
        101.5,
        name="claude-sonnet-5",
        usage=Usage(input_tokens=30, output_tokens=12),
    ),
    event(RunEventKind.TOOL_CALL, 101.6, name="search"),
    event(RunEventKind.TOOL_RESULT, 102.0, name="search"),
    event(RunEventKind.TOOL_CALL, 102.1, name="book"),
    event(RunEventKind.TOOL_ERROR, 103.0, name="book", detail="timeout after 3 attempts"),
    event(RunEventKind.BUDGET_EXCEEDED, 103.5, detail="tokens"),
)


@pytest.fixture(autouse=True)
def _no_default_recorder() -> Iterator[None]:
    previous = install_default_recorder(None)
    yield
    install_default_recorder(previous)


class TestSettings:
    def test_from_env_is_off_without_an_endpoint(self) -> None:
        assert OtlpSettings.from_env({"AGENT_TELEMETRY_PRODUCT": "kora"}) is None
        assert recorder_from_env({}) is None

    def test_from_env_reads_every_prefixed_variable(self) -> None:
        settings = OtlpSettings.from_env(
            {
                "AGENT_TELEMETRY_ENDPOINT": " http://gw:4318/v1/traces ",
                "AGENT_TELEMETRY_PRODUCT": "sre",
                "AGENT_TELEMETRY_SERVICE_NAME": "sre-ai-agent",
                "AGENT_TELEMETRY_ENVIRONMENT": "staging",
                "AGENT_TELEMETRY_RELEASE": "main-1",
                "AGENT_TELEMETRY_QUEUE_SIZE": "128",
                "AGENT_TELEMETRY_FLUSH_INTERVAL_SECONDS": "0.5",
            }
        )
        assert settings is not None
        assert settings.endpoint == "http://gw:4318/v1/traces"
        assert (settings.product, settings.service, settings.environment) == (
            "sre",
            "sre-ai-agent",
            "staging",
        )
        assert (settings.queue_size, settings.flush_interval_seconds) == (128, 0.5)

    def test_an_endpoint_without_a_product_is_a_deployment_error(self) -> None:
        with pytest.raises(ValidationError):
            OtlpSettings.from_env({"AGENT_TELEMETRY_ENDPOINT": "http://gw"})

    def test_service_falls_back_to_the_product(self) -> None:
        assert OtlpSettings(endpoint="http://gw", product="ocr").service == "ocr"

    def test_the_resource_routes_by_product(self) -> None:
        attributes = resource_for(SETTINGS).attributes
        assert attributes["service.namespace"] == "kora"
        assert attributes["service.name"] == "kora-ai-agents"
        assert attributes["tesserix.signal"] == "ai"
        assert attributes["service.version"] == "main-abc123"


class TestSpanShape:
    def test_the_root_span_is_the_agent_with_langfuse_trace_attributes(self) -> None:
        root, *_ = build_spans(finished(), settings=SETTINGS, session_id="chat_9")
        assert root.name == "planner"
        assert root.parent is None
        attributes = attrs(root)
        assert attributes["langfuse.observation.type"] == "agent"
        assert attributes["langfuse.session.id"] == "chat_9"
        assert attributes["langfuse.user.id"] == "u_7"
        assert attributes["langfuse.trace.metadata.tenant"] == "acme"
        assert attributes["langfuse.release"] == "main-abc123"
        assert attributes["langfuse.version"] == "3"
        assert list(attributes["langfuse.tags"]) == ["kora", "planner", "kora-ai-agents"]
        assert json.loads(attributes["langfuse.observation.usage_details"]) == {
            "input": 30,
            "output": 12,
            "cache_read_input_tokens": 4,
        }
        assert root.start_time == 100_000_000_000
        assert root.end_time == 104_000_000_000

    def test_model_and_tool_events_pair_into_child_spans(self) -> None:
        spans = build_spans(finished(events=CONVERSATION), settings=SETTINGS)
        root, *children = spans
        assert [span.name for span in children] == [
            "claude-sonnet-5",
            "search",
            "book",
            "budget_exceeded",
        ]
        assert all(span.parent == root.context for span in children)
        generation, search, book, budget = children
        assert (generation.start_time, generation.end_time) == (
            100_500_000_000,
            101_500_000_000,
        )
        assert attrs(generation)["langfuse.observation.type"] == "generation"
        assert attrs(generation)["gen_ai.usage.output_tokens"] == 12
        assert attrs(search)["langfuse.observation.type"] == "tool"
        assert attrs(book)["langfuse.observation.level"] == "ERROR"
        assert attrs(book)["langfuse.observation.status_message"].startswith("timeout")
        assert book.status.is_ok is False
        assert attrs(budget)["langfuse.observation.type"] == "event"

    def test_ids_are_derived_from_the_run_so_a_replay_upserts(self) -> None:
        first = build_spans(finished(events=CONVERSATION), settings=SETTINGS)
        again = build_spans(finished(events=CONVERSATION), settings=SETTINGS)
        assert [span.context.span_id for span in first] == [span.context.span_id for span in again]
        assert len({span.context.trace_id for span in first}) == 1
        assert len({span.context.span_id for span in first}) == len(first)

    def test_a_failed_run_is_an_error_trace(self) -> None:
        root, *_ = build_spans(finished(RunState.FAILED), settings=SETTINGS)
        assert root.status.is_ok is False
        assert attrs(root)["langfuse.observation.level"] == "ERROR"
        assert attrs(root)["langfuse.trace.metadata.state"] == "failed"

    def test_details_are_scrubbed_before_they_become_attributes(self) -> None:
        leak = event(
            RunEventKind.TOOL_ERROR, 103.0, name="fetch", detail="denied for sk-ant-api03-abcdef"
        )
        *_, span = build_spans(finished(events=(leak,)), settings=SETTINGS)
        assert "sk-ant-api03-abcdef" not in attrs(span)["langfuse.observation.status_message"]


class TestRecorder:
    def test_a_recorded_run_reaches_the_exporter(self) -> None:
        exporter = MemoryExporter()
        recorder = OtlpRecorder(SETTINGS, exporter=exporter)
        with bound_session("chat_1"):
            recorder.record(finished(events=CONVERSATION))
        recorder.shutdown()
        assert exporter.spans[0].name == "planner"
        assert len(exporter.spans) == 5
        assert dict(exporter.spans[0].attributes or {})["langfuse.session.id"] == "chat_1"

    def test_a_broken_exporter_is_counted_not_raised(self) -> None:
        recorder = OtlpRecorder(SETTINGS, exporter=MemoryExporter(fail=True))
        recorder.record(finished())
        recorder.shutdown()
        assert recorder.failures == 0  # export failures are the processor's, not the caller's

    def test_a_run_that_cannot_be_shaped_is_counted_not_raised(self) -> None:
        recorder = OtlpRecorder(SETTINGS, exporter=MemoryExporter())
        recorder.record("not a run")  # type: ignore[arg-type]
        assert recorder.failures == 1

    def test_direct_langfuse_delivery_uses_basic_auth(self) -> None:
        from tesserix_adk.observability.otlp import _exporter

        pair = {"public_key": "pk-lf-" + "1", "secret_key": "sk-lf-" + "2"}
        exporter = _exporter(
            OtlpSettings.model_validate(
                {"endpoint": "http://lf/api/public/otel/v1/traces", "product": "kora", **pair},
                strict=False,
            )
        )
        expected = "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-2").decode()
        assert getattr(exporter, "_headers")["Authorization"] == expected  # noqa: B009


class TestInstallation:
    def test_nothing_is_installed_without_an_endpoint(self) -> None:
        assert install_from_env({}) is None
        assert default_recorder() is None

    def test_the_environment_recorder_becomes_the_default(self) -> None:
        installed = install_from_env(
            {
                "AGENT_TELEMETRY_ENDPOINT": "http://gw:4318/v1/traces",
                "AGENT_TELEMETRY_PRODUCT": "kora",
            }
        )
        assert isinstance(installed, OtlpRecorder)
        assert default_recorder() is installed
        installed.shutdown()

    def test_a_bad_environment_is_logged_not_fatal(self) -> None:
        assert install_from_env({"AGENT_TELEMETRY_ENDPOINT": "http://gw"}) is None
        assert default_recorder() is None


class TestRunnerIntegration:
    async def test_every_finished_run_is_recorded(self) -> None:
        exporter = MemoryExporter()
        recorder = OtlpRecorder(SETTINGS, exporter=exporter)
        runner = AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="Kyoto.")),
            clock=FakeClock(),
            recorder=recorder,
        )
        run = await runner.run(
            Agent(name="planner", instructions="Plan.", free_text=True, model="m"),
            "plan",
            tenant="acme",
            run_id="run_1",
        )
        recorder.shutdown()
        assert run.state is RunState.COMPLETED
        assert exporter.spans[0].name == "planner"
        assert dict(exporter.spans[0].attributes or {})["adk.run_id"] == "run_1"
        assert any(
            attrs(span).get("langfuse.observation.type") == "generation" for span in exporter.spans
        )

    async def test_the_default_recorder_is_used_when_none_is_passed(self) -> None:
        exporter = MemoryExporter()
        install_default_recorder(OtlpRecorder(SETTINGS, exporter=exporter))
        runner = AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="Kyoto.")), clock=FakeClock()
        )
        await runner.run(
            Agent(name="planner", instructions="Plan.", free_text=True, model="m"),
            "plan",
            tenant="acme",
        )
        recorder = default_recorder()
        assert recorder is not None
        recorder.shutdown()
        assert exporter.spans

    async def test_an_exploding_recorder_never_fails_the_run(self) -> None:
        class Exploding:
            def record(self, run: Run[Any]) -> None:
                raise RuntimeError(f"boom {run.id}")

            def shutdown(self) -> None:
                return None

        runner = AgentRunner(
            provider=ScriptedProvider(ModelResponse(content="Kyoto.")),
            clock=FakeClock(),
            recorder=Exploding(),
        )
        run = await runner.run(
            Agent(name="planner", instructions="Plan.", free_text=True, model="m"),
            "plan",
            tenant="acme",
        )
        assert run.state is RunState.COMPLETED
        assert runner.recording_failures == 1
