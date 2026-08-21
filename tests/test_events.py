"""Agent activity as events other systems can consume, carrying no content."""

from __future__ import annotations

from typing import Any

import pytest

from tesserix_adk.core import (
    ALLOWED_ATTRIBUTES,
    EVENT_SCHEMA_VERSION,
    ApprovalDecided,
    ApprovalRequested,
    BudgetExceeded,
    ConfigurationError,
    Delivery,
    EventEnvelope,
    Eventing,
    EventPublisher,
    EventPublishError,
    EventsReplayed,
    EventTooLargeError,
    EventType,
    MemoryErased,
    NullEventPublisher,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    TenantContext,
    ToolCallCompleted,
    ToolCallRequested,
    tenant_scope,
)
from tesserix_adk.testing import FakeClock, InMemoryEventPublisher, assert_events

TENANT = "acme"
RUN = "run_1"


def _eventing(
    publisher: EventPublisher | None = None, **options: object
) -> tuple[Eventing, InMemoryEventPublisher]:
    into: Any = publisher or InMemoryEventPublisher()
    return Eventing(into, clock=FakeClock(), **options), into  # type: ignore[arg-type]


class TestTheSequenceARunEmits:
    async def test_a_run_that_calls_two_tools_is_assertable_in_order(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
            await eventing.emit(ToolCallRequested(run_id=RUN, tool="search", tool_call_id="c1"))
            await eventing.emit(
                ToolCallCompleted(run_id=RUN, tool="search", tool_call_id="c1", state="ok")
            )
            await eventing.emit(RunCompleted(run_id=RUN, iterations=2, tool_calls=1))
        assert_events(
            published.events,
            EventType.RUN_STARTED,
            EventType.TOOL_CALL_REQUESTED,
            EventType.TOOL_CALL_COMPLETED,
            EventType.RUN_COMPLETED,
        )

    async def test_every_event_has_its_own_stable_identifier(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
            await eventing.emit(RunCompleted(run_id=RUN))
        ids = [event.event_id for event in published.events]
        assert len(set(ids)) == 2
        assert ids == sorted(ids)

    async def test_the_catalogue_covers_every_lifecycle_moment(self) -> None:
        emitted = {
            RunStarted(run_id=RUN, agent="desk"),
            RunCompleted(run_id=RUN),
            RunFailed(run_id=RUN, error_code="tool_failed"),
            RunCancelled(run_id=RUN, reason_code="caller"),
            ToolCallRequested(run_id=RUN, tool="t", tool_call_id="c1"),
            ToolCallCompleted(run_id=RUN, tool="t", tool_call_id="c1", state="ok"),
            ApprovalRequested(run_id=RUN, approval_id="a1", tool="refund"),
            ApprovalDecided(run_id=RUN, approval_id="a1", decision="granted"),
            BudgetExceeded(run_id=RUN, scope="run", limit="max_input_tokens"),
            MemoryErased(subject="s-1", records_erased=3),
            EventsReplayed(replay_id="rp_1", records=2),
        }
        assert {payload.type for payload in emitted} == set(EventType)


class TestTheEnvelope:
    async def test_it_carries_the_scope_without_any_publish_site_passing_it(self) -> None:
        eventing, published = _eventing()
        context = TenantContext(tenant=TENANT, user="ada", correlation_id="corr-1")
        with tenant_scope(context):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
        envelope = published.events[0]
        assert (envelope.tenant, envelope.user, envelope.correlation_id) == (
            TENANT,
            "ada",
            "corr-1",
        )

    async def test_it_names_the_run_and_the_trace_it_belongs_to(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
        assert (published.events[0].run_id, published.events[0].trace_id) == (RUN, RUN)

    async def test_it_says_which_event_caused_it(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            first = await eventing.emit(ToolCallRequested(run_id=RUN, tool="t", tool_call_id="c1"))
            await eventing.emit(
                ToolCallCompleted(run_id=RUN, tool="t", tool_call_id="c1", state="ok"),
                caused_by=first,
            )
        assert published.events[1].causation_id == published.events[0].event_id

    async def test_it_says_which_version_of_the_contract_it_speaks(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
        assert published.events[0].schema_version == EVENT_SCHEMA_VERSION

    async def test_it_is_stamped_by_the_injected_clock(self) -> None:
        clock = FakeClock(start=1000.0)
        published = InMemoryEventPublisher()
        with tenant_scope(TENANT):
            await Eventing(published, clock=clock).emit(RunStarted(run_id=RUN, agent="desk"))
        assert published.events[0].occurred_at == 1000.0

    async def test_a_publish_outside_any_tenant_is_a_configuration_error(self) -> None:
        eventing, _ = _eventing()
        with pytest.raises(ConfigurationError, match="tenant"):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))


class TestWhatMayNotTravel:
    async def test_the_allowlist_is_ids_counts_and_spend(self) -> None:
        assert {"run_id", "tool", "input_tokens", "cost_micros", "model"} <= ALLOWED_ATTRIBUTES
        assert not {"messages", "arguments", "content", "prompt", "text"} & ALLOWED_ATTRIBUTES

    async def test_every_payload_field_is_on_the_allowlist(self) -> None:
        for payload in (
            RunStarted(run_id=RUN, agent="desk"),
            ToolCallCompleted(run_id=RUN, tool="t", tool_call_id="c1", state="ok"),
            BudgetExceeded(run_id=RUN, scope="run", limit="max_input_tokens"),
        ):
            assert set(payload.attributes()) <= ALLOWED_ATTRIBUTES

    async def test_an_identifier_in_an_attribute_is_taken_out_before_publish(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(MemoryErased(subject="ada@example.gov", records_erased=1))
        assert "ada@example.gov" not in str(published.events[0].attributes)

    async def test_redaction_happens_before_the_publisher_sees_it(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(
                ApprovalDecided(
                    run_id=RUN, approval_id="a1", decision="granted", approver="ada@example.gov"
                )
            )
        assert "example.gov" not in str(published.events[0].attributes)


class TestWhenTheTransportIsNotThere:
    async def test_best_effort_records_the_drop_and_the_run_continues(self) -> None:
        eventing = Eventing(_Broken(), clock=FakeClock(), delivery=Delivery.BEST_EFFORT)
        with tenant_scope(TENANT):
            assert await eventing.emit(RunStarted(run_id=RUN, agent="desk")) is None
        assert eventing.dropped == 1

    async def test_guaranteed_fails_the_step_so_state_and_events_cannot_diverge(self) -> None:
        eventing = Eventing(_Broken(), clock=FakeClock(), delivery=Delivery.GUARANTEED)
        with tenant_scope(TENANT), pytest.raises(EventPublishError) as refused:
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
        assert refused.value.event_type == EventType.RUN_STARTED

    async def test_the_default_publisher_publishes_nowhere_and_never_fails(self) -> None:
        eventing, _ = _eventing(NullEventPublisher())
        with tenant_scope(TENANT):
            assert await eventing.emit(RunStarted(run_id=RUN, agent="desk")) is not None


class TestWhatIsTooBigToSend:
    async def test_an_event_over_the_transport_ceiling_is_refused_not_truncated(self) -> None:
        eventing, published = _eventing(max_event_bytes=64, delivery=Delivery.GUARANTEED)
        with tenant_scope(TENANT), pytest.raises(EventTooLargeError):
            await eventing.emit(RunStarted(run_id=RUN, agent="x" * 512))
        assert published.events == ()

    async def test_best_effort_drops_it_and_says_so(self) -> None:
        eventing, _ = _eventing(max_event_bytes=64)
        with tenant_scope(TENANT):
            assert await eventing.emit(RunStarted(run_id=RUN, agent="x" * 512)) is None
        assert eventing.dropped == 1


class TestPublishingManyAtOnce:
    async def test_the_batch_arrives_as_one_call(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit_all(
                [RunStarted(run_id=RUN, agent="desk"), RunCompleted(run_id=RUN)]
            )
        assert published.batches == 1
        assert len(published.events) == 2

    async def test_one_event_that_cannot_be_sent_does_not_drop_the_rest(self) -> None:
        eventing, published = _eventing(max_event_bytes=420)
        with tenant_scope(TENANT):
            report = await eventing.emit_all(
                [
                    RunStarted(run_id=RUN, agent="desk"),
                    RunStarted(run_id=RUN, agent="x" * 512),
                    RunCompleted(run_id=RUN),
                ]
            )
        assert len(published.events) == 2
        assert len(report.rejected) == 1

    async def test_guaranteed_publishes_what_it_can_and_then_refuses(self) -> None:
        eventing, published = _eventing(max_event_bytes=420, delivery=Delivery.GUARANTEED)
        with tenant_scope(TENANT), pytest.raises(EventTooLargeError):
            await eventing.emit_all(
                [RunStarted(run_id=RUN, agent="desk"), RunStarted(run_id=RUN, agent="x" * 512)]
            )
        assert len(published.events) == 1


class TestARunThatIsAlreadyOver:
    async def test_a_cancelled_run_still_reports_that_it_was_cancelled(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunCancelled(run_id=RUN, reason_code="caller"))
            await eventing.emit(
                ToolCallCompleted(run_id=RUN, tool="t", tool_call_id="c1", state="abandoned")
            )
        assert [event.type for event in published.events] == [
            EventType.RUN_CANCELLED,
            EventType.TOOL_CALL_COMPLETED,
        ]


class TestTheFakeAndItsHelpers:
    async def test_the_helper_names_the_first_event_that_is_out_of_place(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
        with pytest.raises(AssertionError, match="run_completed"):
            assert_events(published.events, EventType.RUN_COMPLETED)

    async def test_the_helper_notices_an_event_that_never_came(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
        with pytest.raises(AssertionError, match="published"):
            assert_events(published.events, EventType.RUN_STARTED, EventType.RUN_COMPLETED)

    async def test_the_fake_keeps_them_in_the_order_they_were_published(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
            await eventing.emit(RunFailed(run_id=RUN, error_code="tool_failed"))
        assert published.of_type(EventType.RUN_FAILED)[0].attributes["error_code"] == "tool_failed"

    async def test_the_fake_can_be_cleared_between_phases_of_a_test(self) -> None:
        eventing, published = _eventing()
        with tenant_scope(TENANT):
            await eventing.emit(RunStarted(run_id=RUN, agent="desk"))
        published.clear()
        assert published.events == ()


class _Broken:
    """A transport that is not there."""

    async def publish(self, event: EventEnvelope) -> None:
        del event
        raise ConnectionError("no route")

    async def publish_batch(self, events: tuple[EventEnvelope, ...]) -> None:
        del events
        raise ConnectionError("no route")
