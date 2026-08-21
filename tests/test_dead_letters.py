"""What a dead letter records, what an operator may see of it, and what a replay may do.

A recovery that bypasses idempotency causes the second incident, so replay goes through the
same consumer path as live traffic and every guardrail here is about that.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tesserix_adk.adapters.dead_letters import (
    MAX_REPLAY_BATCH,
    DeadLetterQuery,
    DeadLetterRecord,
    InMemoryDeadLetters,
    Replayer,
)
from tesserix_adk.adapters.idempotent_events import IdempotentConsumer, dedupe_key
from tesserix_adk.core.errors import ScopeViolationError
from tesserix_adk.core.events import (
    Delivery,
    EventEnvelope,
    Eventing,
    EventType,
    RunCompleted,
    ToolCallCompleted,
)
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.runtime import MemoryIdempotencyStore
from tesserix_adk.testing import FakeClock, InMemoryEventPublisher

TENANT = "acme"
NOW = 1_000.0


async def _event(
    run_id: str = "run_1", tenant: str = TENANT, tool: str = "search"
) -> EventEnvelope:
    eventing = Eventing(clock=FakeClock(NOW), delivery=Delivery.GUARANTEED)
    with tenant_scope(tenant, user="ada"):
        event = await eventing.emit(
            ToolCallCompleted(run_id=run_id, tool=tool, tool_call_id="c1", state="ok")
        )
    assert event is not None
    return event


async def _run_completed(run_id: str = "run_1") -> EventEnvelope:
    eventing = Eventing(clock=FakeClock(NOW), delivery=Delivery.GUARANTEED)
    with tenant_scope(TENANT, user="ada"):
        event = await eventing.emit(RunCompleted(run_id=run_id, iterations=1))
    assert event is not None
    return event


def _letters(clock: FakeClock | None = None) -> InMemoryDeadLetters:
    return InMemoryDeadLetters(clock=clock or FakeClock(NOW))


def _query(**options: Any) -> DeadLetterQuery:
    return DeadLetterQuery(tenant=TENANT, **options)


class TestWhatIsRecorded:
    async def test_a_buried_event_becomes_a_record(self) -> None:
        letters = _letters()
        event = await _event()

        await letters.bury(event.to_json().encode(), reason="handler_failed", history=("KeyError",))

        [record] = await letters.list(_query())
        assert record.envelope.event_id == event.event_id
        assert record.reason == "handler_failed"
        assert record.last_error == "KeyError"

    async def test_the_same_event_buried_twice_counts_attempts(self) -> None:
        letters = _letters(FakeClock(NOW))
        event = await _event()

        await letters.bury(event.to_json().encode(), reason="handler_failed")
        await letters.bury(event.to_json().encode(), reason="handler_failed")

        [record] = await letters.list(_query())
        assert record.attempts == 2

    async def test_the_record_keeps_when_it_first_and_last_arrived(self) -> None:
        clock = FakeClock(NOW)
        letters = _letters(clock)
        event = await _event()

        await letters.bury(event.to_json().encode(), reason="handler_failed")
        clock.advance(60.0)
        await letters.bury(event.to_json().encode(), reason="handler_failed")

        [record] = await letters.list(_query())
        assert (record.first_seen, record.last_seen) == (NOW, NOW + 60.0)

    async def test_an_undecodable_payload_is_kept_rather_than_dropped(self) -> None:
        letters = _letters()

        await letters.bury(b"{not json", reason="undecodable")

        assert await letters.undecodable() == 1

    async def test_an_event_of_a_type_this_build_never_knew_is_kept_as_unreadable(self) -> None:
        letters = _letters()
        forged = json.loads((await _event()).to_json())
        forged["type"] = "invented_elsewhere"

        await letters.bury(json.dumps(forged).encode(), reason="handler_failed")

        assert await letters.undecodable() == 1

    async def test_the_error_is_the_exception_type_and_not_its_message(self) -> None:
        letters = _letters()
        event = await _event()

        await letters.bury(
            event.to_json().encode(),
            reason="handler_failed",
            history=("ValueError: card 4111 1111 1111 1111 declined",),
        )

        [record] = await letters.list(_query())
        assert record.last_error == "ValueError"


class TestInspection:
    async def test_only_the_tenant_asked_for_is_listed(self) -> None:
        letters = _letters()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        await letters.bury(
            (await _event(tenant="globex")).to_json().encode(), reason="handler_failed"
        )

        assert [record.envelope.tenant for record in await letters.list(_query())] == [TENANT]

    async def test_a_tenant_is_mandatory_because_a_listing_crosses_nothing(self) -> None:
        with pytest.raises(ValueError, match="tenant"):
            DeadLetterQuery(tenant="")

    async def test_filtering_by_event_type(self) -> None:
        letters = _letters()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        await letters.bury((await _run_completed()).to_json().encode(), reason="handler_failed")

        found = await letters.list(_query(event_type=EventType.RUN_COMPLETED))

        assert [record.envelope.type for record in found] == [EventType.RUN_COMPLETED]

    async def test_filtering_by_time_window(self) -> None:
        clock = FakeClock(NOW)
        letters = _letters(clock)
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        clock.advance(3_600.0)
        await letters.bury((await _event("run_2")).to_json().encode(), reason="handler_failed")

        found = await letters.list(_query(since=NOW + 60.0))

        assert [record.envelope.run_id for record in found] == ["run_2"]

    async def test_filtering_by_consumer_group(self) -> None:
        letters = _letters()
        await letters.bury(
            (await _event()).to_json().encode(), reason="handler_failed", group="billing"
        )
        await letters.bury(
            (await _event("run_2")).to_json().encode(), reason="handler_failed", group="search"
        )

        found = await letters.list(_query(group="billing"))

        assert [record.envelope.run_id for record in found] == ["run_1"]

    async def test_a_listing_is_paged_rather_than_unbounded(self) -> None:
        letters = _letters()
        for index in range(10):
            await letters.bury(
                (await _event(f"run_{index}")).to_json().encode(), reason="handler_failed"
            )

        assert len(await letters.list(_query(limit=3))) == 3

    async def test_a_limit_above_the_cap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            DeadLetterQuery(tenant=TENANT, limit=MAX_REPLAY_BATCH + 1)

    async def test_inspection_shows_the_identifiers_and_not_the_body(self) -> None:
        letters = _letters()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")

        [record] = await letters.list(_query())
        rendered = record.inspected()

        assert rendered["run_id"] == "run_1"
        assert rendered["attributes"] == "duration_ms, run_id, state, tool, tool_call_id"
        assert "search" not in str(rendered)

    async def test_a_window_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="before"):
            DeadLetterQuery(tenant=TENANT, since=NOW, until=NOW - 60.0)

    async def test_filtering_by_the_end_of_a_window(self) -> None:
        clock = FakeClock(NOW)
        letters = _letters(clock)
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        clock.advance(3_600.0)
        await letters.bury((await _event("run_2")).to_json().encode(), reason="handler_failed")

        found = await letters.list(_query(until=NOW + 60.0))

        assert [record.envelope.run_id for record in found] == ["run_1"]

    async def test_a_consumer_is_given_a_view_that_records_its_own_group(self) -> None:
        letters = _letters()

        await letters.for_group("billing").bury(
            (await _event()).to_json().encode(), reason="handler_failed"
        )

        [record] = await letters.list(_query(group="billing"))
        assert record.group == "billing"

    async def test_the_arrival_rate_is_counted_even_where_nothing_is_left_buried(self) -> None:
        letters = _letters()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        await letters.forget([record.envelope.event_id for record in await letters.list(_query())])

        stats = await letters.stats(TENANT)

        assert (stats.buried, stats.arrivals) == (0, 1)

    async def test_the_backlog_reports_what_an_operator_alerts_on(self) -> None:
        clock = FakeClock(NOW)
        letters = _letters(clock)
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        clock.advance(120.0)

        stats = await letters.stats(TENANT)

        assert (stats.buried, stats.oldest_seconds) == (1, 120.0)


class TestReplay:
    async def test_a_dry_run_reports_what_would_go_without_sending_it(self) -> None:
        letters, seen = _letters(), _seen()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen)

        plan = await replayer.plan(_query())

        assert plan.replayable == 1
        assert seen == []

    async def test_a_replay_redelivers_through_the_consumer_path(self) -> None:
        letters, seen = _letters(), _seen()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen)

        report = await replayer.replay(_query(), operator="ada")

        assert report.replayed == 1
        assert [event.run_id for event in seen] == ["run_1"]

    async def test_an_already_processed_event_is_suppressed_not_applied_twice(self) -> None:
        letters, seen = _letters(), _seen()
        event = await _event()
        store = MemoryIdempotencyStore(clock=FakeClock(NOW))
        once = IdempotentConsumer(_handler(seen), store=store, group="billing")
        await once.handle(event)
        await letters.bury(event.to_json().encode(), reason="handler_failed", group="billing")
        replayer = Replayer(letters, handler=once.handle, clock=FakeClock(NOW))

        report = await replayer.replay(_query(group="billing"), operator="ada")

        assert len(seen) == 1
        assert report.suppressed == 1

    async def test_an_event_the_consumer_gave_up_on_is_processed_again(self) -> None:
        letters, seen = _letters(), _seen()
        event = await _event()
        store = MemoryIdempotencyStore(clock=FakeClock(NOW))
        broken = IdempotentConsumer(
            _explodes, store=store, group="billing", max_attempts=1, dead_letter=letters
        )
        await broken.handle(event)
        fixed = IdempotentConsumer(_handler(seen), store=store, group="billing")

        report = await Replayer(letters, handler=fixed.handle, clock=FakeClock(NOW)).replay(
            _query(), operator="ada"
        )

        assert (report.replayed, len(seen)) == (1, 1)

    async def test_every_replayed_envelope_is_marked_as_one(self) -> None:
        letters, seen = _letters(), _seen()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen)

        report = await replayer.replay(_query(), operator="ada")

        assert seen[0].replay_id == report.replay_id
        assert report.replay_id != ""

    async def test_live_traffic_carries_no_replay_marker(self) -> None:
        assert (await _event()).replay_id == ""

    async def test_a_replay_without_a_tenant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tenant"):
            DeadLetterQuery(tenant="")

    async def test_a_selection_spanning_tenants_is_refused(self) -> None:
        letters, seen = _letters(), _seen()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        await letters.bury(
            (await _event(tenant="globex")).to_json().encode(), reason="handler_failed"
        )
        letters.leak = True
        replayer = _replayer(letters, seen)

        with pytest.raises(ScopeViolationError, match="globex"):
            await replayer.replay(_query(), operator="ada")

        assert seen == []

    async def test_a_batch_is_capped_rather_than_replaying_a_whole_backlog(self) -> None:
        letters, seen = _letters(), _seen()
        for index in range(5):
            await letters.bury(
                (await _event(f"run_{index}")).to_json().encode(), reason="handler_failed"
            )
        replayer = _replayer(letters, seen)

        report = await replayer.replay(_query(limit=2), operator="ada")

        assert (report.replayed, report.remaining) == (2, 3)

    async def test_a_run_is_replayed_in_the_order_its_events_were_emitted(self) -> None:
        letters, seen = _letters(), _seen()
        first = await _event("run_1", tool="search")
        second = await _event("run_1", tool="write")
        await letters.bury(second.to_json().encode(), reason="handler_failed")
        await letters.bury(first.to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen)

        await replayer.replay(_query(), operator="ada")

        assert [event.event_id for event in seen] == sorted([first.event_id, second.event_id])

    async def test_a_record_this_kit_can_no_longer_read_is_refused_not_guessed(self) -> None:
        letters, seen = _letters(), _seen()
        event = await _event()
        ahead = event.model_copy(update={"schema_version": 99})
        await letters.bury(ahead.to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen)

        report = await replayer.replay(_query(), operator="ada")

        assert (report.replayed, report.refused) == (0, 1)
        assert report.refusals[0][1] == "unsupported_version"
        assert seen == []

    async def test_an_undecodable_record_is_refused_and_the_batch_continues(self) -> None:
        letters, seen = _letters(), _seen()
        await letters.bury(b"{not json", reason="undecodable")
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen)

        report = await replayer.replay(_query(), operator="ada")

        assert (report.replayed, report.refused) == (1, 1)

    async def test_a_record_whose_scope_was_erased_is_refused(self) -> None:
        letters, seen = _letters(), _seen()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen, erased=_always_erased)

        report = await replayer.replay(_query(), operator="ada")

        assert (report.replayed, report.refused) == (0, 1)
        assert report.refusals[0][1] == "erased"
        assert seen == []

    async def test_a_handler_that_fails_again_leaves_the_record_where_it_was(self) -> None:
        letters = _letters()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        replayer = Replayer(letters, handler=_explodes, clock=FakeClock(NOW))

        report = await replayer.replay(_query(), operator="ada")

        assert (report.replayed, report.failed) == (0, 1)
        assert len(await letters.list(_query())) == 1

    async def test_an_event_another_worker_holds_is_left_for_that_worker(self) -> None:
        letters, seen = _letters(), _seen()
        event = await _event()
        await letters.bury(event.to_json().encode(), reason="handler_failed", group="billing")
        store = MemoryIdempotencyStore(clock=FakeClock(NOW))
        await store.begin(dedupe_key(group="billing", event=event), tenant=TENANT, ttl_seconds=60.0)
        held = IdempotentConsumer(_handler(seen), store=store, group="billing")

        report = await Replayer(letters, handler=held.handle, clock=FakeClock(NOW)).replay(
            _query(), operator="ada"
        )

        assert (report.replayed, report.failed, seen) == (0, 1, [])
        assert len(await letters.list(_query())) == 1

    async def test_a_replayed_record_is_forgotten_so_it_is_not_replayed_twice(self) -> None:
        letters, seen = _letters(), _seen()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        replayer = _replayer(letters, seen)

        await replayer.replay(_query(), operator="ada")

        assert await letters.list(_query()) == []


class TestTheAuditTrail:
    async def test_a_replay_records_who_replayed_what(self) -> None:
        letters, seen, published = _letters(), _seen(), InMemoryEventPublisher()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        eventing = Eventing(published, clock=FakeClock(NOW), delivery=Delivery.GUARANTEED)
        replayer = _replayer(letters, seen, eventing=eventing)

        report = await replayer.replay(_query(), operator="ada")

        [audit] = [e for e in published.events if e.type is EventType.EVENTS_REPLAYED]
        assert audit.tenant == TENANT
        assert audit.attributes["approver"] == "ada"
        assert audit.attributes["replay_id"] == report.replay_id
        assert audit.attributes["records"] == "1"

    async def test_the_audit_event_carries_no_payload_content(self) -> None:
        letters, seen, published = _letters(), _seen(), InMemoryEventPublisher()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        eventing = Eventing(published, clock=FakeClock(NOW), delivery=Delivery.GUARANTEED)

        await _replayer(letters, seen, eventing=eventing).replay(_query(), operator="ada")

        [audit] = [e for e in published.events if e.type is EventType.EVENTS_REPLAYED]
        assert "search" not in str(audit.attributes)

    async def test_a_dry_run_records_nothing_because_nothing_happened(self) -> None:
        letters, seen, published = _letters(), _seen(), InMemoryEventPublisher()
        await letters.bury((await _event()).to_json().encode(), reason="handler_failed")
        eventing = Eventing(published, clock=FakeClock(NOW), delivery=Delivery.GUARANTEED)

        await _replayer(letters, seen, eventing=eventing).plan(_query())

        assert published.events == ()


def _seen() -> list[EventEnvelope]:
    return []


def _handler(seen: list[EventEnvelope]) -> Any:
    async def handle(event: EventEnvelope) -> None:
        seen.append(event)

    return handle


async def _explodes(event: EventEnvelope) -> None:  # noqa: ARG001 — the handler's own signature
    raise ConnectionError("still broken")


async def _always_erased(record: DeadLetterRecord) -> bool:  # noqa: ARG001 — the check's own signature
    return True


def _replayer(letters: InMemoryDeadLetters, seen: list[EventEnvelope], **options: Any) -> Replayer:
    return Replayer(letters, handler=_handler(seen), clock=FakeClock(NOW), **options)
