"""What `adk dead-letters` shows an operator, and what it makes them say before a replay.

The command is the recovery path a person uses at three in the morning, so the two things
under test are that a listing never prints a payload and that a replay cannot happen by
accident.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from tesserix_adk.adapters.dead_letters import InMemoryDeadLetters, Replayer
from tesserix_adk.cli.dead_letters import MISUSED, OK, main
from tesserix_adk.core.events import Delivery, Eventing, ToolCallCompleted
from tesserix_adk.core.tenancy import tenant_scope
from tesserix_adk.testing import FakeClock

if TYPE_CHECKING:
    from tesserix_adk.core.events import EventEnvelope

TENANT = "acme"
NOW = 1_000.0


async def _buried(letters: InMemoryDeadLetters, run_id: str = "run_1") -> EventEnvelope:
    eventing = Eventing(clock=FakeClock(NOW), delivery=Delivery.GUARANTEED)
    with tenant_scope(TENANT, user="ada"):
        event = await eventing.emit(
            ToolCallCompleted(run_id=run_id, tool="search", tool_call_id="c1", state="ok")
        )
    assert event is not None
    await letters.bury(event.to_json().encode(), reason="handler_failed", group="billing")
    return event


def _parts() -> tuple[InMemoryDeadLetters, list[EventEnvelope], Replayer, io.StringIO]:
    letters = InMemoryDeadLetters(clock=FakeClock(NOW))
    seen: list[EventEnvelope] = []

    async def handle(event: EventEnvelope) -> None:
        seen.append(event)

    return letters, seen, Replayer(letters, handler=handle, clock=FakeClock(NOW)), io.StringIO()


class TestListing:
    async def test_it_shows_the_identifiers_and_never_the_payload(self) -> None:
        letters, _, replayer, out = _parts()
        event = await _buried(letters)

        code = await main(["list", "--tenant", TENANT], replayer=replayer, out=out)

        assert code == OK
        assert event.event_id in out.getvalue()
        assert "search" not in out.getvalue()

    async def test_an_empty_backlog_says_so(self) -> None:
        _, _, replayer, out = _parts()

        await main(["list", "--tenant", TENANT], replayer=replayer, out=out)

        assert "nothing" in out.getvalue()

    async def test_a_tenant_is_required_because_a_listing_crosses_nothing(self) -> None:
        _, _, replayer, out = _parts()

        assert await main(["list"], replayer=replayer, out=out) == MISUSED

    async def test_show_prints_one_record_field_by_field(self) -> None:
        letters, _, replayer, out = _parts()
        event = await _buried(letters)

        code = await main(
            ["show", "--tenant", TENANT, "--event-id", event.event_id],
            replayer=replayer,
            out=out,
        )

        assert code == OK
        assert "attempts" in out.getvalue()
        assert "search" not in out.getvalue()

    async def test_show_says_plainly_when_there_is_no_such_record(self) -> None:
        _, _, replayer, out = _parts()

        code = await main(
            ["show", "--tenant", TENANT, "--event-id", "nope"], replayer=replayer, out=out
        )

        assert (code, "no record" in out.getvalue()) == (1, True)


class TestReplaying:
    async def test_a_dry_run_reports_what_would_go_and_sends_nothing(self) -> None:
        letters, seen, replayer, out = _parts()
        await _buried(letters)

        code = await main(
            ["replay", "--tenant", TENANT, "--by", "ada", "--dry-run"],
            replayer=replayer,
            out=out,
        )

        assert (code, seen) == (OK, [])
        assert "1" in out.getvalue()

    async def test_a_replay_needs_a_name_to_put_in_the_audit_record(self) -> None:
        letters, seen, replayer, out = _parts()
        await _buried(letters)

        assert await main(["replay", "--tenant", TENANT], replayer=replayer, out=out) == MISUSED
        assert seen == []

    async def test_a_replay_redelivers_and_reports_what_it_did(self) -> None:
        letters, seen, replayer, out = _parts()
        await _buried(letters)

        code = await main(
            ["replay", "--tenant", TENANT, "--by", "ada", "--reason", "consumer_fixed"],
            replayer=replayer,
            out=out,
        )

        assert (code, len(seen)) == (OK, 1)
        assert "replayed 1" in out.getvalue()

    async def test_a_refusal_is_named_so_an_operator_can_act_on_it(self) -> None:
        letters, _, replayer, out = _parts()
        await letters.bury(b"{not json", reason="undecodable")

        await main(["replay", "--tenant", TENANT, "--by", "ada"], replayer=replayer, out=out)

        assert "undecodable" in out.getvalue()
