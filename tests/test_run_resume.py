"""A run paused for two days, resumed on a worker that was not there when it started."""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    Checkpoint,
    CheckpointBoundary,
    CheckpointFormatError,
    HistoryUnavailableError,
    Message,
    PendingCall,
    RunLeaseError,
    TextPart,
    ToolCall,
    ToolDisposition,
    Usage,
)
from tesserix_adk.runtime import (
    Checkpointer,
    HistoryStore,
    MemoryCheckpointStore,
    MemoryLeaseStore,
    Resumer,
    scrubbed,
)
from tesserix_adk.runtime.idempotency import MemoryIdempotencyStore
from tesserix_adk.testing import FakeClock

pytestmark = pytest.mark.anyio


def said(text: str) -> Message:
    """One user turn."""
    return Message(role="user", content=[TextPart(text=text)])


def frontier(**overrides: object) -> Checkpoint:
    """A run paused on an approval, two days ago."""
    fields: dict[str, object] = {
        "run_id": "r1",
        "tenant": "acme",
        "agent_name": "booking",
        "model": "claude-opus-5",
        "boundary": CheckpointBoundary.BEFORE_APPROVAL,
        "messages": (said("book the 18:40"),),
        "usage": Usage(input_tokens=1_200, output_tokens=300),
        "cost_micros": 4_100,
        "iterations": 3,
        "pending_approval": "req-9",
        "grant_id": "grant-2",
        "scopes": ("booking:write",),
        "user": "ada",
    }
    return Checkpoint(**(fields | overrides))  # type: ignore[arg-type]


class Transcripts:
    """A history store that can be asked to have lost what it held."""

    def __init__(self, *, held: tuple[Message, ...] = ()) -> None:
        self.held = held
        self.evicted = False

    async def fetch(self, handle: str, *, tenant: str) -> tuple[Message, ...] | None:
        """Return the transcript, or nothing once the store has been asked to lose it."""
        assert tenant
        assert handle
        return None if self.evicted else self.held


def resumer(
    *,
    store: MemoryCheckpointStore | None = None,
    history: HistoryStore | None = None,
    idempotency: MemoryIdempotencyStore | None = None,
) -> tuple[Resumer, MemoryCheckpointStore, FakeClock]:
    """A resumer over in-memory stores, and the clock its leases expire on."""
    clock = FakeClock()
    checkpoints = store or MemoryCheckpointStore()
    return (
        Resumer(
            checkpoints=Checkpointer(checkpoints, clock=clock),
            leases=MemoryLeaseStore(clock),
            history=history,
            idempotency=idempotency,
        ),
        checkpoints,
        clock,
    )


class TestCarryingARunOn:
    """The run continues where it stopped, on everything it was carrying."""

    async def test_it_resumes_at_the_iteration_it_stopped_at(self) -> None:
        resume, store, _ = resumer()
        await store.put(frontier())

        carried = await resume.resume("r1", tenant="acme", worker="w1")

        assert carried is not None
        assert carried.iterations == 3
        assert carried.checkpoint.usage.input_tokens == 1_200
        assert carried.checkpoint.cost_micros == 4_100

    async def test_the_tenant_and_user_survive_the_restart(self) -> None:
        resume, store, _ = resumer()
        await store.put(frontier())

        carried = await resume.resume("r1", tenant="acme", worker="w1")

        assert carried is not None
        assert carried.checkpoint.tenant == "acme"
        assert carried.checkpoint.user == "ada"
        assert carried.checkpoint.scopes == ("booking:write",)

    async def test_the_approval_the_run_is_waiting_on_comes_back_with_it(self) -> None:
        resume, store, _ = resumer()
        await store.put(frontier())

        carried = await resume.resume("r1", tenant="acme", worker="w1")

        assert carried is not None
        assert carried.awaiting_approval == "req-9"
        assert carried.checkpoint.grant_id == "grant-2"

    async def test_the_approved_call_is_applied_exactly_once(self) -> None:
        idempotency = MemoryIdempotencyStore()
        await idempotency.begin("charge-1", tenant="acme", ttl_seconds=60.0)
        await idempotency.record("charge-1", tenant="acme", outcome="charged", ttl_seconds=60.0)
        resume, store, _ = resumer(idempotency=idempotency)
        await store.put(
            frontier(
                pending=(
                    PendingCall(
                        call=ToolCall(id="c0", name="charge_card"),
                        idempotency_key="charge-1",
                        dispatched=True,
                    ),
                )
            )
        )

        carried = await resume.resume("r1", tenant="acme", worker="w1")

        assert carried is not None
        assert carried.plan.safe is True
        assert [one.disposition for one in carried.plan.completed] == [ToolDisposition.COMPLETED]
        assert carried.plan.to_dispatch == ()

    async def test_a_run_nobody_checkpointed_resumes_as_nothing(self) -> None:
        resume, _, _ = resumer()

        assert await resume.resume("r1", tenant="acme", worker="w1") is None

    async def test_a_run_that_was_never_checkpointed_is_not_left_held(self) -> None:
        resume, _, _ = resumer()

        await resume.resume("r1", tenant="acme", worker="w1")

        second = resume.holder("w2")
        assert (await second.acquire("r1", tenant="acme")).holder == "w2"


class TestTwoWorkersOneRun:
    """The second worker is refused before anything is dispatched."""

    async def test_the_second_resume_is_refused_by_the_lease(self) -> None:
        resume, store, _ = resumer()
        await store.put(frontier())
        await resume.resume("r1", tenant="acme", worker="w1")

        with pytest.raises(RunLeaseError) as refused:
            await resume.resume("r1", tenant="acme", worker="w2")

        assert refused.value.holder == "w1"
        assert refused.value.requested_by == "w2"

    async def test_the_refusal_happens_before_the_frontier_is_even_read(self) -> None:
        class Counting(MemoryCheckpointStore):
            reads = 0

            async def latest(self, run_id: str, *, tenant: str) -> Checkpoint | None:
                type(self).reads += 1
                return await super().latest(run_id, tenant=tenant)

        store = Counting()
        resume, _, _ = resumer(store=store)
        await store.put(frontier())
        await resume.resume("r1", tenant="acme", worker="w1")

        with pytest.raises(RunLeaseError):
            await resume.resume("r1", tenant="acme", worker="w2")

        assert Counting.reads == 1

    async def test_the_run_is_takeable_once_the_holder_lets_go(self) -> None:
        resume, store, _ = resumer()
        await store.put(frontier())
        carried = await resume.resume("r1", tenant="acme", worker="w1")
        assert carried is not None

        holder = resume.holder("w1")
        holder._lease = carried.lease
        await holder.release()

        assert await resume.resume("r1", tenant="acme", worker="w2") is not None


class TestRefusingToGuess:
    """A resume that would have to invent something fails instead."""

    async def test_a_checkpoint_from_a_newer_kit_is_refused(self) -> None:
        resume, store, _ = resumer()
        await store.put(frontier(format_version=99))

        with pytest.raises(CheckpointFormatError) as refused:
            await resume.resume("r1", tenant="acme", worker="w1")

        assert refused.value.format_version == 99

    async def test_an_evicted_transcript_fails_closed(self) -> None:
        transcripts = Transcripts(held=(said("book the 18:40"),))
        transcripts.evicted = True
        resume, store, _ = resumer(history=transcripts)
        await store.put(frontier(history_handle="h-1", messages=()))

        with pytest.raises(HistoryUnavailableError) as refused:
            await resume.resume("r1", tenant="acme", worker="w1")

        assert refused.value.handle == "h-1"
        assert refused.value.retryable is False

    async def test_a_handle_with_no_store_behind_it_fails_closed(self) -> None:
        resume, store, _ = resumer()
        await store.put(frontier(history_handle="h-1", messages=()))

        with pytest.raises(HistoryUnavailableError):
            await resume.resume("r1", tenant="acme", worker="w1")

    async def test_a_resolved_transcript_comes_back_on_the_checkpoint(self) -> None:
        transcripts = Transcripts(held=(said("book the 18:40"), said("and a hotel")))
        resume, store, _ = resumer(history=transcripts)
        await store.put(frontier(history_handle="h-1", messages=()))

        carried = await resume.resume("r1", tenant="acme", worker="w1")

        assert carried is not None
        assert len(carried.checkpoint.messages) == 2

    async def test_a_dispatched_call_nothing_can_decide_is_named(self) -> None:
        resume, store, _ = resumer()
        await store.put(
            frontier(
                pending=(PendingCall(call=ToolCall(id="c0", name="charge_card"), dispatched=True),)
            )
        )

        carried = await resume.resume("r1", tenant="acme", worker="w1")

        assert carried is not None
        assert carried.plan.safe is False

    async def test_the_history_store_protocol_is_satisfied_by_a_plain_object(self) -> None:
        assert isinstance(Transcripts(), HistoryStore)


class TestWhatIsWrittenDown:
    """A checkpoint outlives the run, and gets read by whoever debugs the resume."""

    async def test_a_credential_in_the_transcript_never_reaches_the_store(self) -> None:
        store = MemoryCheckpointStore()
        writer = Checkpointer(store, clock=FakeClock())
        await writer.record(
            frontier(
                boundary=CheckpointBoundary.AFTER_MODEL_CALL,
                messages=(said("use sk-live-0123456789 for the booking"),),
            )
        )

        written = await store.latest("r1", tenant="acme")

        assert written is not None
        assert "sk-live-0123456789" not in written.model_dump_json()
        assert "[redacted]" in written.model_dump_json()

    async def test_a_credential_in_a_tool_argument_never_reaches_the_store(self) -> None:
        masked = scrubbed(
            frontier(
                pending=(
                    PendingCall(
                        call=ToolCall(
                            id="c0",
                            name="charge_card",
                            arguments={
                                "token": "Bearer abc123def456",  # gitleaks:allow
                                "amount": 4_100,
                            },
                        ),
                    ),
                )
            )
        )

        assert masked.pending[0].call.arguments["token"] == "[redacted]"  # noqa: S105
        assert masked.pending[0].call.arguments["amount"] == 4_100

    def test_a_credential_in_an_assistant_tool_call_is_masked_too(self) -> None:
        message = Message(
            role="assistant",
            tool_calls=(
                ToolCall(
                    id="c0",
                    name="pay",
                    arguments={"key": "sk-live-0123456789"},  # gitleaks:allow
                ),
            ),
        )

        masked = scrubbed(frontier(messages=(message,)))

        assert masked.messages[0].tool_calls[0].arguments["key"] == "[redacted]"

    def test_redaction_can_be_told_about_this_deployments_own_shapes(self) -> None:
        masked = scrubbed(frontier(messages=(said("staff id EMP-7781"),)), (r"EMP-\d+",))

        part = masked.messages[0].content[0]
        assert isinstance(part, TextPart)
        assert part.text == "staff id [redacted]"

    async def test_redaction_can_be_turned_off_where_a_store_does_its_own(self) -> None:
        store = MemoryCheckpointStore()
        writer = Checkpointer(store, clock=FakeClock(), redact=False)
        await writer.record(
            frontier(
                boundary=CheckpointBoundary.AFTER_MODEL_CALL,
                messages=(said("use sk-live-0123456789"),),
            )
        )

        written = await store.latest("r1", tenant="acme")

        assert written is not None
        assert "sk-live-0123456789" in written.model_dump_json()

    def test_a_binary_part_carries_no_text_to_mask(self) -> None:
        from tesserix_adk.core import BinaryPart

        part = BinaryPart(media_type="image/png", data=b"\x89PNG")
        masked = scrubbed(frontier(messages=(Message(role="user", content=[part]),)))

        assert masked.messages[0].content[0] == part
