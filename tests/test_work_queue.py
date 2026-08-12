"""Work handed to a worker that then dies, and what the queue does about it."""

from __future__ import annotations

import pytest

from tesserix_adk.core.errors import LeaseLostError, QueueUnavailableError, WorkItemNotFoundError
from tesserix_adk.core.queue import QueuePolicy, WorkItem, WorkPriority, WorkQueue, WorkState
from tesserix_adk.runtime import MemoryWorkQueue
from tesserix_adk.runtime.queue import LEASE_LAPSED, WORKER_RESTARTED
from tesserix_adk.testing import FakeClock, WorkQueueConformance

TENANT = "acme"


def a_queue(
    clock: FakeClock | None = None,
    *,
    lease_seconds: float = 30.0,
    max_attempts: int = 5,
    backoff_seconds: float = 1.0,
    max_lease_seconds: float = 3600.0,
) -> MemoryWorkQueue:
    """A queue whose leases and attempts a test can reason about."""
    policy = QueuePolicy(
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        max_lease_seconds=max_lease_seconds,
    )
    return MemoryWorkQueue(policy, clock or FakeClock())


def an_item(
    item_id: str = "i1",
    *,
    tenant: str = TENANT,
    queue: str = "default",
    priority: WorkPriority = WorkPriority.NORMAL,
    dedupe_key: str | None = None,
    state: WorkState = WorkState.QUEUED,
    worker: str | None = None,
    enqueued_at: float = 0.0,
    available_at: float = 0.0,
    lease_expires_at: float | None = None,
    first_claimed_at: float | None = None,
) -> WorkItem:
    """An item ready to be enqueued."""
    return WorkItem(
        id=item_id,
        tenant=tenant,
        queue=queue,
        priority=priority,
        dedupe_key=dedupe_key,
        state=state,
        worker=worker,
        enqueued_at=enqueued_at,
        available_at=available_at,
        lease_expires_at=lease_expires_at,
        first_claimed_at=first_claimed_at,
    )


class TestWhatAnItemKnowsAboutItself:
    def test_a_new_item_is_waiting(self) -> None:
        assert an_item().state is WorkState.QUEUED

    def test_an_unclaimed_item_is_not_held(self) -> None:
        assert an_item().held is False

    def test_a_claimed_item_names_who_holds_it(self) -> None:
        """A claim nobody is attributed with is a claim no reaper can free."""
        with pytest.raises(ValueError, match="names the worker"):
            an_item(state=WorkState.CLAIMED)

    def test_an_item_nobody_has_claimed_was_never_held(self) -> None:
        assert an_item().held_since == 0.0

    def test_a_claimed_item_remembers_when_it_was_taken(self) -> None:
        assert an_item(state=WorkState.CLAIMED, worker="w1", first_claimed_at=9.0).held_since == 9.0

    def test_a_completed_item_is_terminal(self) -> None:
        assert an_item(state=WorkState.COMPLETED).terminal is True

    def test_a_dead_lettered_item_is_terminal(self) -> None:
        assert an_item(state=WorkState.DEAD_LETTERED).terminal is True

    def test_a_waiting_item_is_not_terminal(self) -> None:
        assert an_item().terminal is False

    def test_an_unclaimed_item_has_not_lapsed(self) -> None:
        assert an_item().lapsed_at(1_000.0) is False

    def test_a_claim_lapses_when_its_lease_runs_out(self) -> None:
        held = an_item(state=WorkState.CLAIMED, worker="w1", lease_expires_at=10.0)
        assert (held.lapsed_at(9.0), held.lapsed_at(11.0)) == (False, True)

    def test_a_claim_with_no_lease_does_not_lapse(self) -> None:
        assert an_item(state=WorkState.CLAIMED, worker="w1").lapsed_at(1_000.0) is False


class TestWhatAPolicySays:
    def test_the_first_failure_waits_the_base(self) -> None:
        assert QueuePolicy(backoff_seconds=2.0).backoff_for(1) == 2.0

    def test_each_further_failure_waits_twice_as_long(self) -> None:
        assert QueuePolicy(backoff_seconds=2.0).backoff_for(3) == 8.0

    def test_the_wait_stops_doubling_at_the_cap(self) -> None:
        assert QueuePolicy(backoff_seconds=2.0, backoff_cap_seconds=5.0).backoff_for(9) == 5.0

    def test_an_item_that_has_not_failed_waits_for_nothing(self) -> None:
        assert QueuePolicy().backoff_for(0) == 0.0

    def test_attempts_run_out_at_the_cap(self) -> None:
        policy = QueuePolicy(max_attempts=2)
        assert (policy.exhausted(1), policy.exhausted(2)) == (False, True)


class TestHandingWorkOver:
    async def test_the_memory_queue_is_a_work_queue(self) -> None:
        assert isinstance(a_queue(), WorkQueue)

    async def test_the_queue_says_what_it_will_do_to_an_item(self) -> None:
        assert a_queue(max_attempts=7).policy.max_attempts == 7

    async def test_an_enqueued_item_is_claimable(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item())
        claimed = await queue.claim(worker="w1")
        assert claimed is not None
        assert (claimed.id, claimed.worker, claimed.state) == ("i1", "w1", WorkState.CLAIMED)

    async def test_an_empty_queue_hands_out_nothing(self) -> None:
        assert await a_queue().claim(worker="w1") is None

    async def test_the_claim_runs_for_the_policy_s_lease(self) -> None:
        claimed = await _claimed(a_queue(lease_seconds=30.0))
        assert claimed.lease_expires_at == 30.0

    async def test_a_worker_may_ask_for_a_longer_lease(self) -> None:
        """A handler that knows it is slow says so, rather than heartbeating in a hot loop."""
        claimed = await _claimed(a_queue(lease_seconds=30.0), lease_seconds=120.0)
        assert claimed.lease_expires_at == 120.0

    async def test_an_item_is_only_handed_to_one_worker(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item())
        await queue.claim(worker="w1")
        assert await queue.claim(worker="w2") is None

    async def test_a_named_queue_does_not_serve_another_s_work(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item(queue="reports"))
        assert await queue.claim(worker="w1", queue="default") is None

    async def test_an_item_that_is_backing_off_is_not_due(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item(available_at=50.0))
        assert await queue.claim(worker="w1") is None

    async def test_the_enqueue_time_a_caller_gave_is_kept(self) -> None:
        """A job scheduled earlier and enqueued late is as old as it says it is."""
        queue = a_queue()
        stored = await queue.enqueue(an_item(enqueued_at=7.0))
        assert stored.enqueued_at == 7.0

    async def test_higher_priority_goes_first(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("low", priority=WorkPriority.LOW))
        await queue.enqueue(an_item("urgent", priority=WorkPriority.URGENT))
        claimed = await queue.claim(worker="w1")
        assert claimed is not None
        assert claimed.id == "urgent"

    async def test_at_equal_priority_the_older_item_goes_first(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("first"))
        await queue.enqueue(an_item("second"))
        claimed = await queue.claim(worker="w1")
        assert claimed is not None
        assert claimed.id == "first"


class TestOneTenantCannotStarveAnother:
    async def test_a_backlog_does_not_hold_the_queue(self) -> None:
        queue = a_queue()
        for index in range(5):
            await queue.enqueue(an_item(f"loud{index}", tenant="loud"))
        await queue.enqueue(an_item("quiet1", tenant="quiet"))
        served = [await queue.claim(worker=f"w{index}") for index in range(3)]
        assert any(item is not None and item.tenant == "quiet" for item in served)

    async def test_priority_orders_a_tenant_s_own_work_and_nothing_else(self) -> None:
        """Otherwise every tenant enqueues at urgent, and rightly."""
        queue = a_queue()
        await queue.enqueue(an_item("quiet1", tenant="quiet", priority=WorkPriority.LOW))
        await queue.enqueue(an_item("loud1", tenant="loud", priority=WorkPriority.URGENT))
        served = [await queue.claim(worker="w1"), await queue.claim(worker="w2")]
        assert {item.tenant for item in served if item is not None} == {"quiet", "loud"}


class TestSayingTheWorkerIsStillAlive:
    async def test_a_heartbeat_extends_the_claim(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=30.0)
        await _claimed(queue)
        clock.advance(20.0)
        renewed = await queue.heartbeat("i1", tenant=TENANT, worker="w1")
        assert renewed.lease_expires_at == 50.0

    async def test_a_heartbeat_on_a_lapsed_claim_is_refused(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=10.0)
        await _claimed(queue)
        clock.advance(11.0)
        with pytest.raises(LeaseLostError) as refused:
            await queue.heartbeat("i1", tenant=TENANT, worker="w1")
        assert refused.value.reason == "expired"

    async def test_a_heartbeat_from_a_worker_that_lost_the_item_names_who_has_it(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=10.0, backoff_seconds=0.0)
        await _claimed(queue)
        clock.advance(11.0)
        await queue.reap()
        await queue.claim(worker="w2")
        with pytest.raises(LeaseLostError) as refused:
            await queue.heartbeat("i1", tenant=TENANT, worker="w1")
        assert (refused.value.reason, refused.value.holder) == ("taken", "w2")

    async def test_a_claim_cannot_be_renewed_forever(self) -> None:
        """A stuck run looks exactly like a busy one to everything except this bound."""
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=100.0, max_lease_seconds=60.0)
        await _claimed(queue)
        clock.advance(61.0)
        with pytest.raises(LeaseLostError) as refused:
            await queue.heartbeat("i1", tenant=TENANT, worker="w1")
        assert refused.value.reason == "capped"

    async def test_losing_a_claim_is_not_worth_retrying(self) -> None:
        assert LeaseLostError("gone").retryable is False

    async def test_a_heartbeat_for_an_item_that_is_not_there_is_refused(self) -> None:
        with pytest.raises(WorkItemNotFoundError):
            await a_queue().heartbeat("absent", tenant=TENANT, worker="w1")


class TestFinishingWork:
    async def test_a_completed_item_is_not_handed_out_again(self) -> None:
        queue = a_queue()
        await _claimed(queue)
        await queue.complete("i1", tenant=TENANT, worker="w1")
        assert await queue.claim(worker="w2") is None

    async def test_completing_an_item_this_worker_lost_is_refused(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=5.0)
        await _claimed(queue)
        clock.advance(6.0)
        await queue.reap()
        with pytest.raises(LeaseLostError):
            await queue.complete("i1", tenant=TENANT, worker="w1")

    async def test_completing_an_item_nobody_claimed_is_refused(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item())
        with pytest.raises(LeaseLostError) as refused:
            await queue.complete("i1", tenant=TENANT, worker="w1")
        assert refused.value.reason == "expired"

    async def test_completing_an_absent_item_is_refused(self) -> None:
        with pytest.raises(WorkItemNotFoundError) as refused:
            await a_queue().complete("absent", tenant=TENANT, worker="w1")
        assert refused.value.item_id == "absent"


class TestWorkThatFailed:
    async def test_a_failure_comes_back_for_another_attempt(self) -> None:
        queue = a_queue()
        await _claimed(queue)
        failed = await queue.fail("i1", tenant=TENANT, worker="w1", error="boom")
        assert (failed.state, failed.attempts, failed.failures) == (
            WorkState.QUEUED,
            1,
            ("boom",),
        )

    async def test_the_retry_waits_out_its_backoff(self) -> None:
        queue = a_queue(backoff_seconds=4.0)
        await _claimed(queue)
        failed = await queue.fail("i1", tenant=TENANT, worker="w1", error="boom")
        assert failed.available_at == 4.0

    async def test_an_item_that_cannot_succeed_skips_its_remaining_attempts(self) -> None:
        queue = a_queue(max_attempts=5)
        await _claimed(queue)
        failed = await queue.fail(
            "i1", tenant=TENANT, worker="w1", error="malformed", retryable=False
        )
        assert failed.state is WorkState.DEAD_LETTERED

    async def test_an_item_that_runs_out_of_attempts_is_dead_lettered(self) -> None:
        queue = a_queue(max_attempts=2, backoff_seconds=0.0)
        await queue.enqueue(an_item())
        for _ in range(2):
            await queue.claim(worker="w1")
            await queue.fail("i1", tenant=TENANT, worker="w1", error="boom")
        dead = await queue.dead_letters(tenant=TENANT)
        assert (dead[0].state, dead[0].attempts) == (WorkState.DEAD_LETTERED, 2)

    async def test_a_dead_lettered_item_keeps_its_failures(self) -> None:
        """The item explains itself without anyone going looking through logs."""
        queue = a_queue(max_attempts=2, backoff_seconds=0.0)
        await queue.enqueue(an_item())
        for message in ("first", "second"):
            await queue.claim(worker="w1")
            await queue.fail("i1", tenant=TENANT, worker="w1", error=message)
        dead = await queue.dead_letters(tenant=TENANT)
        assert dead[0].failures == ("first", "second")

    async def test_a_dead_lettered_item_is_never_claimed_again(self) -> None:
        queue = a_queue(max_attempts=1)
        await _claimed(queue)
        await queue.fail("i1", tenant=TENANT, worker="w1", error="boom")
        assert await queue.claim(worker="w2") is None

    async def test_the_dead_letter_is_paged(self) -> None:
        queue = a_queue(max_attempts=1)
        for index in range(3):
            await _claimed(queue, item=an_item(f"i{index}"))
            await queue.fail(f"i{index}", tenant=TENANT, worker="w1", error="boom")
        assert len(await queue.dead_letters(tenant=TENANT, limit=2)) == 2

    async def test_one_tenant_cannot_read_another_s_dead_letter(self) -> None:
        queue = a_queue(max_attempts=1)
        await _claimed(queue, item=an_item("i1", tenant="one"))
        await queue.fail("i1", tenant="one", worker="w1", error="boom")
        assert await queue.dead_letters(tenant="two") == ()


class TestWorkersThatDie:
    async def test_a_lapsed_claim_is_requeued(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=10.0, backoff_seconds=0.0)
        await _claimed(queue)
        clock.advance(11.0)
        reaped = await queue.reap()
        assert [(item.id, item.state, item.attempts) for item in reaped] == [
            ("i1", WorkState.QUEUED, 1)
        ]

    async def test_the_requeued_item_says_why_it_came_back(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=10.0)
        await _claimed(queue)
        clock.advance(11.0)
        assert (await queue.reap())[0].failures == (LEASE_LAPSED,)

    async def test_a_reaped_item_is_claimable_by_someone_else(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=10.0, backoff_seconds=0.0)
        await _claimed(queue)
        clock.advance(11.0)
        await queue.reap()
        claimed = await queue.claim(worker="w2")
        assert claimed is not None
        assert claimed.worker == "w2"

    async def test_a_living_worker_s_claim_is_left_alone(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=30.0)
        await _claimed(queue)
        clock.advance(10.0)
        assert await queue.reap() == ()

    async def test_an_item_that_has_run_out_of_attempts_is_reaped_to_the_dead_letter(
        self,
    ) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=1.0, max_attempts=1)
        await _claimed(queue)
        clock.advance(2.0)
        assert (await queue.reap())[0].state is WorkState.DEAD_LETTERED

    async def test_reaping_counts_what_it_moved(self) -> None:
        clock = FakeClock()
        queue = a_queue(clock, lease_seconds=1.0)
        await _claimed(queue)
        clock.advance(2.0)
        await queue.reap()
        assert (await queue.stats()).reaped == 1


class TestAWorkerThatComesBack:
    async def test_a_restarted_worker_gives_back_what_it_held(self) -> None:
        queue = a_queue()
        await _claimed(queue)
        adopted = await queue.adopt(worker="w1")
        assert [(item.id, item.state) for item in adopted] == [("i1", WorkState.QUEUED)]

    async def test_the_returned_item_does_not_wait_out_a_backoff(self) -> None:
        """A rolled pod is not a poisonous item, and the work is due now."""
        queue = a_queue(backoff_seconds=60.0)
        await _claimed(queue)
        assert (await queue.adopt(worker="w1"))[0].available_at == 0.0

    async def test_the_attempt_still_counts(self) -> None:
        queue = a_queue()
        await _claimed(queue)
        adopted = await queue.adopt(worker="w1")
        assert (adopted[0].attempts, adopted[0].failures) == (1, (WORKER_RESTARTED,))

    async def test_another_worker_s_items_are_left_alone(self) -> None:
        queue = a_queue()
        await _claimed(queue)
        assert await queue.adopt(worker="w2") == ()

    async def test_a_worker_with_nothing_outstanding_adopts_nothing(self) -> None:
        assert await a_queue().adopt(worker="w1") == ()


class TestTheSameJobTwice:
    async def test_a_duplicate_is_collapsed_into_the_first(self) -> None:
        queue = a_queue()
        first = await queue.enqueue(an_item("i1", dedupe_key="nightly"))
        second = await queue.enqueue(an_item("i2", dedupe_key="nightly"))
        assert second.id == first.id

    async def test_the_duplicate_is_not_a_second_item(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("i1", dedupe_key="nightly"))
        await queue.enqueue(an_item("i2", dedupe_key="nightly"))
        assert (await queue.stats()).depth == 1

    async def test_suppressed_duplicates_are_counted(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("i1", dedupe_key="nightly"))
        await queue.enqueue(an_item("i2", dedupe_key="nightly"))
        assert (await queue.stats()).duplicates_suppressed == 1

    async def test_re_enqueueing_a_live_id_does_not_reset_its_attempts(self) -> None:
        """Otherwise a well-meaning retry hands a poisonous item an unlimited supply."""
        queue = a_queue(backoff_seconds=0.0)
        await _claimed(queue)
        await queue.fail("i1", tenant=TENANT, worker="w1", error="boom")
        assert (await queue.enqueue(an_item())).attempts == 1

    async def test_an_item_whose_id_has_finished_may_be_enqueued_again(self) -> None:
        queue = a_queue()
        await _claimed(queue)
        await queue.complete("i1", tenant=TENANT, worker="w1")
        assert (await queue.enqueue(an_item())).state is WorkState.QUEUED

    async def test_an_item_with_no_key_is_never_a_duplicate(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("i1"))
        await queue.enqueue(an_item("i2"))
        assert (await queue.stats()).depth == 2

    async def test_a_finished_job_may_be_enqueued_again(self) -> None:
        """Nightly means nightly, not once."""
        queue = a_queue()
        await _claimed(queue, item=an_item("i1", dedupe_key="nightly"))
        await queue.complete("i1", tenant=TENANT, worker="w1")
        second = await queue.enqueue(an_item("i2", dedupe_key="nightly"))
        assert second.id == "i2"

    async def test_a_dead_lettered_job_may_be_enqueued_again(self) -> None:
        queue = a_queue(max_attempts=1)
        await _claimed(queue, item=an_item("i1", dedupe_key="nightly"))
        await queue.fail("i1", tenant=TENANT, worker="w1", error="boom")
        second = await queue.enqueue(an_item("i2", dedupe_key="nightly"))
        assert second.id == "i2"

    async def test_one_tenant_s_key_does_not_collapse_another_s_job(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("i1", tenant="one", dedupe_key="nightly"))
        second = await queue.enqueue(an_item("i2", tenant="two", dedupe_key="nightly"))
        assert second.id == "i2"


class TestTenantIsolation:
    async def test_one_tenant_cannot_complete_another_s_item(self) -> None:
        queue = a_queue()
        await _claimed(queue, item=an_item("i1", tenant="one"))
        with pytest.raises(WorkItemNotFoundError):
            await queue.complete("i1", tenant="two", worker="w1")

    async def test_one_tenant_cannot_fail_another_s_item(self) -> None:
        queue = a_queue()
        await _claimed(queue, item=an_item("i1", tenant="one"))
        with pytest.raises(WorkItemNotFoundError):
            await queue.fail("i1", tenant="two", worker="w1", error="boom")


class TestWhatTheQueueLooksLike:
    async def test_depth_counts_what_is_waiting(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("i1"))
        await queue.enqueue(an_item("i2"))
        await queue.claim(worker="w1")
        stats = await queue.stats()
        assert (stats.depth, stats.claimed) == (1, 1)

    async def test_the_oldest_item_s_age_is_reported(self) -> None:
        """Depth alone says nothing: one item an hour old is the incident."""
        clock = FakeClock()
        queue = a_queue(clock)
        await queue.enqueue(an_item("i1"))
        clock.advance(90.0)
        assert (await queue.stats()).oldest_age_seconds == 90.0

    async def test_an_empty_queue_has_no_oldest_item(self) -> None:
        assert (await a_queue().stats()).oldest_age_seconds == 0.0

    async def test_dead_letters_are_counted(self) -> None:
        queue = a_queue(max_attempts=1)
        await _claimed(queue)
        await queue.fail("i1", tenant=TENANT, worker="w1", error="boom")
        assert (await queue.stats()).dead_lettered == 1

    async def test_another_queue_s_work_is_not_counted(self) -> None:
        queue = a_queue()
        await queue.enqueue(an_item("i1", queue="reports"))
        assert (await queue.stats(queue="default")).depth == 0

    async def test_a_queue_without_a_clock_has_no_opinion_about_time(self) -> None:
        queue = MemoryWorkQueue()
        stored = await queue.enqueue(an_item())
        assert stored.enqueued_at == 0.0


class TestLosingTheQueueItself:
    def test_an_unreachable_queue_is_worth_retrying(self) -> None:
        assert QueueUnavailableError("redis is down", queue="default").retryable is True

    def test_it_says_what_it_was_doing(self) -> None:
        refused = QueueUnavailableError("redis is down", queue="runs", operation="enqueue")
        assert (refused.queue, refused.operation) == ("runs", "enqueue")


class TestTheMemoryQueueConforms(WorkQueueConformance):
    def make_queue(self) -> WorkQueue:
        return MemoryWorkQueue(QueuePolicy(max_attempts=3), FakeClock())

    async def advance(self, queue: WorkQueue, seconds: float) -> None:
        assert isinstance(queue, MemoryWorkQueue)
        clock = queue._clock
        assert isinstance(clock, FakeClock)
        clock.advance(seconds)


async def _claimed(
    queue: MemoryWorkQueue,
    *,
    item: WorkItem | None = None,
    worker: str = "w1",
    lease_seconds: float | None = None,
) -> WorkItem:
    """Enqueue an item and take it, which most of these tests start from."""
    await queue.enqueue(item or an_item())
    claimed = await queue.claim(worker=worker, lease_seconds=lease_seconds)
    assert claimed is not None
    return claimed
