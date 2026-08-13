"""A run that stops for three days, and the one decision that carries it on.

The failures this file exists to prevent are a run that holds a worker open across a
weekend, an approval replayed into a second payment, and a decision honoured days after
whatever else the call depended on stopped being true.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTokenError,
    CheckpointBoundary,
    CheckpointPolicy,
    ConfigurationError,
    ModelCapabilities,
    PendingDecision,
    RunEventKind,
    RunState,
    SuspendedRun,
    ToolCall,
    Usage,
    legal_transitions,
    mint_token,
)
from tesserix_adk.core.suspension import DEFAULT_SUSPENSION_SECONDS, digest_of_token
from tesserix_adk.runtime import (
    AgentRunner,
    ApprovalDeferred,
    Checkpointer,
    DeferringGate,
    MemoryCheckpointStore,
    MemorySuspensionStore,
    ModelResponse,
)
from tesserix_adk.testing import FakeClock, ScriptedProvider
from tesserix_adk.tools import ToolRegistry, tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from tesserix_adk.core import ApprovalToken, Run

CAPABLE = ModelCapabilities(tool_calling=True, context_window_tokens=200_000)
NOW = 1_000.0
HELD_FOR = 62 * 3_600.0  # the working week the issue is written against
TENANT = "acme"
IBAN = "GB33BUKB20201555555555"


class Posting:
    """A transport that only puts the question somewhere; the answer arrives elsewhere."""

    def __init__(self) -> None:
        self.delivered: list[ApprovalRecord] = []

    async def deliver(self, record: ApprovalRecord) -> None:
        self.delivered.append(record)


class Answering:
    """A transport that carries the answer back in its own reply."""

    def __init__(self) -> None:
        self.delivered: list[ApprovalRecord] = []

    async def deliver(self, record: ApprovalRecord) -> ApprovalDecision:
        self.delivered.append(record)
        return ApprovalDecision(record_id=record.id, granted=True, decided_by="ada", decided_at=NOW)


class TestWhatAStoppedRunHoldsOnTo:
    """Nothing. That is the whole point of stopping rather than waiting."""

    async def test_a_run_awaiting_a_decision_comes_back_suspended(self) -> None:
        world = _World()

        run = await world.start()

        assert run.state is RunState.SUSPENDED
        assert not run.state.is_terminal

    async def test_it_leaves_no_task_running_behind_it(self) -> None:
        """A worker held across a weekend is a run that dies with the next deploy."""
        world = _World()
        before = len(asyncio.all_tasks())

        await world.start()

        assert len(asyncio.all_tasks()) == before

    async def test_the_tool_never_ran(self) -> None:
        world = _World()

        await world.start()

        assert world.called == []

    async def test_the_question_reached_whoever_decides(self) -> None:
        world = _World()

        await world.start()

        assert [held.tool_name for held in world.transport.delivered] == ["wire_funds"]

    async def test_the_trail_says_what_it_is_waiting_on(self) -> None:
        world = _World()

        run = await world.start()

        stopped = _first(run, RunEventKind.RUN_SUSPENDED)
        assert stopped.name == "wire_funds"
        assert "waiting on" in stopped.detail

    async def test_the_frontier_is_written_down_with_the_held_call_on_it(self) -> None:
        world = _World()

        await world.start()

        assert world.checkpoints is not None
        frontier = await world.checkpoints.latest("run_1", tenant=TENANT)
        assert frontier is not None
        assert frontier.boundary is CheckpointBoundary.BEFORE_APPROVAL
        assert [one.call.name for one in frontier.pending] == ["wire_funds"]
        assert not frontier.pending[0].dispatched

    async def test_the_suspension_records_what_it_stopped_under(self) -> None:
        world = _World()

        await world.start()

        held = await world.suspensions.get("run_1", tenant=TENANT)
        assert held is not None
        assert held.model == "scripted-1"
        assert held.suspended_at == NOW
        assert held.expires_at == NOW + DEFAULT_SUSPENSION_SECONDS

    async def test_it_appears_on_the_rota_of_whoever_decides(self) -> None:
        world = _World()

        await world.start()

        waiting = await world.gate.pending(tenant=TENANT)
        assert [one.record.tool_name for one in waiting] == ["wire_funds"]

    async def test_another_tenant_sees_none_of_it(self) -> None:
        world = _World()

        await world.start()

        assert await world.gate.pending(tenant="globex") == ()


class TestWhatThePersonDecidingIsShown:
    """Enough to decide with, and not the account number."""

    def test_a_pending_decision_carries_the_digest_rather_than_the_payload(self) -> None:
        held = _suspended()

        shown = PendingDecision.of(held)

        assert shown.arguments_digest == held.record.arguments_digest
        assert IBAN not in shown.model_dump_json()

    def test_it_names_what_is_being_asked_and_who_asked(self) -> None:
        shown = PendingDecision.of(_suspended())

        assert shown.tool_name == "wire_funds"
        assert shown.agent_name == "planner"
        assert shown.run_id == "run_1"

    def test_it_says_when_the_question_closes(self) -> None:
        shown = PendingDecision.of(_suspended(expires_at=NOW + 60.0))

        assert shown.expires_at == NOW + 60.0


class TestTheDecisionThatArrivesDaysLater:
    """The original run carries on, once, with everything it had."""

    async def test_an_approval_after_sixty_two_hours_runs_the_held_call(self) -> None:
        world = _World()
        await world.start()
        world.clock.set(NOW + HELD_FOR)

        run = await world.decide(granted=True)

        assert run.state is RunState.COMPLETED
        assert world.called == [{"amount": 500}]

    async def test_it_is_the_same_run_rather_than_a_second_one(self) -> None:
        world = _World()
        stopped = await world.start()
        world.clock.set(NOW + HELD_FOR)

        run = await world.decide(granted=True)

        assert run.id == stopped.id
        assert run.tenant == stopped.tenant
        assert run.user == "ada"

    async def test_it_carries_on_from_the_iteration_it_stopped_at(self) -> None:
        world = _World()
        await world.start()
        world.clock.set(NOW + HELD_FOR)

        run = await world.decide(granted=True)

        resumed = _first(run, RunEventKind.RUN_RESUMED)
        assert resumed.name == str(CheckpointBoundary.BEFORE_APPROVAL)
        assert "from iteration 1" in resumed.detail

    async def test_the_trail_says_how_long_nobody_answered_for(self) -> None:
        world = _World()
        await world.start()
        world.clock.set(NOW + HELD_FOR)

        run = await world.decide(granted=True)

        assert "suspended 223200s" in _first(run, RunEventKind.RUN_RESUMED).detail

    async def test_what_it_had_already_spent_comes_back_with_it(self) -> None:
        world = _World()
        await world.start()
        world.clock.set(NOW + HELD_FOR)

        run = await world.decide(granted=True)

        assert run.usage.input_tokens >= 2

    async def test_a_denial_days_later_refuses_the_call(self) -> None:
        world = _World()
        await world.start()
        world.clock.set(NOW + HELD_FOR)

        run = await world.decide(granted=False, reason="not this quarter")

        assert world.called == []
        assert _first(run, RunEventKind.APPROVAL_DENIED).detail == "not this quarter"

    async def test_a_run_that_stops_a_second_time_stops_the_same_way(self) -> None:
        """One decision buys one action, so the next held call is a fresh question."""
        world = _World()
        await world.start()
        world.clock.set(NOW + HELD_FOR)
        first = world.token

        run = await world.decide(granted=True, then=_calling("wire_funds", amount=90))

        assert run.state is RunState.SUSPENDED
        assert world.called == [{"amount": 500}]
        assert world.token != first

    async def test_the_run_is_no_longer_on_anybody_s_rota(self) -> None:
        world = _World()
        await world.start()

        await world.decide(granted=True)

        assert await world.gate.pending(tenant=TENANT) == ()


class TestATokenPresentedTwice:
    """One decision, one execution. A second presentation buys nothing."""

    async def test_the_second_presentation_is_refused(self) -> None:
        world = _World()
        await world.start()
        await world.decide(granted=True)

        with pytest.raises(ApprovalTokenError, match="already taken"):
            await world.decide(granted=True)

    async def test_the_call_does_not_run_a_second_time(self) -> None:
        world = _World()
        await world.start()
        await world.decide(granted=True)

        with pytest.raises(ApprovalTokenError):
            await world.decide(granted=True)

        assert world.called == [{"amount": 500}]

    async def test_the_attempt_is_recorded_against_whoever_made_it(self) -> None:
        world = _World()
        await world.start()
        await world.decide(granted=True)

        with pytest.raises(ApprovalTokenError):
            await world.decide(granted=True, decided_by="mallory")

        refused = world.suspensions.attempts[-1]
        assert refused.presented_by == "mallory"
        assert not refused.accepted
        assert refused.reason == "the decision it names was already taken"

    async def test_a_token_presented_as_another_tenant_resolves_to_nothing(self) -> None:
        world = _World()
        await world.start()

        with pytest.raises(ApprovalTokenError, match="no suspended run"):
            await world.decide(granted=True, tenant="globex")

    async def test_a_token_nobody_issued_resolves_to_nothing(self) -> None:
        world = _World()
        await world.start()

        with pytest.raises(ApprovalTokenError, match="no suspended run"):
            await world.gate.redeem("not-a-token", tenant=TENANT, presented_by="mallory")

    async def test_an_accepted_presentation_is_recorded_too(self) -> None:
        world = _World()
        await world.start()

        await world.decide(granted=True)

        accepted = world.suspensions.attempts[-1]
        assert accepted.accepted
        assert accepted.presented_by == "ada"
        assert accepted.at == NOW


class TestWhatChangedWhileNobodyWasLooking:
    """A person saying yes is one of the conditions, not all of them."""

    async def test_a_token_answered_past_its_expiry_is_a_denial(self) -> None:
        world = _World()
        await world.start()
        world.clock.set(NOW + DEFAULT_SUSPENSION_SECONDS)

        run = await world.decide(granted=True)

        assert world.called == []
        assert "expired" in _first(run, RunEventKind.APPROVAL_DENIED).detail

    async def test_the_expiry_denial_is_not_attributed_to_the_person(self) -> None:
        """Nobody decided this in time, and the trail must not read as though they did."""
        world = _World()
        await world.start()
        world.clock.set(NOW + DEFAULT_SUSPENSION_SECONDS)

        run = await world.decide(granted=True)

        assert "ada" not in _first(run, RunEventKind.APPROVAL_DENIED).detail

    async def test_a_model_that_moved_under_the_run_stops_the_resume(self) -> None:
        world = _World()
        await world.start()

        with pytest.raises(ConfigurationError, match="scripted-1"):
            await world.decide(granted=True, model="scripted-2")

    async def test_a_caller_may_say_the_drift_is_acceptable(self) -> None:
        world = _World()
        await world.start()

        run = await world.decide(granted=True, model="scripted-2", drift=True)

        assert run.state is RunState.COMPLETED

    async def test_a_payload_the_tool_no_longer_accepts_is_refused(self) -> None:
        """A schema that moved during the wait is a payload nobody approved the shape of."""
        world = _World()
        await world.start()
        world.clock.set(NOW + HELD_FOR)

        run = await world.decide(granted=True, narrowed=True)

        assert world.called == []
        assert _first(run, RunEventKind.SCHEMA_VIOLATION).name == "wire_funds"


class TestAGateThatCannotDefer:
    """Deferring without somewhere to write it down is a promise nobody can keep."""

    async def test_a_transport_that_answers_inline_never_defers(self) -> None:
        world = _World(transport=Answering())

        run = await world.start()

        assert run.state is RunState.COMPLETED
        assert world.called == [{"amount": 500}]

    async def test_a_runner_with_no_checkpointer_refuses_to_defer(self) -> None:
        world = _World(checkpointing=False)

        with pytest.raises(ConfigurationError, match="no checkpointer"):
            await world.start()

    async def test_a_policy_that_skips_the_approval_boundary_refuses_to_defer(self) -> None:
        world = _World(
            policy=CheckpointPolicy(boundaries=frozenset({CheckpointBoundary.AFTER_MODEL_CALL}))
        )

        with pytest.raises(ConfigurationError, match="does not write at"):
            await world.start()

    async def test_a_gate_that_cannot_redeem_cannot_be_resumed_from(self) -> None:
        world = _World()
        await world.start()

        with pytest.raises(ConfigurationError, match="cannot redeem tokens"):
            await world.decide(granted=True, gate=_NoRedeeming())


class TestTheTokenItself:
    """Single-use, tenant-bound, expiring, and bound to what the approver was shown."""

    def test_it_is_minted_over_the_payload_the_approver_saw(self) -> None:
        record = _record()

        token = mint_token(record, issued_at=NOW)

        assert token.arguments_digest == record.arguments_digest
        assert token.record_id == record.id
        assert token.tenant == TENANT

    def test_it_expires_where_the_ttl_says(self) -> None:
        token = mint_token(_record(), issued_at=NOW, ttl_seconds=60.0)

        assert not token.expired_by(NOW + 59.0)
        assert token.expired_by(NOW + 60.0)

    def test_two_tokens_are_never_the_same_secret(self) -> None:
        assert mint_token(_record()).value != mint_token(_record()).value

    def test_a_store_keeps_the_digest_rather_than_the_secret(self) -> None:
        token = mint_token(_record())

        assert token.digest == digest_of_token(token.value)
        assert token.value not in token.digest


class TestTheSuspensionItself:
    def test_it_knows_how_long_it_has_been_stopped(self) -> None:
        assert _suspended().held_for(NOW + HELD_FOR) == HELD_FOR

    def test_a_clock_that_went_backwards_is_no_time_at_all(self) -> None:
        assert _suspended().held_for(NOW - 10.0) == 0.0

    def test_it_knows_when_nobody_answered_in_time(self) -> None:
        held = _suspended(expires_at=NOW + 60.0)

        assert not held.expired_by(NOW + 59.0)
        assert held.expired_by(NOW + 60.0)


class TestTheStoreUnderneath:
    async def test_it_finds_a_suspension_by_token_within_the_tenant(self) -> None:
        store = MemorySuspensionStore()
        held = _suspended()
        await store.put(held)

        assert await store.by_token(held.token_digest, tenant=TENANT) == held
        assert await store.by_token(held.token_digest, tenant="globex") is None

    async def test_a_token_no_suspension_carries_resolves_to_nothing(self) -> None:
        store = MemorySuspensionStore()
        await store.put(_suspended())

        assert await store.by_token(digest_of_token("other"), tenant=TENANT) is None

    async def test_it_lets_one_caller_take_the_decision(self) -> None:
        store = MemorySuspensionStore()
        await store.put(_suspended())

        assert await store.spend("run_1", tenant=TENANT) is True
        assert await store.spend("run_1", tenant=TENANT) is False

    async def test_a_run_nobody_stopped_cannot_be_spent(self) -> None:
        assert await MemorySuspensionStore().spend("run_9", tenant=TENANT) is False

    async def test_a_spent_suspension_is_off_the_rota(self) -> None:
        store = MemorySuspensionStore()
        await store.put(_suspended())
        await store.spend("run_1", tenant=TENANT)

        assert await store.pending(tenant=TENANT) == ()

    async def test_the_rota_is_oldest_first(self) -> None:
        store = MemorySuspensionStore()
        await store.put(_suspended(run_id="run_2", suspended_at=NOW + 10.0))
        await store.put(_suspended(run_id="run_1", suspended_at=NOW))

        assert [one.run_id for one in await store.pending(tenant=TENANT)] == ["run_1", "run_2"]

    async def test_a_forgotten_suspension_is_gone(self) -> None:
        store = MemorySuspensionStore()
        await store.put(_suspended())
        await store.forget("run_1", tenant=TENANT)

        assert await store.get("run_1", tenant=TENANT) is None

    async def test_forgetting_a_run_nobody_stopped_is_not_an_error(self) -> None:
        await MemorySuspensionStore().forget("run_9", tenant=TENANT)

    async def test_a_run_that_never_stopped_has_no_suspension(self) -> None:
        assert await MemorySuspensionStore().get("run_9", tenant=TENANT) is None


class TestWhatARunMayDoNext:
    def test_a_running_run_may_stop_on_a_question(self) -> None:
        assert RunState.SUSPENDED in legal_transitions(RunState.RUNNING)

    def test_a_stopped_run_may_go_again(self) -> None:
        assert RunState.RUNNING in legal_transitions(RunState.SUSPENDED)

    def test_a_stopped_run_is_not_over(self) -> None:
        assert not RunState.SUSPENDED.is_terminal

    def test_a_stopped_run_may_not_complete_without_going_again(self) -> None:
        assert RunState.COMPLETED not in legal_transitions(RunState.SUSPENDED)


class TestTheGateThatDefers:
    def test_the_signal_carries_the_token_and_where_the_run_is_waiting(self) -> None:
        store = MemorySuspensionStore()
        token = mint_token(_record())

        deferred = ApprovalDeferred(token, store)

        assert deferred.token is token
        assert deferred.store is store

    async def test_it_hands_out_a_token_bound_to_the_question(self) -> None:
        gate = DeferringGate(Posting(), MemorySuspensionStore(), clock=FakeClock(start=NOW))
        record = _record()

        with pytest.raises(ApprovalDeferred) as deferred:
            await gate.request(record)

        assert deferred.value.token.record_id == record.id
        assert deferred.value.token.tenant == TENANT

    async def test_a_gate_with_no_clock_stamps_the_token_with_wall_time(self) -> None:
        gate = DeferringGate(Posting(), MemorySuspensionStore())

        with pytest.raises(ApprovalDeferred) as deferred:
            await gate.request(_record())

        assert deferred.value.token.issued_at > NOW

    async def test_a_transport_that_answers_inline_hands_nobody_a_token(self) -> None:
        issued: list[ApprovalToken] = []
        gate = DeferringGate(Answering(), MemorySuspensionStore(), hand_to=_collecting(issued))

        decision = await gate.request(_record())

        assert decision.granted
        assert issued == []

    async def test_it_says_where_its_stopped_runs_wait(self) -> None:
        store = MemorySuspensionStore()

        assert DeferringGate(Posting(), store).store is store


class _NoRedeeming:
    """A gate with nowhere for a decision taken out of band to land."""

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        raise AssertionError(f"no test gets as far as asking about {record.tool_name}")


class _World:
    """One agent that moves money, one gate that defers, and a clock somebody else winds."""

    def __init__(
        self,
        *,
        transport: Any = None,
        checkpointing: bool = True,
        policy: CheckpointPolicy | None = None,
    ) -> None:
        self.called: list[dict[str, Any]] = []
        self.token = ""
        self.transport = transport if transport is not None else Posting()
        self.clock = FakeClock(start=NOW, auto_advance=False)
        self.suspensions = MemorySuspensionStore()
        self.gate = DeferringGate(
            self.transport, self.suspensions, hand_to=self._issued, clock=self.clock
        )
        self.checkpoints = (
            Checkpointer(MemoryCheckpointStore(), policy, self.clock) if checkpointing else None
        )
        self.agent: Agent[Any] = Agent(
            name="planner",
            instructions="Settle the invoice.",
            free_text=True,
            model="scripted-1",
            tools=("wire_funds",),
            approval_required_tools=("wire_funds",),
        )

    async def _issued(self, token: ApprovalToken) -> None:
        self.token = token.value

    async def start(self) -> Run[Any]:
        """Drive the agent up to the point somebody has to decide."""
        registry, release = self._tools(narrowed=False)
        try:
            runner = self._runner(registry, _calling("wire_funds", amount=500), _answering())
            return await runner.run(
                self.agent, "settle it", tenant=TENANT, run_id="run_1", user="ada"
            )
        finally:
            release()

    async def decide(
        self,
        *,
        granted: bool,
        decided_by: str = "ada",
        reason: str = "",
        tenant: str = TENANT,
        model: str | None = None,
        drift: bool = False,
        narrowed: bool = False,
        gate: Any = None,
        then: ModelResponse | None = None,
    ) -> Run[Any]:
        """Answer the question, days later, as whoever is answering it."""
        agent = self.agent if model is None else self.agent.model_copy(update={"model": model})
        registry, release = self._tools(narrowed=narrowed)
        next_up = then if then is not None else _answering()
        try:
            return await self._runner(registry, next_up, gate=gate).resume_with_decision(
                agent,
                "run_1",
                tenant=tenant,
                token=self.token,
                granted=granted,
                decided_by=decided_by,
                reason=reason,
                user="ada",
                allow_model_drift=drift,
            )
        finally:
            release()

    def _tools(self, *, narrowed: bool) -> tuple[ToolRegistry, Callable[[], None]]:
        """The money-moving tool, optionally under a schema that moved while nobody looked."""
        called = self.called

        if narrowed:

            @tool(name="wire_funds", requires_approval=True)
            async def moving(amount: int, mandate: str) -> str:
                """Move money against a standing mandate.

                Args:
                    amount: How much.
                    mandate: Which standing instruction authorises it.
                """
                called.append({"amount": amount, "mandate": mandate})
                return f"sent {amount}"

        else:

            @tool(name="wire_funds", requires_approval=True)
            async def moving(amount: int) -> str:
                """Move money.

                Args:
                    amount: How much.
                """
                called.append({"amount": amount})
                return f"sent {amount}"

        return ToolRegistry((moving,), clock=self.clock), moving.release

    def _runner(
        self, registry: ToolRegistry, *responses: ModelResponse, gate: Any = None
    ) -> AgentRunner:
        return AgentRunner(
            provider=ScriptedProvider(*responses, capabilities=CAPABLE),
            clock=self.clock,
            tools=registry.view(allow=("wire_funds",), agent="planner"),
            approvals=gate if gate is not None else self.gate,
            checkpoints=self.checkpoints,
        )


def _collecting(issued: list[ApprovalToken]) -> Callable[[ApprovalToken], Any]:
    """A hand-off that keeps the tokens instead of sending them anywhere."""

    async def hand_to(token: ApprovalToken) -> None:
        issued.append(token)

    return hand_to


def _record() -> ApprovalRecord:
    """One held call, as it goes to whoever decides."""
    return ApprovalRecord.for_call(
        run_id="run_1",
        tenant=TENANT,
        agent_name="planner",
        tool_name="wire_funds",
        arguments={"amount": 500, "iban": IBAN},
        reason="wire_funds is declared to require approval",
        requested_at=NOW,
    )


def _suspended(
    *, run_id: str = "run_1", suspended_at: float = NOW, expires_at: float = NOW + 60.0
) -> SuspendedRun:
    record = _record()
    return SuspendedRun(
        run_id=run_id,
        tenant=TENANT,
        agent_name="planner",
        record=record,
        call=ToolCall(id="c1", name="wire_funds", arguments={"amount": 500}),
        token_digest=mint_token(record).digest,
        suspended_at=suspended_at,
        expires_at=expires_at,
        model="scripted-1",
    )


def _first(run: Run[Any], kind: RunEventKind) -> Any:
    return next(event for event in run.events if event.kind is kind)


def _calling(name: str, **arguments: object) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(id=f"call_{name}", name=name, arguments=arguments),),
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _answering() -> ModelResponse:
    return ModelResponse(content="Done.", usage=Usage(input_tokens=1, output_tokens=1))
