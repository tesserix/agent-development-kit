"""Policy attached to the loop rather than remembered at a call site.

The failure this file exists to prevent is an agent that is safe in one product and unsafe
in another because the check lived in application code and the next caller did not write
it. Every hook point is enforced by the runner, on every path out of a run.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core import (
    Agent,
    ApprovalDecision,
    ApprovalRecord,
    BudgetLimits,
    ConfigurationError,
    DeadlineConfig,
    Hook,
    HookAction,
    HookChain,
    HookDecision,
    HookPoint,
    HookRegistrationError,
    HookSubject,
    Run,
    RunEventKind,
    RunState,
    ToolCall,
    Usage,
    resolve_hooks,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeBudgetPolicy, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from collections.abc import Sequence


class Recorder:
    """A hook that decides nothing and records every subject it was handed."""

    def __init__(self, name: str = "recorder", points: tuple[HookPoint, ...] | None = None) -> None:
        self._name = name
        self._points = points if points is not None else tuple(HookPoint)
        self.seen: list[HookSubject] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def points(self) -> tuple[HookPoint, ...]:
        return self._points

    async def on(self, subject: HookSubject) -> HookDecision:
        self.seen.append(subject)
        return HookDecision.proceed()


class Deciding:
    """A hook that always returns the decision it was built with."""

    def __init__(self, name: str, point: HookPoint, decision: HookDecision) -> None:
        self._name = name
        self._points = (point,)
        self._decision = decision
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def points(self) -> tuple[HookPoint, ...]:
        return self._points

    async def on(self, subject: HookSubject) -> HookDecision:  # noqa: ARG002 — answers the same way whatever it is asked
        self.calls += 1
        return self._decision


class Appending:
    """A rewriting hook, so chained rewrites are visible in order."""

    def __init__(self, name: str, point: HookPoint, suffix: str) -> None:
        self._name = name
        self._points = (point,)
        self._suffix = suffix

    @property
    def name(self) -> str:
        return self._name

    @property
    def points(self) -> tuple[HookPoint, ...]:
        return self._points

    async def on(self, subject: HookSubject) -> HookDecision:
        return HookDecision.rewrite(subject.content + self._suffix, reason="appended")


class Exploding:
    """A hook that cannot be evaluated. Fail closed: the run stops rather than skipping it."""

    def __init__(self, point: HookPoint = HookPoint.BEFORE_MODEL_CALL) -> None:
        self._points = (point,)

    @property
    def name(self) -> str:
        return "exploding"

    @property
    def points(self) -> tuple[HookPoint, ...]:
        return self._points

    async def on(self, subject: HookSubject) -> HookDecision:  # noqa: ARG002 — answers the same way whatever it is asked
        raise RuntimeError("policy service unreachable")


class SelfRegistering:
    """A hook that tries to add another hook to the chain it is running in."""

    def __init__(self, chain: HookChain) -> None:
        self._chain = chain
        self.rejected: Exception | None = None

    @property
    def name(self) -> str:
        return "self_registering"

    @property
    def points(self) -> tuple[HookPoint, ...]:
        return (HookPoint.BEFORE_MODEL_CALL,)

    async def on(self, subject: HookSubject) -> HookDecision:  # noqa: ARG002 — answers the same way whatever it is asked
        try:
            self._chain.register(Recorder("smuggled"))
        except HookRegistrationError as rejected:
            self.rejected = rejected
        return HookDecision.proceed()


class Gate:
    """An approval gate that answers with whatever it was built with."""

    def __init__(
        self,
        decision: ApprovalDecision | None = None,
        *,
        granted: bool = True,
        decided_at: float | None = None,
    ) -> None:
        self._decision = decision
        self._granted = granted
        self._decided_at = decided_at
        self.requested: list[ApprovalRecord] = []

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        self.requested.append(record)
        if self._decision is not None:
            return self._decision
        return ApprovalDecision(
            record_id=record.id,
            granted=self._granted,
            decided_by="ada",
            decided_at=record.requested_at if self._decided_at is None else self._decided_at,
        )


def agent(**overrides: object) -> Agent:
    fields: dict[str, object] = {
        "name": "planner",
        "instructions": "Plan trips.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answering(content: str = "Kyoto.") -> ModelResponse:
    return ModelResponse(content=content, usage=Usage(input_tokens=10, output_tokens=5))


def calling(tool: str = "search", **arguments: Any) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name=tool, arguments=arguments),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def registry() -> FakeToolRegistry:
    return FakeToolRegistry({"search": lambda **_: "a result", "wire_funds": lambda **_: "sent"})


HOOK_REFUSAL = RunEventKind.HOOK_REFUSAL


def kinds(run: Run) -> list[RunEventKind]:
    return [event.kind for event in run.events]


def details(run: Run, kind: RunEventKind) -> list[str]:
    return [event.detail or "" for event in run.events if event.kind is kind]


class TestTheHookPoints:
    def test_every_documented_point_exists(self) -> None:
        """A point the loop does not offer is a check a consumer writes at a call site."""
        assert set(HookPoint) == {
            HookPoint.BEFORE_PROMPT_ASSEMBLY,
            HookPoint.BEFORE_MODEL_CALL,
            HookPoint.AFTER_MODEL_RESPONSE,
            HookPoint.BEFORE_TOOL_DISPATCH,
            HookPoint.AFTER_TOOL_RESULT,
            HookPoint.BEFORE_OUTPUT_VALIDATION,
            HookPoint.ON_TERMINAL,
        }

    async def test_a_hook_fires_at_each_point_of_a_run_that_calls_a_tool(self) -> None:
        recorder = Recorder()
        provider = ScriptedProvider(calling(), answering())
        runner = AgentRunner(
            provider=provider, tools=registry(), hooks=HookChain([recorder]), approvals=Gate()
        )

        await runner.run(agent(tools=("search",)), "Trains to Kyoto?", tenant="acme")

        fired = [subject.point for subject in recorder.seen]
        assert fired[0] is HookPoint.BEFORE_PROMPT_ASSEMBLY
        assert fired[-1] is HookPoint.ON_TERMINAL
        assert {
            HookPoint.BEFORE_MODEL_CALL,
            HookPoint.AFTER_MODEL_RESPONSE,
            HookPoint.BEFORE_TOOL_DISPATCH,
            HookPoint.AFTER_TOOL_RESULT,
        } <= set(fired)

    async def test_the_terminal_hook_fires_on_a_failed_run_too(self) -> None:
        """A policy that only sees successful runs cannot report what a fleet actually does."""
        recorder = Recorder(points=(HookPoint.ON_TERMINAL,))
        provider = ScriptedProvider(ModelResponse(usage=Usage(input_tokens=1, output_tokens=0)))
        runner = AgentRunner(provider=provider, hooks=HookChain([recorder]))

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert run.state is RunState.FAILED
        assert [subject.state for subject in recorder.seen] == [RunState.FAILED]

    async def test_a_hook_is_only_called_at_the_points_it_declares(self) -> None:
        recorder = Recorder(points=(HookPoint.BEFORE_MODEL_CALL,))
        runner = AgentRunner(provider=ScriptedProvider(answering()), hooks=HookChain([recorder]))

        await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert [subject.point for subject in recorder.seen] == [HookPoint.BEFORE_MODEL_CALL]

    async def test_a_subject_carries_the_run_without_carrying_the_run_object(self) -> None:
        """A hook gets facts, not the run: it cannot widen a scope it cannot reach."""
        recorder = Recorder(points=(HookPoint.BEFORE_MODEL_CALL,))
        runner = AgentRunner(provider=ScriptedProvider(answering()), hooks=HookChain([recorder]))

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme", user="ada")

        subject = recorder.seen[0]
        assert (subject.run_id, subject.tenant, subject.user) == (run.id, "acme", "ada")
        assert subject.agent_name == "planner"
        assert not hasattr(subject, "run")


class TestDecisions:
    def test_a_rewrite_must_carry_its_replacement(self) -> None:
        with pytest.raises(ValueError, match="replacement"):
            HookDecision(action=HookAction.REWRITE, reason="tidied")

    def test_a_refusal_must_say_why(self) -> None:
        """A refusal with no reason is a run that stopped and nobody can say what for."""
        with pytest.raises(ValueError, match="reason"):
            HookDecision(action=HookAction.REFUSE)

    def test_only_a_rewrite_carries_a_replacement(self) -> None:
        with pytest.raises(ValueError, match="replacement"):
            HookDecision(action=HookAction.REFUSE, reason="no", replacement="yes")

    def test_the_constructors_build_what_they_name(self) -> None:
        assert HookDecision.proceed().action is HookAction.CONTINUE
        assert HookDecision.rewrite("x").replacement == "x"
        assert HookDecision.refuse("no").action is HookAction.REFUSE
        assert HookDecision.require_approval("ask").action is HookAction.REQUIRE_APPROVAL

    @pytest.mark.parametrize(
        ("looser", "tighter"),
        [
            (HookDecision.proceed(), HookDecision.rewrite("x")),
            (HookDecision.rewrite("x"), HookDecision.require_approval("ask")),
            (HookDecision.require_approval("ask"), HookDecision.refuse("no")),
        ],
    )
    def test_restrictiveness_is_ordered(self, looser: HookDecision, tighter: HookDecision) -> None:
        assert tighter.restrictiveness > looser.restrictiveness


class TestResolvingConflicts:
    def test_the_most_restrictive_decision_wins(self) -> None:
        """Two hooks disagreeing is not a tie to break by luck: the tighter answer holds."""
        decisions = [HookDecision.proceed(), HookDecision.refuse("secrets"), HookDecision.proceed()]
        assert resolve_hooks(decisions) == HookDecision.refuse("secrets")

    def test_a_refusal_beats_an_approval_request(self) -> None:
        decisions = [HookDecision.require_approval("ask"), HookDecision.refuse("no")]
        assert resolve_hooks(decisions).action is HookAction.REFUSE

    def test_ties_break_toward_the_first_declared(self) -> None:
        """Deterministic, and the same on every process: order of declaration decides."""
        decisions = [HookDecision.refuse("first"), HookDecision.refuse("second")]
        assert resolve_hooks(decisions).reason == "first"

    def test_nothing_to_resolve_is_permission_to_proceed(self) -> None:
        assert resolve_hooks([]) == HookDecision.proceed()

    async def test_every_hook_at_a_point_runs_even_after_one_refuses(self) -> None:
        """Stopping at the first refusal hides the second, so a second run looks different."""
        refusing = Deciding("refusing", HookPoint.BEFORE_MODEL_CALL, HookDecision.refuse("no"))
        later = Deciding("later", HookPoint.BEFORE_MODEL_CALL, HookDecision.proceed())
        runner = AgentRunner(
            provider=ScriptedProvider(answering()), hooks=HookChain([refusing, later])
        )

        await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert later.calls == 1


class TestTheChain:
    def test_hooks_keep_the_order_they_were_declared_in(self) -> None:
        first, second = Recorder("first"), Recorder("second")
        chain = HookChain([first, second])
        assert [hook.name for hook in chain.at(HookPoint.BEFORE_MODEL_CALL)] == ["first", "second"]
        assert len(chain) == 2

    def test_two_hooks_cannot_share_a_name(self) -> None:
        """A name is how a decision is attributed; two owners of one name attribute nothing."""
        with pytest.raises(HookRegistrationError, match="twice"):
            HookChain([Recorder("audit"), Recorder("audit")])

    def test_a_hook_declaring_no_points_is_refused(self) -> None:
        with pytest.raises(HookRegistrationError, match="no hook points"):
            HookChain([Recorder("idle", points=())])

    def test_a_hook_missing_a_protocol_member_is_refused_at_registration(self) -> None:
        class Partial:
            name = "partial"

        with pytest.raises(HookRegistrationError, match="points"):
            HookChain([Partial()])  # type: ignore[list-item]

    def test_a_sealed_chain_refuses_registration(self) -> None:
        chain = HookChain([Recorder("audit")]).sealed()
        with pytest.raises(HookRegistrationError, match="sealed"):
            chain.register(Recorder("late"))

    def test_registering_before_the_run_is_allowed(self) -> None:
        chain = HookChain()
        chain.register(Recorder("audit"))
        assert chain.names == ("audit",)

    async def test_a_hook_cannot_register_another_hook_mid_run(self) -> None:
        """The chain a run started with is the chain it is judged by, start to finish."""
        chain = HookChain()
        smuggler = SelfRegistering(chain)
        chain.register(smuggler)
        runner = AgentRunner(provider=ScriptedProvider(answering()), hooks=chain)

        await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert isinstance(smuggler.rejected, HookRegistrationError)
        assert chain.names == ("self_registering",)


class TestFailClosed:
    async def test_a_hook_that_raises_stops_the_run(self) -> None:
        """A check that could not be evaluated is not a check that passed."""
        runner = AgentRunner(provider=ScriptedProvider(answering()), hooks=HookChain([Exploding()]))

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert run.state is RunState.FAILED
        assert "HookEvaluationError" in (run.events[-1].detail or "")

    async def test_a_hook_that_raises_stops_the_run_before_the_model_is_called(self) -> None:
        provider = ScriptedProvider(answering())
        runner = AgentRunner(provider=provider, hooks=HookChain([Exploding()]))

        await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert provider.requests == []

    async def test_a_hook_that_outlives_its_ceiling_stops_the_run(self) -> None:
        class Stalling(Recorder):
            async def on(self, subject: HookSubject) -> HookDecision:  # noqa: ARG002 — answers the same way whatever it is asked
                await asyncio.sleep(3600)
                return HookDecision.proceed()

        runner = AgentRunner(
            provider=ScriptedProvider(answering()),
            hooks=HookChain([Stalling("stalling", (HookPoint.BEFORE_MODEL_CALL,))]),
            deadlines=DeadlineConfig(hook_seconds=0.01),
        )

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert run.state is RunState.CANCELLED


class TestRewrites:
    async def test_a_rewritten_input_changes_what_is_asked(self) -> None:
        provider = ScriptedProvider(answering())
        rewriting = Deciding(
            "redactor",
            HookPoint.BEFORE_PROMPT_ASSEMBLY,
            HookDecision.rewrite("Trains to [REDACTED]?", reason="pii"),
        )
        runner = AgentRunner(provider=provider, hooks=HookChain([rewriting]))

        await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert "[REDACTED]" in str(provider.requests[0].messages)
        assert "Kyoto" not in str(provider.requests[0].messages)

    async def test_a_rewrite_is_recorded_so_the_prompt_can_be_reproduced(self) -> None:
        """Digests, not content: a replay recomputes them and knows it assembled the same
        prompt, without the log holding what was redacted."""
        rewriting = Deciding(
            "redactor",
            HookPoint.BEFORE_PROMPT_ASSEMBLY,
            HookDecision.rewrite("Trains to [REDACTED]?", reason="pii"),
        )
        runner = AgentRunner(provider=ScriptedProvider(answering()), hooks=HookChain([rewriting]))

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        recorded = details(run, RunEventKind.HOOK_REWRITE)
        assert len(recorded) == 1
        assert "Kyoto" not in recorded[0]
        assert "[REDACTED]" not in recorded[0]
        assert "→" in recorded[0]

    async def test_an_outgoing_message_can_be_redacted_on_its_way_out(self) -> None:
        """The last thing before the wire is the last chance to keep something off it."""
        provider = ScriptedProvider(answering())
        rewriting = Deciding(
            "redactor",
            HookPoint.BEFORE_MODEL_CALL,
            HookDecision.rewrite("Trains to [REDACTED]?", reason="pii"),
        )
        runner = AgentRunner(provider=provider, hooks=HookChain([rewriting]))

        await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert "[REDACTED]" in str(provider.requests[0].messages)
        assert "Kyoto" not in str(provider.requests[0].messages)

    async def test_a_response_can_be_rewritten_before_anything_reads_it(self) -> None:
        rewriting = Deciding(
            "redactor",
            HookPoint.AFTER_MODEL_RESPONSE,
            HookDecision.rewrite("[REDACTED]", reason="pii"),
        )
        runner = AgentRunner(provider=ScriptedProvider(answering()), hooks=HookChain([rewriting]))

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert "[REDACTED]" in str(run.messages[-1].content)

    async def test_a_tool_result_can_be_redacted_before_the_model_reads_it(self) -> None:
        """What comes back from a tool enters the context; policy sees it before it does."""
        provider = ScriptedProvider(calling(), answering())
        rewriting = Deciding(
            "redactor",
            HookPoint.AFTER_TOOL_RESULT,
            HookDecision.rewrite("[REDACTED]", reason="pii"),
        )
        runner = AgentRunner(provider=provider, tools=registry(), hooks=HookChain([rewriting]))

        await runner.run(agent(tools=("search",)), "Trains?", tenant="acme")

        assert "[REDACTED]" in str(provider.requests[1].messages[-1].content)
        assert "a result" not in str(provider.requests[1].messages[-1].content)

    async def test_rewrites_chain_in_declaration_order(self) -> None:
        provider = ScriptedProvider(answering())
        chain = HookChain(
            [
                Appending("first", HookPoint.BEFORE_PROMPT_ASSEMBLY, " one"),
                Appending("second", HookPoint.BEFORE_PROMPT_ASSEMBLY, " two"),
            ]
        )
        runner = AgentRunner(provider=provider, hooks=chain)

        await runner.run(agent(), "ask", tenant="acme")

        assert "ask one two" in str(provider.requests[0].messages)


class TestRefusals:
    async def test_a_refusal_before_the_model_call_ends_the_run(self) -> None:
        refusing = Deciding(
            "policy", HookPoint.BEFORE_MODEL_CALL, HookDecision.refuse("model not approved")
        )
        provider = ScriptedProvider(answering())
        runner = AgentRunner(provider=provider, hooks=HookChain([refusing]))

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert run.state is RunState.FAILED
        assert provider.requests == []
        assert details(run, RunEventKind.HOOK_REFUSAL) == ["model not approved"]

    async def test_a_refusal_before_tool_dispatch_dispatches_nothing(self) -> None:
        refusing = Deciding(
            "policy", HookPoint.BEFORE_TOOL_DISPATCH, HookDecision.refuse("tool not approved")
        )
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(calling(), answering()),
            tools=tools,
            hooks=HookChain([refusing]),
        )

        run = await runner.run(agent(tools=("search",)), "Trains?", tenant="acme")

        assert run.state is RunState.FAILED
        assert tools.calls == []

    async def test_a_refusal_before_output_validation_ends_the_run(self) -> None:
        refusing = Deciding(
            "policy", HookPoint.BEFORE_OUTPUT_VALIDATION, HookDecision.refuse("unsafe answer")
        )
        runner = AgentRunner(provider=ScriptedProvider(answering()), hooks=HookChain([refusing]))

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert run.state is RunState.FAILED
        assert RunEventKind.HOOK_REFUSAL in kinds(run)


class TestApprovalGates:
    async def test_a_tool_declaring_approval_waits_for_a_decision(self) -> None:
        gate = Gate(granted=True)
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(calling("wire_funds", amount=500), answering()),
            tools=tools,
            approvals=gate,
        )

        run = await runner.run(
            agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
            "Pay the deposit.",
            tenant="acme",
        )

        assert run.state is RunState.COMPLETED
        assert [record.tool_name for record in gate.requested] == ["wire_funds"]
        assert RunEventKind.APPROVAL_GRANTED in kinds(run)
        assert [name for name, _ in tools.calls] == ["wire_funds"]

    async def test_a_denied_approval_dispatches_nothing(self) -> None:
        gate = Gate(granted=False)
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(calling("wire_funds", amount=500), answering()),
            tools=tools,
            approvals=gate,
        )

        run = await runner.run(
            agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
            "Pay the deposit.",
            tenant="acme",
        )

        assert run.state is RunState.FAILED
        assert tools.calls == []
        assert RunEventKind.APPROVAL_DENIED in kinds(run)

    async def test_the_record_carries_a_digest_rather_than_the_arguments(self) -> None:
        """An approval queue is a queryable store; account numbers do not belong in one."""
        gate = Gate(granted=True)
        runner = AgentRunner(
            provider=ScriptedProvider(
                calling("wire_funds", iban="GB33BUKB20201555555555"), answering()
            ),
            tools=registry(),
            approvals=gate,
        )

        await runner.run(
            agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
            "Pay the deposit.",
            tenant="acme",
        )

        record = gate.requested[0]
        assert "GB33BUKB20201555555555" not in record.model_dump_json()
        assert len(record.arguments_digest) == 64

    def test_the_same_arguments_digest_the_same_way(self) -> None:
        """A reviewer approving a call must be able to tell it is the call that ran."""
        one = ApprovalRecord.for_call(
            run_id="r",
            tenant="acme",
            agent_name="planner",
            tool_name="wire_funds",
            arguments={"amount": 500, "to": "ada"},
            reason="policy",
            requested_at=1.0,
        )
        other = ApprovalRecord.for_call(
            run_id="r",
            tenant="acme",
            agent_name="planner",
            tool_name="wire_funds",
            arguments={"to": "ada", "amount": 500},
            reason="policy",
            requested_at=1.0,
        )
        assert one.arguments_digest == other.arguments_digest

    async def test_a_hook_can_require_approval_for_a_tool_that_did_not_declare_it(self) -> None:
        gate = Gate(granted=True)
        asking = Deciding(
            "policy",
            HookPoint.BEFORE_TOOL_DISPATCH,
            HookDecision.require_approval("unusual amount"),
        )
        runner = AgentRunner(
            provider=ScriptedProvider(calling(), answering()),
            tools=registry(),
            hooks=HookChain([asking]),
            approvals=gate,
        )

        run = await runner.run(agent(tools=("search",)), "Trains?", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert gate.requested[0].reason == "unusual amount"

    async def test_a_decision_that_arrives_after_the_deadline_is_refused(self) -> None:
        """An approval is permission at a moment. Honouring a stale one runs what nobody
        currently agrees to."""
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(calling("wire_funds"), answering()),
            tools=tools,
            approvals=Gate(granted=True, decided_at=0.0),
            approval_ttl_seconds=30.0,
        )

        run = await runner.run(
            agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
            "Pay the deposit.",
            tenant="acme",
        )

        assert run.state is RunState.FAILED
        assert "ApprovalExpiredError" in (run.events[-1].detail or "")
        assert tools.calls == []

    async def test_a_decision_for_another_record_is_refused(self) -> None:
        mismatched = ApprovalDecision(
            record_id="some_other_run", granted=True, decided_by="ada", decided_at=0.0
        )
        runner = AgentRunner(
            provider=ScriptedProvider(calling("wire_funds"), answering()),
            tools=registry(),
            approvals=Gate(mismatched),
        )

        run = await runner.run(
            agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
            "Pay the deposit.",
            tenant="acme",
        )

        assert run.state is RunState.FAILED

    def test_an_agent_requiring_approval_without_a_gate_does_not_start(self) -> None:
        """Fail at construction, not at the call that would have gone through unapproved."""
        runner = AgentRunner(provider=ScriptedProvider(), tools=registry())

        with pytest.raises(ConfigurationError, match="approval"):
            runner.run_sync(
                agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
                "Pay.",
                tenant="acme",
            )

    def test_approval_can_only_be_required_of_a_declared_tool(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            agent(tools=("search",), approval_required_tools=("wire_funds",))


class TestWhenTheRunIsAlreadyOver:
    async def test_a_terminal_hook_that_raises_is_recorded_rather_than_acted_on(self) -> None:
        """Nothing left to fail closed to: the run is over, so a late failure is evidence."""
        runner = AgentRunner(
            provider=ScriptedProvider(answering()),
            hooks=HookChain([Exploding(HookPoint.ON_TERMINAL)]),
        )

        run = await runner.run(agent(), "Trains to Kyoto?", tenant="acme")

        assert run.state is RunState.COMPLETED
        assert any("could not be evaluated" in detail for detail in details(run, HOOK_REFUSAL))


class TestApprovalsThatCannotBeObtained:
    async def test_a_hook_requiring_approval_with_no_gate_is_a_misconfiguration(self) -> None:
        """Better to refuse to start than to let the call go out unapproved."""
        requiring = Deciding(
            "policy", HookPoint.BEFORE_TOOL_DISPATCH, HookDecision.require_approval("large sum")
        )
        runner = AgentRunner(
            provider=ScriptedProvider(calling("wire_funds"), answering()),
            tools=registry(),
            hooks=HookChain([requiring]),
        )

        with pytest.raises(ConfigurationError, match="unapproved"):
            await runner.run(agent(tools=("wire_funds",)), "Pay it.", tenant="acme")

    async def test_a_gate_that_fails_is_not_a_grant(self) -> None:
        class Broken:
            async def request(self, record: ApprovalRecord) -> ApprovalDecision:  # noqa: ARG002 — it never answers
                raise RuntimeError("approval service unreachable")

        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(calling("wire_funds"), answering()),
            tools=tools,
            approvals=Broken(),
        )

        run = await runner.run(
            agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
            "Pay it.",
            tenant="acme",
        )

        assert run.state is RunState.FAILED
        assert "ApprovalDeniedError" in (run.events[-1].detail or "")
        assert tools.calls == []

    async def test_a_gate_that_never_answers_cancels_rather_than_proceeds(self) -> None:
        class Stalling:
            async def request(self, record: ApprovalRecord) -> ApprovalDecision:  # noqa: ARG002 — it never answers
                await asyncio.sleep(3600)
                raise AssertionError("unreachable")

        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(calling("wire_funds"), answering()),
            tools=tools,
            approvals=Stalling(),
            deadlines=DeadlineConfig(run_seconds=0.01),
        )

        run = await runner.run(
            agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
            "Pay it.",
            tenant="acme",
        )

        assert run.state is RunState.CANCELLED
        assert tools.calls == []


class TestBudgetsAreCheckedBeforeSpending:
    async def test_a_tool_dispatch_reserves_before_it_runs(self) -> None:
        budget = FakeBudgetPolicy()
        runner = AgentRunner(
            provider=ScriptedProvider(calling(), answering()), tools=registry(), budget=budget
        )

        await runner.run(agent(tools=("search",), budget=_a_budget()), "Trains?", tenant="acme")

        assert len(budget.reservations) >= 2

    async def test_a_dispatch_that_would_breach_the_budget_does_not_run(self) -> None:
        """Reported after the fact, an overspend is a bill. Checked before, it is a limit."""
        tools = registry()
        runner = AgentRunner(
            provider=ScriptedProvider(calling("search", q="x" * 4_000), answering()),
            tools=tools,
            budget=FakeBudgetPolicy(limit=100),
        )

        run = await runner.run(
            agent(tools=("search",), budget=_a_budget()), "Trains?", tenant="acme"
        )

        assert run.state is RunState.BUDGET_EXHAUSTED
        assert tools.calls == []


def _a_budget() -> BudgetLimits:
    return BudgetLimits(max_input_tokens=1_000)


class TestConformance:
    def test_a_hook_satisfies_the_protocol(self) -> None:
        assert isinstance(Recorder(), Hook)

    def test_a_chain_is_iterable_in_order(self) -> None:
        hooks: Sequence[Any] = [Recorder("one"), Recorder("two")]
        assert HookChain(hooks).names == ("one", "two")
