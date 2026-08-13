"""A planner that reasons, an executor that acts, and a validated plan between them.

One agent that plans and acts in the same breath executes its own hallucinations. A step
is a sentence, the sentence becomes a call, and nothing in between ever established that
the tool exists, that this agent may call it, or that the arguments are the ones the tool
declared. The failure is not that the model was wrong — models are wrong — it is that
there was nowhere to put the check.

So the two are separated. A planner produces a `Plan`: typed steps, each naming a
registered tool and carrying schema-valid arguments, and no free text anybody executes. A
`PlanExecutor` then validates the whole plan — registry, allowlist, delegated scope,
argument schema, dependency graph — and clears every step that touches the world through
an approval gate or an autonomy grant, all before the first step runs. Nothing is coerced,
guessed or repaired: an invalid plan is refused with what was wrong and the payload that
produced it, and the planner may try again a bounded number of times.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`. The
decisions behind these types are in `docs/planning.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from tesserix_adk.core.autonomy import ActionRequest, AutonomyOutcome
from tesserix_adk.core.errors import (
    ApprovalDeniedError,
    AutonomyRefusedError,
    ConfigurationError,
    IndeterminateOutcomeError,
    PlanValidationError,
)
from tesserix_adk.core.hooks import ApprovalRecord
from tesserix_adk.core.idempotency import idempotency_key
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.run import RunEvent, RunEventKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tesserix_adk.core.agent import Agent
    from tesserix_adk.core.autonomy import AutonomyLadder
    from tesserix_adk.core.hooks import ApprovalGate
    from tesserix_adk.core.idempotency import IdempotencyStore
    from tesserix_adk.core.protocols import Clock, ToolRegistry
    from tesserix_adk.runtime.delegation import Delegation
    from tesserix_adk.runtime.loop import AgentRunner

__all__ = [
    "AgentPlanner",
    "ExecutedPlan",
    "InMemoryPlanStore",
    "Plan",
    "PlanExecutor",
    "PlanRecord",
    "PlanStep",
    "PlanStore",
    "Planner",
    "StepResult",
    "ToolContract",
]

_DEFAULT_TTL = 86_400.0


def _tupled(value: object) -> object:
    """Accept the JSON array a model returns for a field the kit keeps as a tuple."""
    return tuple(value) if isinstance(value, list) else value


class PlanStep(AdkModel):
    """One thing a plan will do, as a call somebody could check before it happens.

    Args:
        id: What this step is called within the plan, so another step can wait for it.
        tool: The registered tool it calls. A name, never a sentence: a step that has to
            be interpreted is a step nothing could have validated.
        arguments: What the tool is called with, checked against the tool's declared
            model before the plan starts.
        depends_on: The steps that have to finish first.
        intent: Why the planner included it, for the person reading the plan back. Nothing
            executes it.
    """

    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    intent: str = ""

    _tupled_depends_on = field_validator("depends_on", mode="before")(_tupled)

    @model_validator(mode="after")
    def _does_not_wait_for_itself(self) -> Self:
        if self.id in self.depends_on:
            raise ValueError(f"step {self.id} waits for itself, so nothing in it could start")
        return self


class Plan(AdkModel):
    """What a planner produced: an ordered set of checkable steps and the goal behind them.

    Args:
        goal: What the plan is for, in the planner's words. Recorded, never executed.
        steps: The steps, in the order the planner wrote them. Execution order comes from
            `depends_on`, not from this.
        revision: Which attempt this is. A replan mints the next one, so two plans for one
            task are distinguishable in a record.
    """

    goal: str = Field(min_length=1)
    steps: tuple[PlanStep, ...] = ()
    revision: int = Field(default=0, ge=0)

    _tupled_steps = field_validator("steps", mode="before")(_tupled)

    @model_validator(mode="after")
    def _every_step_answers_to_one_name(self) -> Self:
        ids = [one.id for one in self.steps]
        twice = sorted({name for name in ids if ids.count(name) > 1})
        if twice:
            raise ValueError(
                f"step id used more than once: {', '.join(twice)}. A step another step "
                f"waits for has to be the one it meant"
            )
        return self

    def step(self, id: str) -> PlanStep | None:  # noqa: A002 — the field it looks up is called id
        """The step called `id`, or nothing where the plan has no such step."""
        return next((one for one in self.steps if one.id == id), None)


@dataclass(frozen=True, slots=True)
class ToolContract:
    """What one tool takes, and whether taking it back is possible.

    Args:
        tool: The registered tool this describes.
        accepts: The model its arguments have to be. Validated strictly: a plan that wrote
            `"2"` where the tool declared an integer is refused rather than coerced, since
            coercion is the executor guessing what the planner meant.
        irreversible: Whether the effect can be undone. An irreversible step is cleared by
            a person or by a matching grant, however confident the planner was.
        key_arguments: Which arguments identify one side effect, where only some do. Used
            to derive the idempotency key a step's result is recorded under.
    """

    tool: str
    accepts: type[BaseModel]
    irreversible: bool = False
    key_arguments: tuple[str, ...] = ()

    def validated(self, arguments: Mapping[str, Any]) -> BaseModel:
        """Return `arguments` as the declared model, or say which fields are wrong.

        Raises:
            PlanValidationError: If an argument is absent, of the wrong type, or one the
                tool never declared. Carries the fields and the payload, because "invalid
                arguments" tells a planner nothing it can plan differently from.
        """
        undeclared = tuple(sorted(set(arguments) - set(self.accepts.model_fields)))
        if undeclared:
            raise PlanValidationError(
                f"{self.tool} was given arguments it does not declare: {', '.join(undeclared)}",
                tool=self.tool,
                reason="arguments",
                violations=undeclared,
                payload=dict(arguments),
            )
        try:
            return self.accepts.model_validate(dict(arguments), strict=True)
        except ValidationError as refused:
            raise PlanValidationError(
                f"{self.tool} was given {refused.error_count()} argument(s) that do not "
                f"satisfy what it declared",
                tool=self.tool,
                reason="arguments",
                violations=tuple(str(one["loc"][0]) for one in refused.errors() if one["loc"]),
                payload=dict(arguments),
            ) from refused


class StepResult(AdkModel):
    """What one step did, in the form a record and a resume both read.

    Args:
        step_id: Which step.
        tool: What it called.
        outcome: What the tool returned, as it was recorded.
        key: The idempotency key the effect was recorded under, where one could be derived.
            Absent means no guarantee, rather than permission to run it again.
        replayed: Whether the outcome came from the record rather than from a fresh call.
        approved_by: Who cleared it, where a person did.
    """

    step_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    outcome: str = ""
    key: str | None = None
    replayed: bool = False
    approved_by: str | None = None


class PlanRecord(AdkModel):
    """A plan and how far it got, written down so another process can carry it on.

    Args:
        run_id: The run the plan belongs to.
        tenant: The isolation boundary.
        agent_name: Which agent the plan is being executed for.
        plan: The plan as it was validated.
        results: The steps that finished, in the order they did.
        created_at: Unix seconds it was first written.
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    plan: Plan
    results: tuple[StepResult, ...] = ()
    created_at: float = 0.0


@runtime_checkable
class PlanStore(Protocol):
    """Where a plan and its progress live between processes.

    Deliberately not the `CheckpointStore`: a checkpoint is a conversation frontier, and a
    plan written into one would resume as a conversation that never happened.
    """

    async def put(self, record: PlanRecord) -> None:
        """Write `record`, replacing whatever was held for the same run."""
        ...

    async def latest(self, run_id: str, *, tenant: str) -> PlanRecord | None:
        """The record held for `run_id`, or nothing where none was written."""
        ...

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Remove the record for `run_id`. Erasure reaches here."""
        ...


class InMemoryPlanStore:
    """A `PlanStore` that lives as long as the process, for tests and single-node runs."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], PlanRecord] = {}

    async def put(self, record: PlanRecord) -> None:
        """Hold `record` under its tenant and run."""
        self._records[record.tenant, record.run_id] = record

    async def latest(self, run_id: str, *, tenant: str) -> PlanRecord | None:
        """What is held for `run_id` under `tenant`."""
        return self._records.get((tenant, run_id))

    async def forget(self, run_id: str, *, tenant: str) -> None:
        """Drop what is held for `run_id` under `tenant`."""
        self._records.pop((tenant, run_id), None)


@runtime_checkable
class Planner(Protocol):
    """Whatever produces a plan. It reasons; it does not call anything."""

    async def plan(self, task: str, *, feedback: str = "") -> Plan:
        """Return a plan for `task`, given what a previous attempt got wrong."""
        ...


@dataclass(frozen=True, slots=True)
class ExecutedPlan:
    """What came of running a plan.

    Args:
        plan: The plan as it was validated.
        results: What each step did, in the order the steps ran.
    """

    plan: Plan
    results: tuple[StepResult, ...]

    @property
    def outcomes(self) -> Mapping[str, str]:
        """What each step returned, by step id."""
        return {one.step_id: one.outcome for one in self.results}

    @property
    def complete(self) -> bool:
        """Whether every step in the plan has a result."""
        return len(self.results) == len(self.plan.steps)


@dataclass(frozen=True, slots=True)
class _Clearance:
    """What the executor established about one step before anything ran."""

    arguments: Mapping[str, Any]
    payload: Mapping[str, Any]
    approved_by: str | None = None


class AgentPlanner:
    """An agent that plans, held to producing a plan and to holding nothing it could call.

    Args:
        runner: Runs the planning agent.
        agent: The planner. It declares `Plan` as its output type and no tools at all —
            an agent that can dispatch is not a planner, and this is checked here rather
            than left to whoever wires it up.
        delegation: Where the tenant, user and run come from, so a plan cannot be produced
            under an identity nobody passed.

    Raises:
        ConfigurationError: If the agent holds tools, or answers as anything but a `Plan`.
    """

    __slots__ = ("_agent", "_delegation", "_runner")

    def __init__(self, runner: AgentRunner, agent: Agent[Plan], *, delegation: Delegation) -> None:
        if agent.tools:
            raise ConfigurationError(
                f"{agent.name} holds {', '.join(agent.tools)}, so it could dispatch what it "
                f"is planning; a planner that can act is the thing this separation exists "
                f"to prevent"
            )
        if agent.output_type is not Plan:
            raise ConfigurationError(
                f"{agent.name} does not answer with a Plan, so what it produces would be "
                f"prose somebody has to interpret into calls"
            )
        self._runner = runner
        self._agent = agent
        self._delegation = delegation

    async def plan(self, task: str, *, feedback: str = "") -> Plan:
        """Plan `task`, telling the model what a previous attempt got wrong.

        Raises:
            PlanValidationError: If the run produced no plan at all.
        """
        caller = self._delegation.context
        run = await self._runner.run(
            self._agent,
            task if not feedback else f"{task}\n\nThe previous plan was refused: {feedback}",
            tenant=caller.tenant.tenant,
            user=caller.tenant.user,
        )
        if run.output is None:
            raise PlanValidationError(
                f"the planning run ended {run.state} without a plan",
                reason="empty",
                run_id=run.id,
                tenant=run.tenant,
            )
        return run.output


class PlanExecutor:
    """Runs a validated plan, and refuses one it cannot validate.

    Args:
        tools: The registry every step is checked against and dispatched through.
        contracts: What each tool takes, and whether it can be undone. Every tool this
            agent may call needs one, or a step naming it could not be checked.
        agent: The agent the plan is being executed for. Its allowlist caps the plan.
        delegation: What the run holds and who it belongs to. A step outside the delegated
            scope is refused even where the agent declares the tool.
        approvals: Where a step that touches the world waits for a person.
        autonomy: The ladder that decides whether a grant covers a step instead. Where a
            tool is in no action class, the contract's `irreversible` decides.
        plans: Where the plan and its progress are written, so a resume is possible.
        idempotency: Where an executed step's outcome is recorded under its key, so a
            resume in another process does not repeat the effect.
        clock: How the record is timed.
        max_steps: The most steps a plan may have. A longer one is refused, never trimmed.
        max_replans: How many times a refused plan may be planned again.
        ttl_seconds: How long an idempotency record stays authoritative.

    Raises:
        ConfigurationError: If a contract names a tool the registry does not hold, or a
            tool the agent may call has no contract.
    """

    __slots__ = (
        "_agent",
        "_approvals",
        "_autonomy",
        "_clock",
        "_contracts",
        "_delegation",
        "_events",
        "_idempotency",
        "_max_replans",
        "_max_steps",
        "_plans",
        "_tools",
        "_ttl",
    )

    def __init__(
        self,
        tools: ToolRegistry,
        contracts: Sequence[ToolContract],
        *,
        agent: Agent[Any],
        delegation: Delegation,
        approvals: ApprovalGate | None = None,
        autonomy: AutonomyLadder | None = None,
        plans: PlanStore | None = None,
        idempotency: IdempotencyStore | None = None,
        clock: Clock | None = None,
        max_steps: int = 24,
        max_replans: int = 2,
        ttl_seconds: float = _DEFAULT_TTL,
    ) -> None:
        registered = {one.name for one in tools.declarations()}
        stray = sorted({one.tool for one in contracts} - registered)
        if stray:
            raise ConfigurationError(
                f"contracts describe tools nothing registers: {', '.join(stray)}. A "
                f"contract for a tool that cannot be called validates nothing"
            )
        described = {one.tool for one in contracts}
        allowed = frozenset(agent.tools) & delegation.scope.tools
        unchecked = sorted(allowed - described)
        if unchecked:
            raise ConfigurationError(
                f"tools this agent may call have no contract: {', '.join(unchecked)}. A "
                f"step naming one could not have its arguments checked"
            )
        self._tools = tools
        self._contracts = {one.tool: one for one in contracts}
        self._agent = agent
        self._delegation = delegation
        self._approvals = approvals
        self._autonomy = autonomy
        self._plans = plans
        self._idempotency = idempotency
        self._clock = clock
        self._max_steps = max_steps
        self._max_replans = max_replans
        self._ttl = ttl_seconds
        self._events: list[RunEvent] = []

    @property
    def allowed(self) -> tuple[str, ...]:
        """What a step may name: the agent's allowlist, capped by the delegated scope."""
        return tuple(tool for tool in self._agent.tools if tool in self._delegation.scope.tools)

    @property
    def events(self) -> tuple[RunEvent, ...]:
        """Every validation, refusal, replan and step, in order, for the run's record."""
        return tuple(self._events)

    def validate(self, plan: Plan) -> Plan:
        """Return `plan` unchanged, or refuse it in full.

        Nothing is repaired here. An executor that dropped an undeclared argument or
        trimmed a plan to fit would be deciding what the planner meant, which is the
        decision this separation exists to keep out of the runtime.

        Raises:
            PlanValidationError: With the step, the tool and what was wrong.
        """
        try:
            self._check(plan)
        except PlanValidationError as refused:
            self._events.append(
                RunEvent(kind=RunEventKind.PLAN_REFUSED, name=refused.step, detail=refused.reason)
            )
            raise
        self._events.append(
            RunEvent(kind=RunEventKind.PLANNED, name=plan.goal, detail=f"{len(plan.steps)} steps")
        )
        return plan

    async def planned(self, planner: Planner, task: str) -> Plan:
        """Ask `planner` for a plan for `task`, and again where the plan does not validate.

        Raises:
            PlanValidationError: With `reason="replan"` where the allowance ran out. A
                planner that keeps regenerating an invalid plan is a loop with a model in
                it, so the last refusal is reported rather than tried again.
        """
        feedback = ""
        refusal: PlanValidationError | None = None
        for attempt in range(self._max_replans + 1):
            if refusal is not None:
                self._events.append(
                    RunEvent(kind=RunEventKind.REPLANNED, name=refusal.step, detail=refusal.reason)
                )
            plan = await planner.plan(task, feedback=feedback)
            try:
                return self.validate(plan.model_copy(update={"revision": attempt}))
            except PlanValidationError as refused:
                refusal = refused
                feedback = str(refused)
        raise PlanValidationError(
            f"the planner produced an invalid plan {self._max_replans + 1} times running; "
            f"the last was refused because {feedback}",
            step=refusal.step if refusal else "",
            tool=refusal.tool if refusal else "",
            reason="replan",
            attempts=self._max_replans + 1,
            payload=refusal.payload if refusal else None,
            run_id=self._delegation.context.run_id,
            tenant=self._delegation.context.tenant.tenant,
        )

    async def execute(self, plan: Plan) -> ExecutedPlan:
        """Validate `plan` in full, clear every step, then run what is left of it.

        Raises:
            PlanValidationError: If the plan does not validate. Nothing runs.
            ApprovalDeniedError: If a person declined a step. Nothing runs.
            AutonomyRefusedError: If a step is one no grant could permit. Nothing runs.
            ConfigurationError: If a step needs a person and there is no gate to ask.
        """
        return await self._carry_out(self.validate(plan), ())

    async def resume(self) -> ExecutedPlan:
        """Carry on the plan written down for this run, revalidating it first.

        A tool's schema may have moved since the plan was made, so the plan is validated
        against the contracts as they are now and refused where it no longer fits. What
        already ran is replayed from the record rather than repeated.

        Raises:
            ConfigurationError: If there is no plan store, or nothing was written for this
                run.
            PlanValidationError: If the plan no longer validates.
        """
        if self._plans is None:
            raise ConfigurationError(
                "this executor was given no plan store, so there is nothing it could resume from"
            )
        caller = self._delegation.context
        record = await self._plans.latest(caller.run_id, tenant=caller.tenant.tenant)
        if record is None:
            raise ConfigurationError(
                f"nothing to resume: no plan was written down for run {caller.run_id}"
            )
        return await self._carry_out(self.validate(record.plan), record.results)

    async def _carry_out(self, plan: Plan, done: tuple[StepResult, ...]) -> ExecutedPlan:
        """Clear what is left, write the plan down, then run the remaining steps."""
        finished = {one.step_id for one in done}
        remaining = [one for one in _ordered(plan) if one.id not in finished]
        cleared = await self._cleared(remaining)
        results = list(done)
        await self._write(plan, results)
        for one in remaining:
            try:
                results.append(await self._executed(one, cleared[one.id]))
            finally:
                await self._write(plan, results)
            self._events.append(
                RunEvent(kind=RunEventKind.STEP_EXECUTED, name=one.tool, detail=one.id)
            )
        return ExecutedPlan(plan=plan, results=tuple(results))

    async def _cleared(self, steps: Sequence[PlanStep]) -> dict[str, _Clearance]:
        """Decide about every step before the first one runs, so none half-happens."""
        cleared: dict[str, _Clearance] = {}
        for one in steps:
            contract = self._contracts[one.tool]
            checked = contract.validated(one.arguments)
            # The tool is called with the types it declared; keys and records digest the
            # JSON rendering, since a Decimal or a date has no one digest otherwise.
            payload = checked.model_dump(mode="json")
            cleared[one.id] = _Clearance(
                arguments=checked.model_dump(),
                payload=payload,
                approved_by=await self._decided(one, contract, payload),
            )
        return cleared

    async def _decided(
        self, step: PlanStep, contract: ToolContract, arguments: Mapping[str, Any]
    ) -> str | None:
        """Who cleared this step, where anybody had to: a grant, a person, or nobody."""
        caller = self._delegation.context
        if self._autonomy is not None and self._autonomy.classify(step.tool) is not None:
            decision = await self._autonomy.decide(
                ActionRequest(
                    tool=step.tool,
                    tenant=caller.tenant.tenant,
                    arguments=dict(arguments),
                    user=caller.tenant.user,
                )
            )
            if decision.outcome is AutonomyOutcome.REFUSE:
                self._events.append(
                    RunEvent(
                        kind=RunEventKind.AUTONOMY_REFUSED, name=step.tool, detail=decision.reason
                    )
                )
                raise AutonomyRefusedError(
                    f"{step.tool} is not something a grant could permit: {decision.reason}",
                    tool=step.tool,
                    action_class=decision.action_class,
                )
            if decision.outcome is AutonomyOutcome.ACT:
                return decision.grant_id
            self._events.append(
                RunEvent(
                    kind=RunEventKind.AUTONOMY_ESCALATED, name=step.tool, detail=decision.reason
                )
            )
            return await self._approved(step, arguments, reason=decision.reason)
        if contract.irreversible or step.tool in self._agent.approval_required_tools:
            return await self._approved(
                step, arguments, reason=f"{step.tool} cannot be undone once it has run"
            )
        return None

    async def _approved(self, step: PlanStep, arguments: Mapping[str, Any], *, reason: str) -> str:
        """Put the step in front of whoever decides, and refuse where nobody does."""
        if self._approvals is None:
            raise ConfigurationError(
                f"step {step.id} calls {step.tool}, which needs a human decision, but the "
                f"executor was given no approval gate to ask"
            )
        caller = self._delegation.context
        record = ApprovalRecord.for_call(
            run_id=caller.run_id,
            tenant=caller.tenant.tenant,
            agent_name=self._agent.name,
            tool_name=step.tool,
            arguments=arguments,
            reason=reason,
            requested_at=self._clock.now() if self._clock else 0.0,
        )
        self._events.append(
            RunEvent(kind=RunEventKind.APPROVAL_REQUIRED, name=step.tool, detail=step.id)
        )
        decision = await self._approvals.request(record)
        if not decision.granted:
            self._events.append(
                RunEvent(kind=RunEventKind.APPROVAL_DENIED, name=step.tool, detail=decision.reason)
            )
            raise ApprovalDeniedError(
                f"{decision.decided_by} declined step {step.id}: "
                f"{decision.reason or 'no reason given'}",
                run_id=caller.run_id,
                tenant=caller.tenant.tenant,
            )
        self._events.append(
            RunEvent(kind=RunEventKind.APPROVAL_GRANTED, name=step.tool, detail=decision.decided_by)
        )
        return decision.decided_by

    async def _executed(self, step: PlanStep, cleared: _Clearance) -> StepResult:
        """Run one step, under its key where the effect has one."""
        caller = self._delegation.context
        contract = self._contracts[step.tool]
        key = idempotency_key(
            tenant=caller.tenant.tenant,
            run_id=caller.run_id,
            tool=step.tool,
            arguments=cleared.payload,
            key_arguments=contract.key_arguments,
        )
        result = StepResult(
            step_id=step.id, tool=step.tool, key=key, approved_by=cleared.approved_by
        )
        recorded = await self._claimed(step, key)
        if recorded is not None:
            return result.model_copy(update={"outcome": recorded, "replayed": True})
        try:
            outcome = _recorded(await self._tools.invoke(step.tool, dict(cleared.arguments)))
        except BaseException:
            await self._abandon(key)
            raise
        await self._record(key, outcome)
        return result.model_copy(update={"outcome": outcome})

    async def _claimed(self, step: PlanStep, key: str | None) -> str | None:
        """What the store already knows about this effect, and whether it may proceed."""
        if self._idempotency is None or key is None:
            return None
        claim = await self._idempotency.begin(
            key, tenant=self._delegation.context.tenant.tenant, ttl_seconds=self._ttl
        )
        if claim.in_flight:
            raise IndeterminateOutcomeError(
                f"step {step.id} calls {step.tool}, which another caller is running under "
                f"the same key; nobody can say yet whether its effect happened"
            )
        return claim.outcome

    async def _record(self, key: str | None, outcome: str) -> None:
        """Record what the step returned, so a repeat returns it rather than running."""
        if self._idempotency is not None and key is not None:
            await self._idempotency.record(
                key,
                tenant=self._delegation.context.tenant.tenant,
                outcome=outcome,
                ttl_seconds=self._ttl,
            )

    async def _abandon(self, key: str | None) -> None:
        """Release the key a step died holding, since it recorded nothing."""
        if self._idempotency is not None and key is not None:
            await self._idempotency.abandon(key, tenant=self._delegation.context.tenant.tenant)

    async def _write(self, plan: Plan, results: Sequence[StepResult]) -> None:
        """Write down the plan and how far it got, where there is anywhere to write it."""
        if self._plans is None:
            return
        caller = self._delegation.context
        await self._plans.put(
            PlanRecord(
                run_id=caller.run_id,
                tenant=caller.tenant.tenant,
                agent_name=self._agent.name,
                plan=plan,
                results=tuple(results),
                created_at=self._clock.now() if self._clock else 0.0,
            )
        )

    def _check(self, plan: Plan) -> None:
        """Every reason a plan is refused, in the order that reports the earliest fault."""
        if not plan.steps:
            raise self._refusal(
                "the planner produced a plan with no steps, which is a planner that did "
                "not plan rather than a task needing nothing",
                reason="empty",
            )
        if len(plan.steps) > self._max_steps:
            raise self._refusal(
                f"the plan has {len(plan.steps)} steps and the ceiling is {self._max_steps}; "
                f"a plan is refused rather than trimmed to fit",
                reason="too_long",
            )
        allowed = frozenset(self.allowed)
        for one in plan.steps:
            if one.tool not in self._contracts:
                raise self._refusal(
                    f"step {one.id} calls {one.tool}, which nothing registers",
                    step=one.id,
                    tool=one.tool,
                    reason="unknown_tool",
                )
            if one.tool not in allowed:
                raise self._refusal(
                    f"step {one.id} calls {one.tool}, which this agent may not call under "
                    f"the scope it holds",
                    step=one.id,
                    tool=one.tool,
                    reason="not_allowed",
                )
            self._checked(one)
        self._graphed(plan)

    def _checked(self, step: PlanStep) -> None:
        """Validate one step's arguments, and say which step they belonged to."""
        try:
            self._contracts[step.tool].validated(step.arguments)
        except PlanValidationError as refused:
            raise self._refusal(
                str(refused),
                step=step.id,
                tool=step.tool,
                reason="arguments",
                violations=refused.violations,
                payload=refused.payload,
            ) from refused

    def _graphed(self, plan: Plan) -> None:
        """Refuse a dependency nothing satisfies, and a loop nothing could start."""
        ids = {one.id for one in plan.steps}
        for one in plan.steps:
            stray = tuple(sorted(set(one.depends_on) - ids))
            if stray:
                raise self._refusal(
                    f"step {one.id} waits for {', '.join(stray)}, which the plan does not contain",
                    step=one.id,
                    tool=one.tool,
                    reason="dependency",
                    violations=stray,
                )
        looped = _looped(plan)
        if looped:
            raise self._refusal(
                f"these steps wait for each other, so none of them could ever start: "
                f"{', '.join(looped)}",
                step=looped[0],
                reason="cycle",
                violations=looped,
            )

    def _refusal(
        self,
        message: str,
        *,
        step: str = "",
        tool: str = "",
        reason: str,
        violations: tuple[str, ...] = (),
        payload: Mapping[str, Any] | None = None,
    ) -> PlanValidationError:
        """A refusal naming the step, the tool and the run it was refused on."""
        caller = self._delegation.context
        return PlanValidationError(
            message,
            step=step,
            tool=tool,
            reason=reason,
            violations=violations,
            payload=payload,
            run_id=caller.run_id,
            tenant=caller.tenant.tenant,
        )


def _sorted(plan: Plan) -> tuple[tuple[PlanStep, ...], tuple[str, ...]]:
    """The steps in dependency order, and the ones that wait on each other for ever."""
    remaining = list(plan.steps)
    placed: set[str] = set()
    ordered: list[PlanStep] = []
    while remaining:
        ready = [one for one in remaining if set(one.depends_on) <= placed]
        if not ready:
            return tuple(ordered), tuple(sorted(one.id for one in remaining))
        ordered.extend(ready)
        placed.update(one.id for one in ready)
        remaining = [one for one in remaining if one.id not in placed]
    return tuple(ordered), ()


def _ordered(plan: Plan) -> tuple[PlanStep, ...]:
    """The steps in an order that satisfies their dependencies, declaration order first."""
    return _sorted(plan)[0]


def _looped(plan: Plan) -> tuple[str, ...]:
    """The steps that wait on each other, or nothing where the graph is acyclic."""
    return _sorted(plan)[1]


def _recorded(outcome: object) -> str:
    """What a tool returned, as the one string a store and a resume both read back."""
    if isinstance(outcome, str):
        return outcome
    return json.dumps(outcome, sort_keys=True, default=str)
