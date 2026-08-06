"""Attach policy to the loop instead of remembering it at a call site.

Four runs: a prompt redacted on its way out, a refusal that stops a run before a token is
spent, a tool call held for a human, and a hook that cannot be reached — which stops the
run rather than being skipped, because a check that did not run is not a check that passed.

Run it with `python examples/hooks.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import (
    Agent,
    ApprovalDecision,
    ApprovalRecord,
    HookChain,
    HookDecision,
    HookPoint,
    HookSubject,
    Run,
    RunEventKind,
    ToolCall,
    Usage,
)
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeToolRegistry, ScriptedProvider

_ACCOUNT = "Wire 500 from account 4111-1111-1111-1111."


class Redactor:
    """Strips a card number out of the prompt before it can reach a provider."""

    name = "redactor"
    points = (HookPoint.BEFORE_PROMPT_ASSEMBLY,)

    async def on(self, subject: HookSubject) -> HookDecision:
        """Rewrite the content if it carries a number, otherwise stand aside."""
        if "4111" not in subject.content:
            return HookDecision.proceed()
        return HookDecision.rewrite(
            subject.content.replace("4111-1111-1111-1111", "[CARD]"), reason="card number"
        )


class ModelAllowList:
    """Refuses any model the court has not approved, wherever the agent was declared."""

    name = "model-allow-list"
    points = (HookPoint.BEFORE_MODEL_CALL,)

    def __init__(self, allowed: tuple[str, ...]) -> None:
        self._allowed = allowed

    async def on(self, subject: HookSubject) -> HookDecision:
        """Stand aside for an approved agent; refuse everything else."""
        if subject.agent_name in self._allowed:
            return HookDecision.proceed()
        return HookDecision.refuse(f"agent {subject.agent_name!r} is not on the allow-list")


class Unreachable:
    """A policy service that is down. It must not become a policy that is off."""

    name = "unreachable"
    points = (HookPoint.BEFORE_MODEL_CALL,)

    async def on(self, subject: HookSubject) -> HookDecision:  # noqa: ARG002 — it never gets that far
        """Fail, every time."""
        raise ConnectionError("policy service unreachable")


class Desk:
    """An approval desk that grants, and remembers what it was asked."""

    def __init__(self) -> None:
        self.asked: list[ApprovalRecord] = []

    async def request(self, record: ApprovalRecord) -> ApprovalDecision:
        """Grant the call and echo the record it answers."""
        self.asked.append(record)
        return ApprovalDecision(
            record_id=record.id,
            granted=True,
            decided_by="registrar",
            decided_at=record.requested_at,
            reason="counter-signed",
        )


def agent(**overrides: object) -> Agent:
    """A clerk that can wire funds, for the runs that need a tool to hold."""
    fields: dict[str, object] = {
        "name": "clerk",
        "instructions": "Handle payments. Cite the authorisation.",
        "free_text": True,
        "model": "claude-sonnet-5",
    }
    return Agent(**{**fields, **overrides})  # type: ignore[arg-type]


def answering(content: str = "Done.") -> ModelResponse:
    """A plain answer."""
    return ModelResponse(content=content, usage=Usage(input_tokens=10, output_tokens=5))


def wiring() -> ModelResponse:
    """A turn that asks to move money — the kind a human should see first."""
    return ModelResponse(
        tool_calls=(ToolCall(id="call_1", name="wire_funds", arguments={"amount": 500}),),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def tools() -> FakeToolRegistry:
    """One tool with a side effect nobody wants twice."""
    return FakeToolRegistry({"wire_funds": lambda **_: "sent"})


def report(title: str, run: Run) -> None:
    """Print how the run ended and what policy said along the way."""
    print(f"\n{title}")  # noqa: T201
    print(f"  state: {run.state}")  # noqa: T201
    for event in run.events:
        if event.kind in _INTERESTING:
            print(f"  {event.kind}: {event.detail}")  # noqa: T201


_INTERESTING = frozenset(
    {
        RunEventKind.HOOK_REWRITE,
        RunEventKind.HOOK_REFUSAL,
        RunEventKind.APPROVAL_REQUIRED,
        RunEventKind.APPROVAL_GRANTED,
        RunEventKind.APPROVAL_DENIED,
        RunEventKind.TERMINATED,
    }
)


async def what_leaves_the_process_is_what_policy_allows() -> None:
    """The card number never reaches the provider, and the log holds digests of both."""
    provider = ScriptedProvider(answering())
    runner = AgentRunner(provider=provider, hooks=HookChain([Redactor()]))

    run = await runner.run(agent(), _ACCOUNT, tenant="acme")
    report("a prompt redacted on its way out", run)
    print(f"  provider saw: {_sent(provider)}")  # noqa: T201


async def a_refusal_costs_nothing() -> None:
    """Refused before the call, so there is no token spent and no request to retract."""
    provider = ScriptedProvider(answering())
    runner = AgentRunner(provider=provider, hooks=HookChain([ModelAllowList(("auditor",))]))

    run = await runner.run(agent(), "Wire it.", tenant="acme")
    report("an agent that is not on the allow-list", run)
    print(f"  model calls: {len(provider.requests)}")  # noqa: T201


async def a_call_with_a_side_effect_waits_for_a_human() -> None:
    """The desk sees a digest of the arguments, never the arguments.

    An approval queue outlives the run and is read by people who are not party to it.
    """
    desk = Desk()
    runner = AgentRunner(
        provider=ScriptedProvider(wiring(), answering()), tools=tools(), approvals=desk
    )

    run = await runner.run(
        agent(tools=("wire_funds",), approval_required_tools=("wire_funds",)),
        "Pay the invoice.",
        tenant="acme",
    )
    report("a tool call held for a decision", run)
    print(f"  desk saw digest: {desk.asked[0].arguments_digest[:12]}…")  # noqa: T201


async def a_check_that_could_not_run_is_not_a_check_that_passed() -> None:
    """The policy service is down, so the run stops. Failing open would be worse."""
    provider = ScriptedProvider(answering())
    runner = AgentRunner(provider=provider, hooks=HookChain([Unreachable()]))

    run = await runner.run(agent(), "Wire it.", tenant="acme")
    report("a hook that cannot be reached", run)
    print(f"  model calls: {len(provider.requests)}")  # noqa: T201


def _sent(provider: ScriptedProvider) -> str:
    """The text the provider was actually handed."""
    if not provider.requests:
        return "nothing"
    return str(provider.requests[0].messages[-1].content)


async def main() -> None:
    """Four runs, one policy chain each, no path around it."""
    await what_leaves_the_process_is_what_policy_allows()
    await a_refusal_costs_nothing()
    await a_call_with_a_side_effect_waits_for_a_human()
    await a_check_that_could_not_run_is_not_a_check_that_passed()


if __name__ == "__main__":
    asyncio.run(main())
