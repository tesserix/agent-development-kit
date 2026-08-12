"""How far one agent may hand work to another, and what the other is allowed to hold.

Four scenarios: a scope that narrows on the way down, an escalation refused, the ceilings
that bound the shape of a run, and a grant that has expired.
Run it with `python examples/delegation.py`.
"""

from __future__ import annotations

from tesserix_adk.core import DelegationLimitError, ScopeEscalationError
from tesserix_adk.runtime import Delegation, DelegationLimits, DelegationScope
from tesserix_adk.testing import FakeClock

TOOLS = frozenset({"search", "summarise", "file_bug"})


def _supervisor(**limits: int) -> Delegation:
    """The agent a run starts at, holding everything the run was granted."""
    return Delegation.root(
        run_id="run_1",
        tenant="acme",
        agent="supervisor",
        scope=DelegationScope(tools=TOOLS),
        limits=DelegationLimits(**limits),
    )


def narrowing_on_the_way_down() -> None:
    """A child holds the intersection of what it asked for and what its parent held."""
    root = _supervisor()
    researcher = root.to("researcher", tools={"search", "summarise"})
    reader = researcher.to("reader", tools={"search"})

    print("=== narrowing on the way down ===")  # noqa: T201
    for agent in (root, researcher, reader):
        print(f"{'/'.join(agent.path)}: {sorted(agent.scope.tools)}")  # noqa: T201
    print(f"tenant stays {reader.context.tenant.tenant}, run stays {reader.context.run_id}")  # noqa: T201


def asking_for_what_the_parent_never_held() -> None:
    """The child's own configuration does not matter; the caller's scope does."""
    print("\n=== asking for what the parent never held ===")  # noqa: T201
    try:
        _supervisor().to("accountant", tools={"search", "wire_transfer"})
    except ScopeEscalationError as refused:
        print(f"{refused} (requested={refused.requested})")  # noqa: T201


def the_shape_of_a_run() -> None:
    """Depth, fan-out and the run's own ceiling each bound something different."""
    print("\n=== the shape of a run ===")  # noqa: T201

    deep = _supervisor(max_depth=2).to("a").to("b")
    try:
        deep.to("c")
    except DelegationLimitError as refused:
        print(f"{refused.reason}: {refused}")  # noqa: T201

    wide = _supervisor(max_fan_out=1)
    wide.to("a")
    try:
        wide.to("b")
    except DelegationLimitError as refused:
        print(f"{refused.reason}: {refused}")  # noqa: T201

    circling = _supervisor(max_depth=8).to("a").to("b")
    try:
        circling.to("a")
    except DelegationLimitError as refused:
        print(f"{refused.reason}: {refused}")  # noqa: T201

    print(f"the run spent {wide.delegations} of its {wide.limits.max_delegations}")  # noqa: T201


def a_grant_that_ran_out() -> None:
    """A scope that outlives its reason is a scope nobody reviewed."""
    root = Delegation.root(
        run_id="run_1",
        tenant="acme",
        agent="supervisor",
        scope=DelegationScope(tools=TOOLS, expires_at=100.0),
        clock=FakeClock(start=101.0),
    )

    print("\n=== a grant that ran out ===")  # noqa: T201
    try:
        root.to("researcher")
    except DelegationLimitError as refused:
        print(f"{refused.reason}: {refused}")  # noqa: T201


def main() -> None:
    """Run every scenario in the order the docs describe them."""
    narrowing_on_the_way_down()
    asking_for_what_the_parent_never_held()
    the_shape_of_a_run()
    a_grant_that_ran_out()


if __name__ == "__main__":
    main()
