"""Reporting token savings as measured or estimated, with a holdout behind the estimate.

Run it with `uv run python examples/savings_accounting.py`.
"""

from __future__ import annotations

from tesserix_adk.observability import Arm, HoldoutPolicy, ShapedRun, account, by_tenant

POLICY = HoldoutPolicy(fraction=0.2)


def traffic(count: int = 200, tenant: str = "acme") -> tuple[ShapedRun, ...]:
    """A window of runs, each assigned by the policy and shaped only where it says so."""
    runs = []
    for index in range(count):
        run_id = f"run-{index}"
        arm = POLICY.arm(run_id)
        shaped = arm is Arm.SHAPED
        runs.append(
            ShapedRun(
                run_id=run_id,
                tenant=tenant,
                arm=arm,
                input_tokens_before=4000,
                input_tokens_after=1500 if shaped else 4000,
                output_tokens=320 + index % 40 if shaped else 480 + index % 40,
            )
        )
    return tuple(runs)


def main() -> None:
    """Report a window with a control, then the same window with none."""
    runs = traffic()
    controlled = account(runs, policy=POLICY)
    print(controlled.summary())  # noqa: T201

    uncontrolled = account(runs, policy=HoldoutPolicy(fraction=0.0))
    print(uncontrolled.output.label())  # noqa: T201

    thin = account(runs[:10], policy=POLICY)
    print(thin.output.label())  # noqa: T201

    mixed = (*runs[:100], *traffic(count=100, tenant="beta"))
    for tenant, report in sorted(by_tenant(mixed, policy=POLICY).items()):
        print(f"{tenant}: input {report.input.label()}")  # noqa: T201


if __name__ == "__main__":
    main()
