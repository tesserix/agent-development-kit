"""Code a model wrote, run somewhere it can do no harm.

Five scenarios: what it produces, what it cannot reach, what happens when it will not
stop, what it leaves behind, and the same thing wired into a tool.
Run it with `python examples/sandbox.py`.
"""

from __future__ import annotations

import asyncio

from tesserix_adk.core import SandboxTimeoutError
from tesserix_adk.tools import (
    SandboxLimits,
    SubprocessSandbox,
    ToolContext,
    ToolRefusal,
    sandbox_tool,
)


async def what_it_produces() -> None:
    """A non-zero exit is a result: the caller asked what the code did, and now knows."""
    sandbox = SubprocessSandbox()

    ran = await sandbox.run("print(sum(range(10)))")
    raised = await sandbox.run("raise ValueError('the input was empty')")

    print("=== what it produces ===")  # noqa: T201
    print(f"clean run: {ran.stdout.strip()} (exit {ran.exit_code})")  # noqa: T201
    print(f"raised:    {raised.stderr.strip().splitlines()[-1]}")  # noqa: T201


async def what_it_cannot_reach() -> None:
    """The environment is built, not inherited, so there is nothing in it worth stealing."""
    sandbox = SubprocessSandbox()

    result = await sandbox.run(
        "import os, socket\n"
        "print('credential:', os.environ.get('AWS_SECRET_ACCESS_KEY'))\n"
        "try:\n"
        "    socket.create_connection(('example.com', 80))\n"
        "except OSError as refused:\n"
        "    print('network:', refused)\n"
        "try:\n"
        "    import tesserix_adk\n"
        "except ImportError:\n"
        "    print('the kit: not importable from in here')"
    )

    print("\n=== what it cannot reach ===")  # noqa: T201
    print(result.stdout.strip())  # noqa: T201


async def when_it_will_not_stop() -> None:
    """Wall time catches waiting, processor time catches spinning. Different diagnoses."""
    sandbox = SubprocessSandbox(limits=SandboxLimits(wall_seconds=0.5, cpu_seconds=30))

    print("\n=== when it will not stop ===")  # noqa: T201
    try:
        await sandbox.run("import select\nselect.select([], [], [], 300)")
    except SandboxTimeoutError as killed:
        print(f"stopped at the {killed.limit} ceiling after {killed.seconds}s")  # noqa: T201


async def what_it_leaves_behind() -> None:
    """Files it wrote come back; files it was given do not, having been paid for once."""
    sandbox = SubprocessSandbox(limits=SandboxLimits(max_artifact_bytes=12))

    result = await sandbox.run(
        "rows = open('in.csv').read().splitlines()\n"
        "open('out.txt', 'w').write('|'.join(reversed(rows)) * 5)",
        files={"in.csv": "alpha\nbeta"},
    )

    print("\n=== what it leaves behind ===")  # noqa: T201
    for artifact in result.artifacts:
        cut = " (capped)" if artifact.truncated else ""
        print(f"{artifact.name}: {artifact.content!r}{cut}")  # noqa: T201


async def as_a_tool() -> None:
    """A ceiling that fires is a refusal: running the same code again hits the same one."""
    run_python = sandbox_tool(
        SubprocessSandbox(), name="run_python_example", limits=SandboxLimits(wall_seconds=0.5)
    )
    context = ToolContext(run_id="run_1", tenant="acme")

    print("\n=== as a tool ===")  # noqa: T201
    print(await run_python.invoke({"code": "print('a tool call like any other')"}, context))  # noqa: T201
    try:
        await run_python.invoke({"code": "import select\nselect.select([], [], [], 300)"}, context)
    except ToolRefusal as refused:
        print(f"{refused.code}: {refused}")  # noqa: T201
    run_python.release()


async def main() -> None:
    """Run every scenario in the order the docs describe them."""
    await what_it_produces()
    await what_it_cannot_reach()
    await when_it_will_not_stop()
    await what_it_leaves_behind()
    await as_a_tool()


if __name__ == "__main__":
    asyncio.run(main())
