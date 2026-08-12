"""What model-generated code can reach, what comes back, and what happens when it misbehaves."""

from __future__ import annotations

import os
import sys

import pytest

from tesserix_adk.core import ConfigurationError, SandboxMemoryError, SandboxTimeoutError
from tesserix_adk.tools import (
    SandboxLimits,
    SandboxResult,
    SubprocessSandbox,
    ToolContext,
    ToolRefusal,
    sandbox_tool,
)

LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux", reason="address-space ceilings are advisory outside Linux"
)


def _fast(**overrides: float | int) -> SandboxLimits:
    """Limits small enough that a test that hits one does not wait around for it."""
    return SandboxLimits(wall_seconds=10.0, cpu_seconds=5, **overrides)  # type: ignore[arg-type]


class TestTheLimits:
    def test_a_ceiling_of_zero_is_refused_where_it_is_written(self) -> None:
        with pytest.raises(ConfigurationError, match="wall"):
            SandboxLimits(wall_seconds=0)

    def test_no_processor_time_at_all_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="cpu"):
            SandboxLimits(cpu_seconds=0)

    def test_no_memory_at_all_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="memory"):
            SandboxLimits(memory_bytes=0)

    def test_no_output_at_all_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="output"):
            SandboxLimits(max_output_chars=0)

    def test_an_artifact_ceiling_of_zero_bytes_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="artifact"):
            SandboxLimits(max_artifact_bytes=0)

    def test_a_negative_artifact_count_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="artifact"):
            SandboxLimits(max_artifacts=-1)


class TestWhatComesBack:
    async def test_output_is_captured_rather_than_written_to_the_host(self) -> None:
        result = await SubprocessSandbox(limits=_fast()).run("print('hello')")

        assert result == SandboxResult(stdout="hello\n", stderr="", exit_code=0)
        assert result.ok

    async def test_a_traceback_comes_back_as_stderr_and_a_failing_exit_code(self) -> None:
        result = await SubprocessSandbox(limits=_fast()).run("raise ValueError('no')")

        assert not result.ok
        assert "ValueError: no" in result.stderr

    async def test_output_larger_than_the_ceiling_is_cut_and_says_so(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast(max_output_chars=32))

        result = await sandbox.run("print('x' * 5_000)")

        assert len(result.stdout) == 32
        assert result.stdout_truncated

    async def test_files_the_code_writes_come_back_as_artifacts(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        result = await sandbox.run("open('report.txt', 'w').write('done')")

        assert [(a.name, a.content) for a in result.artifacts] == [("report.txt", b"done")]

    async def test_an_artifact_larger_than_the_ceiling_is_capped_not_refused(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast(max_artifact_bytes=16))

        result = await sandbox.run("open('big.bin', 'w').write('y' * 4_000)")

        assert result.artifacts[0].content == b"y" * 16
        assert result.artifacts[0].truncated

    async def test_more_artifacts_than_the_ceiling_keeps_the_first_by_name(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast(max_artifacts=2))

        result = await sandbox.run("[open(f'{n}.txt', 'w').write('.') for n in range(5)]")

        assert [artifact.name for artifact in result.artifacts] == ["0.txt", "1.txt"]

    async def test_input_files_are_readable_and_are_not_returned_as_artifacts(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        result = await sandbox.run("print(open('in.csv').read())", files={"in.csv": "a,b"})

        assert result.stdout.strip() == "a,b"
        assert result.artifacts == ()

    async def test_an_input_file_may_not_be_written_outside_the_workspace(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        with pytest.raises(ValueError, match="outside"):
            await sandbox.run("pass", files={"../escape.txt": "no"})


class TestWhatTheCodeCanReach:
    async def test_the_network_is_not_there(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        result = await sandbox.run(
            "import socket\ntry:\n socket.create_connection(('10.0.0.1', 80))\n"
            "except OSError as e:\n print('refused', e)"
        )

        assert "refused" in result.stdout
        assert "network" in result.stdout

    async def test_the_hosts_environment_does_not_follow_the_code_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "the-real-one")
        sandbox = SubprocessSandbox(limits=_fast())

        result = await sandbox.run("import os\nprint(os.environ.get('AWS_SECRET_ACCESS_KEY'))")

        assert result.stdout.strip() == "None"
        assert "the-real-one" not in result.stdout

    async def test_the_code_starts_somewhere_that_is_not_the_hosts_directory(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        result = await sandbox.run("import os\nprint(os.getcwd())")

        assert result.stdout.strip() != os.getcwd()

    async def test_the_workspace_is_gone_once_the_run_is_over(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        result = await sandbox.run("import os\nprint(os.getcwd())")

        assert not os.path.exists(result.stdout.strip())

    async def test_the_kits_own_modules_are_not_importable_from_inside(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        result = await sandbox.run(
            "try:\n import tesserix_adk\nexcept ImportError:\n print('not here')"
        )

        assert result.stdout.strip() == "not here"


class TestWhenItMisbehaves:
    async def test_waiting_forever_is_killed_at_the_wall_clock(self) -> None:
        sandbox = SubprocessSandbox(limits=SandboxLimits(wall_seconds=0.5, cpu_seconds=30))

        with pytest.raises(SandboxTimeoutError) as killed:
            await sandbox.run("import select\nselect.select([], [], [], 300)")

        assert killed.value.limit == "wall"

    async def test_a_burnt_cpu_ceiling_is_the_cpu_ceiling_and_says_so(self) -> None:
        sandbox = SubprocessSandbox(limits=SandboxLimits(wall_seconds=30.0, cpu_seconds=1))

        with pytest.raises(SandboxTimeoutError) as killed:
            await sandbox.run("while True:\n pass")

        assert killed.value.limit == "cpu"

    async def test_running_out_of_memory_is_a_memory_error_not_a_crash(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast())

        with pytest.raises(SandboxMemoryError):
            await sandbox.run("raise MemoryError")

    @LINUX_ONLY
    async def test_an_allocation_past_the_ceiling_is_stopped_by_the_ceiling(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast(memory_bytes=64 * 1024 * 1024))

        with pytest.raises(SandboxMemoryError):
            await sandbox.run("x = bytearray(512 * 1024 * 1024)")

    @LINUX_ONLY
    async def test_the_code_is_told_what_ceiling_it_is_under(self) -> None:
        sandbox = SubprocessSandbox(limits=_fast(memory_bytes=128 * 1024 * 1024))

        result = await sandbox.run(
            "import resource\nprint(resource.getrlimit(resource.RLIMIT_AS)[0])"
        )

        assert result.stdout.strip() == str(128 * 1024 * 1024)


class TestTheTool:
    async def test_the_agent_gets_the_output_and_the_names_of_what_was_written(self) -> None:
        run_code = sandbox_tool(SubprocessSandbox(limits=_fast()), name="run_python_one")

        rendered = await run_code.invoke(
            {"code": "print('hi'); open('out.csv', 'w').write('a')"},
            ToolContext(run_id="run_1", tenant="acme"),
        )

        assert "hi" in rendered
        assert "out.csv" in rendered
        run_code.release()

    async def test_a_failing_exit_reads_as_the_failure_it_was(self) -> None:
        run_code = sandbox_tool(SubprocessSandbox(limits=_fast()), name="run_python_two")

        rendered = await run_code.invoke(
            {"code": "raise ValueError('no')"}, ToolContext(run_id="run_1", tenant="acme")
        )

        assert "exit code 1" in rendered
        assert "ValueError: no" in rendered
        run_code.release()

    async def test_hitting_a_ceiling_refuses_rather_than_returning_half_an_answer(self) -> None:
        sandbox = SubprocessSandbox(limits=SandboxLimits(wall_seconds=0.5, cpu_seconds=30))
        run_code = sandbox_tool(sandbox, name="run_python_three")

        with pytest.raises(ToolRefusal) as refused:
            await run_code.invoke(
                {"code": "while True:\n pass"}, ToolContext(run_id="run_1", tenant="acme")
            )

        assert refused.value.code == "sandbox_limit_exceeded"
        run_code.release()

    async def test_a_run_may_tighten_the_sandboxes_ceilings_but_the_tool_keeps_its_own(
        self,
    ) -> None:
        sandbox = SubprocessSandbox(limits=_fast())
        run_code = sandbox_tool(sandbox, name="run_python_four", limits=_fast(max_output_chars=4))

        rendered = await run_code.invoke(
            {"code": "print('abcdefgh')"}, ToolContext(run_id="run_1", tenant="acme")
        )

        assert "abcd" in rendered
        assert "abcde" not in rendered
        run_code.release()
