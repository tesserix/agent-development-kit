"""Running code a model wrote, somewhere it can do no harm.

Model-generated code is untrusted by construction: whatever wrote it read tool output,
retrieved documents and user text, any of which an attacker may have supplied. Running it
in the agent's own process hands an injected prompt the process's credentials, its network
position and its filesystem. So it runs somewhere else, under ceilings set before it
starts, with an environment that holds nothing worth stealing.

`SubprocessSandbox` is the batteries-included implementation: a fresh interpreter, an empty
environment, a temporary workspace it cannot leave through the API it is given, no network,
and ceilings on processor time, address space, output and artifacts. It is defence in
depth in one process tree, not a virtual machine — determined code with `ctypes` is a
kernel boundary away from the host, and that boundary is the container the sandbox itself
runs in. `Sandbox` is the seam: a deployment that needs gVisor, Kata or a remote executor
binds one, and every caller above it is unchanged.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import resource
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tesserix_adk.core.errors import (
    ConfigurationError,
    SandboxError,
    SandboxMemoryError,
    SandboxTimeoutError,
)
from tesserix_adk.tools.context import (
    ToolContext,  # noqa: TC001 — the decorator reads this annotation at runtime
)
from tesserix_adk.tools.decorator import Tool, tool
from tesserix_adk.tools.errors import ToolRefusal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "DEFAULT_LIMITS",
    "Sandbox",
    "SandboxArtifact",
    "SandboxLimits",
    "SandboxResult",
    "SubprocessSandbox",
    "sandbox_tool",
]

# The soft ceiling raises SIGXCPU; a child that ignores it is killed at the hard one.
_CPU_SIGNALS = (-int(getattr(signal, "SIGXCPU", 24)), -int(signal.SIGKILL))

_BOOTSTRAP = """
import builtins, socket, sys

_REFUSED = "sandbox: the network is not reachable from here"


def _no_network(*_args, **_kwargs):
    raise OSError(_REFUSED)


for _name in ("socket", "create_connection", "create_server", "socketpair", "getaddrinfo"):
    setattr(socket, _name, _no_network)

_code = sys.stdin.read()
sys.argv = ["sandbox"]
exec(compile(_code, "<sandbox>", "exec"), {"__name__": "__main__", "__builtins__": builtins})
"""


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """What generated code may spend before the sandbox takes the process away.

    Args:
        wall_seconds: Elapsed time. Catches code that is waiting for something that will
            never arrive.
        cpu_seconds: Processor time, which is not elapsed time: four threads burn four
            processor seconds in one. Catches code that is spinning, and says so, where
            the wall clock cannot tell spinning from waiting.
        memory_bytes: Address space. Set on the child before any generated code runs, so
            an allocation that would starve the host fails inside the sandbox instead.
            Enforced by the kernel on Linux; macOS refuses address-space ceilings outright,
            where it stands as an intent the time ceilings still bound.
        max_output_chars: How much of each stream is kept. Output is a channel back into
            the conversation, so it is bounded like any other.
        max_artifact_bytes: How much of each written file is returned.
        max_artifacts: How many written files are returned, in name order.

    Example:
        >>> SandboxLimits(wall_seconds=2.0, cpu_seconds=1).memory_bytes
        268435456
    """

    wall_seconds: float = 10.0
    cpu_seconds: int = 5
    memory_bytes: int = 256 * 1024 * 1024
    max_output_chars: int = 16_384
    max_artifact_bytes: int = 1024 * 1024
    max_artifacts: int = 16

    def __post_init__(self) -> None:
        """Refuse ceilings that cannot bound anything, at the point they are written."""
        if self.wall_seconds <= 0:
            raise ConfigurationError("a wall-clock ceiling of zero leaves no time to run in")
        if self.cpu_seconds <= 0:
            raise ConfigurationError("a cpu ceiling of zero leaves no time to run in")
        if self.memory_bytes <= 0:
            raise ConfigurationError("a memory ceiling of zero cannot hold an interpreter")
        if self.max_output_chars <= 0:
            raise ConfigurationError("an output ceiling of zero discards the whole result")
        if self.max_artifact_bytes <= 0 or self.max_artifacts < 0:
            raise ConfigurationError("an artifact ceiling of zero returns an empty file")


DEFAULT_LIMITS = SandboxLimits()
"""The ceilings a sandbox uses when a caller expresses no opinion."""


@dataclass(frozen=True, slots=True)
class SandboxArtifact:
    """A file the generated code wrote, carried back out of the workspace.

    Args:
        name: The path it was written to, relative to the workspace.
        content: What it holds, up to `max_artifact_bytes`.
        truncated: Whether the file was larger than what came back.
    """

    name: str
    content: bytes
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class SandboxResult:
    r"""What running the code produced. A non-zero exit is a result, not an error.

    Args:
        stdout: What the code printed, up to the ceiling.
        stderr: What it printed to the error stream, including a traceback if it raised.
        exit_code: What the interpreter exited with.
        artifacts: Files it wrote, in name order.
        stdout_truncated: Whether output was cut at the ceiling.
        stderr_truncated: Whether error output was cut at the ceiling.

    Example:
        >>> SandboxResult(stdout="4\\n", stderr="", exit_code=0).ok
        True
    """

    stdout: str
    stderr: str
    exit_code: int
    artifacts: tuple[SandboxArtifact, ...] = ()
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def ok(self) -> bool:
        """Whether the code ran to completion without raising."""
        return self.exit_code == 0


@runtime_checkable
class Sandbox(Protocol):
    """Somewhere untrusted code runs and the host does not notice.

    The contract is narrow on purpose, so a stronger isolation backend — a container, a
    microVM, a remote executor — can satisfy it without anything above changing.
    """

    async def run(
        self,
        code: str,
        *,
        limits: SandboxLimits | None = None,
        files: Mapping[str, str] | None = None,
    ) -> SandboxResult:
        """Run `code` with `files` alongside it, and return what it produced.

        Raises `SandboxTimeoutError` or `SandboxMemoryError` when a ceiling fired, because
        then there is no result to report — only the fact that it was stopped.
        """
        ...


@dataclass(frozen=True, slots=True)
class SubprocessSandbox:
    r"""A sandbox that is a fresh interpreter in a temporary directory with nothing in it.

    Args:
        limits: The ceilings applied when a call does not give its own.
        executable: Which interpreter runs the code. Defaults to the running one.

    Example:
        >>> import asyncio
        >>> asyncio.run(SubprocessSandbox().run("print(6 * 7)")).stdout
        '42\n'
    """

    limits: SandboxLimits = field(default_factory=lambda: DEFAULT_LIMITS)
    executable: str = field(default_factory=lambda: sys.executable)

    async def run(
        self,
        code: str,
        *,
        limits: SandboxLimits | None = None,
        files: Mapping[str, str] | None = None,
    ) -> SandboxResult:
        """Run `code` in a workspace that exists only for the length of the call."""
        ceilings = limits or self.limits
        workspace = Path(tempfile.mkdtemp(prefix="adk-sandbox-"))
        try:
            given = _seed(workspace, files or {})
            return await self._execute(code, workspace, ceilings, given)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    async def _execute(
        self, code: str, workspace: Path, ceilings: SandboxLimits, given: frozenset[str]
    ) -> SandboxResult:
        """Start the child, feed it the code, and read back what survived the ceilings."""
        process = await asyncio.create_subprocess_exec(
            self.executable,
            "-I",
            "-S",
            "-c",
            _BOOTSTRAP,
            cwd=workspace,
            env=_environment(workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_ceilings_for(ceilings),
        )
        try:
            out, err = await asyncio.wait_for(
                process.communicate(code.encode("utf-8")), ceilings.wall_seconds
            )
        except TimeoutError as expired:
            await _kill(process)
            raise SandboxTimeoutError(
                f"the code ran past its {ceilings.wall_seconds}s wall-clock ceiling",
                limit="wall",
                seconds=ceilings.wall_seconds,
            ) from expired

        failed = err.decode("utf-8", "replace")
        exit_code = process.returncode or 0
        if exit_code in _CPU_SIGNALS:
            raise SandboxTimeoutError(
                f"the code ran past its {ceilings.cpu_seconds}s cpu ceiling",
                limit="cpu",
                seconds=float(ceilings.cpu_seconds),
            )
        if "MemoryError" in failed:
            raise SandboxMemoryError(
                f"the code asked for more than its {ceilings.memory_bytes} byte ceiling",
                limit_bytes=ceilings.memory_bytes,
            )
        stdout, cut_out = _bounded(out.decode("utf-8", "replace"), ceilings.max_output_chars)
        stderr, cut_err = _bounded(failed, ceilings.max_output_chars)
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            artifacts=_collected(workspace, ceilings, given),
            stdout_truncated=cut_out,
            stderr_truncated=cut_err,
        )


def sandbox_tool(
    sandbox: Sandbox,
    *,
    name: str = "run_python",
    limits: SandboxLimits | None = None,
) -> Tool[..., str]:
    """Build the tool that lets an agent run code in `sandbox`.

    Args:
        sandbox: Where the code runs. The tool never runs anything in this process.
        name: What the model calls it.
        limits: Ceilings for calls through this tool, tighter than the sandbox's own where
            a tool exposed to a model should be more careful than its backend.

    Returns:
        The tool, holding the name until it is released.

    Example:
        >>> run_python = sandbox_tool(SubprocessSandbox(), name="run_python_doc")
        >>> run_python.name
        'run_python_doc'
        >>> run_python.release()
    """

    @tool(name=name, parallel_safe=False)
    async def run_python(code: str, context: ToolContext) -> str:
        """Run Python in an isolated sandbox with no network and no credentials.

        Args:
            code: The program to run. Print what should come back.
            context: The run this call belongs to.
        """
        del context
        try:
            result = await sandbox.run(code, limits=limits)
        except SandboxError as stopped:
            raise ToolRefusal(name, "sandbox_limit_exceeded", str(stopped)) from stopped
        return _rendered(result)

    return run_python


def _rendered(result: SandboxResult) -> str:
    """Say what happened in the order a reader needs it: outcome, output, then artifacts."""
    parts = [f"exit code {result.exit_code}"]
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr}")
    if result.artifacts:
        written = ", ".join(f"{a.name} ({len(a.content)} bytes)" for a in result.artifacts)
        parts.append(f"files written: {written}")
    return "\n\n".join(parts)


def _seed(workspace: Path, files: Mapping[str, str]) -> frozenset[str]:
    """Write the caller's input files, refusing any path that leaves the workspace."""
    written: set[str] = set()
    for name, content in files.items():
        target = (workspace / name).resolve()
        if not target.is_relative_to(workspace.resolve()):
            raise ValueError(f"{name!r} writes outside the sandbox workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.add(str(target.relative_to(workspace.resolve())))
    return frozenset(written)


def _environment(workspace: Path) -> dict[str, str]:
    """Build the child's whole environment, so nothing of the host's leaks into it."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workspace),
        "TMPDIR": str(workspace),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _ceilings_for(limits: SandboxLimits) -> Callable[[], None]:
    """Return the hook that binds the ceilings inside the child, before it runs anything."""

    def bind() -> None:  # pragma: no cover — runs in the child, after the fork
        # Headroom so the kernel signals the soft ceiling before killing at the hard one.
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        for kind in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(kind, (limits.memory_bytes, limits.memory_bytes))
            except (OSError, ValueError):
                continue

    return bind


async def _kill(process: asyncio.subprocess.Process) -> None:
    """Take the whole session, not the one process: the code may have started others."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    await process.wait()


def _bounded(text: str, ceiling: int) -> tuple[str, bool]:
    """Cut a stream at the ceiling and say whether cutting happened."""
    return (text[:ceiling], True) if len(text) > ceiling else (text, False)


def _collected(
    workspace: Path, limits: SandboxLimits, given: frozenset[str]
) -> tuple[SandboxArtifact, ...]:
    """Read back what the code wrote, in name order, capped by count and by size.

    What the caller handed in is not an artifact: returning it would charge the
    conversation twice for content it already had.
    """
    artifacts: list[SandboxArtifact] = []
    for path in sorted(workspace.rglob("*")):
        if len(artifacts) >= limits.max_artifacts:
            break
        name = str(path.relative_to(workspace))
        if not path.is_file() or name in given:
            continue
        content = path.read_bytes()
        artifacts.append(
            SandboxArtifact(
                name=name,
                content=content[: limits.max_artifact_bytes],
                truncated=len(content) > limits.max_artifact_bytes,
            )
        )
    return tuple(artifacts)
