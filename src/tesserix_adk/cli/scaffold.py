"""Create a typed agent or tool from templates shipped with this kit.

Generation is a small filesystem transaction: validation and conflict discovery happen
before staging, and an interrupted replacement restores every consumer-owned byte.  The
command deliberately does not choose a project layout or dependency manager for its user.
"""

from __future__ import annotations

import argparse
import keyword
import os
import re
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tesserix_adk import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO

__all__ = ["main"]

OK = 0
FAILED = 1
MISUSED = 2
TEMPLATES = ("mcp-client", "multi-agent", "single", "tool-using")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")
_VERSION = re.compile(r"\s*(~=|==|!=|<=|>=|<|>)\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?")
_RESERVED = frozenset(
    {
        "a2a",
        "adapters",
        "agent",
        "cli",
        "core",
        "evals",
        "guardrails",
        "hooks",
        "memory",
        "models",
        "observability",
        "providers",
        "rag",
        "runtime",
        "security",
        "state",
        "testing",
        "tool",
        "tools",
        "workflows",
    }
)


@dataclass(frozen=True)
class _Name:
    """One user-facing name and its safe Python forms."""

    display: str
    module: str
    class_name: str


def main(argv: Sequence[str], *, cwd: Path | None = None, out: TextIO | None = None) -> int:
    """Generate one complete template set and return a stable process exit code.

    Args:
        argv: Arguments after ``tesserix-adk``, beginning with ``new``.
        cwd: Existing consumer project directory. Defaults to the process directory.
        out: Human-readable diagnostics. Defaults to stdout.

    Returns:
        ``0`` after generation or listing, ``1`` for a filesystem conflict or failure,
        and ``2`` for a command, name, project, or Python version that cannot be used.
    """
    writer = out if out is not None else sys.stdout
    root = Path.cwd() if cwd is None else cwd
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        writer.write("usage: tesserix-adk new [--list] | {agent,tool} NAME [options]\n")
        return MISUSED
    if parsed.command != "new":
        writer.write("the scaffold command starts with 'new'\n")
        return MISUSED
    if parsed.list_templates:
        writer.write("\n".join(TEMPLATES) + "\n")
        return OK
    if parsed.kind is None or parsed.name is None:
        writer.write("an agent or tool name is required\n")
        return MISUSED

    problem = _project_problem(root)
    if problem is not None:
        writer.write(problem + "\n")
        return MISUSED
    name = _normalised(parsed.name)
    if name is None:
        writer.write("name must be a non-reserved Python identifier or kebab-case identifier\n")
        return MISUSED
    files = _agent_files(name, parsed.template) if parsed.kind == "agent" else _tool_files(name)
    conflicts = tuple(path for path in files if (root / path).exists())
    if conflicts and not parsed.force:
        writer.write("generation aborted; conflicting paths:\n")
        for path in conflicts:
            writer.write(f"  {path}\n")
        return FAILED

    try:
        _commit(root, files, replace=parsed.force)
    except OSError as error:
        writer.write(f"generation failed and was rolled back: {error}\n")
        return FAILED
    writer.write(f"created {len(files)} files from template version {__version__}\n")
    writer.write(_dependency_hint(root) + "\n")
    return OK


def _parser() -> argparse.ArgumentParser:
    """Build the non-exiting parser used by :func:`main`."""
    parser = argparse.ArgumentParser(prog="tesserix-adk", add_help=False)
    commands = parser.add_subparsers(dest="command")
    new = commands.add_parser("new", add_help=False)
    new.add_argument("kind", nargs="?", choices=("agent", "tool"))
    new.add_argument("name", nargs="?")
    new.add_argument("--template", choices=TEMPLATES, default="single")
    new.add_argument("--list", action="store_true", dest="list_templates")
    new.add_argument("--force", action="store_true")
    return parser


def _project_problem(root: Path) -> str | None:
    """Return why ``root`` cannot host generated code, before any write occurs."""
    project = root / "pyproject.toml"
    if not root.is_dir():
        return f"project directory does not exist: {root}"
    if not project.is_file():
        return "pyproject.toml is required; project bootstrapping is intentionally separate"
    try:
        with project.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        return f"pyproject.toml cannot be read: {error}"
    metadata = document.get("project")
    if not isinstance(metadata, dict):
        return "pyproject.toml has no [project] table"
    supported = metadata.get("requires-python")
    if not isinstance(supported, str):
        return "[project].requires-python must declare a supported Python version"
    if not _allows_current_python(supported):
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return f"project requires Python {supported!r}, but scaffolding is running on {version}"
    return None


def _allows_current_python(specifier: str) -> bool:
    """Evaluate the ordinary PEP 440 clauses used by ``requires-python``.

    The package has no runtime dependency on ``packaging``.  Unsupported exotic clauses
    fail closed instead of being guessed at.
    """
    current = sys.version_info[:3]
    clauses = tuple(part.strip() for part in specifier.split(",") if part.strip())
    if not clauses:
        return False
    for clause in clauses:
        matched = _VERSION.fullmatch(clause)
        if matched is None:
            return False
        operator, major, minor, patch = matched.groups()
        target = (int(major), int(minor or 0), int(patch or 0))
        precision = 1 + (minor is not None) + (patch is not None)
        if operator == ">=" and not current >= target:
            return False
        if operator == ">" and not current > target:
            return False
        if operator == "<=" and not current <= target:
            return False
        if operator == "<" and not current < target:
            return False
        if operator == "==" and current[:precision] != target[:precision]:
            return False
        if operator == "!=" and current[:precision] == target[:precision]:
            return False
        if operator == "~=" and not (current >= target and current[0] == target[0]):
            return False
    return True


def _normalised(value: str) -> _Name | None:
    """Validate a public name and return deterministic module/class spellings."""
    if _NAME.fullmatch(value) is None:
        return None
    module = value.replace("-", "_").lower()
    if keyword.iskeyword(module) or module in _RESERVED:
        return None
    parts = module.split("_")
    if any(not part for part in parts):
        return None
    return _Name(
        display=value.replace("_", "-"),
        module=module,
        class_name="".join(map(str.title, parts)),
    )


def _commit(root: Path, files: Mapping[str, str], *, replace: bool) -> None:
    """Replace the whole template set, compensating every partial filesystem move."""
    staging = Path(tempfile.mkdtemp(prefix=".adk-new-", dir=root))
    staged = staging / "new"
    backup = staging / "old"
    staged.mkdir()
    backup.mkdir()
    installed: list[Path] = []
    saved: list[tuple[Path, Path]] = []
    try:
        for relative, content in files.items():
            (staged / relative).write_text(content, encoding="utf-8", newline="\n")
        for relative in files:
            target = root / relative
            previous = backup / relative
            if target.exists():
                if not replace:
                    raise FileExistsError(target)
                os.replace(target, previous)
                saved.append((previous, target))
            os.replace(staged / relative, target)
            installed.append(target)
    except BaseException:
        for target in reversed(installed):
            target.unlink(missing_ok=True)
        for previous, target in reversed(saved):
            os.replace(previous, target)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _dependency_hint(root: Path) -> str:
    """Suggest, but never run, the consumer's dependency-manager command."""
    pin = f"tesserix-adk=={__version__}"
    if (root / "uv.lock").exists():
        return f"next: uv add '{pin}'"
    return f"next: python -m pip install '{pin}' (or use your project dependency manager)"


def _agent_files(name: _Name, template: str) -> dict[str, str]:
    """Render one typed, traced, budgeted agent and its offline test."""
    tool_name = f"{name.module}_lookup"
    module = f'''"""A generated {template} agent pinned to Tesserix ADK {__version__}.

Example:
    >>> build_agent().name
    '{name.display}'
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel

from tesserix_adk import Agent, tool
from tesserix_adk.core import AdkConfig, BudgetLimits, Instrumentation, load_config
from tesserix_adk.runtime import SystemClock

ADK_TEMPLATE_VERSION: Final[str] = "{__version__}"
TEMPLATE_KIND: Final[str] = "{template}"


class {name.class_name}Input(BaseModel):
    """Validated input to the agent.

    Args:
        request: What the caller wants the agent to do.
    """

    request: str


class {name.class_name}Output(BaseModel):
    """Validated answer returned to the caller.

    Args:
        answer: The grounded answer.
    """

    answer: str


@tool(idempotency="read_only")
def {tool_name}(query: str) -> str:
    """Look up deterministic local context for a request.

    Args:
        query: The caller's search phrase.
    """
    return f"local context for {{query}}"


def resolved_config() -> AdkConfig:
    """Resolve configuration through the same typed path used in deployment."""
    return load_config(
        {{"provider": {{"endpoint": "http://127.0.0.1:1"}}}}, env={{}}, start=None
    )


def instrumentation() -> Instrumentation:
    """Build run-rooted, content-safe instrumentation for this agent."""
    return Instrumentation(clock=SystemClock())


def build_agent() -> Agent[{name.class_name}Output]:
    """Build the declaration; provider and credentials stay deployment concerns."""
    config = resolved_config()
    return Agent(
        name="{name.display}",
        instructions="Use {tool_name}, then answer the caller with grounded context.",
        model="replace-with-a-configured-model",
        output_type={name.class_name}Output,
        tools=("{tool_name}",),
        idempotent_tools=("{tool_name}",),
        budget=BudgetLimits(
            max_model_calls=2,
            max_tool_calls=1,
            max_input_tokens=min(config.budget.max_input_tokens or 8_000, 8_000),
            max_output_tokens=min(config.budget.max_output_tokens or 1_000, 1_000),
        ),
    )
'''
    test = f'''"""Offline contract test for the generated {name.display} agent."""

from tesserix_adk import AgentRunner, ToolRegistry
from tesserix_adk.testing import FakeClock, FakeModelProvider, ScriptedTurn

from {name.module}_agent import (
    {name.class_name}Input,
    {name.class_name}Output,
    build_agent,
    {tool_name},
)


def test_{name.module}_agent_runs_offline() -> None:
    """The generated declaration, tool schema, budget and output work together."""
    request = {name.class_name}Input(request="find a safe starting point")
    provider = FakeModelProvider(
        ScriptedTurn.calling("{tool_name}", {{"query": request.request}}),
        ScriptedTurn.returning({{"answer": "A grounded local answer."}}),
    )
    run = AgentRunner(
        provider=provider,
        tools=ToolRegistry(({tool_name},)),
        clock=FakeClock(),
    ).run_sync(build_agent(), request.request, tenant="local-test", user="developer")

    assert run.output == {name.class_name}Output(answer="A grounded local answer.")
    assert {tool_name}.parameters_schema["required"] == ["query"]
'''
    return {f"{name.module}_agent.py": module, f"test_{name.module}_agent.py": test}


def _tool_files(name: _Name) -> dict[str, str]:
    """Render one typed tool and its schema/execution contract test."""
    called = f"{name.module}_tool"
    module = f'''"""A generated tool pinned to Tesserix ADK {__version__}.

Example:
    >>> import asyncio
    >>> asyncio.run({called}("hello"))
    'processed: hello'
"""

from typing import Final

from tesserix_adk import tool

ADK_TEMPLATE_VERSION: Final[str] = "{__version__}"


@tool(idempotency="read_only")
def {called}(value: str) -> str:
    """Process one validated value without network access.

    Args:
        value: The value to process.
    """
    return f"processed: {{value}}"
'''
    test = f'''"""Offline contract test for the generated {name.display} tool."""

import asyncio

from {name.module}_tool import {called}


def test_{called}_schema_and_result() -> None:
    """The callable and the schema consumed by a model stay in lockstep."""
    assert asyncio.run({called}("hello")) == "processed: hello"
    assert {called}.parameters_schema["required"] == ["value"]
'''
    return {f"{name.module}_tool.py": module, f"test_{name.module}_tool.py": test}
