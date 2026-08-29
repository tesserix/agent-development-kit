"""Run extensible, secret-safe environment diagnostics without changing the machine."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field

from tesserix_adk.core import AdkConfig, AdkModel, ConfigurationError
from tesserix_adk.core.redaction import scrub

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
    from typing import TextIO

__all__ = [
    "CheckRegistry",
    "CheckResult",
    "CheckStatus",
    "CredentialPresenceCheck",
    "DiagnosticCheck",
    "DoctorCheck",
    "DoctorContext",
    "ExtraPresenceCheck",
    "ProbeObservation",
    "PythonVersionCheck",
    "main",
]

OK = 0
UNHEALTHY = 1
MISUSED = 2


class CheckStatus(StrEnum):
    """One diagnostic's explicit outcome."""

    PASS = "pass"  # noqa: S105 — diagnostic status, not a credential
    FAIL = "fail"
    WARN = "warn"
    SKIPPED = "skipped"


class CheckResult(AdkModel):
    """A bounded diagnostic finding with one concrete suggested remediation.

    Args:
        name: Stable check name.
        status: Pass, failure, warning or deliberate skip.
        cause: Evidence-based explanation with no credential values.
        remediation: Exactly one next action; successful checks say no action is required.
        required: Whether failure makes the doctor command unhealthy.
        duration_ms: Wall-clock duration, useful for a slow diagnostic itself.
    """

    name: str = Field(min_length=1)
    status: CheckStatus
    cause: str = Field(min_length=1)
    remediation: str = Field(min_length=1)
    required: bool = True
    duration_ms: float = Field(default=0.0, ge=0)


class ProbeObservation(AdkModel):
    """A probe's provider-normalised result before policy marks it required.

    Args:
        healthy: Whether the dependency answered with required capability/entitlement.
        cause: Normalised status, never a response body or credential.
    """

    healthy: bool
    cause: str = Field(min_length=1)

    @classmethod
    def passed(cls, cause: str = "reachable and entitled") -> ProbeObservation:
        """Build a healthy observation."""
        return cls(healthy=True, cause=cause)

    @classmethod
    def failed(cls, cause: str) -> ProbeObservation:
        """Build an unhealthy observation with an exact normalised cause."""
        return cls(healthy=False, cause=cause)


@dataclass(frozen=True)
class DoctorContext:
    """Typed configuration and environment boundary available to checks.

    Environment values remain inside checks; the report surface accepts only scrubbed
    ``CheckResult`` values.
    """

    config: AdkConfig
    environ: Mapping[str, str]


class DoctorCheck(Protocol):
    """A plugin diagnostic registered without changing the command implementation."""

    @property
    def name(self) -> str:
        """Stable unique check name."""
        ...

    @property
    def required(self) -> bool:
        """Whether failure makes the overall environment unhealthy."""
        ...

    @property
    def network(self) -> bool:
        """Whether ``--offline`` skips this check."""
        ...

    async def run(self, context: DoctorContext) -> CheckResult:
        """Inspect one dependency or local prerequisite without mutating it."""
        ...


@dataclass(frozen=True)
class CredentialPresenceCheck:
    """Distinguish a missing credential from one present but never print its value.

    Args:
        variable: Exact environment variable to inspect.
        required: Whether absence fails the overall diagnosis.
    """

    variable: str
    required: bool = True
    network: bool = False

    @property
    def name(self) -> str:
        """Stable check name including the actionable variable."""
        return f"credential:{self.variable}"

    async def run(self, context: DoctorContext) -> CheckResult:
        """Report presence only, leaving validity to the provider probe."""
        present = bool(context.environ.get(self.variable, "").strip())
        status = (
            CheckStatus.PASS if present else CheckStatus.FAIL if self.required else CheckStatus.WARN
        )
        cause = (
            f"{self.variable} is present (value not displayed)"
            if present
            else f"{self.variable} is missing"
        )
        remediation = (
            "no remediation required"
            if present
            else f"set {self.variable} in the process secret store, then rerun doctor"
        )
        return CheckResult(
            name=self.name,
            status=status,
            cause=cause,
            remediation=remediation,
            required=self.required,
        )


@dataclass(frozen=True)
class DiagnosticCheck:
    """Adapt one provider, model, store, MCP, telemetry or clock-skew probe.

    Args:
        name: Stable component name.
        endpoint: Safe endpoint identity. Userinfo, query and fragment are removed in output.
        probe: Async read-only probe returning a normalised observation.
        remediation: One action to take if the observation is unhealthy.
        required: Whether an unhealthy observation fails the overall diagnosis.
        network: Whether ``--offline`` must skip the probe.
    """

    name: str
    endpoint: str
    probe: Callable[[DoctorContext], Awaitable[ProbeObservation]]
    remediation: str
    required: bool = True
    network: bool = True

    async def run(self, context: DoctorContext) -> CheckResult:
        """Execute the read-only probe and attach its safe endpoint identity."""
        observed = await self.probe(context)
        status = CheckStatus.PASS if observed.healthy else CheckStatus.FAIL
        return CheckResult(
            name=self.name,
            status=status,
            cause=f"{_safe_endpoint(self.endpoint)}: {observed.cause}",
            remediation="no remediation required" if observed.healthy else self.remediation,
            required=self.required,
        )


@dataclass(frozen=True)
class ExtraPresenceCheck:
    """Check an optional adapter extra without installing or changing anything.

    Args:
        module: Import module supplied by the extra.
        extra: Pip extra users should select.
        required: Whether this deployment declares the adapter required.
    """

    module: str
    extra: str
    required: bool = True
    network: bool = False

    @property
    def name(self) -> str:
        """Stable optional-dependency check name."""
        return f"extra:{self.extra}"

    async def run(self, context: DoctorContext) -> CheckResult:
        """Inspect import metadata without importing adapter side effects."""
        del context
        try:
            present = importlib.util.find_spec(self.module) is not None
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            present = False
        status = (
            CheckStatus.PASS if present else CheckStatus.FAIL if self.required else CheckStatus.WARN
        )
        return CheckResult(
            name=self.name,
            status=status,
            cause=f"module {self.module!r} is {'installed' if present else 'missing'}",
            remediation=(
                "no remediation required"
                if present
                else f"install project dependency 'tesserix-adk[{self.extra}]'"
            ),
            required=self.required,
        )


@dataclass(frozen=True)
class PythonVersionCheck:
    """Confirm the interpreter remains in this installed kit's supported range."""

    required: bool = True
    network: bool = False
    name: str = "python"

    async def run(self, context: DoctorContext) -> CheckResult:
        """Report the actual interpreter and minimum supported version."""
        del context
        current = sys.version_info[:3]
        healthy = current >= (3, 12, 0)
        return CheckResult(
            name=self.name,
            status=CheckStatus.PASS if healthy else CheckStatus.FAIL,
            cause=f"Python {current[0]}.{current[1]}.{current[2]} (requires >=3.12)",
            remediation=(
                "no remediation required"
                if healthy
                else "run this project with Python 3.12 or newer"
            ),
            required=self.required,
        )


class CheckRegistry:
    """Ordered, duplicate-safe registry for built-in and adapter-contributed checks."""

    def __init__(self, checks: Iterable[DoctorCheck] = ()) -> None:
        self._checks: dict[str, DoctorCheck] = {}
        for check in checks:
            self.register(check)

    def register(self, check: DoctorCheck) -> None:
        """Register one name exactly once.

        Raises:
            ConfigurationError: Another check already owns the same name.
        """
        if check.name in self._checks:
            raise ConfigurationError(f"doctor check {check.name!r} is already registered")
        self._checks[check.name] = check

    @property
    def checks(self) -> tuple[DoctorCheck, ...]:
        """Registered checks in deterministic registration order."""
        return tuple(self._checks.values())

    async def run(
        self, context: DoctorContext, *, offline: bool = False
    ) -> tuple[CheckResult, ...]:
        """Run independent checks concurrently while keeping report order stable."""

        async def execute(check: DoctorCheck) -> CheckResult:
            if offline and check.network:
                return CheckResult(
                    name=check.name,
                    status=CheckStatus.SKIPPED,
                    cause="network diagnostic skipped by --offline",
                    remediation="rerun without --offline when network diagnostics are intended",
                    required=check.required,
                )
            started = time.perf_counter()
            try:
                result = await check.run(context)
            except Exception as error:  # plugin boundary must become an explicit failed finding
                result = CheckResult(
                    name=check.name,
                    status=CheckStatus.FAIL,
                    cause=f"check raised {type(error).__name__}: {scrub(str(error))}",
                    remediation=f"repair or disable the {check.name} diagnostic implementation",
                    required=check.required,
                )
            return result.model_copy(
                update={"duration_ms": max((time.perf_counter() - started) * 1_000, 0.0)}
            )

        return tuple(await asyncio.gather(*(execute(check) for check in self.checks)))


async def main(
    argv: Sequence[str],
    *,
    registry: CheckRegistry,
    context: DoctorContext,
    out: TextIO | None = None,
) -> int:
    """Run registered checks and return non-zero only for required failures.

    Args:
        argv: Arguments after ``doctor``.
        registry: Built-in and adapter-owned checks for this application.
        context: Resolved configuration and environment boundary.
        out: Human or JSON report destination. Defaults to stdout.

    Returns:
        ``0`` when no required check fails, ``1`` otherwise and ``2`` for command misuse.
    """
    writer = out if out is not None else sys.stdout
    try:
        parsed = _parser().parse_args(argv)
    except SystemExit:
        return MISUSED
    results = await registry.run(context, offline=parsed.offline)
    safe = tuple(_safe_result(result) for result in results)
    if parsed.json:
        writer.write(
            json.dumps(
                {
                    "healthy": not any(
                        result.required and result.status is CheckStatus.FAIL for result in safe
                    ),
                    "checks": [result.model_dump(mode="json") for result in safe],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    else:
        for result in safe:
            writer.write(f"{result.status.value:<7} {result.name}: {result.cause}\n")
            writer.write(f"        remediation: {result.remediation}\n")
    return (
        UNHEALTHY
        if any(result.required and result.status is CheckStatus.FAIL for result in safe)
        else OK
    )


def _safe_result(result: CheckResult) -> CheckResult:
    """Apply core redaction even where a third-party check forgot its boundary."""
    return result.model_copy(
        update={
            "name": scrub(result.name),
            "cause": scrub(result.cause),
            "remediation": scrub(result.remediation),
        }
    )


def _safe_endpoint(endpoint: str) -> str:
    """Render endpoint identity without userinfo, query parameters or fragments."""
    parsed = urlsplit(endpoint)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{host}{port}"
    return (
        urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        if parsed.scheme
        else scrub(endpoint)
    )


def _parser() -> argparse.ArgumentParser:
    """Build the ``tesserix-adk doctor`` command line."""
    parser = argparse.ArgumentParser(prog="tesserix-adk doctor", description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip every network probe")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    return parser
