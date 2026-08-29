"""Configuration explains precedence and environment diagnosis gives concrete remedies."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from tesserix_adk.cli.config_command import main as config_main
from tesserix_adk.cli.doctor import (
    CheckRegistry,
    CredentialPresenceCheck,
    DiagnosticCheck,
    DoctorContext,
    ProbeObservation,
)
from tesserix_adk.cli.doctor import (
    main as doctor_main,
)
from tesserix_adk.core import AdkConfig, load_config

if TYPE_CHECKING:
    from pathlib import Path


def configured() -> AdkConfig:
    """One minimal configuration without reading the process environment."""
    return load_config(
        {"provider": {"endpoint": "https://provider.example.invalid"}},
        env={},
        start=None,
    )


def test_config_show_names_the_winning_source_and_never_prints_a_secret() -> None:
    output = io.StringIO()
    credential_value = "-".join(("credential", "fixture", "value"))

    code = config_main(
        ["show", "--json"],
        overrides={"provider": {"endpoint": "https://code.example.invalid"}},
        env={
            "TESSERIX_ADK_PROVIDER__ENDPOINT": "https://env.example.invalid",
            "TESSERIX_ADK_PROVIDER__API_KEY": credential_value,
        },
        start=None,
        out=output,
    )

    assert code == 0
    document = json.loads(output.getvalue())
    assert document["provider.endpoint"] == {
        "source": "code",
        "value": "https://code.example.invalid",
        "overridden": [{"source": "env", "value": "https://env.example.invalid"}],
    }
    assert document["provider.api_key"]["value"] == "**********"
    assert credential_value not in output.getvalue()


def test_config_validate_reports_every_problem_and_the_bad_file_location(tmp_path: Path) -> None:
    config = tmp_path / "adk.toml"
    config.write_text(
        '[provider]\nendpoint = "https://example.invalid"\nunknown = true\n'
        "[budget]\nmax_model_calls = -1\n",
        encoding="utf-8",
    )
    output = io.StringIO()

    assert config_main(["validate", "--config", str(config)], env={}, out=output) == 1

    assert "provider.unknown" in output.getvalue()
    assert "budget.max_model_calls" in output.getvalue()

    config.write_text('[provider\nendpoint = "broken"\n', encoding="utf-8")
    output = io.StringIO()
    assert config_main(["validate", "--config", str(config)], env={}, out=output) == 1
    assert str(config) in output.getvalue()
    assert "line 1" in output.getvalue().lower()


async def test_doctor_reports_missing_credential_and_unreachable_memory_together() -> None:
    credential_variable = "GROQ_API_KEY"

    async def unreachable(_context: DoctorContext) -> ProbeObservation:
        return ProbeObservation.failed("connection refused after 0.2s")

    registry = CheckRegistry(
        (
            CredentialPresenceCheck(credential_variable),
            DiagnosticCheck(
                name="memory",
                endpoint="redis://memory.internal:6379/0",
                probe=unreachable,
                remediation="start the Redis adapter or correct TESSERIX_ADK_STORES__REDIS_URL",
            ),
        )
    )
    output = io.StringIO()

    code = await doctor_main(
        [],
        registry=registry,
        context=DoctorContext(config=configured(), environ={}),
        out=output,
    )

    assert code == 1
    assert credential_variable in output.getvalue()
    assert "redis://memory.internal:6379/0" in output.getvalue()
    assert "connection refused" in output.getvalue()


async def test_present_but_unauthorised_is_not_reported_as_missing_or_generic() -> None:
    credential_value = "sk-live-super-secret-value"

    async def unauthorised(_context: DoctorContext) -> ProbeObservation:
        return ProbeObservation.failed("present but unauthorised (provider status 401)")

    registry = CheckRegistry(
        (
            CredentialPresenceCheck("OPENROUTER_API_KEY"),
            DiagnosticCheck(
                name="provider",
                endpoint="https://openrouter.ai/api/v1",
                probe=unauthorised,
                remediation="verify model entitlement and replace OPENROUTER_API_KEY",
            ),
        )
    )
    output = io.StringIO()

    code = await doctor_main(
        ["--json"],
        registry=registry,
        context=DoctorContext(
            config=configured(),
            environ={"OPENROUTER_API_KEY": credential_value},
        ),
        out=output,
    )

    assert code == 1
    document = json.loads(output.getvalue())
    assert document["checks"][0]["status"] == "pass"
    assert "present (value not displayed)" in document["checks"][0]["cause"]
    assert "present but unauthorised" in document["checks"][1]["cause"]
    assert credential_value not in output.getvalue()


async def test_offline_skips_network_checks_without_hiding_local_failures() -> None:
    called = 0

    async def network(_context: DoctorContext) -> ProbeObservation:
        nonlocal called
        called += 1
        return ProbeObservation.failed("unreachable")

    registry = CheckRegistry(
        (
            CredentialPresenceCheck("OPTIONAL_KEY", required=False),
            DiagnosticCheck(
                name="mcp-handshake",
                endpoint="https://mcp.example.invalid",
                probe=network,
                remediation="check the MCP endpoint",
            ),
        )
    )
    output = io.StringIO()

    code = await doctor_main(
        ["--offline"],
        registry=registry,
        context=DoctorContext(config=configured(), environ={}),
        out=output,
    )

    assert code == 0
    assert called == 0
    assert "skipped" in output.getvalue()
    assert "offline" in output.getvalue()
