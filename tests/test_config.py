"""Configuration resolves the same way everywhere, and says where every value came from.

Each product reading its own environment variables is why an agent that is safe locally
behaves differently in the cluster. These tests pin the precedence — code, environment,
file, default — and the provenance that makes a misconfiguration diagnosable instead of
mysterious.
"""

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from tesserix_adk.core import (
    AdkConfig,
    ConfigError,
    ConfigurationError,
    load_config,
    resolve_config,
)
from tesserix_adk.core.config import ENV_PREFIX, leaf_keys, secret_keys

MINIMAL_ENV = {f"{ENV_PREFIX}PROVIDER__ENDPOINT": "http://vllm.internal:8000"}


def _load(env: dict[str, str] | None = None, **kwargs: object) -> AdkConfig:
    """Load with the one required key already supplied; extra env is merged on top."""
    return load_config(env=MINIMAL_ENV | (env or {}), **kwargs)  # type: ignore[arg-type]


def test_a_config_resolved_from_one_layer_reports_that_layer() -> None:
    resolution = resolve_config(env=MINIMAL_ENV, start=None)
    assert resolution.config.provider.endpoint == "http://vllm.internal:8000"
    assert resolution.provenance["provider.endpoint"].layer == "env"


def test_defaults_are_reported_as_defaults_not_as_absent() -> None:
    """Support needs to distinguish 'nobody set it' from 'someone set it to the default'."""
    resolution = resolve_config(env=MINIMAL_ENV, start=None)
    assert resolution.provenance["budget.max_input_tokens"].layer == "default"


def test_the_environment_beats_the_file_and_the_file_is_recorded_as_overridden(
    tmp_path: Path,
) -> None:
    """The primary scenario: same key in two layers, and both are visible afterwards."""
    (tmp_path / "adk.toml").write_text("[budget]\nmax_input_tokens = 111\n", encoding="utf-8")

    resolution = resolve_config(
        env=MINIMAL_ENV | {f"{ENV_PREFIX}BUDGET__MAX_INPUT_TOKENS": "222"}, start=tmp_path
    )

    assert resolution.config.budget.max_input_tokens == 222
    entry = resolution.provenance["budget.max_input_tokens"]
    assert entry.layer == "env"
    assert entry.overridden == (("file", "111"),)


def test_code_beats_the_environment() -> None:
    resolution = resolve_config(
        {"budget": {"max_input_tokens": 7}},
        env=MINIMAL_ENV | {f"{ENV_PREFIX}BUDGET__MAX_INPUT_TOKENS": "222"},
        start=None,
    )
    assert resolution.config.budget.max_input_tokens == 7
    assert resolution.provenance["budget.max_input_tokens"].layer == "code"
    assert resolution.provenance["budget.max_input_tokens"].overridden == (("env", "222"),)


def test_the_full_precedence_order_holds_for_one_key(tmp_path: Path) -> None:
    """Every layer sets the same key; the winner and the losers are both asserted."""
    (tmp_path / "adk.toml").write_text("[loop]\nmax_repeated_calls = 3\n", encoding="utf-8")

    resolution = resolve_config(
        {"loop": {"max_repeated_calls": 1}},
        env=MINIMAL_ENV | {f"{ENV_PREFIX}LOOP__MAX_REPEATED_CALLS": "2"},
        start=tmp_path,
    )

    entry = resolution.provenance["loop.max_repeated_calls"]
    assert (resolution.config.loop.max_repeated_calls, entry.layer) == (1, "code")
    assert entry.overridden == (("env", "2"), ("file", "3"))


def test_the_pyproject_tool_table_is_read_when_there_is_no_adk_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "app"\n\n[tool.tesserix-adk.budget]\nmax_input_tokens = 55\n',
        encoding="utf-8",
    )
    resolution = resolve_config(env=MINIMAL_ENV, start=tmp_path)
    assert resolution.config.budget.max_input_tokens == 55
    assert resolution.provenance["budget.max_input_tokens"].layer == "file"


def test_adk_toml_wins_over_a_pyproject_table_in_the_same_directory(tmp_path: Path) -> None:
    (tmp_path / "adk.toml").write_text("[budget]\nmax_input_tokens = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[tool.tesserix-adk.budget]\nmax_input_tokens = 2\n", encoding="utf-8"
    )
    assert _load(start=tmp_path).budget.max_input_tokens == 1


def test_a_pyproject_without_the_tool_table_is_not_a_config_file(tmp_path: Path) -> None:
    """Otherwise a worker started one directory down resolves a different configuration."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "app"\n', encoding="utf-8")
    (tmp_path / "adk.toml").write_text("[budget]\nmax_input_tokens = 9\n", encoding="utf-8")
    nested = tmp_path / "src" / "worker"
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text('[project]\nname = "worker"\n', encoding="utf-8")

    assert _load(start=nested).budget.max_input_tokens == 9


def test_an_explicit_path_skips_discovery(tmp_path: Path) -> None:
    (tmp_path / "adk.toml").write_text("[budget]\nmax_input_tokens = 1\n", encoding="utf-8")
    elsewhere = tmp_path / "other.toml"
    elsewhere.write_text("[budget]\nmax_input_tokens = 2\n", encoding="utf-8")

    assert _load(path=elsewhere, start=tmp_path).budget.max_input_tokens == 2


def test_an_explicit_path_that_does_not_exist_is_an_error_not_a_shrug(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        _load(path=tmp_path / "absent.toml")
    assert "absent.toml" in str(caught.value)


def test_a_missing_required_key_is_reported_rather_than_defaulted() -> None:
    """No endpoint is ever invented, so an agent cannot start half-configured."""
    with pytest.raises(ConfigError) as caught:
        load_config(env={}, start=None)

    problems = {p.key: p for p in caught.value.problems}
    assert "provider.endpoint" in problems
    assert problems["provider.endpoint"].layer is None


def test_a_secret_in_a_config_file_is_rejected_outright(tmp_path: Path) -> None:
    """Secrets arrive from the environment. A file gets committed; the environment does not."""
    (tmp_path / "adk.toml").write_text('[provider]\napi_key = "sk-live-123"\n', encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        load_config(env=MINIMAL_ENV, start=tmp_path)

    problem = next(p for p in caught.value.problems if p.key == "provider.api_key")
    assert problem.layer == "file"
    assert "sk-live-123" not in str(caught.value)


def test_every_problem_is_reported_at_once(tmp_path: Path) -> None:
    """The failure scenario: fix one thing, restart, discover the next is a wasted afternoon."""
    (tmp_path / "adk.toml").write_text('[provider]\napi_key = "sk-live-123"\n', encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        load_config(env={}, start=tmp_path)

    assert {p.key for p in caught.value.problems} >= {"provider.api_key", "provider.endpoint"}


def test_an_unknown_environment_key_is_a_typo_not_a_shrug() -> None:
    """Silently ignoring TESSERIX_ADK_REDACTON__ENABLED leaves redaction off."""
    with pytest.raises(ConfigError) as caught:
        _load({f"{ENV_PREFIX}REDACTON__ENABLED": "false"})

    problem = next(p for p in caught.value.problems if "redacton" in p.key)
    assert problem.layer == "env"
    assert "unknown" in problem.reason.lower()


def test_an_unknown_file_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "adk.toml").write_text("[budget]\nmax_tokens = 5\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        _load(start=tmp_path)
    assert next(p for p in caught.value.problems if p.key == "budget.max_tokens").layer == "file"


def test_an_unknown_code_key_is_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        _load(overrides={"budget": {"maximum_tokens": 5}})
    assert next(p for p in caught.value.problems if "maximum_tokens" in p.key).layer == "code"


def test_a_malformed_number_reports_the_literal_and_the_layer() -> None:
    with pytest.raises(ConfigError) as caught:
        _load({f"{ENV_PREFIX}BUDGET__MAX_INPUT_TOKENS": "lots"})

    problem = next(p for p in caught.value.problems if p.key == "budget.max_input_tokens")
    assert problem.layer == "env"
    assert problem.literal == "lots"


def test_a_malformed_duration_reports_the_literal_and_the_layer(tmp_path: Path) -> None:
    (tmp_path / "adk.toml").write_text('[provider]\nrequest_timeout = "soon"\n', encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        _load(start=tmp_path)

    problem = next(p for p in caught.value.problems if p.key == "provider.request_timeout")
    assert (problem.layer, problem.literal) == ("file", "soon")


def test_a_duration_accepts_seconds_and_iso8601() -> None:
    assert _load({f"{ENV_PREFIX}PROVIDER__REQUEST_TIMEOUT": "45"}).provider.request_timeout == (
        timedelta(seconds=45)
    )
    assert _load(
        {f"{ENV_PREFIX}PROVIDER__REQUEST_TIMEOUT": "PT1M30S"}
    ).provider.request_timeout == timedelta(seconds=90)


def test_a_config_error_is_a_configuration_error() -> None:
    """Consumers catch ConfigurationError to mean 'this cannot work', and should keep doing so."""
    with pytest.raises(ConfigurationError):
        load_config(env={}, start=None)


def test_the_error_message_lists_every_problem_with_its_layer() -> None:
    with pytest.raises(ConfigError) as caught:
        _load({f"{ENV_PREFIX}BUDGET__MAX_INPUT_TOKENS": "lots"})
    assert "budget.max_input_tokens" in str(caught.value)
    assert "env" in str(caught.value)


def test_a_secret_is_readable_from_the_environment() -> None:
    config = _load({f"{ENV_PREFIX}PROVIDER__API_KEY": "sk-live-123"})
    assert config.provider.api_key is not None
    assert config.provider.api_key.get_secret_value() == "sk-live-123"


@pytest.mark.parametrize(
    "render",
    [repr, str, lambda c: c.model_dump_json(), lambda c: json.dumps(c.model_dump(mode="json"))],
    ids=["repr", "str", "model_dump_json", "model_dump"],
)
def test_a_secret_never_appears_in_a_rendering_of_the_config(render: object) -> None:
    """repr and dumps are what reach logs and telemetry exporters."""
    config = _load({f"{ENV_PREFIX}PROVIDER__API_KEY": "sk-live-123"})
    assert "sk-live-123" not in render(config)  # type: ignore[operator]


def test_a_secret_value_never_appears_in_the_provenance() -> None:
    resolution = resolve_config(
        env=MINIMAL_ENV | {f"{ENV_PREFIX}PROVIDER__API_KEY": "sk-live-123"}, start=None
    )
    assert "sk-live-123" not in resolution.explain()
    assert "sk-live-123" not in repr(resolution.provenance)


def test_an_overridden_secret_is_not_disclosed_by_the_override_record() -> None:
    """A losing layer's value is shown to explain the override — except for secrets."""
    resolution = resolve_config(
        {"provider": {"api_key": SecretStr("sk-code")}},
        env=MINIMAL_ENV | {f"{ENV_PREFIX}PROVIDER__API_KEY": "sk-env"},
        start=None,
    )
    entry = resolution.provenance["provider.api_key"]
    assert entry.overridden == (("env", "**********"),)


def test_the_secret_key_set_covers_every_secret_field() -> None:
    assert secret_keys(AdkConfig) == {"provider.api_key", "stores.postgres_dsn"}


def test_the_resolved_config_is_frozen() -> None:
    """Resolved once at startup: there is no hot reload, so a change means a restart."""
    config = _load()
    with pytest.raises(ValueError, match="frozen"):
        config.budget = None  # type: ignore[assignment]


def test_explain_names_the_layer_for_every_key() -> None:
    resolution = resolve_config(env=MINIMAL_ENV, start=None)
    explained = resolution.explain()
    for key in resolution.provenance:
        assert key in explained
    assert "provider.endpoint" in explained
    assert "env" in explained


def test_every_resolved_key_has_a_provenance_entry() -> None:
    """A key nobody can attribute is a key nobody can debug."""
    resolution = resolve_config(env=MINIMAL_ENV, start=None)
    assert set(resolution.provenance) == set(leaf_keys(AdkConfig))


def test_two_processes_with_the_same_inputs_resolve_identical_config_and_provenance(
    tmp_path: Path,
) -> None:
    """Workers that disagree about configuration fail in ways nobody can reproduce."""
    (tmp_path / "adk.toml").write_text("[budget]\nmax_input_tokens = 321\n", encoding="utf-8")
    script = (
        "import json,sys;from tesserix_adk.core import resolve_config;"
        "r=resolve_config(env={'TESSERIX_ADK_PROVIDER__ENDPOINT':'http://vllm:8000'},"
        "start=sys.argv[1]);"
        "print(json.dumps([r.config.model_dump(mode='json'), r.explain()]))"
    )
    runs = [
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert '"max_input_tokens": 321' in runs[0]
