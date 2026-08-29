"""The public surface is a contract, so it is stored, diffed and reviewed like one.

Without a declared surface a consumer imports whatever it can reach and every
internal rename becomes someone else's outage. These tests make a surface change
impossible to land accidentally: it fails until the snapshot moves in the same
commit.
"""

import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest
from pydantic import BaseModel
from tools.api_surface import (
    RE_EXPORT_ALLOWLIST,
    SNAPSHOT,
    LeakError,
    collect_surface,
    find_leaks,
    is_public_module,
    main,
    public_modules,
    render,
    third_party_re_exports,
)

import tesserix_adk

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_snapshot_matches_the_current_surface() -> None:
    """The failure message is the diff; regenerate with `make api-snapshot`."""
    expected = SNAPSHOT.read_text(encoding="utf-8")
    actual = render(collect_surface())
    assert actual == expected, (
        "public API surface changed. Run `make api-snapshot`, review the diff, and "
        "record the stability decision in CHANGELOG.md in the same pull request."
    )


def test_every_public_module_declares_its_surface() -> None:
    undeclared = [m.__name__ for m in public_modules() if not hasattr(m, "__all__")]
    assert undeclared == []


@pytest.mark.parametrize("module", public_modules(), ids=lambda m: m.__name__)
def test_no_underscore_name_is_exported(module: ModuleType) -> None:
    exported = getattr(module, "__all__", [])
    assert [n for n in exported if n.startswith("_") and not n.startswith("__")] == []


@pytest.mark.parametrize("module", public_modules(), ids=lambda m: m.__name__)
def test_every_exported_name_actually_resolves(module: ModuleType) -> None:
    """A `__all__` entry that only exists under TYPE_CHECKING breaks at runtime."""
    missing = [n for n in getattr(module, "__all__", []) if not hasattr(module, n)]
    assert missing == []


def test_no_public_signature_leaks_a_vendor_or_concrete_type() -> None:
    assert find_leaks(collect_surface()) == []


def test_a_framework_name_does_not_count_as_a_vendor_type_leak() -> None:
    surface = {
        "example.export_openai_agent": ("def export_openai_agent(agent: 'object') -> 'object'")
    }

    assert find_leaks(surface) == []


def test_a_leaked_vendor_type_is_reported_with_its_symbol() -> None:
    surface = {"tesserix_adk.memory.open_store": "def open_store() -> redis.Redis"}
    leaks = find_leaks(surface)
    assert len(leaks) == 1
    assert "redis" in leaks[0]
    assert "tesserix_adk.memory.open_store" in leaks[0]


def test_a_leaked_fake_is_reported_outside_the_testing_package() -> None:
    surface = {"tesserix_adk.runtime.build": "def build() -> FakeKeyValueStore"}
    assert find_leaks(surface) != []


def test_the_testing_package_may_return_its_own_fakes() -> None:
    surface = {"tesserix_adk.testing.make": "def make() -> FakeKeyValueStore"}
    assert find_leaks(surface) == []


def test_leak_error_names_every_offender_at_once() -> None:
    err = LeakError(("a leaks redis", "b leaks httpx"))
    assert "redis" in str(err)
    assert "httpx" in str(err)


def test_the_experimental_namespace_is_outside_the_snapshot() -> None:
    """Experimental carries no stability promise, so pinning it would be a lie."""
    assert not any(name.startswith("tesserix_adk.experimental") for name in collect_surface())
    assert any(
        m.name == "tesserix_adk.experimental"
        for m in pkgutil.iter_modules(tesserix_adk.__path__, "tesserix_adk.")
    )


def _module(name: str, **members: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def test_every_re_exported_third_party_name_is_listed_deliberately() -> None:
    """A re-exported vendor type makes that vendor's release cadence our problem."""
    assert third_party_re_exports() <= RE_EXPORT_ALLOWLIST


def test_a_re_exported_foreign_symbol_is_detected() -> None:
    module = _module("tesserix_adk.memory", __all__=["BaseModel"], BaseModel=BaseModel)
    assert third_party_re_exports([module]) == {"tesserix_adk.memory.BaseModel"}


def test_a_type_alias_is_not_a_re_export() -> None:
    """`Layer = Literal[...]` is defined here; typing is not a vendor we take releases from."""
    alias = Literal["a", "b"]
    module = _module("tesserix_adk.memory", __all__=["Kind"], Kind=alias)
    assert third_party_re_exports([module]) == set()


def test_a_type_alias_is_described_by_its_members_not_as_a_function() -> None:
    """`Layer` is a closed set of strings; the snapshot has to show which strings."""
    module = _module("tesserix_adk.memory", __all__=["Kind"], Kind=Literal["a", "b"])
    described = "Kind = typing.Literal['a', 'b']"
    assert collect_surface([module]) == {"tesserix_adk.memory.Kind": described}


def test_vendor_names_in_a_closed_string_alias_are_values_not_type_leaks() -> None:
    surface = {"tesserix_adk.core.Provider": "Provider = typing.Literal['openai', 'anthropic']"}
    assert find_leaks(surface) == []


def test_a_name_promised_but_absent_is_not_recorded_as_surface() -> None:
    """`__all__` can outrun the code during a refactor; the snapshot must not lie."""
    module = _module("tesserix_adk.memory", __all__=["gone"])
    assert collect_surface([module]) == {}


@pytest.mark.parametrize(
    ("name", "public"),
    [
        ("tesserix_adk.core", True),
        ("tesserix_adk.core.protocols", True),
        ("tesserix_adk.core._internal", False),
        ("tesserix_adk.experimental", False),
        ("tesserix_adk.experimental.router", False),
    ],
)
def test_module_visibility_rules(name: str, public: bool) -> None:
    assert is_public_module(name) is public


def test_the_check_passes_against_the_committed_snapshot() -> None:
    assert main([]) == 0


def test_the_check_fails_and_prints_a_diff_when_a_symbol_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point: a surface change is red until the snapshot moves with it."""
    stale = tmp_path / "api-surface.txt"
    stale.write_text("tesserix_adk.core.Renamed :: class Renamed(Exception)\n", encoding="utf-8")
    monkeypatch.setattr("tools.api_surface.SNAPSHOT", stale)

    assert main([]) == 1

    stderr = capsys.readouterr().err
    assert "-tesserix_adk.core.Renamed" in stderr
    assert "+tesserix_adk.core.AdkError" in stderr
    assert "make api-snapshot" in stderr


def test_write_regenerates_the_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "nested" / "api-surface.txt"
    monkeypatch.setattr("tools.api_surface.SNAPSHOT", target)

    assert main(["--write"]) == 0
    assert target.read_text(encoding="utf-8") == render(collect_surface())


def test_a_missing_snapshot_is_a_failure_not_a_silent_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tools.api_surface.SNAPSHOT", tmp_path / "absent.txt")
    assert main([]) == 1


def test_the_check_refuses_to_write_a_snapshot_that_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leak must not be recordable as the new normal."""
    monkeypatch.setattr(
        "tools.api_surface.collect_surface",
        lambda: {"tesserix_adk.memory.open_store": "def open_store() -> redis.Redis"},
    )
    with pytest.raises(LeakError, match="redis"):
        main(["--write"])


def test_the_snapshot_is_committed() -> None:
    assert SNAPSHOT.exists(), "run `make api-snapshot` and commit docs/api-surface.txt"
    assert SNAPSHOT.read_text(encoding="utf-8").endswith("\n")
