"""Optional integrations must stay optional, and reaching past one must say so.

A base install that quietly drags in temporalio, redis and psycopg is the reason teams
vendor snippets instead of depending on a kit. These tests hold the base footprint down,
and turn the failure mode — an ImportError naming a transitive module the consumer has
never heard of — into a message naming the extra and the install command.
"""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement

from tesserix_adk.core import AdkError, MissingExtraError, require_extra
from tests.ci_config import pyproject

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tesserix_adk"

BASE_REQUIREMENTS = {"pydantic", "httpx", "opentelemetry-api"}

INTEGRATION_EXTRAS = {"mcp", "temporal", "graphiti", "redis", "postgres"}

# The importable name each extra provides, so the footprint test can prove that a base
# install of the kit reaches none of them.
EXTRA_TOP_LEVEL_MODULES = {
    "mcp": "mcp",
    "temporal": "temporalio",
    "graphiti": "graphiti_core",
    "redis": "redis",
    "postgres": "psycopg",
}

# Distinct packages a base install resolves to, counted from uv.lock. Every one of these
# lands in every consumer's image, so raising the ceiling is a reviewed decision.
TRANSITIVE_CEILING = 16

# Extras gate integrations. They may never gate anything the kit promises unconditionally,
# such as redaction or budget enforcement — those layers must not import an optional SDK.
NON_NEGOTIABLE_PACKAGES = ("core", "runtime", "guardrails", "observability")

# No release of temporalio ships this, so the miss is genuine in every extras leg —
# including the ones where the extra is installed.
ABSENT_MODULE = "temporalio.absent_in_every_release"


def _extras() -> dict[str, list[str]]:
    declared: dict[str, list[str]] = pyproject()["project"].get("optional-dependencies", {})
    return declared


def _names(requirements: list[str]) -> set[str]:
    return {Requirement(r).name for r in requirements}


def test_the_base_install_names_only_the_agreed_requirements() -> None:
    """Anything beyond these three is a cost every consumer pays, whether they use it or not."""
    assert _names(pyproject()["project"]["dependencies"]) == BASE_REQUIREMENTS


def test_no_integration_sdk_appears_in_the_base_requirements() -> None:
    """Provider and store SDKs live in an extra. Always. This is the rule the story exists for."""
    base = _names(pyproject()["project"]["dependencies"])
    for extra in INTEGRATION_EXTRAS:
        assert base.isdisjoint(_names(_extras()[extra]))


def test_the_declared_extras_are_the_agreed_set() -> None:
    assert set(_extras()) == INTEGRATION_EXTRAS | {"all"}


def test_every_extra_requirement_declares_a_floor_and_a_ceiling() -> None:
    """A floor without a ceiling ships the next major release to consumers unreviewed."""
    for extra in INTEGRATION_EXTRAS:
        for raw in _extras()[extra]:
            specifiers = {s.operator for s in Requirement(raw).specifier}
            assert specifiers & {">=", "=="}, f"{extra}: {raw} has no lower bound"
            assert specifiers & {"<", "<=", "==", "~="}, f"{extra}: {raw} has no upper bound"


def test_the_all_extra_is_a_pure_union() -> None:
    """Listing packages directly lets `all` become the tested path while the parts rot."""
    union = f"tesserix-adk[{','.join(sorted(INTEGRATION_EXTRAS))}]"
    assert _extras()["all"] == [union]


def test_every_declared_extra_has_a_known_import_name() -> None:
    """The footprint test can only prove absence of what it can name."""
    assert set(EXTRA_TOP_LEVEL_MODULES) == INTEGRATION_EXTRAS


def _locked_base_packages() -> set[str]:
    """Packages a consumer installing the kit with no extras actually receives.

    Read from `uv.lock` rather than a resolver run: no network, and it is the same
    graph CI installs. Optional-dependency edges are deliberately not followed.
    """
    with (ROOT / "uv.lock").open("rb") as handle:
        locked = {p["name"]: p for p in tomllib.load(handle)["package"]}

    seen: set[str] = set()
    queue = ["tesserix-adk"]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        queue.extend(d["name"] for d in locked[name].get("dependencies", []))
    return seen


def test_the_base_install_stays_under_the_transitive_ceiling() -> None:
    packages = _locked_base_packages()
    assert len(packages) <= TRANSITIVE_CEILING, sorted(packages)


def test_the_locked_base_install_contains_no_optional_sdk() -> None:
    locked = _locked_base_packages()
    for extra in INTEGRATION_EXTRAS:
        assert locked.isdisjoint(_names(_extras()[extra])), extra


def test_importing_the_whole_kit_reaches_no_optional_dependency() -> None:
    """The primary scenario: a base install imports the surface without touching an SDK.

    Run in a fresh interpreter so an import the rest of the suite performed cannot mask
    an eager one here, and asserted against `sys.modules` so it holds in the `all` leg
    too — where the wheels are installed but must still go untouched.
    """
    script = (
        "import importlib, pkgutil, sys; import tesserix_adk;"
        "[importlib.import_module(m.name)"
        " for m in pkgutil.walk_packages(tesserix_adk.__path__, 'tesserix_adk.')];"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    )
    imported = set(result.stdout.split())
    assert imported.isdisjoint(EXTRA_TOP_LEVEL_MODULES.values())


def test_require_extra_returns_the_module_when_it_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stand_in = ModuleType("installed_sdk")
    monkeypatch.setitem(sys.modules, "installed_sdk", stand_in)
    assert require_extra("redis", "installed_sdk") is stand_in


def test_require_extra_names_the_extra_and_the_install_command() -> None:
    """The failure scenario: an actionable message instead of a stranger's traceback."""
    with pytest.raises(MissingExtraError) as caught:
        require_extra("temporal", ABSENT_MODULE)

    message = str(caught.value)
    assert "temporal" in message
    assert "uv add 'tesserix-adk[temporal]'" in message
    assert ABSENT_MODULE in message
    assert caught.value.extra == "temporal"
    assert caught.value.install_command == "uv add 'tesserix-adk[temporal]'"


def test_a_missing_extra_is_catchable_as_either_an_adk_error_or_an_import_error() -> None:
    """Consumers already wrap optional imports in `except ImportError`; keep that working."""
    error = MissingExtraError(extra="redis", module="redis.asyncio")
    assert isinstance(error, AdkError)
    assert isinstance(error, ImportError)
    assert error.name == "redis.asyncio"


def test_a_broken_dependency_inside_an_installed_package_is_not_reported_as_a_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed SDK whose own import fails is that SDK's bug, not an uninstalled extra."""
    package = tmp_path / "brokensdk"
    package.mkdir()
    (package / "__init__.py").write_text("import definitely_absent_module\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as caught:
        require_extra("redis", "brokensdk")

    assert not isinstance(caught.value, MissingExtraError)
    assert caught.value.name == "definitely_absent_module"


def _require_extra_call_sites() -> list[tuple[Path, str]]:
    """Every `require_extra("<extra>", ...)` in the kit, as (file, extra) pairs."""
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            called = isinstance(node, ast.Call) and getattr(node.func, "id", "") == "require_extra"
            if called and isinstance(node, ast.Call) and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    sites.append((path, first.value))
    return sites


def test_every_require_extra_call_site_names_a_declared_extra() -> None:
    """A typo would print an install command that does not work."""
    for path, extra in _require_extra_call_sites():
        assert extra in _extras(), f"{path}: '{extra}' is not a declared extra"


@pytest.mark.parametrize("package", NON_NEGOTIABLE_PACKAGES)
def test_no_unconditional_layer_gates_itself_behind_an_extra(package: str) -> None:
    """Redaction and budget enforcement are promises, not integrations."""
    gated = [p for p, _ in _require_extra_call_sites() if (SRC / package) in p.parents]
    assert gated == []
