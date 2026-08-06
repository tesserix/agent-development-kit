"""The typing guarantee, enforced rather than claimed.

`mypy --strict` proves the code the checker can see. It cannot prove that a public symbol
is annotated at all, that an `Any` in an exported signature was a decision rather than a
slip, or that a `type: ignore` was reviewed by anyone. This gate holds the escape hatches
in one reviewed inventory: every ignore and every public `Any` names a reason, an owner and
a date, and an entry that no longer matches the code is as much a failure as an unlisted
hatch — a stale record is how an inventory stops describing anything.
"""

from __future__ import annotations

import datetime as dt
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tools import typing_gate as gate
from tools.api_surface import collect_surface, public_modules
from tools.typing_gate import (
    ALLOWED_ANY_KINDS,
    POLICY,
    Escape,
    TypingPolicyError,
    annotation_gaps,
    ignores_in_source,
    load_policy,
    main,
    public_anys,
    violations,
)

ROOT = Path(__file__).resolve().parents[1]


def policy() -> dict[str, object]:
    return load_policy(POLICY)


class TestThePolicyIsReadable:
    def test_the_policy_file_is_committed(self) -> None:
        assert POLICY.is_file()

    def test_it_names_the_people_who_may_own_an_entry(self) -> None:
        assert policy()["owners"]

    def test_it_pins_the_checker_it_speaks_for(self) -> None:
        """A rule that tightens in a new mypy release must break on a chosen upgrade."""
        assert policy()["checker"]

    def test_an_unreadable_policy_is_an_error_rather_than_an_empty_one(
        self, tmp_path: Path
    ) -> None:
        """Silently reading no entries would turn the gate off without saying so."""
        with pytest.raises(TypingPolicyError):
            load_policy(tmp_path / "absent.toml")


class TestEveryIgnoreInSourceIsListed:
    def test_the_source_still_contains_ignores_to_govern(self) -> None:
        assert ignores_in_source()

    def test_every_one_of_them_is_declared(self) -> None:
        assert [v for v in violations() if "not declared" in v] == []

    def test_an_undeclared_ignore_fails_the_gate(self) -> None:
        found = (Escape(path="src/tesserix_adk/core/agent.py", code="misc", line=1),)
        assert any("not declared" in v for v in violations(found=found))

    def test_a_declared_ignore_that_no_longer_exists_fails_too(self) -> None:
        """A record outliving its code is how an inventory stops describing anything."""
        assert any("no longer" in v for v in violations(found=()))


class TestEveryPublicAnyIsListed:
    def test_the_surface_still_contains_anys_to_govern(self) -> None:
        assert public_anys(collect_surface())

    def test_every_one_of_them_is_declared(self) -> None:
        assert [v for v in violations() if "undeclared Any" in v] == []

    def test_an_undeclared_any_fails_the_gate(self) -> None:
        assert any("undeclared Any" in v for v in violations(anys=("tesserix_adk.core.Agent",)))

    def test_a_declared_any_that_is_gone_fails_too(self) -> None:
        assert any("no longer" in v for v in violations(anys=()))

    def test_every_declared_any_says_which_kind_it_is(self) -> None:
        kinds = {entry["kind"] for entry in policy()["any"]}  # type: ignore[attr-defined]
        assert kinds <= ALLOWED_ANY_KINDS

    def test_a_provisional_any_names_the_story_that_removes_it(self) -> None:
        """Provisional means someone is coming back; without an issue nobody is."""
        provisional = [e for e in policy()["any"] if e["kind"] == "provisional"]  # type: ignore[attr-defined]
        assert provisional
        assert all(e["removed_by"].startswith("#") for e in provisional)


class TestEveryPublicNameIsAnnotated:
    def test_nothing_exported_is_missing_an_annotation(self) -> None:
        assert annotation_gaps() == []

    def test_the_gate_reports_the_symbol_and_the_parameter(self) -> None:
        """A gap named only by module is a gap the author has to go looking for."""
        assert all(":" in gap for gap in annotation_gaps(check=_unannotated))

    def test_a_callable_with_no_readable_signature_is_passed_over(self) -> None:
        """A C callable cannot be inspected; reporting it as a gap would be unactionable."""
        assert annotation_gaps(modules=[_module("math.log", math.log)]) == []


class TestAnEntryStaysAccountable:
    def test_every_entry_has_an_owner_the_policy_recognises(self) -> None:
        assert [v for v in violations() if "unknown owner" in v] == []

    def test_an_entry_owned_by_nobody_is_flagged_for_reassignment(self) -> None:
        """An owner who has left is a record nobody is going to revisit."""
        stale = {"path": "x", "code": "misc", "reason": "r", "owner": "@gone", "review_by": _AHEAD}
        assert any("unknown owner" in v for v in violations(ignores=(stale,)))

    def test_no_entry_is_overdue_for_review(self) -> None:
        assert [v for v in violations() if "overdue" in v] == []

    def test_an_overdue_entry_fails_the_gate(self) -> None:
        past = {"path": "x", "code": "misc", "reason": "r", "owner": _OWNER, "review_by": _BEHIND}
        assert any("overdue" in v for v in violations(ignores=(past,)))

    def test_an_entry_without_a_reason_fails_the_gate(self) -> None:
        thin = {"path": "x", "code": "misc", "owner": _OWNER, "review_by": _AHEAD}
        assert any("reason" in v for v in violations(ignores=(thin,)))


class TestTheGateReportsWhatItFound:
    def test_a_clean_tree_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main() == 0
        assert capsys.readouterr().err == ""

    def test_a_violation_exits_nonzero_and_says_what_to_do(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An exit code alone leaves the author to rediscover which hatch was undeclared."""
        monkeypatch.setattr(gate, "violations", lambda: ["core/agent.py:1 ignores [misc]"])
        assert main() == 1
        reported = capsys.readouterr().err
        assert "core/agent.py:1" in reported
        assert "typing-policy.toml" in reported


class TestAThirdPartyBoundaryCannotLeakAny:
    def test_the_checker_refuses_a_type_that_came_from_an_unfollowed_import(self) -> None:
        """A dependency that drops its stubs must fail here, not silently widen the surface."""
        assert _mypy_config()["disallow_any_unimported"] is True

    def test_no_override_readmits_an_untyped_dependency_wholesale(self) -> None:
        """`ignore_missing_imports` is how an SDK's `Any` gets back in without a shim."""
        for override in _mypy_config().get("overrides", []):
            assert not override.get("ignore_missing_imports"), override.get("module")
            assert not override.get("follow_untyped_imports"), override.get("module")

    def test_the_policy_speaks_for_the_checker_the_project_pins(self) -> None:
        """A mypy that floats reclassifies ignores on a day nobody chose."""
        pinned = [r for r in _dev_requirements() if r.startswith("mypy")]
        assert pinned == [policy()["checker"]]


class TestTheGateRunsWhereItIsNeeded:
    def test_the_kit_imports_without_any_optional_extra(self) -> None:
        """A consumer who installed none of them must still be able to run the gate."""
        modules = [m.__name__ for m in public_modules()]
        program = "import importlib, sys\n" + "".join(
            f"importlib.import_module({name!r})\n" for name in modules
        )
        finished = subprocess.run(  # noqa: S603
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )
        assert finished.returncode == 0, finished.stderr

    @pytest.mark.parametrize("module", [m.__name__ for m in public_modules()])
    def test_each_public_module_imports_on_its_own(self, module: str) -> None:
        """A `TYPE_CHECKING` import promoted to runtime shows up as a cycle here first."""
        finished = subprocess.run(  # noqa: S603
            [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
        )
        assert finished.returncode == 0, finished.stderr


_OWNER = "@sam123ben"
_AHEAD = dt.date.today() + dt.timedelta(days=365)
_BEHIND = dt.date.today() - dt.timedelta(days=1)


def _unannotated(_: object) -> bool:
    return True


def _module(name: str, member: object) -> ModuleType:
    made = ModuleType(name)
    made.__all__ = ["member"]  # type: ignore[attr-defined]
    made.member = member  # type: ignore[attr-defined]
    return made


def _pyproject() -> dict[str, Any]:
    loaded: dict[str, Any] = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return loaded


def _mypy_config() -> dict[str, Any]:
    config: dict[str, Any] = _pyproject()["tool"]["mypy"]
    return config


def _dev_requirements() -> list[str]:
    groups: dict[str, list[str]] = _pyproject()["dependency-groups"]
    return [r for group in groups.values() for r in group]
