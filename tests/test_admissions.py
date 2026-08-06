"""What is allowed into the dependency surface every consuming product inherits.

A package added here is added to every product that installs the kit, and none of those
teams reviewed it. So a runtime dependency needs a recorded decision naming the
alternative that was rejected, and the graph the lock resolves to is itself committed —
otherwise the base install grows through an innocuous-looking version bump that nobody
reads as a dependency change.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pytest
from tools import admissions
from tools.lockfile import Graph

from tests.ci_config import ROOT, ci_jobs, ci_run_steps

if TYPE_CHECKING:
    from pathlib import Path

TODAY = dt.date(2026, 8, 6)
JOB = "admissions"

RECORD = """
package = "httpx"
profile = "base"
decided = 2026-07-01
owner = "@sam123ben"
need = "An async HTTP client for every provider transport the kit ships."
alternatives = '''
urllib and http.client are synchronous, and the kit's transport is async end to end;
writing an async client over asyncio streams is a maintenance burden nobody wants.
'''
maintenance = "Active: monthly releases, several maintainers, part of the encode stack."
licence = "BSD-3-Clause"
transitive = 6
native_build = false
security_history = "No unfixed advisories; past advisories fixed inside a week."
review_by = 2027-07-01
"""


def lock(runtime: tuple[str, ...] = ("httpx",), **labels: tuple[str, ...]) -> Graph:
    packages: list[dict[str, Any]] = [
        {
            "name": "tesserix-adk",
            "dependencies": [{"name": name} for name in runtime],
            "optional-dependencies": {
                extra.removeprefix("extra_"): [{"name": name} for name in names]
                for extra, names in labels.items()
                if extra.startswith("extra_")
            },
            "dev-dependencies": {
                group.removeprefix("group_"): [{"name": name} for name in names]
                for group, names in labels.items()
                if group.startswith("group_")
            },
        }
    ]
    named = set(runtime)
    for names in labels.values():
        named.update(names)
    packages += [{"name": name, "version": "1.0", "dependencies": []} for name in sorted(named)]
    return Graph.from_lock({"package": packages})


def write(tmp_path: Path, body: str = RECORD, name: str = "httpx.toml") -> Path:
    directory = tmp_path / "admissions"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(body, encoding="utf-8")
    return directory


def check(
    tmp_path: Path,
    *,
    graph: Graph | None = None,
    body: str = RECORD,
    inventory: dict[str, str] | None = None,
    direct: dict[str, list[str]] | None = None,
) -> list[str]:
    graph = graph or lock()
    return admissions.violations(
        graph,
        records=admissions.records(write(tmp_path, body)),
        recorded=admissions.inventory(graph) if inventory is None else inventory,
        direct=direct or {"": ["httpx>=0.27"]},
        today=TODAY,
    )


class TestADecisionIsRequired:
    def test_a_recorded_runtime_dependency_passes(self, tmp_path: Path) -> None:
        assert check(tmp_path) == []

    def test_an_unrecorded_runtime_dependency_is_a_violation(self, tmp_path: Path) -> None:
        """Every consuming product inherits it without having reviewed it."""
        graph = lock(("httpx", "orjson"))
        found = check(tmp_path, direct={"": ["httpx>=0.27", "orjson>=3"]}, graph=graph)
        assert any("orjson" in violation for violation in found)

    def test_the_violation_names_what_the_addition_drags_in(self, tmp_path: Path) -> None:
        """A one-line requirement can be twenty packages, which is the number that matters."""
        graph = lock(("httpx", "orjson"))
        found = check(tmp_path, direct={"": ["httpx>=0.27", "orjson>=3"]}, graph=graph)
        assert any("every consumer" in violation for violation in found)

    def test_a_development_only_dependency_needs_no_decision(self, tmp_path: Path) -> None:
        """It is never in a consumer's resolution, so the bar is the maintainer's own."""
        graph = lock(("httpx",), group_dev=("ruff",))
        assert check(tmp_path, graph=graph, inventory=admissions.inventory(graph)) == []


class TestTheGraphIsCommitted:
    def test_a_package_the_lock_grew_since_the_last_review_is_a_violation(
        self, tmp_path: Path
    ) -> None:
        """A version bump that drags in a new component is still a dependency change."""
        graph = lock(("httpx", "sniffio"))
        found = check(tmp_path, graph=graph, inventory={"httpx": "runtime"})
        assert any("sniffio" in violation for violation in found)

    def test_a_package_that_left_the_graph_is_a_violation(self, tmp_path: Path) -> None:
        """A stale inventory stops being the thing a reviewer can read the diff of."""
        found = check(tmp_path, inventory={"httpx": "runtime", "departed": "runtime"})
        assert any("departed" in violation for violation in found)

    def test_a_package_that_changed_profile_is_a_violation(self, tmp_path: Path) -> None:
        """Reaching base from an extra is the creep this gate exists to catch."""
        found = check(tmp_path, inventory={"httpx": "extra:mcp"})
        assert any("httpx" in violation for violation in found)

    def test_development_packages_are_not_in_the_inventory(self) -> None:
        graph = lock(("httpx",), group_dev=("ruff",))
        assert "ruff" not in admissions.inventory(graph)


class TestPreferenceOrder:
    def test_an_integration_sdk_in_the_base_install_is_a_violation(self, tmp_path: Path) -> None:
        """Provider and store SDKs live behind an extra and behind a protocol. Always."""
        graph = lock(("httpx", "mcp"))
        found = check(tmp_path, graph=graph, direct={"": ["httpx>=0.27", "mcp>=1.9"]})
        assert any("mcp" in violation and "extra" in violation for violation in found)

    def test_an_integration_sdk_behind_its_extra_is_fine(self, tmp_path: Path) -> None:
        graph = lock(("httpx",), extra_mcp=("mcp",))
        found = check(tmp_path, graph=graph, direct={"": ["httpx>=0.27"], "mcp": ["mcp>=1.9"]})
        assert not any("behind an extra" in violation for violation in found)


class TestApprovalExpires:
    def test_a_record_past_its_review_date_is_a_violation(self, tmp_path: Path) -> None:
        """An approved dependency that goes unmaintained stays approved until this fires."""
        expired = RECORD.replace("review_by = 2027-07-01", "review_by = 2026-01-01")
        found = check(tmp_path, body=expired)
        assert any("re-review" in violation for violation in found)


class TestRecords:
    def test_a_record_missing_the_rejected_alternative_is_refused(self, tmp_path: Path) -> None:
        """Without it the record says a package was added, not that a choice was made."""
        body = RECORD.replace("alternatives = '''", "unused = '''")
        with pytest.raises(admissions.AdmissionError, match="alternatives"):
            admissions.records(write(tmp_path, body))

    def test_a_record_missing_its_owner_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(admissions.AdmissionError, match="owner"):
            admissions.records(write(tmp_path, RECORD.replace('owner = "@sam123ben"', "")))

    def test_a_record_with_a_field_nobody_recognises_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(admissions.AdmissionError, match="licence_"):
            admissions.records(write(tmp_path, RECORD + 'licence_ = "MIT"\n'))

    def test_a_record_that_is_not_valid_toml_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(admissions.AdmissionError, match="TOML"):
            admissions.records(write(tmp_path, "package = \n"))

    def test_a_record_naming_a_profile_that_does_not_exist_is_refused(self, tmp_path: Path) -> None:
        body = RECORD.replace('profile = "base"', 'profile = "somewhere"')
        with pytest.raises(admissions.AdmissionError, match="somewhere"):
            admissions.records(write(tmp_path, body))

    def test_a_record_for_a_package_nothing_depends_on_is_a_violation(self, tmp_path: Path) -> None:
        """A dropped dependency leaves an approval that reads as current and is not."""
        found = check(tmp_path, direct={"": []})
        assert any("httpx" in violation and "no longer" in violation for violation in found)

    def test_a_vendored_record_needs_no_matching_requirement(self, tmp_path: Path) -> None:
        """The point of vendoring is that there is no requirement to match it against."""
        body = RECORD.replace('profile = "base"', 'profile = "vendored"')
        found = check(tmp_path, body=body, direct={"": []})
        assert not any("no longer" in violation for violation in found)

    def test_a_review_date_that_is_not_a_date_is_refused(self, tmp_path: Path) -> None:
        """A quoted date is a string that no comparison ever fires on."""
        body = RECORD.replace("review_by = 2027-07-01", 'review_by = "soon"')
        with pytest.raises(admissions.AdmissionError, match="review_by"):
            admissions.records(write(tmp_path, body))

    def test_vendoring_is_a_recordable_outcome(self, tmp_path: Path) -> None:
        """Copying forty lines in, with the licence, beats inheriting a package for them."""
        body = RECORD.replace('profile = "base"', 'profile = "vendored"')
        assert admissions.records(write(tmp_path, body))[0].profile == "vendored"


class TestTheRepositoryItself:
    def test_every_runtime_dependency_this_kit_publishes_is_recorded(self) -> None:
        assert admissions.main([]) == 0

    def test_the_committed_inventory_matches_the_lock(self) -> None:
        assert admissions.load_inventory() == admissions.inventory(admissions.graph())


class TestReadingTheInventory:
    def test_a_missing_inventory_says_how_to_produce_one(self, tmp_path: Path) -> None:
        with pytest.raises(admissions.AdmissionError, match="make admissions"):
            admissions.load_inventory(tmp_path / "absent.toml")

    def test_an_inventory_that_is_not_valid_toml_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "inventory.toml"
        path.write_text('"httpx" = \n', encoding="utf-8")
        with pytest.raises(admissions.AdmissionError, match="TOML"):
            admissions.load_inventory(path)


class TestDocumentation:
    def test_the_preference_order_is_written_down(self) -> None:
        """A contributor decides at the point of adding, not at review."""
        page = (ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8").lower()
        for step in ("standard library", "existing dependency", "optional extra", "vendor"):
            assert step in page

    def test_the_questions_a_record_must_answer_are_written_down(self) -> None:
        page = (ROOT / "docs" / "dependencies.md").read_text(encoding="utf-8").lower()
        for field in ("licence", "maintenance", "transitive", "review"):
            assert field in page


class TestTheGate:
    def test_ci_runs_the_admission_check_on_every_pull_request(self) -> None:
        """The lockfile diff is reviewed when it lands, not on the next nightly."""
        assert "tools.admissions" in " ".join(ci_run_steps(JOB))

    def test_the_gate_needs_the_lockfile_to_be_current_first(self) -> None:
        """Checking a stale lock reviews a graph nobody is about to install."""
        assert ci_jobs()[JOB]["needs"] == "lockfile"

    def test_the_lowest_direct_job_does_not_check_the_committed_inventory(self) -> None:
        """That job re-resolves the lock on purpose, so the two describe different graphs."""
        steps = " ".join(ci_run_steps("lowest-direct"))
        assert "--deselect tests/test_admissions.py::TestTheRepositoryItself" in steps


class TestCommandLine:
    def test_a_violation_fails_the_job(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(admissions, "load_inventory", lambda: {})
        assert admissions.main([]) == 1
        assert "make admissions" in capsys.readouterr().out

    def test_write_regenerates_the_inventory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / "inventory.toml"
        monkeypatch.setattr(admissions, "INVENTORY", path)
        assert admissions.main(["--write"]) == 0
        assert "httpx" in path.read_text(encoding="utf-8")
