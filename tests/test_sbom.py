"""What a consuming reviewer can answer without installing the kit.

The question that arrives on the day of a widely reported vulnerability is "do we ship
that package". Answering it by installing the kit and looking is not an answer a security
review accepts, and it is not available at all for a version released two years ago.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

import pytest
from tools import licences, sbom

if TYPE_CHECKING:
    from pathlib import Path

BUILT_AT = dt.datetime(2026, 8, 6, 12, 0, tzinfo=dt.UTC)
WHEEL = "redis-6.4.0-cp313-cp313"

LOCK: dict[str, Any] = {
    "package": [
        {
            "name": "tesserix-adk",
            "dependencies": [{"name": "httpx"}],
            "optional-dependencies": {"redis": [{"name": "redis"}]},
            "dev-dependencies": {"test": [{"name": "pytest"}]},
        },
        {
            "name": "httpx",
            "version": "0.28.1",
            "dependencies": [{"name": "certifi"}],
            "sdist": {"url": "https://f/httpx-0.28.1.tar.gz", "hash": "sha256:aaa"},
            "wheels": [{"url": "https://f/httpx-0.28.1-py3-none-any.whl", "hash": "sha256:bbb"}],
        },
        {"name": "certifi", "version": "2024.8.30"},
        {
            "name": "redis",
            "version": "6.4.0",
            "wheels": [
                {"url": f"https://f/{WHEEL}-manylinux_x86_64.whl", "hash": "sha256:c"},
                {"url": f"https://f/{WHEEL}-macosx_11_0_arm64.whl", "hash": "sha256:d"},
            ],
        },
        {"name": "pytest", "version": "8.3.4"},
    ]
}

LICENCES = {"httpx": "BSD-3-Clause", "certifi": "MPL-2.0", "redis": "MIT"}


def build(lock: dict[str, Any] | None = None) -> dict[str, Any]:
    return sbom.build(
        lock if lock is not None else LOCK,
        version="0.3.0",
        licence_of=LICENCES.get,
        built_at=BUILT_AT,
    )


def component(document: dict[str, Any], name: str) -> dict[str, Any]:
    found: dict[str, Any] = next(item for item in document["components"] if item["name"] == name)
    return found


def properties(item: dict[str, Any], key: str) -> list[str]:
    return [entry["value"] for entry in item.get("properties", []) if entry["name"] == key]


class TestFormat:
    def test_the_document_is_cyclonedx(self) -> None:
        """A standard interchange format, so a reviewer's existing tooling reads it."""
        document = build()
        assert document["bomFormat"] == "CycloneDX"
        assert document["specVersion"] == "1.6"

    def test_the_document_names_the_release_it_describes(self) -> None:
        assert build()["metadata"]["component"]["version"] == "0.3.0"

    def test_the_build_time_is_recorded(self) -> None:
        assert build()["metadata"]["timestamp"] == "2026-08-06T12:00:00+00:00"

    def test_the_document_is_serialisable_as_json(self) -> None:
        assert json.loads(json.dumps(build()))["bomFormat"] == "CycloneDX"

    def test_components_are_ordered_so_two_builds_of_one_lock_agree(self) -> None:
        names = [item["name"] for item in build()["components"]]
        assert names == sorted(names)


class TestWhatIsIncluded:
    def test_a_runtime_dependency_is_a_component(self) -> None:
        assert component(build(), "httpx")["version"] == "0.28.1"

    def test_a_transitive_dependency_is_a_component_too(self) -> None:
        """The vulnerability is usually four levels down, not in what we chose."""
        assert component(build(), "certifi")["version"] == "2024.8.30"

    def test_a_dependency_of_an_extra_is_a_component(self) -> None:
        assert component(build(), "redis")["version"] == "6.4.0"

    def test_a_development_only_dependency_is_excluded(self) -> None:
        """It is not part of a consumer's exposure, so listing it overstates the surface."""
        assert [item for item in build()["components"] if item["name"] == "pytest"] == []


class TestProfiles:
    def test_a_base_component_says_it_arrives_with_the_base_install(self) -> None:
        assert properties(component(build(), "httpx"), "tesserix:profile") == ["base"]

    def test_an_extra_component_names_the_extra_that_pulls_it_in(self) -> None:
        assert properties(component(build(), "redis"), "tesserix:profile") == ["extra:redis"]


class TestComponentDetail:
    def test_a_component_carries_its_purl(self) -> None:
        assert component(build(), "httpx")["purl"] == "pkg:pypi/httpx@0.28.1"

    def test_a_component_carries_its_licence(self) -> None:
        assert component(build(), "httpx")["licenses"] == [{"expression": "BSD-3-Clause"}]

    def test_a_component_with_no_readable_licence_says_so_rather_than_omitting_it(self) -> None:
        """An absent field reads as "no obligation"; NOASSERTION reads as "go and look"."""
        unknown = sbom.build(LOCK, version="0.3.0", licence_of=lambda _: None, built_at=BUILT_AT)
        assert component(unknown, "httpx")["licenses"] == [{"expression": "NOASSERTION"}]

    def test_a_legacy_licence_string_is_recorded_as_an_spdx_identifier(self) -> None:
        """A reviewer's tooling matches identifiers, not the prose a 2016 setup.py wrote."""
        document = sbom.build(
            LOCK, version="0.3.0", licence_of=lambda _: "3-Clause BSD License", built_at=BUILT_AT
        )
        assert component(document, "httpx")["licenses"] == [{"expression": "BSD-3-Clause"}]

    def test_the_source_artefact_hash_is_recorded(self) -> None:
        assert component(build(), "httpx")["hashes"] == [{"alg": "SHA-256", "content": "aaa"}]

    def test_every_built_artefact_is_listed_with_its_own_hash(self) -> None:
        """One hash per file, because that is what a consumer actually verifies."""
        listed = properties(component(build(), "redis"), "tesserix:artefact")
        assert len(listed) == 2
        assert any("manylinux_x86_64" in entry and "sha256:c" in entry for entry in listed)

    def test_a_platform_specific_component_is_recorded_per_platform(self) -> None:
        """Not just the platform the release machine happened to build on."""
        assert properties(component(build(), "redis"), "tesserix:platform") == [
            "macosx_11_0_arm64",
            "manylinux_x86_64",
        ]

    def test_a_pure_python_component_says_it_runs_anywhere(self) -> None:
        assert properties(component(build(), "httpx"), "tesserix:platform") == ["any"]


class TestDiff:
    def _older(self) -> dict[str, Any]:
        return sbom.build(
            {
                "package": [
                    {"name": "tesserix-adk", "dependencies": [{"name": "httpx"}]},
                    {"name": "httpx", "version": "0.27.0", "dependencies": [{"name": "gone"}]},
                    {"name": "gone", "version": "1.0"},
                ]
            },
            version="0.2.0",
            licence_of=LICENCES.get,
            built_at=BUILT_AT,
        )

    def test_a_new_component_is_reported_as_added(self) -> None:
        difference = sbom.diff(self._older(), build())
        assert "redis" in difference.added

    def test_a_dropped_component_is_reported_as_removed(self) -> None:
        assert "gone" in sbom.diff(self._older(), build()).removed

    def test_a_version_change_is_reported_with_both_versions(self) -> None:
        changed = sbom.diff(self._older(), build()).changed
        assert changed["httpx"] == ("0.27.0", "0.28.1")

    def test_dependency_growth_is_stated_in_the_summary(self) -> None:
        """The number nobody watches until it is 400."""
        rendered = sbom.render(sbom.diff(self._older(), build()))
        assert "3 components" in rendered
        assert "2 added" in rendered

    def test_an_unchanged_graph_says_so(self) -> None:
        assert "no dependency changes" in sbom.render(sbom.diff(build(), build()))


class TestCommandLine:
    def test_the_document_is_written_where_asked(self, tmp_path: Path) -> None:
        target = tmp_path / "sbom.cdx.json"
        assert sbom.main(["--version", "0.3.0", "--output", str(target)]) == 0
        assert json.loads(target.read_text(encoding="utf-8"))["bomFormat"] == "CycloneDX"

    def test_asking_for_a_document_without_saying_which_release_is_refused(self) -> None:
        """A document naming no version cannot be matched to what was published."""
        with pytest.raises(SystemExit):
            sbom.main([])

    def test_the_real_lock_produces_a_document_with_the_kit_in_it(self, tmp_path: Path) -> None:
        target = tmp_path / "sbom.cdx.json"
        sbom.main(["--version", "0.3.0", "--output", str(target)])
        document = json.loads(target.read_text(encoding="utf-8"))
        assert document["metadata"]["component"]["name"] == "tesserix-adk"
        assert any(item["name"] == "pydantic" for item in document["components"])

    def test_two_documents_can_be_diffed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        for name, version in (("old.json", "0.2.0"), ("new.json", "0.3.0")):
            (tmp_path / name).write_text(
                json.dumps(
                    sbom.build(LOCK, version=version, licence_of=LICENCES.get, built_at=BUILT_AT)
                ),
                encoding="utf-8",
            )
        code = sbom.main(["--diff", str(tmp_path / "old.json"), str(tmp_path / "new.json")])
        assert code == 0
        assert "no dependency changes" in capsys.readouterr().out


class TestTheRepositoryItself:
    def test_every_installed_component_of_the_real_lock_has_a_licence(self) -> None:
        """A NOASSERTION here is a gap in the licence gate. Only the installed set is
        checked: the complete graph needs every extra, which the licences CI job syncs."""
        document = sbom.build(sbom.lock(), version="0.0.0", built_at=BUILT_AT)
        unlicensed = [
            item["name"]
            for item in document["components"]
            if item["licenses"] == [{"expression": "NOASSERTION"}]
            and licences.declared(item["name"]) is not None
        ]
        assert unlicensed == []
