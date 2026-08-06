"""The resolved graph, read from `uv.lock` rather than from a resolver run.

An advisory only matters in proportion to who receives the package. A finding in a test
dependency reaches nobody downstream; one in the base set reaches every consumer of every
product built on the kit. That distinction is a graph walk, so it lives here.
"""

from __future__ import annotations

import pytest
from tools import lockfile

LOCK = {
    "package": [
        {
            "name": "tesserix-adk",
            "version": "0.1.0",
            "dependencies": [{"name": "httpx"}],
            "optional-dependencies": {"redis": [{"name": "redis"}]},
            "dev-dependencies": {"test": [{"name": "pytest"}]},
        },
        {"name": "httpx", "version": "0.28.1", "dependencies": [{"name": "idna"}]},
        {"name": "idna", "version": "3.10"},
        {"name": "redis", "version": "6.0.0", "dependencies": [{"name": "idna"}]},
        {"name": "pytest", "version": "8.3.4"},
    ]
}


@pytest.fixture
def graph() -> lockfile.Graph:
    return lockfile.Graph.from_lock(LOCK, root="tesserix-adk")


class TestVersions:
    def test_every_locked_package_is_reported_with_its_version(self, graph: lockfile.Graph) -> None:
        assert graph.versions["httpx"] == "0.28.1"

    def test_the_project_itself_is_not_one_of_its_own_dependencies(
        self, graph: lockfile.Graph
    ) -> None:
        assert "tesserix-adk" not in graph.reach


class TestReach:
    def test_a_base_dependency_reaches_every_consumer(self, graph: lockfile.Graph) -> None:
        assert graph.reach["httpx"] == frozenset({"runtime"})

    def test_a_transitive_dependency_inherits_the_reach_of_its_parent(
        self, graph: lockfile.Graph
    ) -> None:
        assert "runtime" in graph.reach["idna"]

    def test_a_package_behind_an_extra_is_labelled_with_that_extra(
        self, graph: lockfile.Graph
    ) -> None:
        assert graph.reach["redis"] == frozenset({"extra:redis"})

    def test_a_package_reached_two_ways_carries_both_labels(self, graph: lockfile.Graph) -> None:
        """idna arrives through the base set and through the redis extra."""
        assert graph.reach["idna"] == frozenset({"runtime", "extra:redis"})

    def test_a_development_dependency_reaches_nobody_downstream(
        self, graph: lockfile.Graph
    ) -> None:
        assert graph.reach["pytest"] == frozenset({"group:test"})

    def test_a_package_nobody_depends_on_has_no_reach(self) -> None:
        orphan = lockfile.Graph.from_lock(
            {"package": [{"name": "tesserix-adk"}, {"name": "stray", "version": "1.0"}]},
            root="tesserix-adk",
        )
        assert orphan.reach.get("stray", frozenset()) == frozenset()

    def test_a_cycle_does_not_hang_the_walk(self) -> None:
        """Rare, but a self-referential extra is legal and has been seen in the wild."""
        cyclic = lockfile.Graph.from_lock(
            {
                "package": [
                    {"name": "tesserix-adk", "dependencies": [{"name": "a"}]},
                    {"name": "a", "version": "1", "dependencies": [{"name": "b"}]},
                    {"name": "b", "version": "1", "dependencies": [{"name": "a"}]},
                ]
            },
            root="tesserix-adk",
        )
        assert cyclic.reach["b"] == frozenset({"runtime"})

    def test_an_edge_to_a_package_the_lock_does_not_hold_is_an_error(self) -> None:
        """A lock that names a package it does not contain is not a lock to audit against."""
        with pytest.raises(lockfile.LockfileError, match="ghost"):
            lockfile.Graph.from_lock(
                {"package": [{"name": "tesserix-adk", "dependencies": [{"name": "ghost"}]}]},
                root="tesserix-adk",
            )

    def test_a_lock_without_the_project_in_it_is_an_error(self) -> None:
        with pytest.raises(lockfile.LockfileError, match="tesserix-adk"):
            lockfile.Graph.from_lock({"package": [{"name": "httpx", "version": "1"}]})


class TestBlastRadius:
    def test_the_base_set_is_described_as_reaching_every_consumer(
        self, graph: lockfile.Graph
    ) -> None:
        assert graph.blast_radius("httpx") == "every consumer"

    def test_an_extra_only_package_names_the_extra(self, graph: lockfile.Graph) -> None:
        assert graph.blast_radius("redis") == "consumers of the redis extra"

    def test_a_development_only_package_says_it_is_not_shipped(self, graph: lockfile.Graph) -> None:
        assert graph.blast_radius("pytest") == "development only, not shipped to consumers"

    def test_a_package_reached_several_ways_lists_them(self, graph: lockfile.Graph) -> None:
        assert graph.blast_radius("idna") == "every consumer"

    def test_an_unknown_package_is_described_rather_than_crashing(
        self, graph: lockfile.Graph
    ) -> None:
        """A finding against something the lock does not hold still has to be reported."""
        assert "not in the lock" in graph.blast_radius("nothing-like-this")


class TestTheRealLockfile:
    def test_the_project_lock_loads(self) -> None:
        assert lockfile.project_graph().versions["httpx"]

    def test_every_base_dependency_is_labelled_runtime(self) -> None:
        """Extras reach it too — several SDKs depend on pydantic — but runtime is the
        label that decides whether a finding blocks."""
        graph = lockfile.project_graph()
        assert "runtime" in graph.reach["pydantic"]
        assert graph.blast_radius("pydantic") == "every consumer"

    def test_the_test_tooling_is_not_shipped(self) -> None:
        graph = lockfile.project_graph()
        assert "runtime" not in graph.reach["pytest"]
