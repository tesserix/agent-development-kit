"""What the kit's published requirements are allowed to say.

A library has two opposing obligations. Its own builds must be reproducible, which the
lockfile handles. Its *published* requirements must not over-constrain a consumer who
already depends on the same packages — and the constraint that does the damage is the
speculative upper bound, because it turns an upgrade the consumer chose into a
resolution error they cannot fix without forking the kit.

So a floor is required and justified, and a cap is an exception with a name against it
and a trigger for removing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tools import dependency_policy as policy

if TYPE_CHECKING:
    from pathlib import Path

POLICY = """
[[floor]]
package = "pydantic"
floor = "2.9"
reason = "First release with the discriminated-union behaviour the models rely on."

[[floor]]
package = "graphiti-core"
floor = "0.14"
reason = "First release with the episode API the memory adapter uses."

[[cap]]
package = "graphiti-core"
cap = "1"
incompatibility = "0.x carries no compatibility promise; a minor is a breaking change."
trigger = "Remove when 1.0 ships and the graphiti extra's suite passes against it."
owner = "@sam123ben"
"""


def write(tmp_path: Path, body: str = POLICY) -> Path:
    path = tmp_path / "dependencies.toml"
    path.write_text(body, encoding="utf-8")
    return path


# The fixture records a cap for it, and a record with nothing depending on it is itself
# a violation, so every case declares it unless it is deliberately testing that rule.
GRAPHITI = {"graphiti": ["graphiti-core>=0.14,<1"]}


def check(tmp_path: Path, published: dict[str, list[str]], body: str = POLICY) -> list[str]:
    profiles = {**GRAPHITI, **published}
    return policy.violations(profiles, policy=policy.load(write(tmp_path, body)))


class TestCaps:
    def test_a_requirement_with_only_a_floor_passes(self, tmp_path: Path) -> None:
        assert check(tmp_path, {"": ["pydantic>=2.9"]}) == []

    def test_an_unrecorded_cap_is_a_violation(self, tmp_path: Path) -> None:
        """A speculative cap breaks a consumer's upgrade for a break nobody has seen."""
        found = check(tmp_path, {"": ["pydantic>=2.9,<3"]})
        assert len(found) == 1
        assert "pydantic" in found[0]

    def test_a_recorded_cap_passes(self, tmp_path: Path) -> None:
        assert check(tmp_path, {"graphiti": ["graphiti-core>=0.14,<1"]}) == []

    def test_a_cap_that_does_not_match_the_record_is_a_violation(self, tmp_path: Path) -> None:
        """Tightening a cap without revisiting the reason is how the reason goes stale."""
        found = check(tmp_path, {"graphiti": ["graphiti-core>=0.14,<0.20"]})
        assert found
        assert "graphiti-core" in found[0]

    def test_a_cap_in_an_extra_is_checked_like_any_other(self, tmp_path: Path) -> None:
        """A consumer inherits it the moment they install that extra."""
        assert check(tmp_path, {"redis": ["redis>=5.2,<7"]})

    def test_an_exception_for_a_package_no_longer_required_is_a_violation(
        self, tmp_path: Path
    ) -> None:
        """A record nobody removes outlives the dependency and misleads the next reader."""
        found = check(tmp_path, {"": ["pydantic>=2.9"], "graphiti": []})
        assert any("graphiti-core" in violation for violation in found)


class TestUnreadableRequirements:
    def test_a_requirement_that_cannot_be_parsed_is_reported_not_skipped(
        self, tmp_path: Path
    ) -> None:
        """Silently skipping it would let an unrecorded cap through in a malformed line."""
        found = check(tmp_path, {"": ["pydantic >= = 2.9"]})
        assert any("is not a requirement" in violation for violation in found)


class TestFloors:
    def test_a_requirement_with_no_floor_is_a_violation(self, tmp_path: Path) -> None:
        """An unbounded floor is untestable: no resolution proves the oldest still works."""
        found = check(tmp_path, {"": ["pydantic"]})
        assert found
        assert "floor" in found[0]

    def test_a_floor_the_policy_does_not_justify_is_a_violation(self, tmp_path: Path) -> None:
        found = check(tmp_path, {"": ["httpx>=0.27"]})
        assert found
        assert "httpx" in found[0]

    def test_a_floor_that_disagrees_with_the_record_is_a_violation(self, tmp_path: Path) -> None:
        found = check(tmp_path, {"": ["pydantic>=2.4"]})
        assert found
        assert "2.9" in found[0]


class TestLoading:
    def test_a_record_without_a_reason_is_rejected(self, tmp_path: Path) -> None:
        body = '[[floor]]\npackage = "pydantic"\nfloor = "2.9"\n'
        with pytest.raises(policy.PolicyError, match="reason"):
            policy.load(write(tmp_path, body))

    def test_a_cap_without_a_removal_trigger_is_rejected(self, tmp_path: Path) -> None:
        """A cap with no trigger is a permanent cap that nobody decided to make permanent."""
        body = POLICY.replace(
            'trigger = "Remove when 1.0 ships and the graphiti extra\'s suite passes against it."',
            "",
        )
        with pytest.raises(policy.PolicyError, match="trigger"):
            policy.load(write(tmp_path, body))

    def test_a_cap_without_an_owner_is_rejected(self, tmp_path: Path) -> None:
        body = POLICY.replace('owner = "@sam123ben"', "")
        with pytest.raises(policy.PolicyError, match="owner"):
            policy.load(write(tmp_path, body))

    def test_a_record_with_a_field_nobody_recognises_is_rejected(self, tmp_path: Path) -> None:
        """A misspelt field is a justification that silently does nothing."""
        body = POLICY.replace('floor = "2.9"', 'floor = "2.9"\nminimum = "2.9"', 1)
        with pytest.raises(policy.PolicyError, match="minimum"):
            policy.load(write(tmp_path, body))

    def test_a_missing_policy_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(policy.PolicyError, match=r"dependencies\.toml"):
            policy.load(tmp_path / "dependencies.toml")

    def test_a_policy_that_is_not_valid_toml_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(policy.PolicyError, match="TOML"):
            policy.load(write(tmp_path, "[[floor]\n"))


class TestReadingPyproject:
    def test_the_base_requirements_are_published(self) -> None:
        assert "pydantic" in " ".join(policy.published()[""])

    def test_each_extra_is_published_under_its_own_name(self) -> None:
        assert any("redis" in requirement for requirement in policy.published()["redis"])

    def test_the_union_extra_is_not_a_requirement_of_its_own(self) -> None:
        """`all` is a pure union of the others, so its entry constrains nothing."""
        assert policy.published()["all"] == []

    def test_development_groups_are_not_published(self) -> None:
        """A cap in a dev group constrains nobody: it is never in a consumer's resolution."""
        published = [text for profile in policy.published().values() for text in profile]
        assert "ruff" not in " ".join(published)


class TestTheRepositoryItself:
    def test_the_published_requirements_satisfy_the_policy(self) -> None:
        assert policy.violations(policy.published(), policy=policy.load()) == []


class TestCommandLine:
    def test_a_clean_set_of_requirements_passes(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert policy.main([]) == 0
        assert "no dependency policy violations" in capsys.readouterr().out

    def test_a_violation_fails_the_job(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(policy, "published", lambda: {"": ["pydantic>=2.9,<3"]})
        assert policy.main([]) == 1
        assert "pydantic" in capsys.readouterr().out
