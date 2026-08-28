"""The alpha workflow holds publish rights, so its shape is asserted, not reviewed by eye.

Everything here is a property that cannot be checked by running the workflow: that it
publishes without a long-lived token, that no event input reaches a shell, and that a
consumer's own red suite cannot fail the job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.ci_config import load_yaml, triggers

ALPHA = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "alpha.yml"


def alpha_jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = load_yaml(ALPHA)["jobs"]
    return jobs


def run_steps(job: str) -> list[str]:
    return [step["run"] for step in alpha_jobs()[job].get("steps", []) if "run" in step]


def steps(job: str) -> list[dict[str, Any]]:
    listed: list[dict[str, Any]] = alpha_jobs()[job]["steps"]
    return listed


class TestTrigger:
    def test_an_alpha_is_published_on_every_merge_to_main(self) -> None:
        assert triggers(ALPHA)["push"]["branches"] == ["main"]

    def test_an_alpha_can_also_be_cut_by_hand(self) -> None:
        """The first consumer always needs one before the next merge happens to land."""
        assert "workflow_dispatch" in triggers(ALPHA)

    def test_a_tag_push_does_not_trigger_it(self) -> None:
        """Tags belong to release.yml; two workflows racing for one version is a yank."""
        assert "tags" not in triggers(ALPHA).get("push", {})


class TestGates:
    def test_the_full_gates_run_before_anything_is_published(self) -> None:
        assert alpha_jobs()["gates"]["uses"] == "./.github/workflows/ci.yml"

    def test_ci_does_not_also_run_itself_on_a_push_to_main(self) -> None:
        """Both would land in CI's own concurrency group and cancel each other, which
        reads on the branch as a failed run. Alpha runs the same gates on the same push."""
        assert "push" not in triggers(ALPHA.with_name("ci.yml"))

    def test_the_build_waits_for_the_gates(self) -> None:
        assert "gates" in alpha_jobs()["build"]["needs"]

    def test_the_publish_waits_for_the_build(self) -> None:
        assert "build" in alpha_jobs()["publish"]["needs"]


class TestVersioning:
    def test_the_version_comes_from_the_alpha_numbering_tool(self) -> None:
        assert any("tools.alpha" in step for step in run_steps("build"))

    def test_the_alpha_is_tagged_so_hatch_vcs_derives_the_same_version(self) -> None:
        """Without a tag, hatch-vcs produces a dev version that PyPI will not accept twice."""
        assert any("git tag" in step for step in run_steps("build"))

    def test_the_history_is_fetched_in_full(self) -> None:
        checkout = next(step for step in steps("build") if "checkout" in str(step.get("uses")))
        assert checkout["with"]["fetch-depth"] == 0

    def test_the_build_rechecks_for_a_release_tag_after_waiting_for_the_gates(self) -> None:
        build = alpha_jobs()["build"]
        listed = steps("build")
        recheck = next(
            index
            for index, step in enumerate(listed)
            if "describe --exact-match" in str(step.get("run", ""))
        )
        version = next(
            index for index, step in enumerate(listed) if "tools.alpha" in str(step.get("run", ""))
        )

        assert recheck < version
        assert build["outputs"]["alpha"] == "${{ steps.recheck.outputs.alpha }}"
        assert all(
            step.get("if") == "steps.recheck.outputs.alpha == 'true'"
            for step in listed[recheck + 1 :]
        )
        assert "needs.build.outputs.alpha == 'true'" in alpha_jobs()["publish"]["if"]


class TestPublishing:
    def test_publishing_uses_a_minted_token_not_a_stored_one(self) -> None:
        assert alpha_jobs()["publish"]["permissions"]["id-token"] == "write"

    def test_no_upload_token_exists_to_be_stolen(self) -> None:
        assert "PYPI_API_TOKEN" not in ALPHA.read_text(encoding="utf-8")

    def test_the_publish_is_gated_by_an_environment(self) -> None:
        assert alpha_jobs()["publish"]["environment"] == "pypi"

    def test_the_upload_waits_for_the_trusted_publisher_to_be_configured(self) -> None:
        """One-time setup nobody can do from here. Failing every merge until it happens
        trains the team to ignore a red main, which costs more than the delay."""
        assert "PUBLISH_ALPHAS" in alpha_jobs()["publish"]["if"]

    def test_the_build_is_gated_only_on_the_commit_being_a_release(self) -> None:
        """Every other merge is proven publishable; a release already has a version, and an
        alpha built from it would number below the release it was cut from."""
        assert alpha_jobs()["build"]["if"] == "needs.whether.outputs.alpha == 'true'"
        assert any("describe --exact-match" in step for step in run_steps("whether"))

    def test_the_workflow_cannot_write_to_the_repository_by_default(self) -> None:
        assert load_yaml(ALPHA)["permissions"] == {"contents": "read"}

    def test_the_build_may_not_publish_and_the_publish_may_not_build(self) -> None:
        """Separate jobs so the one holding id-token never runs repository code."""
        assert not any("uv build" in step for step in run_steps("publish"))


class TestDownstream:
    def test_the_consumer_suite_runs_against_the_published_alpha(self) -> None:
        assert "publish" in alpha_jobs()["downstream"]["needs"]

    def test_a_missing_consumer_repository_skips_rather_than_fails(self) -> None:
        """There is no consumer yet; a job that fails until there is teaches nothing."""
        assert "DOWNSTREAM_REPO" in alpha_jobs()["downstream"]["if"]

    def test_the_verdict_comes_from_comparing_the_two_runs(self) -> None:
        assert any("tools.downstream" in step for step in run_steps("downstream"))

    def test_the_consumer_suite_itself_cannot_fail_the_job(self) -> None:
        """Only the comparison decides; a red baseline is the consumer's business."""
        pytest_steps = [step for step in steps("downstream") if "pytest" in str(step.get("run"))]
        assert pytest_steps
        assert all(step.get("continue-on-error") is True for step in pytest_steps)


class TestInjection:
    def test_no_event_input_is_interpolated_into_a_shell(self) -> None:
        """A branch or actor name reaching a shell is command injection in the one
        workflow that can publish."""
        for job in alpha_jobs():
            assert not any("${{" in step for step in run_steps(job))


class TestProvenance:
    """An unattested channel is the one an attacker publishes through."""

    def test_an_alpha_is_attested_like_a_release(self) -> None:
        assert any(
            "attest-build-provenance" in str(step.get("uses", "")) for step in steps("build")
        )

    def test_the_alpha_upload_carries_its_own_attestation(self) -> None:
        publish = next(
            step for step in steps("publish") if "gh-action-pypi-publish" in str(step.get("uses"))
        )
        assert publish["with"]["attestations"] is True

    def test_the_consumer_verifies_the_alpha_before_installing_it(self) -> None:
        assert "attestation verify" in " ".join(run_steps("downstream"))
