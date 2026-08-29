"""The publish path is the one workflow nobody gets to rehearse.

An index artefact cannot be replaced, so the properties that make the path safe — no
upload token in existence, the full gates before the build, a smoke install from the
index itself — are asserted here rather than reviewed once and assumed after.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.ci_config import (
    RELEASE,
    load_yaml,
    pyproject,
    release_jobs,
    release_run_steps,
    supported_minors,
    triggers,
)

GUARD = "guard"
GATES = "gates"
BUILD = "build"
PUBLISH = "publish"
MIRROR = "mirror"
NOTES = "notes"
SMOKE = "smoke"
MIRROR_SMOKE = "smoke-mirror"
DIVERGENCE = "divergence"

# Both channels are installed from before the release is called done; the mirror is the
# one consumers use while the index publish is switched off.
SMOKE_JOBS = (SMOKE, MIRROR_SMOKE)

WORKFLOW = load_yaml(RELEASE)
TEXT = RELEASE.read_text(encoding="utf-8")


def _needs(job: str) -> set[str]:
    declared = release_jobs()[job].get("needs", [])
    return {declared} if isinstance(declared, str) else set(declared)


def test_only_a_version_tag_releases() -> None:
    """Any other trigger publishes something nobody decided to release."""
    declared = triggers(RELEASE)
    assert set(declared) == {"push"}
    assert declared["push"]["tags"] == ["v*"]


def test_the_guard_runs_before_anything_irreversible() -> None:
    assert "tools.release_guard" in " ".join(release_run_steps(GUARD))
    assert GUARD in _needs(GATES)


def test_the_guard_can_see_main_to_compare_the_tag_against() -> None:
    """This job runs on a tag push and nowhere else, so a command git rejects sits
    unnoticed until the one moment somebody is trying to release."""
    steps = " ".join(release_run_steps(GUARD))
    assert "git fetch origin main" in steps
    assert "--depth=0" not in steps
    assert release_jobs()[GUARD]["steps"][0]["with"]["fetch-depth"] == 0


def test_the_full_gates_run_before_the_build() -> None:
    """Publishing a version the matrix never saw is how a bad release becomes permanent."""
    assert release_jobs()[GATES]["uses"].endswith("ci.yml")
    assert GATES in _needs(BUILD)


def test_the_gates_are_the_same_workflow_pull_requests_run() -> None:
    """A second copy of the gates drifts, and the release is where it would show."""
    assert "workflow_call" in triggers(RELEASE.with_name("ci.yml"))


def test_the_build_validates_the_metadata_before_upload() -> None:
    """PyPI rejects a bad long description on upload, after the gates have all passed."""
    assert any("twine check --strict" in step for step in release_run_steps(BUILD))


def test_publishing_uses_workflow_identity() -> None:
    """Trusted publishing: nothing to leak, nothing to rotate, nothing to revoke."""
    assert release_jobs()[PUBLISH]["permissions"]["id-token"] == "write"


def test_no_upload_token_exists_anywhere_in_the_path() -> None:
    """The property is the absence, so it has to be asserted as an absence."""
    for forbidden in ("PYPI_API_TOKEN", "TWINE_PASSWORD", "password:"):
        assert forbidden not in TEXT


def test_publishing_is_gated_by_a_reviewed_environment() -> None:
    assert release_jobs()[PUBLISH]["environment"] == "pypi"


def test_the_index_publish_is_switched_on_deliberately_rather_than_by_default() -> None:
    """The trusted publisher is a one-time setup on PyPI; until it exists, an unguarded
    upload step fails the release before the mirror is written."""
    assert release_jobs()[PUBLISH]["if"] == "vars.PUBLISH_TO_PYPI == 'true'"


def test_the_mirror_does_not_wait_on_the_index_publish() -> None:
    """The mirror is what consumers install from, so it cannot be downstream of a job
    that is switched off — a skipped dependency skips everything behind it."""
    assert PUBLISH not in _needs(MIRROR)


def test_a_divergence_between_the_two_indexes_fails_the_release() -> None:
    """One index holding a version the other does not is worse than no release at all."""
    condition = release_jobs()[DIVERGENCE]["if"]
    assert f"needs.{PUBLISH}.result" in condition
    assert f"needs.{MIRROR}.result" in condition
    assert "docs/releasing.md" in " ".join(release_run_steps(DIVERGENCE))


def test_the_release_body_is_assembled_rather_than_written_by_hand() -> None:
    """Hand-written notes leave out the breaking entries; that is the whole point."""
    steps = " ".join(release_run_steps(NOTES))
    assert "tools.release_notes" in steps
    assert "--output" in steps


def test_the_notes_are_assembled_before_anything_is_published() -> None:
    """An undocumented change must block the release, not be discovered after it."""
    assert NOTES in _needs(PUBLISH)


def test_the_release_uses_the_assembled_notes() -> None:
    steps = " ".join(release_run_steps(MIRROR))
    assert "--notes-file" in steps
    assert "--generate-notes" not in steps


@pytest.mark.parametrize("job", SMOKE_JOBS)
def test_the_smoke_job_installs_what_was_published(job: str) -> None:
    """Installing the working tree would prove nothing about what consumers get."""
    steps = " ".join(release_run_steps(job))
    assert "tesserix-adk" in steps
    assert "pip install -e" not in steps
    assert "uv sync" not in steps


def test_the_mirror_is_smoke_tested_the_way_a_consumer_reaches_it() -> None:
    """Consumers resolve the release assets, not the index, while publishing is off."""
    steps = " ".join(release_run_steps(MIRROR_SMOKE))
    assert "gh release download" in steps
    assert "--find-links" in steps


@pytest.mark.parametrize("job", SMOKE_JOBS)
def test_the_smoke_job_installs_the_exact_version_that_was_published(job: str) -> None:
    """ "Latest" would pass against whatever the index already had."""
    assert "==$VERSION" in " ".join(release_run_steps(job))
    assert release_jobs()[job]["env"]["VERSION"] == "${{ needs.build.outputs.version }}"


def test_no_step_interpolates_a_ref_into_a_shell_command() -> None:
    """A tag name reaching a shell unquoted is a command-injection seam in the one
    workflow that holds publish rights."""
    for job in release_jobs():
        assert all("${{" not in step for step in release_run_steps(job))


@pytest.mark.parametrize("job", SMOKE_JOBS)
def test_the_smoke_job_covers_every_extra(job: str) -> None:
    """Extras must resolve from the published channel, not only from the repository."""
    declared = set(pyproject()["project"].get("optional-dependencies", {})) - {"all"}
    legs = set(release_jobs()[job]["strategy"]["matrix"]["extra"])
    assert declared <= legs
    assert {"none", "all"} <= legs


@pytest.mark.parametrize("job", SMOKE_JOBS)
def test_the_smoke_job_runs_the_getting_started_example(job: str) -> None:
    assert "getting_started.py" in " ".join(release_run_steps(job))


@pytest.mark.parametrize("job", SMOKE_JOBS)
def test_the_smoke_job_does_not_stop_at_the_first_failing_extra(job: str) -> None:
    assert release_jobs()[job]["strategy"]["fail-fast"] is False


@pytest.mark.parametrize("job", SMOKE_JOBS)
def test_release_smoke_uses_the_newest_supported_python(job: str) -> None:
    setup = next(
        step
        for step in release_jobs()[job]["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    assert setup["with"]["python-version"] == supported_minors()[-1]


def test_the_workflow_reads_no_more_than_it_needs_by_default() -> None:
    """Write scopes are granted per job, so a compromised step cannot reach the rest."""
    assert WORKFLOW["permissions"] == {"contents": "read"}


SBOM = "sbom"


def test_the_bill_of_materials_is_generated_in_the_release_build() -> None:
    """Regenerating it later describes a resolution, not the artefact that was published."""
    assert "tools.sbom" in " ".join(release_run_steps(SBOM))


def test_the_licence_policy_is_checked_before_anything_is_published() -> None:
    assert "tools.licences" in " ".join(release_run_steps(SBOM))
    assert SBOM in _needs(PUBLISH) | _needs(NOTES)


def test_the_licence_check_sees_every_extra() -> None:
    """A licence arriving through an extra is exposure a consumer still inherits."""
    assert any("--all-extras" in step for step in release_run_steps(SBOM))


def test_the_bill_of_materials_is_published_with_the_release() -> None:
    """An SBOM a reviewer cannot fetch answers nobody's question."""
    assert "sbom.cdx.json" in " ".join(release_run_steps(MIRROR))


def test_the_dependency_diff_reaches_the_release_notes() -> None:
    assert any("--diff" in step for step in release_run_steps(SBOM))
    assert "sbom-diff.md" in " ".join(release_run_steps(NOTES))


def _steps(job: str) -> list[dict[str, Any]]:
    listed: list[dict[str, Any]] = release_jobs()[job].get("steps", [])
    return listed


def _uses(job: str) -> list[str]:
    return [str(step["uses"]) for step in _steps(job) if "uses" in step]


def _step_using(job: str, action: str) -> dict[str, Any]:
    return next(step for step in _steps(job) if action in str(step.get("uses", "")))


class TestProvenance:
    """A consumer cannot tell a workflow's artefact from anyone's upload without this."""

    def test_the_built_artefacts_are_attested(self) -> None:
        assert any("attest-build-provenance" in step for step in _uses(BUILD))

    def test_the_attestation_covers_every_artefact_that_is_published(self) -> None:
        attest = _step_using(BUILD, "attest-build-provenance")
        assert attest["with"]["subject-path"] == "dist/*"

    def test_the_bill_of_materials_is_attested_too(self) -> None:
        """An unattested component list is a list an attacker can edit after the fact."""
        assert any("attest-build-provenance" in step for step in _uses(SBOM))

    def test_signing_needs_no_stored_key(self) -> None:
        """Keyless: workflow identity, so there is no signing key to steal or rotate."""
        assert release_jobs()[BUILD]["permissions"]["id-token"] == "write"
        assert release_jobs()[BUILD]["permissions"]["attestations"] == "write"

    def test_the_index_upload_carries_its_own_attestation(self) -> None:
        publish = _step_using(PUBLISH, "gh-action-pypi-publish")
        assert publish["with"]["attestations"] is True

    def test_the_attestation_is_mirrored_with_the_release(self) -> None:
        """A mirrored or air-gapped install has no attestation store to reach."""
        assert "attestation download" in " ".join(release_run_steps(MIRROR))

    @pytest.mark.parametrize("job", SMOKE_JOBS)
    def test_the_published_artefact_is_verified_before_it_is_installed(self, job: str) -> None:
        """Verification that runs after the install protects nothing."""
        assert "attestation verify" in " ".join(release_run_steps(job))

    @pytest.mark.parametrize("job", SMOKE_JOBS)
    def test_verification_names_the_workflow_that_may_sign(self, job: str) -> None:
        """Without it, any workflow in the repository is an acceptable signer."""
        assert "--signer-workflow" in " ".join(release_run_steps(job))


class TestVerificationDocumentation:
    """A verification command a consumer cannot copy is a command nobody runs."""

    DOC = RELEASE.parents[2] / "docs" / "verifying.md"

    def test_the_documented_command_names_the_signing_workflow(self) -> None:
        """A workflow or repository rename breaks every consumer's verification at once."""
        assert "tesserix/agent-development-kit/.github/workflows/release.yml" in self.DOC.read_text(
            encoding="utf-8"
        )

    def test_the_alpha_channel_is_documented_as_verifiable_too(self) -> None:
        assert "alpha.yml" in self.DOC.read_text(encoding="utf-8")

    def test_the_mirrored_bundle_is_documented_for_an_offline_check(self) -> None:
        assert "--bundle" in self.DOC.read_text(encoding="utf-8")
