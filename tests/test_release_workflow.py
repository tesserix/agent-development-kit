"""The publish path is the one workflow nobody gets to rehearse.

An index artefact cannot be replaced, so the properties that make the path safe — no
upload token in existence, the full gates before the build, a smoke install from the
index itself — are asserted here rather than reviewed once and assumed after.
"""

from __future__ import annotations

from tests.ci_config import RELEASE, load_yaml, pyproject, release_jobs, release_run_steps, triggers

GUARD = "guard"
GATES = "gates"
BUILD = "build"
PUBLISH = "publish"
MIRROR = "mirror"
NOTES = "notes"
SMOKE = "smoke"
DIVERGENCE = "divergence"

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


def test_the_mirror_follows_the_index_publish() -> None:
    assert PUBLISH in _needs(MIRROR)


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


def test_the_smoke_job_installs_from_the_index() -> None:
    """Installing the working tree would prove nothing about what consumers get."""
    steps = " ".join(release_run_steps(SMOKE))
    assert "tesserix-adk" in steps
    assert "pip install -e" not in steps
    assert "uv sync" not in steps


def test_the_smoke_job_installs_the_exact_version_that_was_published() -> None:
    """ "Latest" would pass against whatever the index already had."""
    assert "==$VERSION" in " ".join(release_run_steps(SMOKE))
    assert release_jobs()[SMOKE]["env"]["VERSION"] == "${{ needs.build.outputs.version }}"


def test_no_step_interpolates_a_ref_into_a_shell_command() -> None:
    """A tag name reaching a shell unquoted is a command-injection seam in the one
    workflow that holds publish rights."""
    for job in release_jobs():
        assert all("${{" not in step for step in release_run_steps(job))


def test_the_smoke_job_covers_every_extra() -> None:
    """Extras must resolve from the index, not only from the repository."""
    declared = set(pyproject()["project"].get("optional-dependencies", {})) - {"all"}
    legs = set(release_jobs()[SMOKE]["strategy"]["matrix"]["extra"])
    assert declared <= legs
    assert {"none", "all"} <= legs


def test_the_smoke_job_runs_the_getting_started_example() -> None:
    assert "getting_started.py" in " ".join(release_run_steps(SMOKE))


def test_the_smoke_job_does_not_stop_at_the_first_failing_extra() -> None:
    assert release_jobs()[SMOKE]["strategy"]["fail-fast"] is False


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
