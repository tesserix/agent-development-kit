# Verifying a release

Every artefact the kit publishes is signed by the workflow that built it and carries a
build provenance attestation naming the repository, the workflow, the commit and the build
inputs. Without it, nothing distinguishes an artefact built by the release workflow from
one uploaded by anybody who holds index access.

There is no signing key. Signing is keyless, against the workflow's own identity, so there
is nothing to steal, store or rotate.

## What a consumer runs

```bash
pip download --no-deps --dest ./downloaded 'tesserix-adk==0.3.0'

gh attestation verify ./downloaded/tesserix_adk-0.3.0-py3-none-any.whl \
  --repo tesserix/agent-development-kit \
  --signer-workflow tesserix/agent-development-kit/.github/workflows/release.yml
```

A correct result names the digest, the repository and the workflow:

```
Loaded digest sha256:… for file://tesserix_adk-0.3.0-py3-none-any.whl
Loaded 1 attestation from GitHub API

The following policy criteria will be enforced:
- Predicate type must match:................ https://slsa.dev/provenance/v1
- Source Repository must match:............. https://github.com/tesserix/agent-development-kit
- Workflow must match:...................... …/.github/workflows/release.yml

✓ Verification succeeded!
```

A modified artefact fails on the digest, which is the first line checked, and an artefact
built anywhere else fails on the workflow. Both exit non-zero.

**`--signer-workflow` is not optional.** Without it, any workflow in the repository is an
acceptable signer, which is a much weaker claim than the one you meant to check.

## Pre-releases

Alphas go through the same signing path — no channel is unattested — but they are built by
a different workflow, so the identity you pin is different:

```bash
gh attestation verify ./downloaded/tesserix_adk-0.4.0a3-py3-none-any.whl \
  --repo tesserix/agent-development-kit \
  --signer-workflow tesserix/agent-development-kit/.github/workflows/alpha.yml
```

## The bill of materials is covered too

`sbom.cdx.json` is attested by the same run, so the component list is tamper-evident
rather than a file anybody can edit after the fact:

```bash
gh attestation verify sbom.cdx.json \
  --repo tesserix/agent-development-kit \
  --signer-workflow tesserix/agent-development-kit/.github/workflows/release.yml
```

## Offline, mirrored and air-gapped installs

`gh attestation verify` reaches GitHub for the attestation by default. An install that
cannot reach GitHub would otherwise degrade silently to trusting the mirror, so every
release also carries its attestation bundles as release assets (`*.jsonl`). Mirror them
alongside the artefacts and verify against the local copy:

```bash
gh attestation verify ./tesserix_adk-0.3.0-py3-none-any.whl \
  --bundle ./sha256:….jsonl \
  --repo tesserix/agent-development-kit \
  --signer-workflow tesserix/agent-development-kit/.github/workflows/release.yml
```

If your mirror does not carry the bundles, verification is not happening — say so in your
own runbook rather than assuming the artefact was checked upstream.

## In a consumer pipeline

Verify **before** installing, and let a failure stop the pipeline. Verification that runs
after the install protects nothing, and a warning nobody blocks on is not protection:

```yaml
- run: pip download --no-deps --dest downloaded "tesserix-adk==$VERSION"
- run: |
    gh attestation verify downloaded/*.whl \
      --repo tesserix/agent-development-kit \
      --signer-workflow tesserix/agent-development-kit/.github/workflows/release.yml
- run: pip install "tesserix-adk==$VERSION"
```

The kit's own release workflow does exactly this in its `smoke` job, and the alpha
workflow does it in `downstream`, so the documented command is the one that runs.

## PyPI attestations

Uploads also carry a PEP 740 attestation to PyPI itself, visible on the file's page on the
index. That is a second, independent path to the same claim; the GitHub attestation is the
one the commands above check.

## If the signing identity changes

Renaming the repository, renaming a workflow file, or moving the org changes the identity
in every future attestation and breaks every pinned verification at once. Any such change
must land here, in the same pull request, with the old and new identities both stated and
a note of which release is the first to use the new one. Artefacts released before the
change keep verifying against the old identity — attestations are not reissued.

## Known limitations

- Provenance proves *where* an artefact was built, not that the source was uncompromised.
  A malicious commit on `main` produces a perfectly valid attestation.
- The chain holds only as far as the runner. GitHub-hosted ephemeral runners and the
  absence of any manual upload path are what make it worth having.
- `gh attestation verify` requires the `gh` CLI; there is no `pip`-native equivalent that
  checks the GitHub attestation, though `pip` can check the PyPI one.
