"""Incremental adoption remains measurable and reversible at every stage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "migration.md"
EXAMPLE = ROOT / "examples" / "migrate_legacy_provider.py"


def test_first_step_keeps_the_legacy_client_and_adds_runtime_controls() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "provider: legacy-client" in result.stdout
    assert "tenant attribution: acme" in result.stdout
    assert "model-call ceiling: 1" in result.stdout
    assert "legacy tools changed: False" in result.stdout


def test_each_migration_stage_has_value_evidence_and_rollback() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    for stage in (
        "Stage 0 — Observe the existing agent",
        "Stage 1 — Add the model gateway",
        "Stage 2 — Adopt the tool registry",
        "Stage 3 — Adopt the runtime loop",
        "Stage 4 — Add memory and guardrails",
    ):
        assert stage in guide
    assert guide.count("**Immediate value:**") >= 5
    assert guide.count("**Verify:**") >= 5
    assert guide.count("**Rollback:**") >= 5
    assert "eval" in guide.lower()
    assert "cost per run" in guide.lower()
    assert "p95 latency" in guide.lower()


def test_migration_guide_covers_halfway_state_personal_data_and_gaps() -> None:
    guide = GUIDE.read_text(encoding="utf-8").lower()
    assert "orchestration can stay outside" in guide
    assert "embeddings" in guide
    assert "erasure" in guide
    assert "alpha" in guide
    assert "before 1.0" in guide
    assert "do not" in guide
    assert "interop" in guide
    assert "github issue" in guide
    assert "local workaround" in guide
    assert "escape hatch" in guide
    assert "decision checklist" in guide
    assert "adr/0004-incremental-adoption.md" in guide
