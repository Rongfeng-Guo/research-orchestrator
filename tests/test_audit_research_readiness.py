from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_research_readiness import analyze_repository_research_readiness


def test_research_readiness_audit_reports_expected_findings() -> None:
    summary = analyze_repository_research_readiness(PROJECT_ROOT)

    assert summary["benchmark"]["num_questions"] == 35
    assert summary["benchmark"]["num_domains"] >= 10
    assert summary["policy_artifacts"]["default_artifact"]["num_examples"] == 20
    assert summary["search_cache_residue"]["num_manifests"] >= 1
    assert any("默认 policy artifact" in finding for finding in summary["key_findings"])
