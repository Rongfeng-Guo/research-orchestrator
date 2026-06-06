from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_research_readiness import analyze_repository_research_readiness, search_cache_residue_summary


def test_research_readiness_audit_reports_expected_findings() -> None:
    summary = analyze_repository_research_readiness(PROJECT_ROOT)

    assert summary["benchmark"]["num_questions"] == 35
    assert summary["benchmark"]["num_domains"] >= 10
    assert summary["policy_artifacts"]["default_artifact"]["num_examples"] == 20
    assert summary["search_cache_residue"]["num_manifests"] >= 0
    assert isinstance(summary["search_cache_residue"]["suspicious_test_like_manifests"], list)
    assert any("默认 policy artifact" in finding for finding in summary["key_findings"])


def test_search_cache_residue_detects_pytest_manifest(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "data" / "search_cache" / "citation_resurfaced"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "num_records": 1,
        "num_changed": 1,
        "analysis_summary": {"risk_flags": ["single_record"]},
        "input_paths": [str(tmp_path / "pytest-123" / "cache.jsonl")],
    }
    (manifest_dir / "resurfaced_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = search_cache_residue_summary(tmp_path)

    assert summary["num_manifests"] == 1
    assert summary["suspicious_test_like_manifests"][0]["file"] == "resurfaced_manifest.json"
