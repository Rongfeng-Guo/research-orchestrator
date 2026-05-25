from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_paper_readiness import audit_readiness, render_markdown  # noqa: E402


def _analysis(**overrides) -> dict:
    summary = {
        "input_paths": ["cache.jsonl"],
        "num_success": 12,
        "browser_query_coverage": 1.0,
        "nonzero_composite_query_ratio": 1.0,
        "avg_citation_coverage": 0.12,
        "avg_citation_density": 0.30,
        "avg_citation_quality_score": 0.60,
        "real_source_query_ratio": 1.0,
        "mock_source_query_ratio": 0.0,
        "avg_real_sources_per_success_query": 2.4,
        "relevant_source_query_ratio": 1.0,
        "avg_relevant_sources_per_success_query": 2.2,
        "avg_source_relevance": 0.35,
        "risk_flags": [],
    }
    summary.update(overrides)
    return summary


def _train(**overrides) -> dict:
    summary = {
        "training_allowed": True,
        "headline_claim_ready": True,
        "status": "trained",
        "training_readiness_failures": [],
        "headline_readiness_failures": [],
    }
    summary.update(overrides)
    return summary


def test_audit_readiness_passes_when_all_gates_pass() -> None:
    audit = audit_readiness([_analysis()], training_summary=_train())

    assert audit["status"] == "PASS"
    assert audit["failed_checks"] == []
    assert "Proceed to held-out head-to-head" in audit["next_actions"][0]


def test_audit_readiness_blocks_mock_and_missing_training() -> None:
    audit = audit_readiness(
        [_analysis(real_source_query_ratio=0.0, mock_source_query_ratio=1.0)],
        training_summary=None,
    )

    failed_names = {item["name"] for item in audit["failed_checks"]}
    assert audit["status"] == "BLOCKED"
    assert "real_source_query_ratio" in failed_names
    assert "mock_source_query_ratio" in failed_names
    assert "training_summary_present" in failed_names


def test_render_markdown_contains_failed_checks() -> None:
    audit = audit_readiness(
        [_analysis(avg_real_sources_per_success_query=1.0, avg_relevant_sources_per_success_query=1.0)],
        training_summary=_train(status="blocked", training_allowed=False),
    )

    md = render_markdown(audit)

    assert "# Paper Readiness Audit" in md
    assert "avg_real_sources_per_query" in md
    assert "avg_relevant_sources_per_query" in md
    assert "training_status" in md


def test_audit_cli_writes_report_and_exits_blocked(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis.json"
    train_path = tmp_path / "train.json"
    output_dir = tmp_path / "audit"
    output_json = output_dir / "stable.json"
    output_md = output_dir / "stable.md"
    analysis_path.write_text(json.dumps(_analysis(mock_source_query_ratio=1.0), ensure_ascii=False), encoding="utf-8")
    train_path.write_text(json.dumps(_train(), ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_paper_readiness.py"),
            "--analysis-json",
            str(analysis_path),
            "--train-json",
            str(train_path),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "BLOCKED" in result.stdout
    assert output_json.exists()
    assert output_md.exists()
