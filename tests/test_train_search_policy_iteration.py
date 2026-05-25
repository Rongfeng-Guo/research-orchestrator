from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_search_policy_iteration import evaluate_headline_readiness, evaluate_training_readiness  # noqa: E402


def test_evaluate_training_readiness_blocks_low_browser_ratio() -> None:
    summary = {
        "num_success": 8,
        "label_ratios": {"browser": 0.03},
        "browser_query_coverage": 0.25,
    }

    reasons = evaluate_training_readiness(
        summary,
        min_success_queries=8,
        min_browser_step_ratio=0.10,
        min_browser_query_coverage=0.25,
    )

    assert len(reasons) == 1
    assert "browser_step_ratio_below_threshold" in reasons[0]


def test_evaluate_training_readiness_passes_when_thresholds_met() -> None:
    summary = {
        "num_success": 10,
        "label_ratios": {"browser": 0.12},
        "browser_query_coverage": 0.40,
    }

    reasons = evaluate_training_readiness(
        summary,
        min_success_queries=8,
        min_browser_step_ratio=0.10,
        min_browser_query_coverage=0.25,
    )

    assert reasons == []


def test_evaluate_training_readiness_blocks_mock_sources_when_present() -> None:
    summary = {
        "num_success": 10,
        "label_ratios": {"browser": 0.20},
        "browser_query_coverage": 0.80,
        "mock_source_query_ratio": 1.0,
    }

    reasons = evaluate_training_readiness(summary)

    assert len(reasons) == 1
    assert "mock_source_query_ratio_above_threshold" in reasons[0]


def test_evaluate_headline_readiness_blocks_low_citation_and_low_nonzero_ratio() -> None:
    summary = {
        "avg_citation_coverage": 0.0038,
        "nonzero_composite_query_ratio": 0.3333,
    }

    reasons = evaluate_headline_readiness(
        summary,
        min_avg_citation_coverage=0.01,
        min_nonzero_composite_ratio=0.50,
    )

    assert len(reasons) == 2
    assert "avg_citation_coverage_below_threshold" in reasons[0]
    assert "nonzero_composite_query_ratio_below_threshold" in reasons[1]


def test_evaluate_headline_readiness_passes_when_evidence_thresholds_met() -> None:
    summary = {
        "avg_citation_coverage": 0.021,
        "nonzero_composite_query_ratio": 0.75,
    }

    reasons = evaluate_headline_readiness(
        summary,
        min_avg_citation_coverage=0.01,
        min_nonzero_composite_ratio=0.50,
    )

    assert reasons == []


def test_evaluate_headline_readiness_blocks_mock_or_missing_real_sources() -> None:
    summary = {
        "avg_citation_coverage": 0.05,
        "nonzero_composite_query_ratio": 1.0,
        "real_source_query_ratio": 0.0,
        "mock_source_query_ratio": 1.0,
        "avg_real_sources_per_success_query": 0.0,
        "relevant_source_query_ratio": 0.0,
        "avg_relevant_sources_per_success_query": 0.0,
        "avg_source_relevance": 0.0,
        "avg_citation_quality_score": 0.0,
    }

    reasons = evaluate_headline_readiness(summary)

    assert len(reasons) == 7
    assert "real_source_query_ratio_below_threshold" in reasons[0]
    assert "mock_source_query_ratio_above_threshold" in reasons[1]
    assert "avg_real_sources_per_query_below_threshold" in reasons[2]
    assert "relevant_source_query_ratio_below_threshold" in reasons[3]
    assert "avg_relevant_sources_per_query_below_threshold" in reasons[4]
    assert "avg_source_relevance_below_threshold" in reasons[5]
    assert "avg_citation_quality_score_below_threshold" in reasons[6]


def test_evaluate_headline_readiness_requires_multi_source_citation_quality() -> None:
    summary = {
        "avg_citation_coverage": 0.05,
        "nonzero_composite_query_ratio": 1.0,
        "real_source_query_ratio": 1.0,
        "mock_source_query_ratio": 0.0,
        "avg_real_sources_per_success_query": 1.0,
        "relevant_source_query_ratio": 1.0,
        "avg_relevant_sources_per_success_query": 1.0,
        "avg_source_relevance": 0.30,
        "avg_citation_quality_score": 0.10,
    }

    reasons = evaluate_headline_readiness(summary)

    assert len(reasons) == 3
    assert "avg_real_sources_per_query_below_threshold" in reasons[0]
    assert "avg_relevant_sources_per_query_below_threshold" in reasons[1]
    assert "avg_citation_quality_score_below_threshold" in reasons[2]


def test_evaluate_headline_readiness_blocks_low_source_relevance() -> None:
    summary = {
        "avg_citation_coverage": 0.05,
        "nonzero_composite_query_ratio": 1.0,
        "real_source_query_ratio": 1.0,
        "mock_source_query_ratio": 0.0,
        "avg_real_sources_per_success_query": 3.0,
        "relevant_source_query_ratio": 0.20,
        "avg_relevant_sources_per_success_query": 0.5,
        "avg_source_relevance": 0.04,
        "avg_citation_quality_score": 0.40,
    }

    reasons = evaluate_headline_readiness(summary)

    assert len(reasons) == 3
    assert "relevant_source_query_ratio_below_threshold" in reasons[0]
    assert "avg_relevant_sources_per_query_below_threshold" in reasons[1]
    assert "avg_source_relevance_below_threshold" in reasons[2]


def test_train_search_policy_iteration_writes_stable_blocked_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "cache.jsonl"
    output_summary = tmp_path / "summary.json"
    output_model = tmp_path / "model.json"
    record = {
        "status": "success",
        "cache_id": "cache_001",
        "query_id": "q1",
        "report_content": "短报告",
        "report_metadata": {
            "policy_trace": [
                {
                    "action_type": "tool_call",
                    "tool_name": "web_search",
                    "backend": "mock",
                    "result_count": 1,
                    "top_urls": ["https://example.com/mock"],
                }
            ]
        },
        "evaluation": {
            "composite_score": 0.5,
            "metrics": {"citation_coverage": 0.0},
        },
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_search_policy_iteration.py"),
            "--inputs",
            str(input_path),
            "--output-model",
            str(output_model),
            "--output-summary",
            str(output_summary),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
