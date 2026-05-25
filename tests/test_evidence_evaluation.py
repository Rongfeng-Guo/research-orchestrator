from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.research_bench import ResearchBench  # noqa: E402
from evaluation.metrics.rule_based import RuleBasedMetrics  # noqa: E402
from src.evolution.collector import TrajectoryCollector  # noqa: E402
from src.orchestrator.schemas import ResearchReport  # noqa: E402


def _sample_snapshot() -> dict:
    return {
        "claims": [
            {
                "claim_id": "c1",
                "claim": "GPT-4o has stronger multimodal performance.",
                "source_id": "s1",
                "source_trust": 0.88,
                "support_score": 0.91,
                "confidence": 0.90,
            },
            {
                "claim_id": "c2",
                "claim": "Claude 3.5 remains competitive on coding.",
                "source_id": "s2",
                "source_trust": 0.82,
                "support_score": 0.85,
                "confidence": 0.84,
            },
        ],
        "support_edges": [
            {"source_id": "s1", "claim_id": "c1", "weight": 0.91},
            {"source_id": "s2", "claim_id": "c2", "weight": 0.85},
        ],
        "contradiction_edges": [
            {"claim_id_1": "c1", "claim_id_2": "c2", "severity": 0.25}
        ],
        "source_trusts": [
            {"source_id": "s1", "trust_score": 0.88},
            {"source_id": "s2", "trust_score": 0.82},
        ],
        "candidate_actions": [
            {"name": "verify", "score": 0.74},
            {"name": "search", "score": 0.31},
        ],
    }


def test_evidence_graph_breakdown_tracks_claim_and_edge_quality() -> None:
    breakdown = RuleBasedMetrics.evidence_graph_breakdown(
        {"evidence_snapshot": _sample_snapshot()}
    )

    assert breakdown["claim_count"] == 2.0
    assert breakdown["support_edge_count"] == 2.0
    assert breakdown["contradiction_edge_count"] == 1.0
    assert breakdown["claim_support_coverage"] == 1.0
    assert breakdown["support_edge_precision"] == 1.0
    assert breakdown["contradiction_edge_precision"] == 1.0
    assert breakdown["source_trust_mean"] > 0.8
    assert breakdown["candidate_action_margin"] == pytest.approx(0.43, rel=1e-3)
    assert breakdown["evidence_graph_quality"] > 0.75


def test_composite_score_uses_factual_aliases_without_penalizing_missing_evidence_metric() -> None:
    score = RuleBasedMetrics.composite_score(
        {
            "factual_accuracy_str": 0.40,
            "factual_accuracy_sem": 0.80,
            "logical_consistency": 0.50,
            "citation_coverage": 0.50,
            "bias": 0.50,
            "comprehensiveness": 0.50,
        }
    )

    assert score == pytest.approx(0.548, rel=1e-3)


def test_trajectory_collector_exports_transition_summary_and_rewards() -> None:
    collector = TrajectoryCollector(system_prompt="research")
    report = ResearchReport(
        query="compare frontier models",
        content="final report",
        metadata={
            "evidence_transition_trace": [
                {
                    "step": 1,
                    "task_id": "search_1",
                    "status": "success",
                    "before": {
                        "uncertainty": 0.95,
                        "coverage_ratio": 0.10,
                        "open_conflicts": 2,
                        "claim_count": 1,
                        "support_edge_count": 1,
                        "avg_source_trust": 0.55,
                        "evidence_strength": 0.32,
                    },
                    "after": {
                        "uncertainty": 0.72,
                        "coverage_ratio": 0.42,
                        "open_conflicts": 1,
                        "claim_count": 2,
                        "support_edge_count": 2,
                        "avg_source_trust": 0.70,
                        "evidence_strength": 0.58,
                        "recommended_action": "verify",
                    },
                },
                {
                    "step": 2,
                    "task_id": "verify_1",
                    "status": "success",
                    "before": {
                        "uncertainty": 0.72,
                        "coverage_ratio": 0.42,
                        "open_conflicts": 1,
                        "claim_count": 2,
                        "support_edge_count": 2,
                        "avg_source_trust": 0.70,
                        "evidence_strength": 0.58,
                    },
                    "after": {
                        "uncertainty": 0.36,
                        "coverage_ratio": 0.78,
                        "open_conflicts": 0,
                        "claim_count": 3,
                        "support_edge_count": 3,
                        "avg_source_trust": 0.81,
                        "evidence_strength": 0.81,
                        "recommended_action": "stop",
                    },
                },
            ]
        },
    )

    data = collector.collect("compare frontier models", report, trajectories=[])
    verl_row = collector.to_verl_format(data)

    assert len(data["evidence_transitions"]) == 2
    assert len(data["process_reward_trace"]) == 2
    assert data["evidence_transitions"][0]["before"]["evidence_graph_quality"] > 0.0
    assert data["evidence_transition_summary"]["coverage_gain"] > 0.6
    assert data["evidence_transition_summary"]["uncertainty_reduction"] > 0.5
    assert data["evidence_transition_summary"]["terminal_action"] == "stop"
    assert verl_row["metadata"]["transition_count"] == 2
    assert len(verl_row["evidence_transitions"]) == 2


def test_trajectory_collector_recovers_source_inventory_from_tool_trajectories() -> None:
    collector = TrajectoryCollector(system_prompt="research")
    report = ResearchReport(
        query="compare models",
        content="final report",
        sources=[{"url": "https://example.com/a", "title": "Existing A"}],
    )

    data = collector.collect(
        "compare models",
        report,
        trajectories=[
            {
                "role": "tool",
                "result": {
                    "backend": "serpapi",
                    "results": [
                        {
                            "title": "Existing A duplicate",
                            "url": "https://example.com/a",
                            "snippet": "duplicate",
                        },
                        {
                            "title": "New B",
                            "url": "https://example.org/b",
                            "snippet": "new source",
                        },
                    ],
                },
            },
            {
                "action_type": "tool_call",
                "tool_name": "browser",
                "backend": "browser",
                "top_urls": ["https://example.net/c"],
            },
        ],
    )

    urls = [source["url"] for source in data["sources"]]
    assert urls == [
        "https://example.com/a",
        "https://example.org/b",
        "https://example.net/c",
    ]
    assert data["source_count"] == 3


def test_research_bench_batch_evaluate_aggregates_evidence_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RuleBasedMetrics,
        "semantic_fact_accuracy",
        staticmethod(lambda report, ground_truth=None, threshold=0.65: 0.75),
    )

    bench = ResearchBench()
    results = bench.batch_evaluate(
        [
            {
                "question_id": "tech_001",
                "report": "GPT-4o and Claude 3.5 differ in long context and coding.",
                "report_metadata": {"evidence_snapshot": _sample_snapshot()},
            }
        ]
    )

    assert "average_evidence_metrics" in results
    assert results["average_evidence_metrics"]["claim_count"] == 2.0
    assert results["average_metrics"]["evidence_graph_quality"] > 0.0
