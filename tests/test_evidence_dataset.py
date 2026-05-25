from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evolution.evidence_dataset import EvidenceTransitionDatasetBuilder  # noqa: E402


def _sample_collected_item() -> dict:
    return {
        "query": "compare frontier models",
        "report_metadata": {"evidence_snapshot": {}},
        "final_evidence_snapshot": {"recommended_action": "stop"},
        "process_reward_trace": [0.35, 0.72],
        "evidence_transition_trace": [
            {
                "step": 1,
                "task_id": "search_1",
                "task_type": "search",
                "status": "success",
                "label_action": "search",
                "before": {
                    "uncertainty": 0.95,
                    "coverage_ratio": 0.10,
                    "evidence_strength": 0.28,
                    "open_conflicts": 2,
                    "claim_count": 1,
                    "support_edge_count": 1,
                    "contradiction_edge_count": 0,
                    "avg_source_trust": 0.52,
                    "missing_terms": ["models", "benchmarks"],
                    "candidate_actions": [
                        {"name": "search", "score": 0.82},
                        {"name": "verify", "score": 0.24},
                        {"name": "stop", "score": 0.10},
                    ],
                    "evidence_graph_quality": 0.21,
                    "claim_support_coverage": 0.20,
                    "consistency_score": 0.33,
                },
                "after": {
                    "uncertainty": 0.74,
                    "coverage_ratio": 0.40,
                    "evidence_strength": 0.56,
                    "open_conflicts": 1,
                    "claim_count": 2,
                    "support_edge_count": 2,
                    "contradiction_edge_count": 0,
                    "avg_source_trust": 0.68,
                    "missing_terms": ["benchmarks"],
                    "candidate_actions": [
                        {"name": "verify", "score": 0.74},
                        {"name": "search", "score": 0.31},
                        {"name": "stop", "score": 0.17},
                    ],
                    "recommended_action": "verify",
                    "evidence_graph_quality": 0.49,
                    "claim_support_coverage": 0.50,
                    "consistency_score": 0.58,
                },
                "delta": {
                    "uncertainty": -0.21,
                    "coverage_ratio": 0.30,
                    "open_conflicts": -1,
                    "evidence_strength": 0.28,
                },
            },
            {
                "step": 2,
                "task_id": "verify_1",
                "task_type": "verify",
                "status": "success",
                "label_action": "stop",
                "before": {
                    "uncertainty": 0.74,
                    "coverage_ratio": 0.40,
                    "evidence_strength": 0.56,
                    "open_conflicts": 1,
                    "claim_count": 2,
                    "support_edge_count": 2,
                    "contradiction_edge_count": 0,
                    "avg_source_trust": 0.68,
                    "missing_terms": ["benchmarks"],
                    "candidate_actions": [
                        {"name": "stop", "score": 0.71},
                        {"name": "verify", "score": 0.64},
                        {"name": "search", "score": 0.20},
                    ],
                    "evidence_graph_quality": 0.49,
                    "claim_support_coverage": 0.50,
                    "consistency_score": 0.58,
                },
                "after": {
                    "uncertainty": 0.31,
                    "coverage_ratio": 0.83,
                    "evidence_strength": 0.82,
                    "open_conflicts": 0,
                    "claim_count": 3,
                    "support_edge_count": 3,
                    "contradiction_edge_count": 0,
                    "avg_source_trust": 0.81,
                    "missing_terms": [],
                    "candidate_actions": [
                        {"name": "stop", "score": 0.88},
                        {"name": "verify", "score": 0.35},
                        {"name": "search", "score": 0.08},
                    ],
                    "recommended_action": "stop",
                    "evidence_graph_quality": 0.84,
                    "claim_support_coverage": 1.0,
                    "consistency_score": 0.90,
                },
                "delta": {
                    "uncertainty": -0.43,
                    "coverage_ratio": 0.43,
                    "open_conflicts": -1,
                    "evidence_strength": 0.26,
                },
            },
        ],
    }


def test_dataset_builder_flattens_transitions_and_roundtrips_jsonl(tmp_path: Path) -> None:
    builder = EvidenceTransitionDatasetBuilder()
    rows = builder.build_from_collected_batch([_sample_collected_item()])

    assert len(rows) == 2
    assert rows[0]["sample_id"] == "0000:0000"
    assert rows[0]["label_action"] == "search"
    assert rows[1]["label_action"] == "stop"
    assert len(rows[0]["feature_vector"]) == len(builder.FEATURE_NAMES)
    assert rows[0]["features"]["candidate_action_margin"] > 0.0
    assert rows[1]["reward"] > rows[0]["reward"]

    path = tmp_path / "evidence_transition_dataset.jsonl"
    exported = builder.export_jsonl(rows, path)
    loaded = builder.load_jsonl(exported)

    assert Path(exported).exists()
    assert len(loaded) == 2
    assert loaded[0]["query"] == "compare frontier models"

    summary = builder.summarize(loaded)
    assert summary["num_examples"] == 2
    assert summary["label_distribution"]["search"] == 1
    assert summary["label_distribution"]["stop"] == 1
