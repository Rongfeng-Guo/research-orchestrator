from __future__ import annotations

import json
from pathlib import Path

import pytest


def write_sample_research_outputs(outputs_dir: Path) -> Path:
    """Create a minimal research-output tree for reporting tests."""
    audit_dir = outputs_dir / "research_audit"
    head2head_dir = outputs_dir / "policy_head2head_natural_train_v1_balanced"
    cv_dir = outputs_dir / "policy_head2head_cv_fast_probe8_live"
    audit_dir.mkdir(parents=True, exist_ok=True)
    head2head_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    audit_payload = {
        "created_at": "2026-05-31T16:53:21",
        "benchmark": {
            "num_questions": 35,
            "num_domains": 3,
            "domain_distribution": {"technology": 2, "medicine": 2, "finance": 1},
            "domain_imbalance_ratio": 2.0,
        },
        "policy_artifacts": {
            "strongest_eval_artifact": {
                "artifact": "search_policy_20260515_natural_train_v1.json",
                "eval_accuracy": 0.75,
                "train_accuracy": 0.9,
                "num_examples": 120,
                "browser_examples": 20,
                "label_imbalance_ratio": 1.5,
            },
            "default_artifact": {
                "artifact": "search_policy.json",
                "num_examples": 20,
                "browser_examples": 1,
            },
            "artifacts": [
                {
                    "artifact": "search_policy_20260515_natural_train_v1.json",
                    "eval_accuracy": 0.75,
                    "train_accuracy": 0.9,
                    "num_examples": 120,
                    "browser_examples": 20,
                    "label_imbalance_ratio": 1.5,
                }
            ],
        },
        "key_findings": [
            "Default policy artifact is too small for stable browser decisions.",
            "Natural train policy is currently the strongest evaluated artifact.",
        ],
    }
    (audit_dir / "research_readiness_audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "research_readiness_audit.md").write_text("# Research Readiness Audit\n", encoding="utf-8")

    head2head_payload = {
        "created_at": "2026-05-31T16:53:21",
        "config_path": "configs/real_evidence_fast.yaml",
        "query_count": 5,
        "sampling_strategy": "balanced_domains",
        "queries": [
            {"id": "q1", "domain": "technology"},
            {"id": "q2", "domain": "technology"},
            {"id": "q3", "domain": "medicine"},
            {"id": "q4", "domain": "medicine"},
            {"id": "q5", "domain": "finance"},
        ],
        "runs": [
            {
                "mode": "heuristic",
                "summary": {
                    "num_success": 5,
                    "avg_composite_score": 0.45,
                    "domain_macro_avg_composite_score": 0.44,
                },
                "preflight": {"is_ready": True, "missing_backends": []},
            },
            {
                "mode": "learned",
                "summary": {
                    "num_success": 0,
                    "avg_composite_score": 0.45,
                    "domain_macro_avg_composite_score": 0.44,
                },
                "preflight": {"is_ready": False, "missing_backends": ["openai"]},
            },
        ],
    }
    (head2head_dir / "head2head_20260531_165321.json").write_text(
        json.dumps(head2head_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (head2head_dir / "head2head_20260531_165321.md").write_text("# Head-to-Head\n", encoding="utf-8")

    cv_payload = {
        "created_at": "2026-05-23T11:00:00",
        "config_path": "configs/real_evidence_fast.yaml",
        "source_label": "file:data/queries/fast_probe8.jsonl",
        "fold_count": 4,
        "query_count": 8,
        "folds": [
            {
                "summary": {
                    "train": {"domain_distribution": {"technology": 1, "medicine": 1}},
                    "heldout": {"domain_distribution": {"finance": 1}, "query_ids": ["q1", "q2"]},
                }
            }
        ],
        "aggregate_head2head": {
            "runs": [
                {
                    "mode": "heuristic",
                    "summary": {
                        "avg_composite_score": 0.5,
                        "domain_macro_avg_composite_score": 0.48,
                    },
                    "preflight": {"is_ready": True, "missing_backends": []},
                },
                {
                    "mode": "learned",
                    "summary": {
                        "avg_composite_score": 0.555,
                        "domain_macro_avg_composite_score": 0.53,
                    },
                    "preflight": {"is_ready": True, "missing_backends": []},
                },
            ]
        },
    }
    (cv_dir / "cv_summary.json").write_text(
        json.dumps(cv_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (cv_dir / "cv_summary.md").write_text("# Policy Head-to-Head Cross-Fold Report\n", encoding="utf-8")
    return outputs_dir


@pytest.fixture
def sample_research_outputs(tmp_path: Path) -> Path:
    return write_sample_research_outputs(tmp_path / "sample_outputs")
