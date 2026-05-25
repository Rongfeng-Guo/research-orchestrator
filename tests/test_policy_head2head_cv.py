from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_policy_head2head_cv import aggregate_cv_runs, build_query_folds, render_cv_markdown  # noqa: E402


def _record(
    query_id: str,
    *,
    domain: str,
    browser_calls: int,
    citation_coverage: float,
    composite_score: float = 0.5,
    factual_accuracy: float = 0.3,
) -> dict:
    return {
        "status": "success",
        "cache_id": f"cache_{query_id}",
        "query_id": query_id,
        "query": f"query for {query_id}",
        "domain": domain,
        "source_label": "file:data/queries/fast_probe8.jsonl",
        "expected_topics": [query_id],
        "ground_truth": {query_id: "fact"},
        "confidence": 0.5,
        "search_cost": {
            "browser_calls": browser_calls,
            "search_calls": 3,
        },
        "evaluation": {
            "composite_score": composite_score,
            "metrics": {
                "citation_coverage": citation_coverage,
                "factual_accuracy": factual_accuracy,
            },
        },
    }


def test_build_query_folds_creates_disjoint_balanced_heldout_sets() -> None:
    records = [
        _record("ai_001", domain="AI", browser_calls=1, citation_coverage=0.04),
        _record("ai_002", domain="AI", browser_calls=1, citation_coverage=0.03),
        _record("law_001", domain="法律", browser_calls=1, citation_coverage=0.03),
        _record("law_002", domain="法律", browser_calls=1, citation_coverage=0.02),
        _record("energy_001", domain="能源", browser_calls=1, citation_coverage=0.03),
        _record("energy_002", domain="能源", browser_calls=1, citation_coverage=0.02),
        _record("fin_001", domain="金融", browser_calls=1, citation_coverage=0.03),
        _record("fin_002", domain="金融", browser_calls=1, citation_coverage=0.02),
    ]

    prepared = build_query_folds(
        records,
        source_label="file:data/queries/fast_probe8.jsonl",
        num_folds=4,
        grounded_min_browser_calls=1,
        grounded_min_citation_coverage=0.02,
    )

    folds = prepared["folds"]
    heldout_sets = [set(fold["summary"]["heldout"]["query_ids"]) for fold in folds]
    assert len(folds) == 4
    assert all(len(heldout) == 2 for heldout in heldout_sets)

    union = set().union(*heldout_sets)
    assert len(union) == 8
    for idx, heldout in enumerate(heldout_sets):
        train_ids = set(folds[idx]["summary"]["train"]["query_ids"])
        assert heldout.isdisjoint(train_ids)
        for other_idx, other in enumerate(heldout_sets):
            if other_idx == idx:
                continue
            assert heldout.isdisjoint(other)

    assert prepared["summary"]["grounded_query_count"] == 8


def test_aggregate_cv_runs_merges_records_by_mode() -> None:
    payload = aggregate_cv_runs(
        [
            {
                "runs": [
                    {
                        "mode": "heuristic",
                        "records": [
                            {
                                "status": "success",
                                "score": {
                                    "composite_score": 0.4,
                                    "factual_accuracy": 0.0,
                                    "citation_coverage": 0.2,
                                    "search_policy_score": 0.5,
                                },
                                "cost": {"tool_calls": 2, "search_calls": 1, "browser_calls": 1, "estimated_token_cost": 100},
                                "metadata_summary": {"policy_stats": {"policy_advice_count": 1, "guardrail_trigger_count": 0, "policy_enforce_stop_count": 0}},
                                "elapsed_seconds": 10,
                            }
                        ],
                    },
                    {
                        "mode": "learned",
                        "records": [
                            {
                                "status": "success",
                                "score": {
                                    "composite_score": 0.6,
                                    "factual_accuracy": 0.0,
                                    "citation_coverage": 0.3,
                                    "search_policy_score": 0.55,
                                },
                                "cost": {"tool_calls": 2, "search_calls": 1, "browser_calls": 1, "estimated_token_cost": 90},
                                "metadata_summary": {"policy_stats": {"policy_advice_count": 2, "guardrail_trigger_count": 1, "policy_enforce_stop_count": 1}},
                                "elapsed_seconds": 8,
                            }
                        ],
                    },
                ]
            },
            {
                "runs": [
                    {
                        "mode": "heuristic",
                        "records": [
                            {
                                "status": "success",
                                "score": {
                                    "composite_score": 0.5,
                                    "factual_accuracy": 0.0,
                                    "citation_coverage": 0.25,
                                    "search_policy_score": 0.52,
                                },
                                "cost": {"tool_calls": 3, "search_calls": 2, "browser_calls": 1, "estimated_token_cost": 120},
                                "metadata_summary": {"policy_stats": {"policy_advice_count": 1, "guardrail_trigger_count": 0, "policy_enforce_stop_count": 0}},
                                "elapsed_seconds": 12,
                            }
                        ],
                    }
                ]
            },
        ]
    )

    by_mode = {run["mode"]: run for run in payload["runs"]}
    assert by_mode["heuristic"]["summary"]["num_total"] == 2
    assert by_mode["heuristic"]["summary"]["avg_composite_score"] == 0.45
    assert by_mode["heuristic"]["summary"]["avg_estimated_token_cost"] == 110.0
    assert by_mode["heuristic"]["summary"]["avg_policy_advice_count"] == 1.0
    assert by_mode["learned"]["summary"]["num_total"] == 1
    assert by_mode["learned"]["summary"]["avg_composite_score"] == 0.6
    assert by_mode["learned"]["summary"]["avg_guardrail_trigger_count"] == 1.0


def test_render_cv_markdown_includes_fold_delta_table() -> None:
    markdown = render_cv_markdown(
        {
            "created_at": "2026-05-23T00:00:00",
            "config_path": "configs/test.yaml",
            "source_label": "file:data/queries/test.jsonl",
            "fold_count": 1,
            "query_count": 2,
            "folds": [
                {
                    "fold_index": 1,
                    "summary": {
                        "fold_index": 1,
                        "train": {"num_queries": 6, "grounded_queries": 6},
                        "heldout": {
                            "num_queries": 2,
                            "grounded_queries": 2,
                            "query_ids": ["q1", "q2"],
                        },
                    },
                    "runs": [
                        {
                            "mode": "heuristic",
                            "summary": {
                                "avg_composite_score": 0.5,
                                "avg_citation_coverage": 0.2,
                                "avg_estimated_token_cost": 100.0,
                                "avg_tool_calls": 4.0,
                            },
                        },
                        {
                            "mode": "learned",
                            "summary": {
                                "avg_composite_score": 0.6,
                                "avg_citation_coverage": 0.3,
                                "avg_estimated_token_cost": 80.0,
                                "avg_tool_calls": 3.0,
                            },
                        },
                    ],
                }
            ],
        }
    )

    assert "## Fold Deltas" in markdown
    assert "| fold_01 | q1, q2 | +0.100 | +0.100 | -20.0 | -1.00 | quality up, cost down |" in markdown
