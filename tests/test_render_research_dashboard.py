from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_research_dashboard import build_dashboard, write_html, write_markdown  # noqa: E402


def test_build_dashboard_surfaces_blockers_and_leaderboard() -> None:
    brief_payload = {
        "benchmark_size": 35,
        "benchmark_domain_count": 11,
        "benchmark_domain_imbalance_ratio": 8.0,
        "recommended_policy_artifact": "search_policy_20260515_natural_train_v1.json",
        "recommended_policy_eval_accuracy": 0.9516,
        "head2head_label": "policy_head2head_natural_train_v1_balanced",
        "head2head_query_count": 5,
        "head2head_domains": ["科技", "医疗", "金融"],
        "quality_delta": 0.0,
        "macro_quality_delta": 0.0,
        "blocked_by_preflight": True,
        "missing_backends": ["openai"],
        "cross_fold_label": "policy_head2head_cv_fast_probe8_live",
        "cross_fold_query_count": 8,
        "cross_fold_count": 4,
        "cross_fold_quality_delta": 0.005,
        "cross_fold_macro_quality_delta": 0.004,
        "key_findings": ["finding-a"],
    }
    index_payload = {
        "artifacts": [
            {
                "artifact_type": "policy_head2head",
                "label": "policy_head2head_natural_train_v1_balanced",
                "created_at": "2026-05-31T16:53:21",
                "query_count": 5,
                "sampling_strategy": "balanced_domains",
                "blocked_by_preflight": True,
                "missing_backends": ["openai"],
                "quality_delta": 0.0,
                "macro_quality_delta": 0.0,
                "domains": ["科技", "医疗", "金融"],
            },
            {
                "artifact_type": "policy_head2head_cv",
                "label": "policy_head2head_cv_fast_probe8_live",
                "created_at": "2026-05-23T11:00:00",
                "query_count": 8,
                "fold_count": 4,
                "source_label": "file:data/queries/fast_probe8.jsonl",
                "blocked_by_preflight": False,
                "missing_backends": [],
                "quality_delta": 0.005,
                "macro_quality_delta": 0.004,
                "domains": ["科技", "医疗", "金融", "能源"],
            }
        ]
    }
    audit_payload = {
        "benchmark": {
            "domain_distribution": {
                "科技": 8,
                "医疗": 6,
                "金融": 6,
                "传媒": 1,
            }
        },
        "policy_artifacts": {
            "strongest_eval_artifact": {
                "artifact": "search_policy_20260515_natural_train_v1.json",
                "eval_accuracy": 0.9516,
            },
            "artifacts": [
                {
                    "artifact": "search_policy.json",
                    "eval_accuracy": 0.7143,
                    "train_accuracy": 0.85,
                    "num_examples": 20,
                    "browser_examples": 1,
                    "label_imbalance_ratio": 12.0,
                },
                {
                    "artifact": "search_policy_20260515_natural_train_v1.json",
                    "eval_accuracy": 0.9516,
                    "train_accuracy": 0.9358,
                    "num_examples": 187,
                    "browser_examples": 60,
                    "label_imbalance_ratio": 1.067,
                },
            ],
        },
    }

    dashboard = build_dashboard(brief_payload, index_payload, audit_payload)

    assert dashboard["blockers"]["blocked_by_preflight"] is True
    assert dashboard["blockers"]["missing_backends"] == ["openai"]
    assert dashboard["policy_leaderboard"][0]["artifact"] == "search_policy_20260515_natural_train_v1.json"
    assert dashboard["benchmark_domain_table"][0]["domain"] == "科技"
    assert dashboard["benchmark_domain_table"][-1]["domain"] == "传媒"
    assert dashboard["run_registry"][0]["sampling_strategy"] == "balanced_domains"
    assert dashboard["run_registry"][1]["artifact_type"] == "policy_head2head_cv"
    assert dashboard["run_registry"][1]["fold_count"] == 4
    assert dashboard["run_registry"][1]["sampling_strategy"] == "cross_fold"
    assert dashboard["headline_metrics"]["latest_cross_fold_label"] == "policy_head2head_cv_fast_probe8_live"
    assert dashboard["headline_metrics"]["latest_cross_fold_count"] == 4
    assert dashboard["headline_metrics"]["latest_cross_fold_quality_delta"] == 0.005


def test_write_html_renders_major_sections(tmp_path: Path) -> None:
    dashboard = {
        "created_at": "2026-05-31T17:20:00",
        "headline_metrics": {
            "benchmark_size": 35,
            "benchmark_domain_count": 11,
            "benchmark_domain_imbalance_ratio": 8.0,
            "recommended_policy_artifact": "search_policy_20260515_natural_train_v1.json",
            "recommended_policy_eval_accuracy": 0.9516,
            "latest_head2head_label": "policy_head2head_natural_train_v1_balanced",
            "latest_head2head_query_count": 5,
            "latest_quality_delta": 0.0,
            "latest_macro_quality_delta": 0.0,
            "latest_cross_fold_label": "policy_head2head_cv_fast_probe8_live",
            "latest_cross_fold_query_count": 8,
            "latest_cross_fold_count": 4,
            "latest_cross_fold_quality_delta": 0.005,
            "latest_cross_fold_macro_quality_delta": 0.004,
        },
        "blockers": {
            "blocked_by_preflight": True,
            "missing_backends": ["openai"],
            "summary": "Live learned-vs-heuristic evaluation is blocked by missing backend configuration: openai.",
        },
        "benchmark_domain_table": [{"domain": "科技", "count": 8, "bar": "########################"}],
        "policy_leaderboard": [
            {
                "artifact": "search_policy_20260515_natural_train_v1.json",
                "eval_accuracy": 0.9516,
                "train_accuracy": 0.9358,
                "num_examples": 187,
                "browser_examples": 60,
                "label_imbalance_ratio": 1.067,
            }
        ],
        "run_registry": [
            {
                "artifact_type": "policy_head2head",
                "label": "policy_head2head_natural_train_v1_balanced",
                "created_at": "2026-05-31T16:53:21",
                "query_count": 5,
                "fold_count": 0,
                "sampling_strategy": "balanced_domains",
                "blocked_by_preflight": True,
                "quality_delta": 0.0,
                "macro_quality_delta": 0.0,
                "missing_backends": ["openai"],
            },
            {
                "artifact_type": "policy_head2head_cv",
                "label": "policy_head2head_cv_fast_probe8_live",
                "created_at": "2026-05-23T11:00:00",
                "query_count": 8,
                "fold_count": 4,
                "sampling_strategy": "cross_fold",
                "blocked_by_preflight": False,
                "quality_delta": 0.005,
                "macro_quality_delta": 0.004,
                "missing_backends": [],
            }
        ],
        "latest_run_domains": ["科技", "医疗"],
        "key_findings": ["finding-a"],
        "next_actions": ["action-a"],
    }

    output_path = tmp_path / "research_dashboard.html"
    write_html(dashboard, output_path)
    rendered = output_path.read_text(encoding="utf-8")

    assert "<title>Research Dashboard</title>" in rendered
    assert "Policy Artifact Leaderboard" in rendered
    assert "balanced_domains" in rendered
    assert "policy_head2head_cv_fast_probe8_live" in rendered
    assert "cross_fold" in rendered
    assert "openai" in rendered


def test_write_markdown_includes_cross_fold_headline(tmp_path: Path) -> None:
    dashboard = {
        "created_at": "2026-05-31T17:20:00",
        "headline_metrics": {
            "benchmark_size": 35,
            "benchmark_domain_count": 11,
            "benchmark_domain_imbalance_ratio": 8.0,
            "recommended_policy_artifact": "search_policy_20260515_natural_train_v1.json",
            "recommended_policy_eval_accuracy": 0.9516,
            "latest_head2head_label": "policy_head2head_natural_train_v1_balanced",
            "latest_head2head_query_count": 5,
            "latest_quality_delta": 0.0,
            "latest_macro_quality_delta": 0.0,
            "latest_cross_fold_label": "policy_head2head_cv_fast_probe8_live",
            "latest_cross_fold_query_count": 8,
            "latest_cross_fold_count": 4,
            "latest_cross_fold_quality_delta": 0.005,
            "latest_cross_fold_macro_quality_delta": 0.004,
        },
        "blockers": {"summary": "No blockers recorded."},
        "benchmark_domain_table": [],
        "policy_leaderboard": [],
        "run_registry": [],
        "key_findings": [],
        "next_actions": [],
    }
    output_path = tmp_path / "research_dashboard.md"

    write_markdown(dashboard, output_path)
    markdown = output_path.read_text(encoding="utf-8")

    assert "Latest cross-fold" in markdown
    assert "policy_head2head_cv_fast_probe8_live" in markdown
    assert "Cross-fold deltas: quality=+0.0050, macro=+0.0040" in markdown
