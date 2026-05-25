from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mine_natural_browser_queries import render_markdown, select_natural_browser_queries  # noqa: E402


def _record(
    query_id: str,
    *,
    source_label: str,
    domain: str,
    browser_calls: int,
    composite_score: float,
    citation_coverage: float,
    citation_grounding: float = 0.0,
    factual_accuracy: float = 0.3,
) -> dict:
    return {
        "status": "success",
        "cache_id": f"cache_{query_id}",
        "query_id": query_id,
        "query": f"query for {query_id}",
        "domain": domain,
        "source_label": source_label,
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
            "search_policy_metrics": {
                "citation_grounding": citation_grounding,
            },
        },
    }


def test_select_natural_browser_queries_filters_to_research_bench() -> None:
    records = [
        _record(
            "tech_001",
            source_label="research_bench",
            domain="科技",
            browser_calls=2,
            composite_score=0.52,
            citation_coverage=0.04,
            citation_grounding=1.0,
        ),
        _record(
            "tech_002",
            source_label="research_bench",
            domain="科技",
            browser_calls=0,
            composite_score=0.50,
            citation_coverage=0.02,
        ),
        _record(
            "ctl_001",
            source_label="file:data/queries/browser_positive_control.jsonl",
            domain="受控浏览",
            browser_calls=3,
            composite_score=0.0,
            citation_coverage=0.0,
            citation_grounding=1.0,
        ),
    ]

    selected, summary = select_natural_browser_queries(records)

    assert len(selected) == 1
    assert selected[0]["query_id"] == "tech_001"
    assert summary["num_browser_positive_candidates"] == 1
    assert summary["selected_domain_distribution"] == {"科技": 1}


def test_select_natural_browser_queries_respects_domain_cap() -> None:
    records = [
        _record(
            "tech_001",
            source_label="research_bench",
            domain="科技",
            browser_calls=2,
            composite_score=0.53,
            citation_coverage=0.01,
            citation_grounding=1.0,
        ),
        _record(
            "tech_002",
            source_label="research_bench",
            domain="科技",
            browser_calls=1,
            composite_score=0.49,
            citation_coverage=0.03,
            citation_grounding=0.5,
        ),
        _record(
            "fin_001",
            source_label="research_bench",
            domain="金融",
            browser_calls=1,
            composite_score=0.51,
            citation_coverage=0.02,
            citation_grounding=0.5,
        ),
    ]

    selected, summary = select_natural_browser_queries(records, max_per_domain=1)

    assert len(selected) == 2
    assert {item["domain"] for item in selected} == {"科技", "金融"}
    assert summary["exclusion_counts"]["trimmed_by_domain_cap"] == 1


def test_render_markdown_contains_selected_queries_table() -> None:
    _, summary = select_natural_browser_queries(
        [
            _record(
                "tech_001",
                source_label="research_bench",
                domain="科技",
                browser_calls=1,
                composite_score=0.52,
                citation_coverage=0.04,
                citation_grounding=1.0,
            )
        ]
    )

    md = render_markdown(summary)
    assert "## Selected Queries" in md
    assert "tech_001" in md
    assert "grounding" in md


def test_select_natural_browser_queries_can_mine_latent_grounded_queries() -> None:
    records = [
        _record(
            "edu_001",
            source_label="research_bench",
            domain="教育",
            browser_calls=2,
            composite_score=0.52,
            citation_coverage=0.04,
            citation_grounding=1.0,
        ),
        _record(
            "edu_002",
            source_label="research_bench",
            domain="教育",
            browser_calls=2,
            composite_score=0.50,
            citation_coverage=0.0,
            citation_grounding=1.0,
        ),
        _record(
            "edu_003",
            source_label="research_bench",
            domain="教育",
            browser_calls=2,
            composite_score=0.48,
            citation_coverage=0.0,
            citation_grounding=0.0,
        ),
    ]

    selected, summary = select_natural_browser_queries(
        records,
        min_citation_grounding=0.5,
        max_citation_coverage=0.0,
    )

    assert [item["query_id"] for item in selected] == ["edu_002"]
    assert summary["exclusion_counts"]["above_max_citation_threshold"] == 1
    assert summary["exclusion_counts"]["below_grounding_threshold"] == 1
