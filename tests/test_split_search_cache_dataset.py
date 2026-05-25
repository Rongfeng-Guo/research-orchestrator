from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.split_search_cache_dataset import render_markdown, split_records  # noqa: E402


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
        "source_label": "research_bench",
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


def test_split_records_preserves_grounded_examples_in_both_splits() -> None:
    records = [
        _record("tech_001", domain="科技", browser_calls=3, citation_coverage=0.04),
        _record("tech_002", domain="科技", browser_calls=3, citation_coverage=0.03),
        _record("edu_001", domain="教育", browser_calls=2, citation_coverage=0.03),
        _record("edu_002", domain="教育", browser_calls=2, citation_coverage=0.02),
        _record("med_001", domain="医疗", browser_calls=2, citation_coverage=0.025),
        _record("fin_001", domain="金融", browser_calls=3, citation_coverage=0.0),
        _record("fin_002", domain="金融", browser_calls=3, citation_coverage=0.0),
        _record("law_001", domain="法律", browser_calls=3, citation_coverage=0.0),
        _record("law_002", domain="法律", browser_calls=3, citation_coverage=0.0),
        _record("auto_001", domain="汽车", browser_calls=3, citation_coverage=0.0),
    ]

    split = split_records(
        records,
        heldout_ratio=0.3,
        min_heldout_size=3,
        grounded_min_browser_calls=2,
        grounded_min_citation_coverage=0.02,
        min_grounded_heldout=2,
        min_grounded_train=2,
    )

    summary = split["summary"]
    train_ids = set(summary["train"]["query_ids"])
    heldout_ids = set(summary["heldout"]["query_ids"])

    assert train_ids.isdisjoint(heldout_ids)
    assert len(train_ids) + len(heldout_ids) == 10
    assert summary["heldout"]["grounded_queries"] == 2
    assert summary["train"]["grounded_queries"] == 3


def test_split_records_keeps_query_metadata_for_followup_runs() -> None:
    split = split_records(
        [
            _record("tech_001", domain="科技", browser_calls=3, citation_coverage=0.04),
            _record("fin_001", domain="金融", browser_calls=3, citation_coverage=0.0),
            _record("law_001", domain="法律", browser_calls=3, citation_coverage=0.0),
        ],
        heldout_ratio=0.34,
        min_heldout_size=1,
        grounded_min_browser_calls=2,
        grounded_min_citation_coverage=0.02,
        min_grounded_heldout=1,
        min_grounded_train=0,
    )

    train_query = split["train_queries"][0]
    assert "query" in train_query
    assert "expected_topics" in train_query
    assert "ground_truth" in train_query


def test_render_markdown_contains_split_sections() -> None:
    split = split_records(
        [
            _record("tech_001", domain="科技", browser_calls=3, citation_coverage=0.04),
            _record("fin_001", domain="金融", browser_calls=3, citation_coverage=0.0),
            _record("law_001", domain="法律", browser_calls=3, citation_coverage=0.0),
        ],
        heldout_ratio=0.34,
        min_heldout_size=1,
        grounded_min_browser_calls=2,
        grounded_min_citation_coverage=0.02,
        min_grounded_heldout=1,
        min_grounded_train=0,
    )

    md = render_markdown(split["summary"])
    assert "## Split Sizes" in md
    assert "## Held-out Query IDs" in md
