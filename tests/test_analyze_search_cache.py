from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_search_cache import render_markdown, summarize_cache_records  # noqa: E402


def _record(query_id: str, *, browser: bool, citation: float, grounding: float = 0.0) -> dict:
    policy_trace = [
        {"action_type": "assistant_response", "tool_calls_count": 1},
        {
            "action_type": "tool_call",
            "tool_name": "web_search",
            "query_text": f"query {query_id}",
            "backend": "mock",
            "latency_ms": 100,
            "result_count": 5,
            "top_urls": ["https://example.com/a"],
            "estimated_token_cost": 100,
        },
    ]
    if browser:
        policy_trace.append(
            {
                "action_type": "tool_call",
                "tool_name": "browser",
                "query_text": "",
                "backend": "browser",
                "latency_ms": 120,
                "result_count": 1,
                "top_urls": ["https://example.com/a"],
                "estimated_token_cost": 60,
            }
        )
    policy_trace.append({"action_type": "stop", "stop_reason": "done"})

    return {
        "status": "success",
        "cache_id": f"cache_{query_id}",
        "query_id": query_id,
        "query": f"query {query_id}",
        "domain": "科技",
        "source_label": "research_bench",
        "confidence": 0.42,
        "report_metadata": {"policy_trace": policy_trace},
        "evaluation": {
            "composite_score": 0.51,
            "metrics": {
                "citation_coverage": citation,
                "factual_accuracy": 0.33,
            },
            "search_policy_metrics": {
                "citation_grounding": grounding,
            },
        },
    }


def test_summarize_cache_records_flags_low_browser_and_zero_citation() -> None:
    records = [
        _record("tech_001", browser=False, citation=0.0),
        _record("tech_002", browser=False, citation=0.0),
    ]

    summary = summarize_cache_records(records, [Path("sample.jsonl")])

    assert summary["num_success"] == 2
    assert summary["label_distribution"]["search"] == 2
    assert summary["label_distribution"]["stop"] == 2
    assert summary["browser_query_coverage"] == 0.0
    assert summary["nonzero_composite_query_ratio"] == 1.0
    assert summary["nonzero_citation_query_ratio"] == 0.0
    assert summary["mock_source_query_ratio"] == 1.0
    assert summary["real_source_query_ratio"] == 0.0
    assert "browser_query_coverage_below_25pct" in summary["risk_flags"]
    assert "citation_coverage_near_zero" in summary["risk_flags"]
    assert "mock_source_detected" in summary["risk_flags"]
    assert "real_source_query_ratio_below_80pct" in summary["risk_flags"]


def test_render_markdown_contains_policy_and_query_tables() -> None:
    summary = summarize_cache_records(
        [_record("tech_001", browser=True, citation=0.2, grounding=1.0)],
        [Path("sample.jsonl")],
    )

    md = render_markdown(summary)
    assert "## Policy Labels" in md
    assert "## Evidence Integrity" in md
    assert "## Per Query Snapshot" in md
    assert "tech_001" in md
    assert "grounding" in md


def test_summarize_cache_records_surfaces_grounding_render_gap() -> None:
    summary = summarize_cache_records(
        [
            _record("tech_001", browser=True, citation=0.0, grounding=1.0),
            _record("tech_002", browser=True, citation=0.0, grounding=1.0),
        ],
        [Path("sample.jsonl")],
    )

    assert summary["nonzero_citation_grounding_query_ratio"] == 1.0
    assert summary["explicit_grounded_query_ratio"] == 0.0
    assert summary["latent_grounded_query_ratio"] == 1.0
    assert summary["avg_citation_grounding"] == 1.0
    assert summary["avg_grounding_render_gap"] == 1.0
    assert "grounding_render_gap_high" in summary["risk_flags"]


def test_summarize_cache_records_distinguishes_real_sources() -> None:
    record = _record("tech_real", browser=True, citation=0.2, grounding=1.0)
    record["query"] = "OpenAI research example source"
    trace = record["report_metadata"]["policy_trace"]
    for action in trace:
        if action.get("action_type") == "tool_call":
            action["backend"] = "serpapi" if action.get("tool_name") == "web_search" else "browser"
            action["top_urls"] = ["https://openai.com/research/example"]
    record["sources"] = [
        {
            "url": "https://openai.com/research/example",
            "title": "Real source",
            "snippet": "A real source snippet.",
        }
    ]
    record["report_content"] = (
        "这一段使用真实来源支撑关键结论。 [1]\n\n"
        "## 参考来源\n"
        "1. [Real source](https://openai.com/research/example)\n"
    )

    summary = summarize_cache_records([record], [Path("sample.jsonl")])

    assert summary["mock_source_query_ratio"] == 0.0
    assert summary["real_source_query_ratio"] == 1.0
    assert summary["avg_real_sources_per_success_query"] == 1.0
    assert summary["avg_relevant_sources_per_success_query"] == 1.0
    assert summary["relevant_source_query_ratio"] == 1.0
    assert summary["avg_cited_sources_per_success_query"] == 1.0
    assert summary["avg_source_utilization"] == 1.0
    assert "mock_source_detected" not in summary["risk_flags"]


def test_summarize_cache_records_flags_unrelated_real_sources() -> None:
    record = _record("tech_unrelated", browser=False, citation=0.2, grounding=1.0)
    record["query"] = "GitHub Copilot developer productivity code quality"
    trace = record["report_metadata"]["policy_trace"]
    for action in trace:
        if action.get("action_type") == "tool_call":
            action["backend"] = "bing_html"
            action["top_urls"] = ["https://example.org/cooking"]
    record["sources"] = [
        {
            "url": "https://example.org/cooking",
            "title": "Holiday cooking guide",
            "snippet": "Recipes and kitchen tips for a family dinner.",
        }
    ]

    summary = summarize_cache_records([record], [Path("sample.jsonl")])

    assert summary["real_source_query_ratio"] == 1.0
    assert summary["relevant_source_query_ratio"] == 0.0
    assert summary["avg_source_relevance"] == 0.0
    assert "source_relevance_low" in summary["risk_flags"]
    assert "relevant_source_inventory_thin" in summary["risk_flags"]
