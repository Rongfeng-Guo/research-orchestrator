#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/mine_natural_browser_queries.py
================================================================================
Mine naturally occurring browser-positive queries from existing search-cache JSONL.

This script is intentionally cache-first:
  1) run a natural benchmark slice with build_search_cache.py
  2) mine browser-positive natural queries from that cache
  3) write a reusable query file for the next cache-building round

Usage:
  python scripts/mine_natural_browser_queries.py \
    --inputs data/search_cache/natural_heuristic_20260515 \
    --source-label research_bench \
    --output-dir outputs/natural_browser_query_mining_20260515 \
    --output-queries-file data/queries/natural_browser_positive_20260515.jsonl
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_search_cache import _load_records, _resolve_jsonl_inputs


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_policy_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    structured = record.get("structured_actions", [])
    if isinstance(structured, list) and structured:
        return [item for item in structured if isinstance(item, dict)]

    metadata = record.get("report_metadata", {})
    if isinstance(metadata, dict):
        policy_trace = metadata.get("policy_trace", [])
        if isinstance(policy_trace, list):
            return [item for item in policy_trace if isinstance(item, dict)]

    return []


def _extract_search_policy_metrics(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation", {})
    if isinstance(evaluation, dict):
        search_policy_metrics = evaluation.get("search_policy_metrics", {})
        if isinstance(search_policy_metrics, dict):
            return search_policy_metrics
    return {}


def _count_browser_calls(record: dict[str, Any]) -> int:
    search_cost = record.get("search_cost", {})
    if isinstance(search_cost, dict) and search_cost.get("browser_calls") is not None:
        return _safe_int(search_cost.get("browser_calls"))

    metadata = record.get("report_metadata", {})
    if isinstance(metadata, dict):
        nested_cost = metadata.get("search_cost", {})
        if isinstance(nested_cost, dict) and nested_cost.get("browser_calls") is not None:
            return _safe_int(nested_cost.get("browser_calls"))

    count = 0
    for action in _extract_policy_trace(record):
        if str(action.get("action_type", "") or "") != "tool_call":
            continue
        if str(action.get("tool_name", "") or "") == "browser":
            count += 1
    return count


def _count_search_calls(record: dict[str, Any]) -> int:
    search_cost = record.get("search_cost", {})
    if isinstance(search_cost, dict) and search_cost.get("search_calls") is not None:
        return _safe_int(search_cost.get("search_calls"))

    metadata = record.get("report_metadata", {})
    if isinstance(metadata, dict):
        nested_cost = metadata.get("search_cost", {})
        if isinstance(nested_cost, dict) and nested_cost.get("search_calls") is not None:
            return _safe_int(nested_cost.get("search_calls"))

    count = 0
    for action in _extract_policy_trace(record):
        if str(action.get("action_type", "") or "") != "tool_call":
            continue
        if str(action.get("tool_name", "") or "") in {"web_search", "arxiv_reader"}:
            count += 1
    return count


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation", {})
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    search_policy_metrics = _extract_search_policy_metrics(record)
    browser_calls = _count_browser_calls(record)
    search_calls = _count_search_calls(record)
    citation_coverage = _safe_float(metrics.get("citation_coverage", 0.0))
    citation_grounding = _safe_float(search_policy_metrics.get("citation_grounding", 0.0))

    return {
        "query_id": str(record.get("query_id", record.get("cache_id", "")) or ""),
        "query": str(record.get("query", "") or ""),
        "domain": str(record.get("domain", "") or ""),
        "source_label": str(record.get("source_label", "") or ""),
        "browser_calls": browser_calls,
        "search_calls": search_calls,
        "composite_score": _safe_float(evaluation.get("composite_score", 0.0)),
        "citation_coverage": citation_coverage,
        "citation_grounding": citation_grounding,
        "grounding_render_gap": citation_grounding - citation_coverage,
        "factual_accuracy": _safe_float(metrics.get("factual_accuracy", 0.0)),
        "confidence": _safe_float(record.get("confidence", 0.0)),
        "expected_topics": record.get("expected_topics"),
        "ground_truth": record.get("ground_truth"),
    }


def select_natural_browser_queries(
    records: list[dict[str, Any]],
    *,
    source_label: str = "research_bench",
    min_browser_calls: int = 1,
    min_composite_score: float = 0.0,
    min_citation_coverage: float = 0.0,
    min_citation_grounding: float = 0.0,
    min_factual_accuracy: float = 0.0,
    max_citation_coverage: float | None = None,
    max_per_domain: int | None = None,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_candidates: list[dict[str, Any]] = []
    counts = Counter()

    for record in records:
        if str(record.get("status", "success") or "success") != "success":
            counts["failed_or_non_success"] += 1
            continue

        summary = summarize_record(record)
        if summary["source_label"] != source_label:
            counts["non_target_source"] += 1
            continue
        if not summary["query"]:
            counts["missing_query"] += 1
            continue
        if summary["browser_calls"] < min_browser_calls:
            counts["below_browser_threshold"] += 1
            continue
        if summary["composite_score"] < min_composite_score:
            counts["below_composite_threshold"] += 1
            continue
        if summary["citation_coverage"] < min_citation_coverage:
            counts["below_citation_threshold"] += 1
            continue
        if summary["citation_grounding"] < min_citation_grounding:
            counts["below_grounding_threshold"] += 1
            continue
        if max_citation_coverage is not None and summary["citation_coverage"] > max_citation_coverage:
            counts["above_max_citation_threshold"] += 1
            continue
        if summary["factual_accuracy"] < min_factual_accuracy:
            counts["below_factual_threshold"] += 1
            continue

        selected_candidates.append(summary)

    selected_candidates.sort(
        key=lambda item: (
            item["citation_grounding"],
            item["citation_coverage"],
            item["browser_calls"],
            item["composite_score"],
            item["factual_accuracy"],
            item["confidence"],
        ),
        reverse=True,
    )

    final_selected: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    for item in selected_candidates:
        domain = item["domain"]
        if max_per_domain is not None and domain_counts[domain] >= max_per_domain:
            counts["trimmed_by_domain_cap"] += 1
            continue
        final_selected.append(item)
        domain_counts[domain] += 1
        if top_k is not None and len(final_selected) >= top_k:
            break

    summary = {
        "created_at": datetime.now().isoformat(),
        "source_label": source_label,
        "num_input_records": len(records),
        "num_browser_positive_candidates": len(selected_candidates),
        "num_selected_queries": len(final_selected),
        "filters": {
            "min_browser_calls": min_browser_calls,
            "min_composite_score": min_composite_score,
            "min_citation_coverage": min_citation_coverage,
            "min_citation_grounding": min_citation_grounding,
            "min_factual_accuracy": min_factual_accuracy,
            "max_citation_coverage": max_citation_coverage,
            "max_per_domain": max_per_domain,
            "top_k": top_k,
        },
        "exclusion_counts": dict(counts),
        "selected_domain_distribution": dict(sorted(domain_counts.items())),
        "selected_queries": final_selected,
    }
    return final_selected, summary


def render_markdown(summary: dict[str, Any]) -> str:
    filters = summary.get("filters", {})
    lines = [
        "# Natural Browser-Positive Query Mining",
        "",
        f"- created_at: {summary.get('created_at', '')}",
        f"- source_label: {summary.get('source_label', '')}",
        f"- num_input_records: {summary.get('num_input_records', 0)}",
        f"- num_browser_positive_candidates: {summary.get('num_browser_positive_candidates', 0)}",
        f"- num_selected_queries: {summary.get('num_selected_queries', 0)}",
        "",
        "## Filters",
        "",
        f"- min_browser_calls: {filters.get('min_browser_calls', 0)}",
        f"- min_composite_score: {filters.get('min_composite_score', 0.0):.4f}",
        f"- min_citation_coverage: {filters.get('min_citation_coverage', 0.0):.4f}",
        f"- min_citation_grounding: {filters.get('min_citation_grounding', 0.0):.4f}",
        f"- min_factual_accuracy: {filters.get('min_factual_accuracy', 0.0):.4f}",
        f"- max_citation_coverage: {filters.get('max_citation_coverage')}",
        f"- max_per_domain: {filters.get('max_per_domain')}",
        f"- top_k: {filters.get('top_k')}",
        "",
        "## Selected Queries",
        "",
        "| query_id | domain | browser_calls | citation | grounding | gap | composite | factual |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in summary.get("selected_queries", []):
        lines.append(
            f"| {item.get('query_id', '')} | {item.get('domain', '')} | "
            f"{item.get('browser_calls', 0)} | {item.get('citation_coverage', 0.0):.4f} | "
            f"{item.get('citation_grounding', 0.0):.4f} | {item.get('grounding_render_gap', 0.0):.4f} | "
            f"{item.get('composite_score', 0.0):.4f} | {item.get('factual_accuracy', 0.0):.4f} |"
        )

    lines.extend(["", "## Exclusion Counts", ""])
    exclusion_counts = summary.get("exclusion_counts", {})
    if exclusion_counts:
        for key, value in exclusion_counts.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def _write_queries_file(path: Path, queries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine natural browser-positive queries from search-cache JSONL")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL files or directories")
    parser.add_argument("--source-label", type=str, default="research_bench", help="only keep records from this source label")
    parser.add_argument("--min-browser-calls", type=int, default=1)
    parser.add_argument("--min-composite-score", type=float, default=0.0)
    parser.add_argument("--min-citation-coverage", type=float, default=0.0)
    parser.add_argument("--min-citation-grounding", type=float, default=0.0)
    parser.add_argument("--min-factual-accuracy", type=float, default=0.0)
    parser.add_argument("--max-citation-coverage", type=float, default=None)
    parser.add_argument("--max-per-domain", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-queries-file", type=str, default=None)
    args = parser.parse_args()

    input_paths = _resolve_jsonl_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("no JSONL inputs found")

    records = _load_records(input_paths)
    selected, summary = select_natural_browser_queries(
        records,
        source_label=args.source_label,
        min_browser_calls=args.min_browser_calls,
        min_composite_score=args.min_composite_score,
        min_citation_coverage=args.min_citation_coverage,
        min_citation_grounding=args.min_citation_grounding,
        min_factual_accuracy=args.min_factual_accuracy,
        max_citation_coverage=args.max_citation_coverage,
        max_per_domain=args.max_per_domain,
        top_k=args.top_k,
    )
    summary["input_paths"] = [str(path) for path in input_paths]

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        json_path = output_dir / f"natural_browser_query_mining_{timestamp}.json"
        md_path = output_dir / f"natural_browser_query_mining_{timestamp}.md"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(summary), encoding="utf-8")
        print(str(md_path))

    if args.output_queries_file:
        output_query_path = Path(args.output_queries_file)
        _write_queries_file(output_query_path, selected)
        print(str(output_query_path))


if __name__ == "__main__":
    main()
