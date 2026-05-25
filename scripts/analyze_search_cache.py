#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/analyze_search_cache.py
================================================================================
Analyze search-cache JSONL artifacts for policy-training readiness.

The script summarizes:
  - record/query counts and success rate
  - step-level label distribution (search / browser / stop)
  - browser usage coverage
  - average evaluation metrics when available
  - experiment risk flags such as low browser coverage or too-small sample size

Usage:
  python scripts/analyze_search_cache.py --inputs data/search_cache/smoke_20260514
  python scripts/analyze_search_cache.py --inputs file1.jsonl file2.jsonl --output_dir outputs/search_cache_analysis
================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.search_policy import SearchPolicyMetrics
from src.search_policy.dataset import SearchPolicyDatasetBuilder


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_jsonl_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file() and path.suffix.lower() == ".jsonl":
            paths.append(path)
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    builder = SearchPolicyDatasetBuilder()
    records: list[dict[str, Any]] = []
    for path in paths:
        for record in builder.load_jsonl(path):
            if isinstance(record, dict):
                record = dict(record)
                record["_source_path"] = str(path)
                records.append(record)
    return records


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
    metadata = record.get("report_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    sources = record.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    computed = SearchPolicyMetrics.breakdown(
        str(record.get("report_content", "") or ""),
        metadata,
        sources,
    )

    evaluation = record.get("evaluation", {})
    if isinstance(evaluation, dict):
        search_policy_metrics = evaluation.get("search_policy_metrics", {})
        if isinstance(search_policy_metrics, dict):
            return {**computed, **search_policy_metrics}
    return computed


def _extract_route_stats(record: dict[str, Any]) -> dict[str, Any]:
    route_stats = record.get("route_stats", {})
    if isinstance(route_stats, dict):
        return route_stats

    metadata = record.get("report_metadata", {})
    if isinstance(metadata, dict):
        route_stats = metadata.get("route_stats", {})
        if isinstance(route_stats, dict):
            return route_stats

    return {}


def _is_mock_backend(value: Any) -> bool:
    backend = str(value or "").strip().lower()
    return backend in {"mock", "fake", "fixture", "stub"} or "mock" in backend


def _is_mock_url(value: Any) -> bool:
    url = str(value or "").strip().lower()
    if not url:
        return False
    return (
        "example.com/mock" in url
        or url.startswith("mock://")
        or "localhost/mock" in url
    )


def _is_mock_source(source: Any) -> bool:
    if not isinstance(source, dict):
        return False
    title = str(source.get("title", "") or "").lower()
    snippet = str(source.get("snippet", "") or "").lower()
    return _is_mock_url(source.get("url", "")) or "mock result" in title or "mock search result" in snippet


_LATIN_STOPWORDS = {
    "and", "are", "for", "from", "how", "into", "the", "this", "that", "with",
    "2023", "2024", "2025", "2026", "vs", "www", "com", "http", "https",
}


def _relevance_terms(text: Any) -> set[str]:
    raw = str(text or "").lower()
    terms: set[str] = set()
    for token in re.findall(r"[a-z][a-z0-9+#.-]{2,}", raw):
        cleaned = token.strip(".-_")
        if cleaned and cleaned not in _LATIN_STOPWORDS:
            terms.add(cleaned)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", raw):
        if len(chunk) <= 4:
            terms.add(chunk)
        for idx in range(0, max(len(chunk) - 1, 0)):
            terms.add(chunk[idx:idx + 2])
    return terms


def _record_relevance_terms(record: dict[str, Any]) -> set[str]:
    parts: list[str] = [str(record.get("query", "") or "")]
    expected_topics = record.get("expected_topics", [])
    if isinstance(expected_topics, list):
        parts.extend(str(topic or "") for topic in expected_topics)
    route_stats = _extract_route_stats(record)
    queries_used = route_stats.get("queries_used", [])
    if isinstance(queries_used, list):
        parts.extend(str(query or "") for query in queries_used[:3])
    return _relevance_terms(" ".join(parts))


def _score_source_relevance(source: dict[str, Any], query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    text = " ".join(
        [
            str(source.get("title", "") or ""),
            str(source.get("snippet", "") or ""),
            str(source.get("url", "") or ""),
            str(source.get("domain", "") or ""),
        ]
    )
    source_terms = _relevance_terms(text)
    if not source_terms:
        return 0.0
    overlap = len(query_terms & source_terms)
    denominator = max(4, min(12, len(query_terms)))
    return min(1.0, overlap / denominator)


def _source_relevance(record: dict[str, Any]) -> dict[str, Any]:
    sources = record.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    query_terms = _record_relevance_terms(record)
    scores = [
        _score_source_relevance(source, query_terms)
        for source in sources
        if isinstance(source, dict) and not _is_mock_source(source)
    ]
    relevant_scores = [score for score in scores if score >= 0.15]
    return {
        "avg_source_relevance": _mean(scores),
        "relevant_source_count": len(relevant_scores),
        "has_relevant_source": bool(relevant_scores),
    }


def _evidence_integrity(record: dict[str, Any], policy_trace: list[dict[str, Any]]) -> dict[str, Any]:
    route_stats = _extract_route_stats(record)
    sources = record.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    uses_mock_backend = False
    mock_urls: set[str] = set()
    real_urls: set[str] = set()

    for backend in route_stats.get("search_backends", []) if isinstance(route_stats.get("search_backends"), list) else []:
        if _is_mock_backend(backend):
            uses_mock_backend = True

    route_top_urls = route_stats.get("top_urls", [])
    if not isinstance(route_top_urls, list):
        route_top_urls = []
    for url in route_top_urls:
        url_text = str(url or "").strip()
        if not url_text:
            continue
        if uses_mock_backend or _is_mock_url(url_text):
            mock_urls.add(url_text)
        else:
            real_urls.add(url_text)

    for source in sources:
        if not isinstance(source, dict):
            continue
        url_text = str(source.get("url", "") or "").strip()
        if not url_text:
            continue
        if _is_mock_source(source):
            mock_urls.add(url_text)
        else:
            real_urls.add(url_text)

    for action in policy_trace:
        if str(action.get("action_type", "") or "") != "tool_call":
            continue
        action_uses_mock = _is_mock_backend(action.get("backend", ""))
        if action_uses_mock:
            uses_mock_backend = True
        top_urls = action.get("top_urls", [])
        if not isinstance(top_urls, list):
            continue
        for url in top_urls:
            url_text = str(url or "").strip()
            if not url_text:
                continue
            if action_uses_mock or _is_mock_url(url_text):
                mock_urls.add(url_text)
            else:
                real_urls.add(url_text)

    return {
        "source_count": len(sources),
        "real_source_count": len(real_urls),
        "mock_source_count": len(mock_urls),
        "uses_mock_backend": uses_mock_backend,
        "has_mock_source": uses_mock_backend or bool(mock_urls) or any(_is_mock_source(source) for source in sources),
        "has_real_source": bool(real_urls),
    }


def summarize_cache_records(records: list[dict[str, Any]], source_paths: list[Path]) -> dict[str, Any]:
    builder = SearchPolicyDatasetBuilder()
    rows = builder.build_rows_from_records(records)

    success_records = [
        record for record in records
        if str(record.get("status", "success") or "success") == "success"
    ]
    failed_records = [
        record for record in records
        if str(record.get("status", "success") or "success") != "success"
    ]

    per_query: list[dict[str, Any]] = []
    domains = Counter()
    source_labels = Counter()
    label_counts = Counter(row.get("label_action", "") for row in rows)
    query_ids_with_browser = 0
    search_only_queries = 0
    no_step_queries = 0
    nonzero_composite_queries = 0
    nonzero_citation_queries = 0
    nonzero_citation_grounding_queries = 0
    explicit_grounded_queries = 0
    latent_grounded_queries = 0
    citation_scores: list[float] = []
    citation_grounding_scores: list[float] = []
    citation_density_scores: list[float] = []
    citation_quality_scores: list[float] = []
    source_utilization_scores: list[float] = []
    cited_source_counts: list[float] = []
    reference_source_counts: list[float] = []
    factual_scores: list[float] = []
    composite_scores: list[float] = []
    report_confidences: list[float] = []
    source_counts: list[float] = []
    real_source_counts: list[float] = []
    source_relevance_scores: list[float] = []
    relevant_source_counts: list[float] = []
    queries_with_mock_sources = 0
    queries_with_real_sources = 0
    queries_with_relevant_sources = 0
    queries_with_mock_backend = 0

    for record in success_records:
        query_id = str(record.get("query_id", record.get("cache_id", "")) or "")
        domain = str(record.get("domain", "") or "")
        source_label = str(record.get("source_label", "") or "")
        if domain:
            domains[domain] += 1
        if source_label:
            source_labels[source_label] += 1

        policy_trace = _extract_policy_trace(record)
        tool_actions = [
            action for action in policy_trace
            if str(action.get("action_type", "") or "") == "tool_call"
        ]
        search_calls = 0
        browser_calls = 0
        stop_calls = 0
        for action in policy_trace:
            label = builder._action_to_label(action)  # noqa: SLF001
            if label == "search":
                search_calls += 1
            elif label == "browser":
                browser_calls += 1
            elif label == "stop":
                stop_calls += 1

        if browser_calls > 0:
            query_ids_with_browser += 1
        if search_calls > 0 and browser_calls == 0:
            search_only_queries += 1
        if not policy_trace:
            no_step_queries += 1

        integrity = _evidence_integrity(record, policy_trace)
        source_counts.append(_safe_float(integrity.get("source_count", 0.0)))
        real_source_counts.append(_safe_float(integrity.get("real_source_count", 0.0)))
        if integrity.get("has_mock_source"):
            queries_with_mock_sources += 1
        if integrity.get("has_real_source"):
            queries_with_real_sources += 1
        if integrity.get("uses_mock_backend"):
            queries_with_mock_backend += 1

        relevance = _source_relevance(record)
        source_relevance_scores.append(_safe_float(relevance.get("avg_source_relevance", 0.0)))
        relevant_source_counts.append(_safe_float(relevance.get("relevant_source_count", 0.0)))
        if relevance.get("has_relevant_source"):
            queries_with_relevant_sources += 1

        search_policy_metrics = _extract_search_policy_metrics(record)
        citation_grounding = _safe_float(search_policy_metrics.get("citation_grounding", 0.0))
        citation_grounding_scores.append(citation_grounding)
        citation_density_scores.append(_safe_float(search_policy_metrics.get("citation_density", 0.0)))
        citation_quality_scores.append(_safe_float(search_policy_metrics.get("citation_quality_score", 0.0)))
        source_utilization_scores.append(_safe_float(search_policy_metrics.get("source_utilization", 0.0)))
        cited_source_counts.append(_safe_float(search_policy_metrics.get("cited_source_count", 0.0)))
        reference_source_counts.append(_safe_float(search_policy_metrics.get("reference_source_count", 0.0)))

        evaluation = record.get("evaluation", {})
        if isinstance(evaluation, dict):
            composite_score = _safe_float(evaluation.get("composite_score", 0.0))
            composite_scores.append(composite_score)
            metrics = evaluation.get("metrics", {})
            if isinstance(metrics, dict):
                citation_score = _safe_float(metrics.get("citation_coverage", 0.0))
                citation_scores.append(citation_score)
                factual_scores.append(_safe_float(metrics.get("factual_accuracy", 0.0)))
                if citation_score > 0.0:
                    nonzero_citation_queries += 1
                if citation_grounding > 0.0:
                    nonzero_citation_grounding_queries += 1
                    if citation_score > 0.0:
                        explicit_grounded_queries += 1
                    else:
                        latent_grounded_queries += 1
            if composite_score > 0.0:
                nonzero_composite_queries += 1

        report_confidences.append(_safe_float(record.get("confidence", 0.0)))
        citation_coverage = _safe_float(record.get("evaluation", {}).get("metrics", {}).get("citation_coverage", 0.0))
        per_query.append(
            {
                "query_id": query_id,
                "domain": domain,
                "search_calls": search_calls,
                "browser_calls": browser_calls,
                "stop_calls": stop_calls,
                "tool_calls": len(tool_actions),
                "has_browser": browser_calls > 0,
                "composite_score": _safe_float(record.get("evaluation", {}).get("composite_score", 0.0)),
                "citation_coverage": citation_coverage,
                "citation_grounding": citation_grounding,
                "citation_density": _safe_float(search_policy_metrics.get("citation_density", 0.0)),
                "citation_quality_score": _safe_float(search_policy_metrics.get("citation_quality_score", 0.0)),
                "source_utilization": _safe_float(search_policy_metrics.get("source_utilization", 0.0)),
                "cited_source_count": int(search_policy_metrics.get("cited_source_count", 0) or 0),
                "factual_accuracy": _safe_float(record.get("evaluation", {}).get("metrics", {}).get("factual_accuracy", 0.0)),
                "grounding_without_explicit_citation": citation_grounding > 0.0 and citation_coverage <= 0.0,
                "source_count": int(integrity.get("source_count", 0) or 0),
                "real_source_count": int(integrity.get("real_source_count", 0) or 0),
                "avg_source_relevance": _safe_float(relevance.get("avg_source_relevance", 0.0)),
                "relevant_source_count": int(relevance.get("relevant_source_count", 0) or 0),
                "mock_source_count": int(integrity.get("mock_source_count", 0) or 0),
                "uses_mock_backend": bool(integrity.get("uses_mock_backend", False)),
                "has_mock_source": bool(integrity.get("has_mock_source", False)),
                "has_real_source": bool(integrity.get("has_real_source", False)),
                "has_relevant_source": bool(relevance.get("has_relevant_source", False)),
            }
        )

    label_total = sum(label_counts.values())
    label_ratios = {
        label: round(count / label_total, 4) if label_total else 0.0
        for label, count in sorted(label_counts.items())
    }

    success_count = len(success_records)
    risk_flags: list[str] = []
    if success_count < 8:
        risk_flags.append("success_queries_below_8")
    if success_count > 0 and query_ids_with_browser / success_count < 0.25:
        risk_flags.append("browser_query_coverage_below_25pct")
    if label_total > 0 and label_counts.get("browser", 0) / label_total < 0.10:
        risk_flags.append("browser_step_ratio_below_10pct")
    if success_count > 0 and nonzero_composite_queries / success_count < 0.50:
        risk_flags.append("nonzero_composite_query_ratio_below_50pct")
    if citation_scores and _mean(citation_scores) <= 0.01:
        risk_flags.append("citation_coverage_near_zero")
    if citation_grounding_scores and (_mean(citation_grounding_scores) - _mean(citation_scores)) >= 0.25:
        risk_flags.append("grounding_render_gap_high")
    if success_count > 0 and queries_with_mock_sources / success_count > 0.0:
        risk_flags.append("mock_source_detected")
    if success_count > 0 and queries_with_real_sources / success_count < 0.80:
        risk_flags.append("real_source_query_ratio_below_80pct")
    if source_counts and _mean(source_counts) < 2.0:
        risk_flags.append("source_inventory_thin")
    if real_source_counts and _mean(real_source_counts) < 2.0:
        risk_flags.append("real_source_inventory_thin")
    if source_relevance_scores and _mean(source_relevance_scores) < 0.15:
        risk_flags.append("source_relevance_low")
    if relevant_source_counts and _mean(relevant_source_counts) < 2.0:
        risk_flags.append("relevant_source_inventory_thin")
    if citation_density_scores and _mean(citation_density_scores) < 0.20:
        risk_flags.append("citation_density_low")
    if reference_source_counts and _mean(reference_source_counts) >= 2.0 and _mean(source_utilization_scores) < 0.50:
        risk_flags.append("citation_source_utilization_low")

    return {
        "created_at": datetime.now().isoformat(),
        "input_paths": [str(path) for path in source_paths],
        "num_files": len(source_paths),
        "num_records": len(records),
        "num_success": success_count,
        "num_failed": len(failed_records),
        "num_step_rows": len(rows),
        "label_distribution": dict(sorted(label_counts.items())),
        "label_ratios": label_ratios,
        "browser_query_coverage": round(query_ids_with_browser / success_count, 4) if success_count else 0.0,
        "search_only_query_ratio": round(search_only_queries / success_count, 4) if success_count else 0.0,
        "nonzero_composite_query_ratio": round(nonzero_composite_queries / success_count, 4) if success_count else 0.0,
        "nonzero_citation_query_ratio": round(nonzero_citation_queries / success_count, 4) if success_count else 0.0,
        "nonzero_citation_grounding_query_ratio": round(nonzero_citation_grounding_queries / success_count, 4) if success_count else 0.0,
        "explicit_grounded_query_ratio": round(explicit_grounded_queries / success_count, 4) if success_count else 0.0,
        "latent_grounded_query_ratio": round(latent_grounded_queries / success_count, 4) if success_count else 0.0,
        "queries_without_policy_trace": no_step_queries,
        "mock_source_query_ratio": round(queries_with_mock_sources / success_count, 4) if success_count else 0.0,
        "mock_backend_query_ratio": round(queries_with_mock_backend / success_count, 4) if success_count else 0.0,
        "real_source_query_ratio": round(queries_with_real_sources / success_count, 4) if success_count else 0.0,
        "relevant_source_query_ratio": round(queries_with_relevant_sources / success_count, 4) if success_count else 0.0,
        "avg_sources_per_success_query": round(_mean(source_counts), 4),
        "avg_real_sources_per_success_query": round(_mean(real_source_counts), 4),
        "avg_source_relevance": round(_mean(source_relevance_scores), 4),
        "avg_relevant_sources_per_success_query": round(_mean(relevant_source_counts), 4),
        "avg_search_calls_per_success_query": round(_mean([row["search_calls"] for row in per_query]), 4),
        "avg_browser_calls_per_success_query": round(_mean([row["browser_calls"] for row in per_query]), 4),
        "avg_stop_calls_per_success_query": round(_mean([row["stop_calls"] for row in per_query]), 4),
        "avg_tool_calls_per_success_query": round(_mean([row["tool_calls"] for row in per_query]), 4),
        "avg_composite_score": round(_mean(composite_scores), 4),
        "avg_citation_coverage": round(_mean(citation_scores), 4),
        "avg_citation_grounding": round(_mean(citation_grounding_scores), 4),
        "avg_grounding_render_gap": round(_mean(citation_grounding_scores) - _mean(citation_scores), 4),
        "avg_citation_density": round(_mean(citation_density_scores), 4),
        "avg_citation_quality_score": round(_mean(citation_quality_scores), 4),
        "avg_source_utilization": round(_mean(source_utilization_scores), 4),
        "avg_cited_sources_per_success_query": round(_mean(cited_source_counts), 4),
        "avg_reference_sources_per_success_query": round(_mean(reference_source_counts), 4),
        "avg_factual_accuracy": round(_mean(factual_scores), 4),
        "avg_report_confidence": round(_mean(report_confidences), 4),
        "source_label_distribution": dict(sorted(source_labels.items())),
        "domain_distribution": dict(sorted(domains.items())),
        "risk_flags": risk_flags,
        "per_query": per_query,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Search Cache Analysis",
        "",
        f"- created_at: {summary.get('created_at', '')}",
        f"- files: {summary.get('num_files', 0)}",
        f"- records: {summary.get('num_records', 0)}",
        f"- success: {summary.get('num_success', 0)}",
        f"- failed: {summary.get('num_failed', 0)}",
        f"- step_rows: {summary.get('num_step_rows', 0)}",
        "",
        "## Policy Labels",
        "",
        "| label | count | ratio |",
        "|---|---:|---:|",
    ]
    for label, count in summary.get("label_distribution", {}).items():
        ratio = summary.get("label_ratios", {}).get(label, 0.0)
        lines.append(f"| {label} | {count} | {ratio:.4f} |")

    lines.extend(
        [
            "",
            "## Quality And Coverage",
            "",
            f"- avg_composite_score: {summary.get('avg_composite_score', 0.0):.4f}",
            f"- avg_factual_accuracy: {summary.get('avg_factual_accuracy', 0.0):.4f}",
            f"- avg_citation_coverage: {summary.get('avg_citation_coverage', 0.0):.4f}",
            f"- avg_citation_grounding: {summary.get('avg_citation_grounding', 0.0):.4f}",
            f"- avg_grounding_render_gap: {summary.get('avg_grounding_render_gap', 0.0):.4f}",
            f"- avg_citation_density: {summary.get('avg_citation_density', 0.0):.4f}",
            f"- avg_citation_quality_score: {summary.get('avg_citation_quality_score', 0.0):.4f}",
            f"- avg_source_utilization: {summary.get('avg_source_utilization', 0.0):.4f}",
            f"- avg_report_confidence: {summary.get('avg_report_confidence', 0.0):.4f}",
            "",
            "## Evidence Integrity",
            "",
            f"- mock_source_query_ratio: {summary.get('mock_source_query_ratio', 0.0):.4f}",
            f"- mock_backend_query_ratio: {summary.get('mock_backend_query_ratio', 0.0):.4f}",
            f"- real_source_query_ratio: {summary.get('real_source_query_ratio', 0.0):.4f}",
            f"- relevant_source_query_ratio: {summary.get('relevant_source_query_ratio', 0.0):.4f}",
            f"- avg_sources_per_success_query: {summary.get('avg_sources_per_success_query', 0.0):.4f}",
            f"- avg_real_sources_per_success_query: {summary.get('avg_real_sources_per_success_query', 0.0):.4f}",
            f"- avg_source_relevance: {summary.get('avg_source_relevance', 0.0):.4f}",
            f"- avg_relevant_sources_per_success_query: {summary.get('avg_relevant_sources_per_success_query', 0.0):.4f}",
            f"- avg_reference_sources_per_success_query: {summary.get('avg_reference_sources_per_success_query', 0.0):.4f}",
            f"- avg_cited_sources_per_success_query: {summary.get('avg_cited_sources_per_success_query', 0.0):.4f}",
            "",
            "## Search Behavior",
            "",
            f"- browser_query_coverage: {summary.get('browser_query_coverage', 0.0):.4f}",
            f"- search_only_query_ratio: {summary.get('search_only_query_ratio', 0.0):.4f}",
            f"- nonzero_composite_query_ratio: {summary.get('nonzero_composite_query_ratio', 0.0):.4f}",
            f"- nonzero_citation_query_ratio: {summary.get('nonzero_citation_query_ratio', 0.0):.4f}",
            f"- nonzero_citation_grounding_query_ratio: {summary.get('nonzero_citation_grounding_query_ratio', 0.0):.4f}",
            f"- explicit_grounded_query_ratio: {summary.get('explicit_grounded_query_ratio', 0.0):.4f}",
            f"- latent_grounded_query_ratio: {summary.get('latent_grounded_query_ratio', 0.0):.4f}",
            f"- avg_search_calls_per_success_query: {summary.get('avg_search_calls_per_success_query', 0.0):.4f}",
            f"- avg_browser_calls_per_success_query: {summary.get('avg_browser_calls_per_success_query', 0.0):.4f}",
            f"- avg_stop_calls_per_success_query: {summary.get('avg_stop_calls_per_success_query', 0.0):.4f}",
            f"- avg_tool_calls_per_success_query: {summary.get('avg_tool_calls_per_success_query', 0.0):.4f}",
            "",
            "## Risk Flags",
            "",
        ]
    )
    risk_flags = summary.get("risk_flags", [])
    if risk_flags:
        for flag in risk_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Per Query Snapshot",
            "",
            "| query_id | domain | search | browser | stop | tool_calls | sources | real_sources | relevant_sources | cited_sources | mock | citation | density | grounding | composite |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("per_query", []):
        lines.append(
            f"| {row.get('query_id', '')} | {row.get('domain', '')} | "
            f"{row.get('search_calls', 0)} | {row.get('browser_calls', 0)} | "
            f"{row.get('stop_calls', 0)} | {row.get('tool_calls', 0)} | "
            f"{row.get('source_count', 0)} | {row.get('real_source_count', 0)} | "
            f"{row.get('relevant_source_count', 0)} | "
            f"{row.get('cited_source_count', 0)} | "
            f"{'yes' if row.get('has_mock_source', False) else 'no'} | "
            f"{row.get('citation_coverage', 0.0):.4f} | {row.get('citation_density', 0.0):.4f} | "
            f"{row.get('citation_grounding', 0.0):.4f} | "
            f"{row.get('composite_score', 0.0):.4f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze search-cache JSONL artifacts")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL files or directories")
    parser.add_argument("--output_dir", type=str, default=None, help="optional output directory")
    args = parser.parse_args()

    input_paths = _resolve_jsonl_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("no JSONL inputs found")

    records = _load_records(input_paths)
    summary = summarize_cache_records(records, input_paths)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        json_path = output_dir / f"search_cache_analysis_{timestamp}.json"
        md_path = output_dir / f"search_cache_analysis_{timestamp}.md"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(summary), encoding="utf-8")
        print(str(md_path))


if __name__ == "__main__":
    main()
