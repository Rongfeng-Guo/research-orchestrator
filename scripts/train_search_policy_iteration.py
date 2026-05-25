#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_search_policy_iteration.py
================================================================================
Controlled search-policy iteration runner.

This wrapper enforces simple experiment hygiene:
  1) analyze cache quality first
  2) block training by default when browser coverage is too low
  3) train only when the cache passes minimum readiness thresholds

Usage:
  python scripts/train_search_policy_iteration.py \
    --inputs data/search_cache/mini_20260515 \
    --output-model artifacts/search_policy_v2.json
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_search_cache import _load_records, _resolve_jsonl_inputs, summarize_cache_records
from src.search_policy.dataset import SearchPolicyDatasetBuilder
from src.search_policy.train_policy import SearchPolicyTrainer


def evaluate_training_readiness(
    summary: dict[str, Any],
    *,
    min_success_queries: int = 8,
    min_browser_step_ratio: float = 0.10,
    min_browser_query_coverage: float = 0.25,
    max_mock_source_query_ratio: float = 0.20,
) -> list[str]:
    reasons: list[str] = []
    if int(summary.get("num_success", 0) or 0) < min_success_queries:
        reasons.append(
            f"success_queries_below_threshold({summary.get('num_success', 0)} < {min_success_queries})"
        )

    browser_step_ratio = float(summary.get("label_ratios", {}).get("browser", 0.0) or 0.0)
    if browser_step_ratio < min_browser_step_ratio:
        reasons.append(
            f"browser_step_ratio_below_threshold({browser_step_ratio:.4f} < {min_browser_step_ratio:.4f})"
        )

    browser_query_coverage = float(summary.get("browser_query_coverage", 0.0) or 0.0)
    if browser_query_coverage < min_browser_query_coverage:
        reasons.append(
            f"browser_query_coverage_below_threshold({browser_query_coverage:.4f} < {min_browser_query_coverage:.4f})"
        )

    if "mock_source_query_ratio" in summary:
        mock_source_query_ratio = float(summary.get("mock_source_query_ratio", 0.0) or 0.0)
        if mock_source_query_ratio > max_mock_source_query_ratio:
            reasons.append(
                f"mock_source_query_ratio_above_threshold({mock_source_query_ratio:.4f} > {max_mock_source_query_ratio:.4f})"
            )

    return reasons


def evaluate_headline_readiness(
    summary: dict[str, Any],
    *,
    min_avg_citation_coverage: float = 0.01,
    min_nonzero_composite_ratio: float = 0.50,
    min_real_source_query_ratio: float = 0.80,
    max_mock_source_query_ratio: float = 0.0,
    min_avg_real_sources_per_query: float = 2.0,
    min_relevant_source_query_ratio: float = 0.80,
    min_avg_relevant_sources_per_query: float = 2.0,
    min_avg_source_relevance: float = 0.15,
    min_avg_citation_quality_score: float = 0.20,
) -> list[str]:
    reasons: list[str] = []

    avg_citation_coverage = float(summary.get("avg_citation_coverage", 0.0) or 0.0)
    if avg_citation_coverage < min_avg_citation_coverage:
        reasons.append(
            f"avg_citation_coverage_below_threshold({avg_citation_coverage:.4f} < {min_avg_citation_coverage:.4f})"
        )

    nonzero_composite_ratio = float(summary.get("nonzero_composite_query_ratio", 0.0) or 0.0)
    if nonzero_composite_ratio < min_nonzero_composite_ratio:
        reasons.append(
            f"nonzero_composite_query_ratio_below_threshold({nonzero_composite_ratio:.4f} < {min_nonzero_composite_ratio:.4f})"
        )

    if "real_source_query_ratio" in summary:
        real_source_query_ratio = float(summary.get("real_source_query_ratio", 0.0) or 0.0)
        if real_source_query_ratio < min_real_source_query_ratio:
            reasons.append(
                f"real_source_query_ratio_below_threshold({real_source_query_ratio:.4f} < {min_real_source_query_ratio:.4f})"
            )

    if "mock_source_query_ratio" in summary:
        mock_source_query_ratio = float(summary.get("mock_source_query_ratio", 0.0) or 0.0)
        if mock_source_query_ratio > max_mock_source_query_ratio:
            reasons.append(
                f"mock_source_query_ratio_above_threshold({mock_source_query_ratio:.4f} > {max_mock_source_query_ratio:.4f})"
            )

    if "avg_real_sources_per_success_query" in summary:
        avg_real_sources = float(summary.get("avg_real_sources_per_success_query", 0.0) or 0.0)
        if avg_real_sources < min_avg_real_sources_per_query:
            reasons.append(
                f"avg_real_sources_per_query_below_threshold({avg_real_sources:.4f} < {min_avg_real_sources_per_query:.4f})"
            )

    if "relevant_source_query_ratio" in summary:
        relevant_source_query_ratio = float(summary.get("relevant_source_query_ratio", 0.0) or 0.0)
        if relevant_source_query_ratio < min_relevant_source_query_ratio:
            reasons.append(
                f"relevant_source_query_ratio_below_threshold({relevant_source_query_ratio:.4f} < {min_relevant_source_query_ratio:.4f})"
            )

    if "avg_relevant_sources_per_success_query" in summary:
        avg_relevant_sources = float(summary.get("avg_relevant_sources_per_success_query", 0.0) or 0.0)
        if avg_relevant_sources < min_avg_relevant_sources_per_query:
            reasons.append(
                f"avg_relevant_sources_per_query_below_threshold({avg_relevant_sources:.4f} < {min_avg_relevant_sources_per_query:.4f})"
            )

    if "avg_source_relevance" in summary:
        avg_source_relevance = float(summary.get("avg_source_relevance", 0.0) or 0.0)
        if avg_source_relevance < min_avg_source_relevance:
            reasons.append(
                f"avg_source_relevance_below_threshold({avg_source_relevance:.4f} < {min_avg_source_relevance:.4f})"
            )

    if "avg_citation_quality_score" in summary:
        avg_citation_quality = float(summary.get("avg_citation_quality_score", 0.0) or 0.0)
        if avg_citation_quality < min_avg_citation_quality_score:
            reasons.append(
                f"avg_citation_quality_score_below_threshold({avg_citation_quality:.4f} < {min_avg_citation_quality_score:.4f})"
            )

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Train search policy only when cache quality passes readiness checks")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL files or directories")
    parser.add_argument("--output-model", type=str, required=True, help="target model path")
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--min-success-queries", type=int, default=8)
    parser.add_argument("--min-browser-step-ratio", type=float, default=0.10)
    parser.add_argument("--min-browser-query-coverage", type=float, default=0.25)
    parser.add_argument("--min-avg-citation-coverage", type=float, default=0.01)
    parser.add_argument("--min-nonzero-composite-ratio", type=float, default=0.50)
    parser.add_argument("--min-real-source-query-ratio", type=float, default=0.80)
    parser.add_argument("--max-mock-source-query-ratio", type=float, default=0.0)
    parser.add_argument("--max-training-mock-source-query-ratio", type=float, default=0.20)
    parser.add_argument("--min-avg-real-sources-per-query", type=float, default=2.0)
    parser.add_argument("--min-relevant-source-query-ratio", type=float, default=0.80)
    parser.add_argument("--min-avg-relevant-sources-per-query", type=float, default=2.0)
    parser.add_argument("--min-avg-source-relevance", type=float, default=0.15)
    parser.add_argument("--min-avg-citation-quality-score", type=float, default=0.20)
    parser.add_argument("--output-dir", type=str, default="outputs/search_policy_iteration", help="summary output directory")
    parser.add_argument("--output-summary", type=str, default=None, help="optional stable JSON summary path")
    parser.add_argument("--force", action="store_true", help="train even if readiness checks fail")
    args = parser.parse_args()

    input_paths = _resolve_jsonl_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("no JSONL inputs found")

    records = _load_records(input_paths)
    summary = summarize_cache_records(records, input_paths)
    readiness_failures = evaluate_training_readiness(
        summary,
        min_success_queries=args.min_success_queries,
        min_browser_step_ratio=args.min_browser_step_ratio,
        min_browser_query_coverage=args.min_browser_query_coverage,
        max_mock_source_query_ratio=args.max_training_mock_source_query_ratio,
    )
    headline_failures = evaluate_headline_readiness(
        summary,
        min_avg_citation_coverage=args.min_avg_citation_coverage,
        min_nonzero_composite_ratio=args.min_nonzero_composite_ratio,
        min_real_source_query_ratio=args.min_real_source_query_ratio,
        max_mock_source_query_ratio=args.max_mock_source_query_ratio,
        min_avg_real_sources_per_query=args.min_avg_real_sources_per_query,
        min_relevant_source_query_ratio=args.min_relevant_source_query_ratio,
        min_avg_relevant_sources_per_query=args.min_avg_relevant_sources_per_query,
        min_avg_source_relevance=args.min_avg_source_relevance,
        min_avg_citation_quality_score=args.min_avg_citation_quality_score,
    )
    summary["training_readiness_failures"] = readiness_failures
    summary["headline_readiness_failures"] = headline_failures
    summary["training_allowed"] = not readiness_failures or args.force
    summary["headline_claim_ready"] = not headline_failures

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(args.output_summary) if args.output_summary else output_dir / f"search_policy_iteration_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if readiness_failures and not args.force:
        summary["status"] = "blocked"
        summary["message"] = "training blocked because cache readiness thresholds were not met"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    builder = SearchPolicyDatasetBuilder()
    rows = builder.build_rows_from_paths(input_paths)
    trainer = SearchPolicyTrainer(model_path=args.output_model)
    training_stats = trainer.fit(
        rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        validation_ratio=args.validation_ratio,
    )

    summary["status"] = "trained"
    summary["training_stats"] = training_stats
    summary["output_model"] = args.output_model
    if headline_failures:
        summary["message"] = "training completed for bootstrap purposes, but cache is not headline-evidence ready"
    else:
        summary["message"] = "training completed and cache passed headline-evidence readiness checks"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
