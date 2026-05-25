#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_policy_head2head_cv.py
================================================================================
Prepare query-level cross-fold policy evaluation from an existing search cache and
optionally execute held-out heuristic vs learned head-to-head runs.

This script is intentionally cache-first:
  1) build deterministic query-level folds from a resurfaced cache
  2) train one search policy per fold on the train partition
  3) optionally run head-to-head on each held-out fold
  4) aggregate cross-fold metrics into a single JSON/Markdown report
================================================================================
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_search_cache import _load_records, _resolve_jsonl_inputs, summarize_cache_records
from scripts.mine_natural_browser_queries import summarize_record
from scripts.run_policy_head2head import _render_markdown as _render_head2head_markdown
from scripts.run_policy_head2head import _run_mode
from scripts.train_search_policy_iteration import evaluate_headline_readiness, evaluate_training_readiness
from src.core.runner import load_config, setup_logging
from src.search_policy.dataset import SearchPolicyDatasetBuilder
from src.search_policy.train_policy import SearchPolicyTrainer


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _selection_priority(summary: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        _safe_float(summary.get("citation_coverage", 0.0)),
        _safe_float(summary.get("composite_score", 0.0)),
        _safe_float(summary.get("factual_accuracy", 0.0)),
        _safe_float(summary.get("confidence", 0.0)),
        str(summary.get("query_id", "") or ""),
    )


def _is_grounded(
    summary: dict[str, Any],
    *,
    min_browser_calls: int,
    min_citation_coverage: float,
) -> bool:
    return _safe_int(summary.get("browser_calls", 0)) >= min_browser_calls and _safe_float(
        summary.get("citation_coverage", 0.0)
    ) >= min_citation_coverage


def _record_to_query_item(record: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(record.get("query_id", record.get("cache_id", "")) or ""),
        "query": str(record.get("query", "") or ""),
        "domain": record.get("domain"),
        "expected_topics": record.get("expected_topics"),
        "ground_truth": record.get("ground_truth"),
        "browser_calls": summary.get("browser_calls", 0),
        "search_calls": summary.get("search_calls", 0),
        "citation_coverage": summary.get("citation_coverage", 0.0),
        "composite_score": summary.get("composite_score", 0.0),
    }


def _domain_buckets(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[str(item.get("domain", "") or "")].append(item)
    for domain_items in buckets.values():
        domain_items.sort(key=_selection_priority, reverse=True)
    return dict(buckets)


def _interleave_domains(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mutable = {domain: list(domain_items) for domain, domain_items in _domain_buckets(items).items()}
    ordered: list[dict[str, Any]] = []

    while True:
        nonempty = [(domain, bucket) for domain, bucket in mutable.items() if bucket]
        if not nonempty:
            break
        nonempty.sort(key=lambda pair: (-len(pair[1]), pair[0]))
        for _, bucket in nonempty:
            ordered.append(bucket.pop(0))

    return ordered


def build_query_folds(
    records: list[dict[str, Any]],
    *,
    source_label: str,
    num_folds: int,
    grounded_min_browser_calls: int,
    grounded_min_citation_coverage: float,
) -> dict[str, Any]:
    eligible_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped = Counter()

    for record in records:
        if str(record.get("status", "success") or "success") != "success":
            skipped["non_success"] += 1
            continue
        summary = summarize_record(record)
        if summary["source_label"] != source_label:
            skipped["non_target_source"] += 1
            continue
        if not summary["query"]:
            skipped["missing_query"] += 1
            continue
        eligible_pairs.append((record, summary))

    total = len(eligible_pairs)
    if total < 2:
        raise ValueError("not enough eligible records to create cross-fold splits")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    if total < num_folds:
        raise ValueError(f"not enough eligible queries ({total}) for {num_folds} folds")

    ordered_summaries = _interleave_domains([summary for _, summary in eligible_pairs])
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(num_folds)]
    for idx, summary in enumerate(ordered_summaries):
        assignments[idx % num_folds].append(summary)

    record_by_id = {str(summary["query_id"]): record for record, summary in eligible_pairs}
    summary_by_id = {str(summary["query_id"]): summary for _, summary in eligible_pairs}
    all_query_ids = [str(summary["query_id"]) for summary in ordered_summaries]
    grounded_total = sum(
        1
        for summary in ordered_summaries
        if _is_grounded(
            summary,
            min_browser_calls=grounded_min_browser_calls,
            min_citation_coverage=grounded_min_citation_coverage,
        )
    )

    folds: list[dict[str, Any]] = []
    for fold_index, heldout_summaries in enumerate(assignments, 1):
        heldout_ids = {str(item["query_id"]) for item in heldout_summaries}
        train_ids = [query_id for query_id in all_query_ids if query_id not in heldout_ids]
        heldout_id_list = [str(item["query_id"]) for item in heldout_summaries]

        train_records = [record_by_id[query_id] for query_id in train_ids]
        heldout_records = [record_by_id[query_id] for query_id in heldout_id_list]
        train_queries = [_record_to_query_item(record_by_id[query_id], summary_by_id[query_id]) for query_id in train_ids]
        heldout_queries = [
            _record_to_query_item(record_by_id[query_id], summary_by_id[query_id]) for query_id in heldout_id_list
        ]

        train_records.sort(key=lambda item: str(item.get("query_id", "")))
        heldout_records.sort(key=lambda item: str(item.get("query_id", "")))
        train_queries.sort(key=lambda item: str(item.get("id", "")))
        heldout_queries.sort(key=lambda item: str(item.get("id", "")))

        train_domain_counts = Counter(str(summary_by_id[qid].get("domain", "") or "") for qid in train_ids)
        heldout_domain_counts = Counter(str(summary_by_id[qid].get("domain", "") or "") for qid in heldout_id_list)
        train_grounded = sum(
            1
            for query_id in train_ids
            if _is_grounded(
                summary_by_id[query_id],
                min_browser_calls=grounded_min_browser_calls,
                min_citation_coverage=grounded_min_citation_coverage,
            )
        )
        heldout_grounded = sum(
            1
            for query_id in heldout_id_list
            if _is_grounded(
                summary_by_id[query_id],
                min_browser_calls=grounded_min_browser_calls,
                min_citation_coverage=grounded_min_citation_coverage,
            )
        )

        folds.append(
            {
                "fold_index": fold_index,
                "train_records": train_records,
                "heldout_records": heldout_records,
                "train_queries": train_queries,
                "heldout_queries": heldout_queries,
                "summary": {
                    "fold_index": fold_index,
                    "train": {
                        "num_queries": len(train_ids),
                        "grounded_queries": train_grounded,
                        "domain_distribution": dict(sorted(train_domain_counts.items())),
                        "query_ids": train_ids,
                    },
                    "heldout": {
                        "num_queries": len(heldout_id_list),
                        "grounded_queries": heldout_grounded,
                        "domain_distribution": dict(sorted(heldout_domain_counts.items())),
                        "query_ids": heldout_id_list,
                    },
                },
            }
        )

    summary = {
        "created_at": datetime.now().isoformat(),
        "source_label": source_label,
        "num_folds": num_folds,
        "num_eligible_queries": total,
        "grounded_query_count": grounded_total,
        "grounded_definition": {
            "min_browser_calls": grounded_min_browser_calls,
            "min_citation_coverage": grounded_min_citation_coverage,
        },
        "fold_sizes": [len(fold["summary"]["heldout"]["query_ids"]) for fold in folds],
        "skipped_records": dict(skipped),
    }
    return {"summary": summary, "folds": folds}


def aggregate_cv_runs(fold_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    mode_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold_payload in fold_payloads:
        for run in fold_payload.get("runs", []):
            mode = str(run.get("mode", "") or "")
            if not mode:
                continue
            mode_to_records[mode].extend(run.get("records", []))

    runs: list[dict[str, Any]] = []
    for mode in sorted(mode_to_records):
        records = list(mode_to_records[mode])
        success_rows = [row for row in records if row.get("status") == "success"]
        runs.append(
            {
                "mode": mode,
                "summary": {
                    "num_total": len(records),
                    "num_success": len(success_rows),
                    "num_failed": len(records) - len(success_rows),
                    "avg_composite_score": _mean(
                        [_safe_float(row.get("score", {}).get("composite_score", 0.0)) for row in success_rows]
                    ),
                    "avg_factual_accuracy": _mean(
                        [_safe_float(row.get("score", {}).get("factual_accuracy", 0.0)) for row in success_rows]
                    ),
                    "avg_citation_coverage": _mean(
                        [_safe_float(row.get("score", {}).get("citation_coverage", 0.0)) for row in success_rows]
                    ),
                    "avg_search_policy_score": _mean(
                        [_safe_float(row.get("score", {}).get("search_policy_score", 0.0)) for row in success_rows]
                    ),
                    "avg_tool_calls": _mean(
                        [_safe_float(row.get("cost", {}).get("tool_calls", 0.0)) for row in success_rows]
                    ),
                    "avg_search_calls": _mean(
                        [_safe_float(row.get("cost", {}).get("search_calls", 0.0)) for row in success_rows]
                    ),
                    "avg_browser_calls": _mean(
                        [_safe_float(row.get("cost", {}).get("browser_calls", 0.0)) for row in success_rows]
                    ),
                    "avg_estimated_token_cost": _mean(
                        [_safe_float(row.get("cost", {}).get("estimated_token_cost", 0.0)) for row in success_rows]
                    ),
                    "avg_elapsed_seconds": _mean([_safe_float(row.get("elapsed_seconds", 0.0)) for row in success_rows]),
                    "avg_policy_advice_count": _mean(
                        [
                            _safe_float(
                                row.get("metadata_summary", {}).get("policy_stats", {}).get("policy_advice_count", 0.0)
                            )
                            for row in success_rows
                        ]
                    ),
                    "avg_guardrail_trigger_count": _mean(
                        [
                            _safe_float(
                                row.get("metadata_summary", {}).get("policy_stats", {}).get("guardrail_trigger_count", 0.0)
                            )
                            for row in success_rows
                        ]
                    ),
                    "avg_policy_enforce_stop_count": _mean(
                        [
                            _safe_float(
                                row.get("metadata_summary", {}).get("policy_stats", {}).get("policy_enforce_stop_count", 0.0)
                            )
                            for row in success_rows
                        ]
                    ),
                },
                "records": records,
            }
        )
    return {"runs": runs}


def render_cv_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Policy Head-to-Head Cross-Fold Report",
        "",
        f"- created_at: {payload.get('created_at', '')}",
        f"- config: {payload.get('config_path', '')}",
        f"- source_label: {payload.get('source_label', '')}",
        f"- fold_count: {payload.get('fold_count', 0)}",
        f"- query_count: {payload.get('query_count', 0)}",
        "",
    ]
    aggregate = payload.get("aggregate_head2head")
    if isinstance(aggregate, dict) and aggregate.get("runs"):
        head2head_payload = {
            "created_at": payload.get("created_at", ""),
            "config_path": payload.get("config_path", ""),
            "query_count": payload.get("query_count", 0),
            "runs": aggregate.get("runs", []),
        }
        lines.append(_render_head2head_markdown(head2head_payload).strip())
        lines.append("")

    lines.extend(["## Fold Overview", ""])
    for fold in payload.get("folds", []):
        summary = fold.get("summary", {})
        train = summary.get("train", {})
        heldout = summary.get("heldout", {})
        lines.append(
            f"- fold_{summary.get('fold_index', 0):02d}: "
            f"train={train.get('num_queries', 0)} "
            f"(grounded={train.get('grounded_queries', 0)}), "
            f"heldout={heldout.get('num_queries', 0)} "
            f"(grounded={heldout.get('grounded_queries', 0)})"
        )
    fold_delta_rows = _fold_delta_rows(payload.get("folds", []))
    if fold_delta_rows:
        lines.extend(
            [
                "",
                "## Fold Deltas",
                "",
                "| fold | heldout_ids | quality_delta | citation_delta | token_delta | tool_delta | interpretation |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in fold_delta_rows:
            lines.append(
                f"| fold_{row['fold_index']:02d} | {row['heldout_ids']} "
                f"| {row['quality_delta']:+.3f}"
                f" | {row['citation_delta']:+.3f}"
                f" | {row['token_delta']:+.1f}"
                f" | {row['tool_delta']:+.2f}"
                f" | {row['interpretation']} |"
            )
    return "\n".join(lines).strip() + "\n"


def _summary_by_mode(fold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(run.get("mode", "")): run.get("summary", {}) for run in fold.get("runs", [])}


def _fold_delta_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        by_mode = _summary_by_mode(fold)
        if "learned" not in by_mode or "heuristic" not in by_mode:
            continue
        learned = by_mode["learned"]
        heuristic = by_mode["heuristic"]
        quality_delta = _safe_float(learned.get("avg_composite_score")) - _safe_float(
            heuristic.get("avg_composite_score")
        )
        citation_delta = _safe_float(learned.get("avg_citation_coverage")) - _safe_float(
            heuristic.get("avg_citation_coverage")
        )
        token_delta = _safe_float(learned.get("avg_estimated_token_cost")) - _safe_float(
            heuristic.get("avg_estimated_token_cost")
        )
        tool_delta = _safe_float(learned.get("avg_tool_calls")) - _safe_float(heuristic.get("avg_tool_calls"))
        if quality_delta > 0 and citation_delta >= 0 and token_delta <= 0:
            interpretation = "quality up, cost down"
        elif quality_delta > 0 and citation_delta >= 0:
            interpretation = "quality up, cost up"
        elif quality_delta <= 0 and token_delta < 0:
            interpretation = "cost down, quality down"
        else:
            interpretation = "mixed or unclear"

        heldout_ids = fold.get("summary", {}).get("heldout", {}).get("query_ids", [])
        rows.append(
            {
                "fold_index": _safe_int(fold.get("fold_index", fold.get("summary", {}).get("fold_index", 0))),
                "heldout_ids": ", ".join(str(query_id) for query_id in heldout_ids),
                "quality_delta": quality_delta,
                "citation_delta": citation_delta,
                "token_delta": token_delta,
                "tool_delta": tool_delta,
                "interpretation": interpretation,
            }
        )
    return rows


def _train_fold(
    *,
    train_records: list[dict[str, Any]],
    train_cache_path: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = summarize_cache_records(train_records, [train_cache_path])
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
    summary["training_allowed"] = not readiness_failures or args.force_train
    summary["headline_claim_ready"] = not headline_failures

    if readiness_failures and not args.force_train:
        summary["status"] = "blocked"
        summary["message"] = "training blocked because cache readiness thresholds were not met"
        return summary

    builder = SearchPolicyDatasetBuilder()
    rows = builder.build_rows_from_paths([train_cache_path])
    trainer = SearchPolicyTrainer(model_path=str(model_path))
    training_stats = trainer.fit(
        rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        validation_ratio=args.validation_ratio,
    )
    summary["status"] = "trained"
    summary["training_stats"] = training_stats
    summary["output_model"] = str(model_path)
    if headline_failures:
        summary["message"] = "training completed for bootstrap purposes, but cache is not headline-evidence ready"
    else:
        summary["message"] = "training completed and cache passed headline-evidence readiness checks"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query-level cross-fold policy preparation and optional head-to-head")
    parser.add_argument("--config", type=str, required=True, help="config path for live head-to-head")
    parser.add_argument("--cache-inputs", nargs="+", required=True, help="JSONL files or directories")
    parser.add_argument("--source-label", type=str, required=True, help="search-cache source label to keep")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory for folds and summaries")
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--modes", type=str, default="heuristic,learned", help="comma-separated policy modes")
    parser.add_argument("--run-head2head", action="store_true", help="execute live held-out evaluation after training")
    parser.add_argument("--force-train", action="store_true", help="train on small folds even if readiness checks fail")
    parser.add_argument("--grounded-min-browser-calls", type=int, default=1)
    parser.add_argument("--grounded-min-citation-coverage", type=float, default=0.02)
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
    parser.add_argument("--evidence-policy-mode", type=str, choices=["heuristic", "learned"], default=None)
    parser.add_argument("--evidence-policy-model-path", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    input_paths = _resolve_jsonl_inputs(args.cache_inputs)
    if not input_paths:
        raise SystemExit("no JSONL inputs found")

    records = _load_records(input_paths)
    prepared = build_query_folds(
        records,
        source_label=args.source_label,
        num_folds=args.num_folds,
        grounded_min_browser_calls=args.grounded_min_browser_calls,
        grounded_min_citation_coverage=args.grounded_min_citation_coverage,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.config)
    modes = [mode.strip().lower() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        if mode not in {"off", "heuristic", "learned"}:
            raise SystemExit(f"unsupported mode: {mode}")

    fold_payloads: list[dict[str, Any]] = []
    for fold in prepared["folds"]:
        fold_index = int(fold["fold_index"])
        fold_dir = output_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_cache_path = fold_dir / "train_cache.jsonl"
        heldout_cache_path = fold_dir / "heldout_cache.jsonl"
        train_queries_path = fold_dir / "train_queries.jsonl"
        heldout_queries_path = fold_dir / "heldout_queries.jsonl"
        model_path = fold_dir / f"search_policy_fold_{fold_index:02d}.json"
        train_summary_path = fold_dir / "train_summary.json"
        fold_summary_path = fold_dir / "fold_summary.json"

        _write_jsonl(train_cache_path, fold["train_records"])
        _write_jsonl(heldout_cache_path, fold["heldout_records"])
        _write_jsonl(train_queries_path, fold["train_queries"])
        _write_jsonl(heldout_queries_path, fold["heldout_queries"])

        train_summary = _train_fold(
            train_records=fold["train_records"],
            train_cache_path=train_cache_path,
            model_path=model_path,
            args=args,
        )
        train_summary_path.write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        fold_payload: dict[str, Any] = {
            "fold_index": fold_index,
            "summary": deepcopy(fold["summary"]),
            "artifacts": {
                "train_cache": str(train_cache_path),
                "heldout_cache": str(heldout_cache_path),
                "train_queries": str(train_queries_path),
                "heldout_queries": str(heldout_queries_path),
                "train_summary": str(train_summary_path),
                "model_path": str(model_path) if model_path.exists() else None,
            },
            "train_summary": train_summary,
            "runs": [],
        }

        if args.run_head2head and model_path.exists():
            fold_args = argparse.Namespace(**vars(args))
            fold_args.search_policy_model_path = str(model_path)
            fold_args.output_dir = str(fold_dir)
            runs = []
            for mode in modes:
                runs.append(
                    _run_mode(
                        mode=mode,
                        base_config=base_config,
                        args=fold_args,
                        query_items=fold["heldout_queries"],
                    )
                )
            head2head_payload = {
                "created_at": datetime.now().isoformat(),
                "config_path": args.config,
                "query_count": len(fold["heldout_queries"]),
                "queries": fold["heldout_queries"],
                "runs": runs,
            }
            head2head_json = fold_dir / "head2head.json"
            head2head_md = fold_dir / "head2head.md"
            head2head_json.write_text(json.dumps(head2head_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            head2head_md.write_text(_render_head2head_markdown(head2head_payload), encoding="utf-8")
            fold_payload["artifacts"]["head2head_json"] = str(head2head_json)
            fold_payload["artifacts"]["head2head_md"] = str(head2head_md)
            fold_payload["runs"] = runs

        fold_summary_path.write_text(json.dumps(fold_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_payloads.append(fold_payload)

    aggregate_head2head = aggregate_cv_runs(fold_payloads) if args.run_head2head else None
    query_count = sum(int(fold["summary"]["heldout"]["num_queries"]) for fold in fold_payloads)
    payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": args.config,
        "source_label": args.source_label,
        "fold_count": len(fold_payloads),
        "query_count": query_count,
        "prepared_summary": prepared["summary"],
        "folds": fold_payloads,
        "aggregate_head2head": aggregate_head2head,
    }

    summary_json = output_dir / "cv_summary.json"
    summary_md = output_dir / "cv_summary.md"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_cv_markdown(payload), encoding="utf-8")
    print(str(summary_md))


if __name__ == "__main__":
    main()
