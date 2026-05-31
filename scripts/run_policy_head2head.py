#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_policy_head2head.py
================================================================================
Run head-to-head comparisons for policy modes:
  - off
  - heuristic
  - learned

The script runs the same query set under each mode and writes:
  1) raw JSON results
  2) markdown summary table + concise conclusion hints
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.research_bench import ResearchBench
from src.core.runner import initialize_modules, load_config, run_research, setup_logging
from src.models.model_router import ModelRouter


def _interleave_by_domain(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        domain = str(item.get("domain", "") or "")
        buckets.setdefault(domain, []).append(item)

    ordered: list[dict[str, Any]] = []
    while True:
        nonempty = [(domain, bucket) for domain, bucket in buckets.items() if bucket]
        if not nonempty:
            break
        nonempty.sort(key=lambda pair: (-len(pair[1]), pair[0]))
        for _, bucket in nonempty:
            ordered.append(bucket.pop(0))
    return ordered


def _normalize_query_items(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(q.get("id", f"bench_{i:04d}")),
            "query": str(q.get("query", "")),
            "domain": q.get("domain"),
            "expected_topics": q.get("expected_topics"),
            "ground_truth": q.get("ground_truth"),
        }
        for i, q in enumerate(questions, 1)
    ]


def _required_backends(config: dict[str, Any]) -> list[str]:
    model_cfg = config.get("model", {}) or {}
    default_backend = str(model_cfg.get("backend", "") or "").strip().lower()
    backend_mapping = model_cfg.get("backend_mapping", {}) or {}

    ordered: list[str] = []
    seen: set[str] = set()
    for module_name in ["solver", "planner", "summarizer", "judge", "red_agent", "blue_agent", "compressor"]:
        backend = str(backend_mapping.get(module_name) or default_backend or "").strip().lower()
        if not backend or backend in seen:
            continue
        seen.add(backend)
        ordered.append(backend)
    return ordered


def _preflight_backend_requirements(config: dict[str, Any]) -> dict[str, Any]:
    required = _required_backends(config)
    configured = [name for name in required if ModelRouter._is_backend_configured(name)]
    missing = [name for name in required if name not in configured]
    return {
        "required_backends": required,
        "configured_backends": configured,
        "missing_backends": missing,
        "is_ready": not missing,
    }


def _load_queries(
    *,
    query: str | None,
    queries_file: str | None,
    num_questions: int,
    domain: str | None,
    sampling_strategy: str,
) -> list[dict[str, Any]]:
    if query:
        return [{"id": "single_0001", "query": query.strip()}]

    if queries_file:
        path = Path(queries_file)
        if not path.exists():
            raise FileNotFoundError(f"queries file not found: {queries_file}")
        items: list[dict[str, str]] = []
        if path.suffix.lower() == ".jsonl":
            for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                q = str(obj.get("query", "") or "").strip()
                if q:
                    item = {"id": str(obj.get("id", f"file_{idx:04d}")), "query": q}
                    if isinstance(obj.get("domain"), str):
                        item["domain"] = obj.get("domain")
                    if isinstance(obj.get("expected_topics"), list):
                        item["expected_topics"] = obj.get("expected_topics")
                    if isinstance(obj.get("ground_truth"), dict):
                        item["ground_truth"] = obj.get("ground_truth")
                    items.append(item)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("queries"), list):
                data = data["queries"]
            if not isinstance(data, list):
                raise ValueError("queries file must be a list/jsonl or object with `queries` field")
            for idx, item in enumerate(data, 1):
                if isinstance(item, str):
                    q = item.strip()
                    if q:
                        items.append({"id": f"file_{idx:04d}", "query": q})
                elif isinstance(item, dict):
                    q = str(item.get("query", "") or "").strip()
                    if q:
                        normalized = {"id": str(item.get("id", f"file_{idx:04d}")), "query": q}
                        if isinstance(item.get("domain"), str):
                            normalized["domain"] = item.get("domain")
                        if isinstance(item.get("expected_topics"), list):
                            normalized["expected_topics"] = item.get("expected_topics")
                        if isinstance(item.get("ground_truth"), dict):
                            normalized["ground_truth"] = item.get("ground_truth")
                        items.append(normalized)
        if not items:
            raise ValueError("no usable queries found in queries_file")
        return items

    bench = ResearchBench()
    if sampling_strategy == "balanced_domains" and not domain:
        questions = bench.get_questions(domain=None, n=None)
        questions = _interleave_by_domain(list(questions))
        return _normalize_query_items(questions[:num_questions])

    questions = bench.get_questions(domain=domain, n=num_questions)
    return _normalize_query_items(list(questions))


def _apply_mode_overrides(config: dict[str, Any], mode: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(config)

    evidence_cfg = dict(cfg.get("evidence_policy", {}) or {})
    evidence_mode = args.evidence_policy_mode if args.evidence_policy_mode else evidence_cfg.get("mode", "heuristic")
    evidence_cfg["mode"] = evidence_mode
    if args.evidence_policy_model_path:
        evidence_cfg["model_path"] = args.evidence_policy_model_path
    cfg["evidence_policy"] = evidence_cfg

    search_cfg = dict(cfg.get("search_policy", {}) or {})
    search_cfg["mode"] = mode
    if mode == "off":
        search_cfg["enabled"] = False
    elif mode == "heuristic":
        search_cfg["enabled"] = True
        search_cfg["heuristic_fallback"] = True
    elif mode == "learned":
        search_cfg["enabled"] = True
        search_cfg["heuristic_fallback"] = False
    if args.search_policy_model_path:
        search_cfg["model_path"] = args.search_policy_model_path
    cfg["search_policy"] = search_cfg
    return cfg


def _evaluate_custom_query_report(report_obj: Any, query_item: dict[str, Any]) -> dict[str, float]:
    from evaluation.metrics.rule_based import RuleBasedMetrics
    from evaluation.metrics.search_policy import SearchPolicyMetrics

    bench = ResearchBench()
    report_text, metadata, report_sources = bench._extract_report_payload(
        report_obj,
        getattr(report_obj, "metadata", {}) or {},
    )
    expected_topics = query_item.get("expected_topics", []) if isinstance(query_item.get("expected_topics"), list) else []
    ground_truth = query_item.get("ground_truth", {}) if isinstance(query_item.get("ground_truth"), dict) else {}

    if ground_truth:
        factual_str = RuleBasedMetrics.fact_accuracy(report_text, ground_truth)
        factual_sem = RuleBasedMetrics.semantic_fact_accuracy(report_text, ground_truth, threshold=0.65)
        factual = RuleBasedMetrics.combine_factual_accuracy(factual_str, factual_sem)
    else:
        factual_str = 0.0
        factual_sem = 0.0
        factual = 0.0

    hallucination = RuleBasedMetrics.hallucination_rate(report_text)
    citation = RuleBasedMetrics.citation_coverage(report_text)
    logic = RuleBasedMetrics.logical_consistency(report_text)
    comprehensive = RuleBasedMetrics.comprehensiveness(report_text, expected_topics)
    bias_score = max(0.0, 1.0 - hallucination)
    evidence_breakdown = RuleBasedMetrics.evidence_graph_breakdown(metadata)
    evidence_transition_quality = RuleBasedMetrics.evidence_transition_quality(metadata)
    search_policy_breakdown = SearchPolicyMetrics.breakdown(report_text, metadata, report_sources)

    metrics = {
        "factual_accuracy": factual,
        "factual_accuracy_str": factual_str,
        "factual_accuracy_sem": factual_sem,
        "logical_consistency": logic,
        "citation_coverage": citation,
        "bias": bias_score,
        "comprehensiveness": comprehensive,
        "evidence_graph_quality": evidence_breakdown["evidence_graph_quality"],
        "evidence_transition_quality": evidence_transition_quality,
    }
    if search_policy_breakdown.get("has_policy_trace"):
        metrics["search_policy_score"] = search_policy_breakdown["search_policy_score"]

    if ground_truth:
        weights = None
    else:
        weights = {
            "logical_consistency": 0.24,
            "citation_coverage": 0.22,
            "bias": 0.18,
            "comprehensiveness": 0.18,
            "evidence_graph_quality": 0.10,
        }
        if metadata.get("evidence_transition_trace"):
            weights["evidence_transition_quality"] = 0.08
        if search_policy_breakdown.get("has_policy_trace"):
            weights["search_policy_score"] = 0.10

    composite = RuleBasedMetrics.composite_score(metrics, weights)
    return {
        "question_id": str(query_item.get("id", "")),
        "domain": str(query_item.get("domain", "") or ""),
        "metrics": metrics,
        "evidence_metrics": evidence_breakdown,
        "search_policy_metrics": search_policy_breakdown,
        "composite_score": composite,
        "hallucination_rate": hallucination,
        "metadata_available": bool(metadata),
        "evaluation_mode": "custom_query_fallback",
    }


def _evaluate_report(report_obj: Any, question_ref: str | dict[str, Any]) -> dict[str, float]:
    bench = ResearchBench()
    if isinstance(question_ref, dict):
        question_id = str(question_ref.get("id", "") or "")
    else:
        question_id = str(question_ref)

    try:
        evaluated = bench.evaluate_report(
            report_obj,
            question_id=question_id,
            report_metadata=getattr(report_obj, "metadata", {}) or {},
        )
    except ValueError as exc:
        if isinstance(question_ref, dict) and "未找到题目 ID" in str(exc):
            evaluated = _evaluate_custom_query_report(report_obj, question_ref)
        else:
            raise

    metrics = evaluated.get("metrics", {}) if isinstance(evaluated, dict) else {}
    search_metrics = evaluated.get("search_policy_metrics", {}) if isinstance(evaluated, dict) else {}
    composite = float(evaluated.get("composite_score", 0.0) or 0.0) if isinstance(evaluated, dict) else 0.0
    return {
        "composite_score": composite,
        "factual_accuracy": float(metrics.get("factual_accuracy", 0.0) or 0.0),
        "citation_coverage": float(metrics.get("citation_coverage", 0.0) or 0.0),
        "logical_consistency": float(metrics.get("logical_consistency", 0.0) or 0.0),
        "search_policy_score": float(search_metrics.get("search_policy_score", 0.0) or 0.0),
        "budget_efficiency": float(search_metrics.get("budget_efficiency", 0.0) or 0.0),
    }


def _extract_cost(report_obj: Any) -> dict[str, float]:
    metadata = getattr(report_obj, "metadata", {}) or {}
    search_cost = metadata.get("search_cost", {}) if isinstance(metadata, dict) else {}
    return {
        "tool_calls": float(search_cost.get("tool_calls", 0.0) or 0.0),
        "search_calls": float(search_cost.get("search_calls", 0.0) or 0.0),
        "browser_calls": float(search_cost.get("browser_calls", 0.0) or 0.0),
        "estimated_token_cost": float(search_cost.get("estimated_token_cost", 0.0) or 0.0),
    }


def _extract_metadata_summary(report_obj: Any) -> dict[str, Any]:
    metadata = getattr(report_obj, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    route_stats = metadata.get("route_stats", {})
    if not isinstance(route_stats, dict):
        route_stats = {}
    policy_stats = metadata.get("policy_stats", {})
    if not isinstance(policy_stats, dict):
        policy_stats = {}
    return {
        "route_stats": route_stats,
        "policy_stats": policy_stats,
    }


def _mean(items: list[float]) -> float:
    if not items:
        return 0.0
    return float(statistics.fmean(items))


def _summarize_by_domain(success_rows: list[dict[str, Any]], query_items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    domain_by_query_id = {
        str(item.get("id", "")): str(item.get("domain", "") or "")
        for item in query_items
    }
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in success_rows:
        query_id = str(row.get("query_id", "") or "")
        domain = domain_by_query_id.get(query_id, "")
        buckets.setdefault(domain, []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for domain, rows in sorted(buckets.items()):
        summary[domain] = {
            "num_success": len(rows),
            "avg_composite_score": _mean([r["score"]["composite_score"] for r in rows]),
            "avg_factual_accuracy": _mean([r["score"]["factual_accuracy"] for r in rows]),
            "avg_citation_coverage": _mean([r["score"]["citation_coverage"] for r in rows]),
            "avg_estimated_token_cost": _mean([r["cost"]["estimated_token_cost"] for r in rows]),
            "avg_tool_calls": _mean([r["cost"]["tool_calls"] for r in rows]),
        }
    return summary


def _run_mode(
    *,
    mode: str,
    base_config: dict[str, Any],
    args: argparse.Namespace,
    query_items: list[dict[str, str]],
) -> dict[str, Any]:
    logger = logging.getLogger("head2head")
    cfg = _apply_mode_overrides(base_config, mode, args)
    preflight = _preflight_backend_requirements(cfg)
    mode_records: list[dict[str, Any]] = []

    if not preflight["is_ready"]:
        missing = ", ".join(preflight["missing_backends"])
        error = (
            "preflight_failed: missing backend configuration for "
            f"{missing}. Add the corresponding *_API_KEY and/or *_BASE_URL in .env or .env.local."
        )
        logger.warning("[%s] %s", mode, error)
        for item in query_items:
            mode_records.append(
                {
                    "query_id": item["id"],
                    "query": item["query"],
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "error": error,
                }
            )
        return {
            "mode": mode,
            "summary": {
                "num_total": len(mode_records),
                "num_success": 0,
                "num_failed": len(mode_records),
                "avg_composite_score": 0.0,
                "avg_factual_accuracy": 0.0,
                "avg_citation_coverage": 0.0,
                "avg_search_policy_score": 0.0,
                "avg_tool_calls": 0.0,
                "avg_search_calls": 0.0,
                "avg_browser_calls": 0.0,
                "avg_estimated_token_cost": 0.0,
                "avg_elapsed_seconds": 0.0,
                "avg_policy_advice_count": 0.0,
                "avg_guardrail_trigger_count": 0.0,
                "avg_policy_enforce_stop_count": 0.0,
            },
            "records": mode_records,
            "preflight": preflight,
        }

    for idx, item in enumerate(query_items, 1):
        query_id = item["id"]
        query = item["query"]
        session_id = f"head2head_{mode}_{idx:04d}"
        t0 = time.time()
        try:
            modules = initialize_modules(cfg, session_id=session_id)
            _, report_obj = asyncio.run(run_research(query, cfg, modules, return_report=True))
            elapsed = time.time() - t0
            score = _evaluate_report(report_obj, item)
            cost = _extract_cost(report_obj)
            metadata_summary = _extract_metadata_summary(report_obj)
            mode_records.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "status": "success",
                    "elapsed_seconds": round(elapsed, 3),
                    "score": score,
                    "cost": cost,
                    "metadata_summary": metadata_summary,
                }
            )
            logger.info("[%s][%s/%s] composite=%.3f, tokens=%.1f",
                        mode, idx, len(query_items), score["composite_score"], cost["estimated_token_cost"])
        except Exception as exc:
            elapsed = time.time() - t0
            mode_records.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logger.warning("[%s][%s/%s] failed: %s", mode, idx, len(query_items), exc)

    success_rows = [r for r in mode_records if r.get("status") == "success"]
    domain_summary = _summarize_by_domain(success_rows, query_items)
    summary = {
        "num_total": len(mode_records),
        "num_success": len(success_rows),
        "num_failed": len(mode_records) - len(success_rows),
        "avg_composite_score": _mean([r["score"]["composite_score"] for r in success_rows]),
        "avg_factual_accuracy": _mean([r["score"]["factual_accuracy"] for r in success_rows]),
        "avg_citation_coverage": _mean([r["score"]["citation_coverage"] for r in success_rows]),
        "avg_search_policy_score": _mean([r["score"]["search_policy_score"] for r in success_rows]),
        "avg_tool_calls": _mean([r["cost"]["tool_calls"] for r in success_rows]),
        "avg_search_calls": _mean([r["cost"]["search_calls"] for r in success_rows]),
        "avg_browser_calls": _mean([r["cost"]["browser_calls"] for r in success_rows]),
        "avg_estimated_token_cost": _mean([r["cost"]["estimated_token_cost"] for r in success_rows]),
        "avg_elapsed_seconds": _mean([r["elapsed_seconds"] for r in success_rows]),
        "avg_policy_advice_count": _mean([
            float(r.get("metadata_summary", {}).get("policy_stats", {}).get("policy_advice_count", 0.0) or 0.0)
            for r in success_rows
        ]),
        "avg_guardrail_trigger_count": _mean([
            float(r.get("metadata_summary", {}).get("policy_stats", {}).get("guardrail_trigger_count", 0.0) or 0.0)
            for r in success_rows
        ]),
        "avg_policy_enforce_stop_count": _mean([
            float(r.get("metadata_summary", {}).get("policy_stats", {}).get("policy_enforce_stop_count", 0.0) or 0.0)
            for r in success_rows
        ]),
        "domain_macro_avg_composite_score": _mean(
            [item["avg_composite_score"] for item in domain_summary.values()]
        ),
    }
    return {
        "mode": mode,
        "summary": summary,
        "records": mode_records,
        "domain_summary": domain_summary,
        "preflight": preflight,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    now = payload.get("created_at", "")
    query_items = payload.get("queries", []) if isinstance(payload.get("queries"), list) else []
    lines = [
        "# Policy Head-to-Head Report",
        "",
        f"- created_at: {now}",
        f"- config: {payload.get('config_path')}",
        f"- query_count: {payload.get('query_count')}",
        "",
    ]
    if query_items:
        lines.extend([
            "## Query Set",
            "",
            "| query_id | domain | query |",
            "|---|---|---|",
        ])
        for item in query_items:
            lines.append(
                f"| {str(item.get('id', '') or '')} | {str(item.get('domain', '') or '')} | {str(item.get('query', '') or '').replace('|', '/')} |"
            )
        lines.append("")

    lines.extend([
        "## Summary",
        "",
        "| mode | success | avg_composite | avg_factual | avg_citation | avg_policy_score | avg_tokens | avg_tool_calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in payload.get("runs", []):
        s = run.get("summary", {})
        lines.append(
            f"| {run.get('mode')} | {s.get('num_success', 0)}/{s.get('num_total', 0)} "
            f"| {s.get('avg_composite_score', 0.0):.3f}"
            f" | {s.get('avg_factual_accuracy', 0.0):.3f}"
            f" | {s.get('avg_citation_coverage', 0.0):.3f}"
            f" | {s.get('avg_search_policy_score', 0.0):.3f}"
            f" | {s.get('avg_estimated_token_cost', 0.1):.1f}"
            f" | {s.get('avg_tool_calls', 0.1):.2f} |"
        )

    has_domain_summary = any(run.get("domain_summary") for run in payload.get("runs", []))
    if has_domain_summary:
        lines.extend([
            "",
            "## Domain Summary",
            "",
            "| mode | domain_macro_avg_composite | covered_domains |",
            "|---|---:|---:|",
        ])
        for run in payload.get("runs", []):
            s = run.get("summary", {})
            domain_summary = run.get("domain_summary", {}) if isinstance(run.get("domain_summary"), dict) else {}
            lines.append(
                f"| {run.get('mode')} | {s.get('domain_macro_avg_composite_score', 0.0):.3f} | {len(domain_summary)} |"
            )

    preflight_rows = [
        run for run in payload.get("runs", [])
        if isinstance(run.get("preflight"), dict) and run["preflight"].get("required_backends")
    ]
    if preflight_rows:
        lines.extend([
            "",
            "## Backend Preflight",
            "",
            "| mode | ready | required_backends | missing_backends |",
            "|---|---:|---|---|",
        ])
        for run in preflight_rows:
            preflight = run.get("preflight", {})
            lines.append(
                f"| {run.get('mode')} | {'yes' if preflight.get('is_ready') else 'no'} "
                f"| {', '.join(preflight.get('required_backends', [])) or '-'} "
                f"| {', '.join(preflight.get('missing_backends', [])) or '-'} |"
            )

    if any(run.get("summary", {}).get("avg_policy_advice_count", 0.0) for run in payload.get("runs", [])):
        lines.extend([
            "",
            "## Policy Instrumentation",
            "",
            "| mode | avg_policy_advice | avg_guardrail_triggers | avg_enforce_stop |",
            "|---|---:|---:|---:|",
        ])
        for run in payload.get("runs", []):
            s = run.get("summary", {})
            lines.append(
                f"| {run.get('mode')} "
                f"| {s.get('avg_policy_advice_count', 0.0):.2f}"
                f" | {s.get('avg_guardrail_trigger_count', 0.0):.2f}"
                f" | {s.get('avg_policy_enforce_stop_count', 0.0):.2f} |"
            )

    lines.extend(["", "## Conclusion Hints", ""])

    by_mode = {r["mode"]: r.get("summary", {}) for r in payload.get("runs", [])}
    best_quality = max(by_mode.items(), key=lambda x: x[1].get("avg_composite_score", 0.0))[0] if by_mode else "n/a"
    best_cost = min(by_mode.items(), key=lambda x: x[1].get("avg_estimated_token_cost", 10**18))[0] if by_mode else "n/a"
    lines.append(f"- Best quality mode (avg composite): `{best_quality}`")
    lines.append(f"- Lowest cost mode (avg estimated tokens): `{best_cost}`")

    if by_mode and "learned" in by_mode and "heuristic" in by_mode:
        learned_run = next((run for run in payload.get("runs", []) if run.get("mode") == "learned"), {})
        heuristic_run = next((run for run in payload.get("runs", []) if run.get("mode") == "heuristic"), {})
        learned_preflight = learned_run.get("preflight", {}) if isinstance(learned_run.get("preflight"), dict) else {}
        heuristic_preflight = heuristic_run.get("preflight", {}) if isinstance(heuristic_run.get("preflight"), dict) else {}
        if not learned_preflight.get("is_ready", True) or not heuristic_preflight.get("is_ready", True):
            lines.append("- Experiment status: blocked before live execution because required backend env vars are missing.")
        quality_gap = by_mode["learned"].get("avg_composite_score", 0.0) - by_mode["heuristic"].get("avg_composite_score", 0.0)
        macro_quality_gap = by_mode["learned"].get("domain_macro_avg_composite_score", 0.0) - by_mode["heuristic"].get("domain_macro_avg_composite_score", 0.0)
        factual_gap = by_mode["learned"].get("avg_factual_accuracy", 0.0) - by_mode["heuristic"].get("avg_factual_accuracy", 0.0)
        citation_gap = by_mode["learned"].get("avg_citation_coverage", 0.0) - by_mode["heuristic"].get("avg_citation_coverage", 0.0)
        cost_gap = by_mode["learned"].get("avg_estimated_token_cost", 0.0) - by_mode["heuristic"].get("avg_estimated_token_cost", 0.0)
        tool_gap = by_mode["learned"].get("avg_tool_calls", 0.0) - by_mode["heuristic"].get("avg_tool_calls", 0.0)
        elapsed_gap = by_mode["learned"].get("avg_elapsed_seconds", 0.0) - by_mode["heuristic"].get("avg_elapsed_seconds", 0.0)
        lines.append(
            f"- Learned vs Heuristic: quality_delta={quality_gap:+.3f}, macro_quality_delta={macro_quality_gap:+.3f}, factual_delta={factual_gap:+.3f}, citation_delta={citation_gap:+.3f}, token_delta={cost_gap:+.1f}, tool_delta={tool_gap:+.2f}, elapsed_delta={elapsed_gap:+.2f}s"
        )
        if quality_gap > 0 and citation_gap >= 0 and cost_gap <= 0:
            lines.append("- Interpretation: learned policy improved quality while reducing or keeping cost.")
        elif quality_gap > 0 and citation_gap >= 0 and cost_gap > 0:
            lines.append("- Interpretation: learned policy improved quality with extra cost (quality-cost tradeoff).")
        elif quality_gap <= 0 and cost_gap < 0:
            lines.append("- Interpretation: learned policy reduced cost but also reduced quality.")
        else:
            lines.append("- Interpretation: no clear win yet; collect more queries or tune thresholds.")
        if quality_gap > 0 and citation_gap < 0:
            lines.append("- Risk flag: quality improved but citation coverage dropped, which often indicates premature stopping or reward mismatch.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run off/heuristic/learned head-to-head comparison")
    parser.add_argument("--config", type=str, default="configs/aliyun_smoke.yaml", help="config path")
    parser.add_argument("--query", type=str, default=None, help="single query")
    parser.add_argument("--queries_file", type=str, default=None, help="json/jsonl query file")
    parser.add_argument("--num_questions", type=int, default=5, help="number of benchmark queries if no query provided")
    parser.add_argument("--domain", type=str, default=None, help="benchmark domain filter")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["benchmark_order", "balanced_domains"],
        default="balanced_domains",
        help="benchmark query sampling strategy when no queries_file/query is provided",
    )
    parser.add_argument("--modes", type=str, default="off,heuristic,learned", help="comma-separated modes")
    parser.add_argument("--search-policy-model-path", type=str, default=None, help="override search policy model path")
    parser.add_argument("--evidence-policy-mode", type=str, choices=["heuristic", "learned"], default=None)
    parser.add_argument("--evidence-policy-model-path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/policy_head2head", help="output directory")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("head2head")

    base_config = load_config(args.config)
    query_items = _load_queries(
        query=args.query,
        queries_file=args.queries_file,
        num_questions=args.num_questions,
        domain=args.domain,
        sampling_strategy=args.sampling_strategy,
    )
    if not query_items:
        raise SystemExit("no queries to run")

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in {"off", "heuristic", "learned"}:
            raise SystemExit(f"unsupported mode: {m}")

    os.makedirs(args.output_dir, exist_ok=True)
    runs = []
    for mode in modes:
        logger.info("Running mode: %s", mode)
        runs.append(
            _run_mode(
                mode=mode,
                base_config=base_config,
                args=args,
                query_items=query_items,
            )
        )

    payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": args.config,
        "query_count": len(query_items),
        "sampling_strategy": args.sampling_strategy,
        "queries": query_items,
        "runs": runs,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"head2head_{ts}.json")
    md_path = os.path.join(args.output_dir, f"head2head_{ts}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(payload))

    logger.info("Saved JSON: %s", json_path)
    logger.info("Saved Markdown: %s", md_path)
    print(md_path)


if __name__ == "__main__":
    main()
