#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.rule_based import RuleBasedMetrics  # noqa: E402
from evaluation.metrics.search_policy import SearchPolicyMetrics  # noqa: E402
from evaluation.metrics.search_policy_reward import SearchPolicyReward  # noqa: E402
from scripts.analyze_search_cache import (  # noqa: E402
    _load_records,
    _resolve_jsonl_inputs,
    render_markdown as render_cache_analysis_markdown,
    summarize_cache_records,
)
from src.agents.summarizer import SummarizerAgent  # noqa: E402

MAX_ANALYSIS_STEM_CHARS = 48


def analysis_stem_for_output(output_path: Path, *, output_file_requested: bool, timestamp: str) -> str:
    stem = (
        f"search_cache_analysis_{output_path.stem}"
        if output_file_requested
        else f"search_cache_analysis_resurfaced_{timestamp}"
    )
    if len(stem) <= MAX_ANALYSIS_STEM_CHARS:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:MAX_ANALYSIS_STEM_CHARS - 11]}_{digest}"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _recompute_composite(
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    search_policy_metrics: dict[str, Any],
) -> float:
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    if search_policy_metrics.get("has_policy_trace"):
        numeric_metrics["search_policy_score"] = _safe_float(
            search_policy_metrics.get("search_policy_score", 0.0)
        )

    weights = {
        "factual_accuracy": 0.25,
        "logical_consistency": 0.20,
        "citation_coverage": 0.20,
        "bias": 0.20,
        "comprehensiveness": 0.15,
    }
    if isinstance(metadata, dict) and metadata.get("evidence_snapshot"):
        weights.setdefault("evidence_graph_quality", 0.10)
        if metadata.get("evidence_transition_trace"):
            weights.setdefault("evidence_transition_quality", 0.05)
    if search_policy_metrics.get("has_policy_trace"):
        weights.setdefault("search_policy_score", 0.10)

    return RuleBasedMetrics.composite_score(numeric_metrics, weights)


def _replace_formatted_report_content(
    formatted_report: str,
    old_content: str,
    new_content: str,
) -> tuple[str, bool]:
    if not formatted_report or not old_content or old_content == new_content:
        return formatted_report, False
    if old_content not in formatted_report:
        return formatted_report, False
    return formatted_report.replace(old_content, new_content, 1), True


def resurface_record(
    record: dict[str, Any],
    *,
    agent: SummarizerAgent | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        return record

    updated = deepcopy(record)
    if str(updated.get("status", "success") or "success") != "success":
        return updated

    content = str(updated.get("report_content", "") or "")
    sources = updated.get("sources", [])
    if not content or not isinstance(sources, list) or not sources:
        updated["citation_resurface"] = {
            "status": "skipped",
            "reason": "missing_report_content_or_sources",
        }
        return updated

    evaluation = updated.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}

    agent = agent or SummarizerAgent(name="citation_resurface", policy=None)
    old_metrics = evaluation.get("metrics", {})
    if not isinstance(old_metrics, dict):
        old_metrics = {}
    old_citation = _safe_float(old_metrics.get("citation_coverage", 0.0))
    old_composite = _safe_float(evaluation.get("composite_score", 0.0))
    old_reward = _safe_float(updated.get("reward", 0.0))

    new_content = agent._surface_explicit_citations(content, sources)
    new_citation = RuleBasedMetrics.citation_coverage(new_content)

    updated["report_content"] = new_content
    formatted_report = str(updated.get("formatted_report", "") or "")
    new_formatted, formatted_replaced = _replace_formatted_report_content(
        formatted_report,
        content,
        new_content,
    )
    if formatted_replaced:
        updated["formatted_report"] = new_formatted

    metadata = updated.get("report_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    report_sources = updated.get("sources", []) if isinstance(updated.get("sources", []), list) else []

    metrics = evaluation.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    metrics = dict(metrics)
    metrics["citation_coverage"] = new_citation

    search_policy_metrics = SearchPolicyMetrics.breakdown(
        new_content,
        metadata,
        report_sources,
    )
    evaluation["metrics"] = metrics
    evaluation["search_policy_metrics"] = search_policy_metrics
    evaluation["composite_score"] = _recompute_composite(
        metrics,
        metadata,
        search_policy_metrics,
    )
    updated["evaluation"] = evaluation

    process_reward_trace = updated.get("process_reward_trace", [])
    if not isinstance(process_reward_trace, list):
        process_reward_trace = []
    reward_breakdown = SearchPolicyReward.breakdown(
        report_text=new_content,
        report_metadata=metadata,
        report_sources=report_sources,
        evaluation_result=evaluation,
        process_reward_trace=process_reward_trace,
        confidence=_safe_float(updated.get("confidence", 0.0)),
    )
    updated["reward_breakdown"] = reward_breakdown
    updated["reward"] = reward_breakdown.get("reward", 0.0)

    updated["citation_resurface"] = {
        "status": "updated" if new_content != content else "unchanged",
        "old_citation_coverage": old_citation,
        "new_citation_coverage": new_citation,
        "citation_delta": round(new_citation - old_citation, 6),
        "old_composite_score": old_composite,
        "new_composite_score": evaluation["composite_score"],
        "composite_delta": round(evaluation["composite_score"] - old_composite, 6),
        "old_reward": old_reward,
        "new_reward": updated["reward"],
        "reward_delta": round(updated["reward"] - old_reward, 6),
        "content_changed": new_content != content,
        "formatted_report_replaced": formatted_replaced,
    }
    return updated


def resurface_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    agent = SummarizerAgent(name="citation_resurface", policy=None)
    updated_records = [resurface_record(record, agent=agent) for record in records]

    deltas: list[float] = []
    composite_deltas: list[float] = []
    reward_deltas: list[float] = []
    changed = 0
    improved = 0
    for record in updated_records:
        info = record.get("citation_resurface", {})
        if not isinstance(info, dict) or info.get("status") == "skipped":
            continue
        delta = _safe_float(info.get("citation_delta", 0.0))
        deltas.append(delta)
        composite_deltas.append(_safe_float(info.get("composite_delta", 0.0)))
        reward_deltas.append(_safe_float(info.get("reward_delta", 0.0)))
        if info.get("content_changed"):
            changed += 1
        if delta > 0.0:
            improved += 1

    summary = {
        "created_at": datetime.now().isoformat(),
        "num_records": len(records),
        "num_changed": changed,
        "num_citation_improved": improved,
        "avg_citation_delta": round(_mean(deltas), 6),
        "avg_composite_delta": round(_mean(composite_deltas), 6),
        "avg_reward_delta": round(_mean(reward_deltas), 6),
    }
    return updated_records, summary


def _save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-surface inline citations in existing search-cache JSONL files.",
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL files or directories")
    parser.add_argument("--output-dir", type=str, default="data/search_cache/citation_resurfaced")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument(
        "--analysis-output-dir",
        type=str,
        default=None,
        help="Optional directory for post-resurface search-cache analysis.",
    )
    args = parser.parse_args()

    input_paths = _resolve_jsonl_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("no JSONL inputs found")

    records = _load_records(input_paths)
    updated_records, resurface_summary = resurface_records(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir)
    output_path = Path(args.output_file) if args.output_file else output_dir / f"search_cache_resurfaced_{timestamp}.jsonl"
    manifest_path = output_dir / f"search_cache_resurfaced_manifest_{timestamp}.json"
    _save_jsonl(updated_records, output_path)

    analysis_summary = summarize_cache_records(updated_records, [output_path])
    manifest = {
        **resurface_summary,
        "input_paths": [str(path) for path in input_paths],
        "output_file": str(output_path),
        "analysis_summary": analysis_summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.analysis_output_dir:
        analysis_dir = Path(args.analysis_output_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_stem = analysis_stem_for_output(
            output_path,
            output_file_requested=bool(args.output_file),
            timestamp=timestamp,
        )
        analysis_json = analysis_dir / f"{analysis_stem}.json"
        analysis_md = analysis_dir / f"{analysis_stem}.md"
        analysis_json.write_text(json.dumps(analysis_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        analysis_md.write_text(render_cache_analysis_markdown(analysis_summary), encoding="utf-8")

    print(f"[resurface] JSONL written: {output_path}")
    print(f"[resurface] Manifest written: {manifest_path}")
    print(
        "[resurface] Summary: "
        f"changed={resurface_summary['num_changed']}, "
        f"citation_improved={resurface_summary['num_citation_improved']}, "
        f"avg_citation_delta={resurface_summary['avg_citation_delta']:.4f}"
    )


if __name__ == "__main__":
    main()
