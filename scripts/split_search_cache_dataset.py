#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/split_search_cache_dataset.py
================================================================================
Create a deterministic train / held-out split from existing search-cache JSONL.

Design goals:
  1) query-level no-leakage
  2) preserve at least some browser-grounded examples in both train and held-out
  3) keep held-out domain coverage reasonably broad via round-robin selection

Usage:
  python scripts/split_search_cache_dataset.py \
    --inputs data/search_cache/natural_heuristic_20260515 \
    --source-label research_bench \
    --heldout-ratio 0.2 \
    --output-dir outputs/search_cache_splits_20260515 \
    --output-cache-dir data/search_cache/natural_split_20260515 \
    --output-query-dir data/queries
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_search_cache import _load_records, _resolve_jsonl_inputs
from scripts.mine_natural_browser_queries import summarize_record


def _is_grounded(
    summary: dict[str, Any],
    *,
    min_browser_calls: int,
    min_citation_coverage: float,
) -> bool:
    return int(summary.get("browser_calls", 0) or 0) >= min_browser_calls and float(
        summary.get("citation_coverage", 0.0) or 0.0
    ) >= min_citation_coverage


def _selection_priority(summary: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(summary.get("citation_coverage", 0.0) or 0.0),
        float(summary.get("composite_score", 0.0) or 0.0),
        float(summary.get("factual_accuracy", 0.0) or 0.0),
        float(summary.get("confidence", 0.0) or 0.0),
        str(summary.get("query_id", "") or ""),
    )


def _group_and_sort(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[str(item.get("domain", "") or "")].append(item)
    for domain_items in buckets.values():
        domain_items.sort(key=_selection_priority, reverse=True)
    return dict(buckets)


def _round_robin_pick(buckets: dict[str, list[dict[str, Any]]], target: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    mutable = {domain: list(items) for domain, items in buckets.items()}

    while len(picked) < target:
        nonempty = [(domain, items) for domain, items in mutable.items() if items]
        if not nonempty:
            break
        nonempty.sort(key=lambda pair: (-len(pair[1]), pair[0]))
        progressed = False
        for domain, items in nonempty:
            if len(picked) >= target:
                break
            picked.append(items.pop(0))
            progressed = True
        if not progressed:
            break

    return picked


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


def split_records(
    records: list[dict[str, Any]],
    *,
    source_label: str = "research_bench",
    heldout_ratio: float = 0.2,
    min_heldout_size: int = 1,
    grounded_min_browser_calls: int = 2,
    grounded_min_citation_coverage: float = 0.02,
    min_grounded_heldout: int = 2,
    min_grounded_train: int = 2,
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
        raise ValueError("not enough eligible records to create a train/held-out split")

    heldout_target = max(min_heldout_size, int(math.ceil(total * heldout_ratio)))
    heldout_target = min(heldout_target, total - 1)

    grounded_pairs = [
        pair for pair in eligible_pairs
        if _is_grounded(
            pair[1],
            min_browser_calls=grounded_min_browser_calls,
            min_citation_coverage=grounded_min_citation_coverage,
        )
    ]
    nongrounded_pairs = [pair for pair in eligible_pairs if pair not in grounded_pairs]

    grounded_target = max(min_grounded_heldout, int(round(len(grounded_pairs) * heldout_ratio)))
    grounded_target = min(grounded_target, heldout_target)
    grounded_target = min(grounded_target, max(0, len(grounded_pairs) - min_grounded_train))
    grounded_target = max(0, grounded_target)

    grounded_buckets = _group_and_sort([summary for _, summary in grounded_pairs])
    selected_grounded = _round_robin_pick(grounded_buckets, grounded_target)
    selected_grounded_ids = {str(item["query_id"]) for item in selected_grounded}

    nongrounded_summaries = [
        summary for _, summary in nongrounded_pairs
        if str(summary["query_id"]) not in selected_grounded_ids
    ]
    remaining_target = heldout_target - len(selected_grounded)
    remaining_buckets = _group_and_sort(nongrounded_summaries)
    selected_remaining = _round_robin_pick(remaining_buckets, remaining_target)

    if len(selected_remaining) < remaining_target:
        leftover_grounded_summaries = [
            summary for _, summary in grounded_pairs
            if str(summary["query_id"]) not in selected_grounded_ids
        ]
        extra_needed = remaining_target - len(selected_remaining)
        extra_grounded = _round_robin_pick(_group_and_sort(leftover_grounded_summaries), extra_needed)
        selected_remaining.extend(extra_grounded)

    heldout_ids = selected_grounded_ids | {str(item["query_id"]) for item in selected_remaining}

    train_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []
    train_queries: list[dict[str, Any]] = []
    heldout_queries: list[dict[str, Any]] = []
    train_grounded = 0
    heldout_grounded = 0
    train_domain_counts = Counter()
    heldout_domain_counts = Counter()

    summary_by_id = {str(summary["query_id"]): summary for _, summary in eligible_pairs}
    record_by_id = {str(summary["query_id"]): record for record, summary in eligible_pairs}

    for query_id, summary in summary_by_id.items():
        record = record_by_id[query_id]
        query_item = _record_to_query_item(record, summary)
        grounded = _is_grounded(
            summary,
            min_browser_calls=grounded_min_browser_calls,
            min_citation_coverage=grounded_min_citation_coverage,
        )
        if query_id in heldout_ids:
            heldout_records.append(record)
            heldout_queries.append(query_item)
            heldout_domain_counts[str(summary.get("domain", "") or "")] += 1
            if grounded:
                heldout_grounded += 1
        else:
            train_records.append(record)
            train_queries.append(query_item)
            train_domain_counts[str(summary.get("domain", "") or "")] += 1
            if grounded:
                train_grounded += 1

    train_queries.sort(key=lambda item: str(item.get("id", "")))
    heldout_queries.sort(key=lambda item: str(item.get("id", "")))
    train_records.sort(key=lambda item: str(item.get("query_id", "")))
    heldout_records.sort(key=lambda item: str(item.get("query_id", "")))

    split_summary = {
        "created_at": datetime.now().isoformat(),
        "source_label": source_label,
        "num_eligible_queries": total,
        "heldout_target": heldout_target,
        "grounded_query_count": len(grounded_pairs),
        "grounded_target_heldout": grounded_target,
        "grounded_definition": {
            "min_browser_calls": grounded_min_browser_calls,
            "min_citation_coverage": grounded_min_citation_coverage,
        },
        "train": {
            "num_queries": len(train_queries),
            "grounded_queries": train_grounded,
            "domain_distribution": dict(sorted(train_domain_counts.items())),
            "query_ids": [str(item.get("id", "")) for item in train_queries],
        },
        "heldout": {
            "num_queries": len(heldout_queries),
            "grounded_queries": heldout_grounded,
            "domain_distribution": dict(sorted(heldout_domain_counts.items())),
            "query_ids": [str(item.get("id", "")) for item in heldout_queries],
        },
        "skipped_records": dict(skipped),
    }

    return {
        "summary": split_summary,
        "train_records": train_records,
        "heldout_records": heldout_records,
        "train_queries": train_queries,
        "heldout_queries": heldout_queries,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    train = summary.get("train", {})
    heldout = summary.get("heldout", {})
    grounded_definition = summary.get("grounded_definition", {})
    lines = [
        "# Search Cache Split",
        "",
        f"- created_at: {summary.get('created_at', '')}",
        f"- source_label: {summary.get('source_label', '')}",
        f"- num_eligible_queries: {summary.get('num_eligible_queries', 0)}",
        f"- heldout_target: {summary.get('heldout_target', 0)}",
        f"- grounded_query_count: {summary.get('grounded_query_count', 0)}",
        f"- grounded_target_heldout: {summary.get('grounded_target_heldout', 0)}",
        "",
        "## Grounded Definition",
        "",
        f"- min_browser_calls: {grounded_definition.get('min_browser_calls', 0)}",
        f"- min_citation_coverage: {grounded_definition.get('min_citation_coverage', 0.0):.4f}",
        "",
        "## Split Sizes",
        "",
        f"- train_queries: {train.get('num_queries', 0)}",
        f"- train_grounded_queries: {train.get('grounded_queries', 0)}",
        f"- heldout_queries: {heldout.get('num_queries', 0)}",
        f"- heldout_grounded_queries: {heldout.get('grounded_queries', 0)}",
        "",
        "## Train Query IDs",
        "",
    ]
    for query_id in train.get("query_ids", []):
        lines.append(f"- {query_id}")
    lines.extend(["", "## Held-out Query IDs", ""])
    for query_id in heldout.get("query_ids", []):
        lines.append(f"- {query_id}")
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train / held-out splits from search-cache JSONL")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL files or directories")
    parser.add_argument("--source-label", type=str, default="research_bench")
    parser.add_argument("--heldout-ratio", type=float, default=0.2)
    parser.add_argument("--min-heldout-size", type=int, default=1)
    parser.add_argument("--grounded-min-browser-calls", type=int, default=2)
    parser.add_argument("--grounded-min-citation-coverage", type=float, default=0.02)
    parser.add_argument("--min-grounded-heldout", type=int, default=2)
    parser.add_argument("--min-grounded-train", type=int, default=2)
    parser.add_argument("--output-dir", type=str, required=True, help="summary output directory")
    parser.add_argument("--output-cache-dir", type=str, required=True, help="train/heldout cache output directory")
    parser.add_argument("--output-query-dir", type=str, required=True, help="train/heldout query output directory")
    parser.add_argument("--prefix", type=str, default="natural_split")
    args = parser.parse_args()

    input_paths = _resolve_jsonl_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("no JSONL inputs found")

    records = _load_records(input_paths)
    split = split_records(
        records,
        source_label=args.source_label,
        heldout_ratio=args.heldout_ratio,
        min_heldout_size=args.min_heldout_size,
        grounded_min_browser_calls=args.grounded_min_browser_calls,
        grounded_min_citation_coverage=args.grounded_min_citation_coverage,
        min_grounded_heldout=args.min_grounded_heldout,
        min_grounded_train=args.min_grounded_train,
    )
    summary = dict(split["summary"])
    summary["input_paths"] = [str(path) for path in input_paths]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / f"{prefix}_split_{timestamp}.json"
    summary_md = output_dir / f"{prefix}_split_{timestamp}.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    cache_dir = Path(args.output_cache_dir)
    train_cache_path = cache_dir / f"{prefix}_train_{timestamp}.jsonl"
    heldout_cache_path = cache_dir / f"{prefix}_heldout_{timestamp}.jsonl"
    _write_jsonl(train_cache_path, split["train_records"])
    _write_jsonl(heldout_cache_path, split["heldout_records"])

    query_dir = Path(args.output_query_dir)
    train_queries_path = query_dir / f"{prefix}_train_{timestamp}.jsonl"
    heldout_queries_path = query_dir / f"{prefix}_heldout_{timestamp}.jsonl"
    _write_jsonl(train_queries_path, split["train_queries"])
    _write_jsonl(heldout_queries_path, split["heldout_queries"])

    payload = {
        **summary,
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "train_cache": str(train_cache_path),
            "heldout_cache": str(heldout_cache_path),
            "train_queries": str(train_queries_path),
            "heldout_queries": str(heldout_queries_path),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
