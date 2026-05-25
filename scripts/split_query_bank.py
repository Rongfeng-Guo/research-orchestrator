#!/usr/bin/env python3
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


def load_query_bank(path: str | Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if isinstance(item, str):
                item = {"query": item}
            if not isinstance(item, dict):
                raise ValueError(f"invalid query item at line {line_no}: {item!r}")
            query = str(item.get("query", "") or "").strip()
            if not query:
                raise ValueError(f"missing query at line {line_no}")
            normalized = dict(item)
            normalized.setdefault("id", f"query_{line_no:04d}")
            normalized.setdefault("domain", "unknown")
            normalized["query"] = query
            queries.append(normalized)
    return queries


def _domain_buckets(queries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in queries:
        buckets[str(item.get("domain", "") or "unknown")].append(item)
    for items in buckets.values():
        items.sort(key=lambda item: str(item.get("id", "")))
    return dict(sorted(buckets.items()))


def _round_robin_pick(buckets: dict[str, list[dict[str, Any]]], target: int) -> list[dict[str, Any]]:
    mutable = {domain: list(items) for domain, items in buckets.items()}
    picked: list[dict[str, Any]] = []
    while len(picked) < target:
        nonempty = [(domain, items) for domain, items in mutable.items() if items]
        if not nonempty:
            break
        nonempty.sort(key=lambda pair: (-len(pair[1]), pair[0]))
        for _domain, items in nonempty:
            if len(picked) >= target:
                break
            picked.append(items.pop(0))
    return picked


def split_queries(
    queries: list[dict[str, Any]],
    *,
    heldout_ratio: float = 0.25,
    min_heldout_size: int = 8,
    min_train_size: int = 8,
) -> dict[str, Any]:
    if len(queries) < min_train_size + 1:
        raise ValueError("not enough queries to create a train/heldout split")

    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in queries:
        query_id = str(item.get("id", "") or "")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query id: {query_id}")
        seen_ids.add(query_id)
        deduped.append(dict(item))

    heldout_target = max(min_heldout_size, math.ceil(len(deduped) * heldout_ratio))
    heldout_target = min(heldout_target, len(deduped) - min_train_size)
    if heldout_target <= 0:
        raise ValueError("heldout target is empty; lower min_train_size or add more queries")

    heldout = _round_robin_pick(_domain_buckets(deduped), heldout_target)
    heldout_ids = {str(item.get("id", "")) for item in heldout}
    train = [item for item in deduped if str(item.get("id", "")) not in heldout_ids]
    train.sort(key=lambda item: str(item.get("id", "")))
    heldout.sort(key=lambda item: str(item.get("id", "")))

    summary = {
        "created_at": datetime.now().isoformat(),
        "num_queries": len(deduped),
        "heldout_ratio": heldout_ratio,
        "min_heldout_size": min_heldout_size,
        "min_train_size": min_train_size,
        "train": {
            "num_queries": len(train),
            "domain_distribution": dict(sorted(Counter(str(item.get("domain", "") or "unknown") for item in train).items())),
            "query_ids": [str(item.get("id", "")) for item in train],
        },
        "heldout": {
            "num_queries": len(heldout),
            "domain_distribution": dict(sorted(Counter(str(item.get("domain", "") or "unknown") for item in heldout).items())),
            "query_ids": [str(item.get("id", "")) for item in heldout],
        },
    }
    return {"summary": summary, "train_queries": train, "heldout_queries": heldout}


def render_markdown(summary: dict[str, Any]) -> str:
    train = summary.get("train", {})
    heldout = summary.get("heldout", {})
    lines = [
        "# Query Bank Split",
        "",
        f"- created_at: {summary.get('created_at', '')}",
        f"- num_queries: {summary.get('num_queries', 0)}",
        f"- heldout_ratio: {summary.get('heldout_ratio', 0.0):.4f}",
        f"- train_queries: {train.get('num_queries', 0)}",
        f"- heldout_queries: {heldout.get('num_queries', 0)}",
        "",
        "## Domain Distribution",
        "",
        "| split | domain | count |",
        "|---|---|---:|",
    ]
    for split_name, payload in [("train", train), ("heldout", heldout)]:
        for domain, count in payload.get("domain_distribution", {}).items():
            lines.append(f"| {split_name} | {domain} | {count} |")

    lines.extend(["", "## Heldout Query IDs", ""])
    for query_id in heldout.get("query_ids", []):
        lines.append(f"- {query_id}")
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/heldout query files from a JSONL query bank.")
    parser.add_argument("--queries-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/queries")
    parser.add_argument("--summary-dir", type=str, default="outputs/query_bank_splits")
    parser.add_argument("--train-output-file", type=str, default=None)
    parser.add_argument("--heldout-output-file", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--summary-md", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="query_bank")
    parser.add_argument("--heldout-ratio", type=float, default=0.25)
    parser.add_argument("--min-heldout-size", type=int, default=8)
    parser.add_argument("--min-train-size", type=int, default=8)
    args = parser.parse_args()

    split = split_queries(
        load_query_bank(args.queries_file),
        heldout_ratio=args.heldout_ratio,
        min_heldout_size=args.min_heldout_size,
        min_train_size=args.min_train_size,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    train_path = Path(args.train_output_file) if args.train_output_file else output_dir / f"{args.prefix}_train_{timestamp}.jsonl"
    heldout_path = Path(args.heldout_output_file) if args.heldout_output_file else output_dir / f"{args.prefix}_heldout_{timestamp}.jsonl"
    _write_jsonl(train_path, split["train_queries"])
    _write_jsonl(heldout_path, split["heldout_queries"])

    summary = {
        **split["summary"],
        "input_file": str(args.queries_file),
        "artifacts": {
            "train_queries": str(train_path),
            "heldout_queries": str(heldout_path),
        },
    }
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json) if args.summary_json else summary_dir / f"{args.prefix}_split_{timestamp}.json"
    summary_md = Path(args.summary_md) if args.summary_md else summary_dir / f"{args.prefix}_split_{timestamp}.md"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    payload = {
        **summary,
        "artifacts": {
            **summary["artifacts"],
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
