#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a concise paper-style research brief from current indexed outputs.

Usage:
  python scripts/render_research_brief.py
  python scripts/render_research_brief.py --index_json outputs/research_index/research_output_index.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_present(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return _load_json(path)


def _latest_head2head(index_payload: dict[str, Any]) -> dict[str, Any] | None:
    for item in index_payload.get("artifacts", []):
        if item.get("artifact_type") == "policy_head2head":
            return item
    return None


def build_brief(index_payload: dict[str, Any]) -> dict[str, Any]:
    latest = _latest_head2head(index_payload)
    audit_entry = next((item for item in index_payload.get("artifacts", []) if item.get("artifact_type") == "research_audit"), None)
    audit_payload = _load_json_if_present(Path(audit_entry["path_json"])) if audit_entry and audit_entry.get("path_json") else {}
    latest_payload = _load_json_if_present(Path(latest["path_json"])) if latest and latest.get("path_json") else {}

    benchmark = audit_payload.get("benchmark", {}) if isinstance(audit_payload.get("benchmark"), dict) else {}
    strongest = (
        (audit_payload.get("policy_artifacts", {}) or {}).get("strongest_eval_artifact", {})
        if isinstance(audit_payload.get("policy_artifacts"), dict)
        else {}
    )
    findings = audit_payload.get("key_findings", []) if isinstance(audit_payload.get("key_findings"), list) else []

    runs = latest_payload.get("runs", []) if isinstance(latest_payload.get("runs"), list) else []
    by_mode = {str(run.get("mode", "")): run for run in runs}
    heuristic = by_mode.get("heuristic", {})
    learned = by_mode.get("learned", {})
    heuristic_summary = heuristic.get("summary", {}) if isinstance(heuristic.get("summary"), dict) else {}
    learned_summary = learned.get("summary", {}) if isinstance(learned.get("summary"), dict) else {}
    learned_preflight = learned.get("preflight", {}) if isinstance(learned.get("preflight"), dict) else {}
    heuristic_preflight = heuristic.get("preflight", {}) if isinstance(heuristic.get("preflight"), dict) else {}

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_size": benchmark.get("num_questions", 0),
        "benchmark_domain_count": benchmark.get("num_domains", 0),
        "benchmark_domain_imbalance_ratio": benchmark.get("domain_imbalance_ratio"),
        "recommended_policy_artifact": strongest.get("artifact"),
        "recommended_policy_eval_accuracy": strongest.get("eval_accuracy"),
        "head2head_label": latest.get("label") if latest else None,
        "head2head_query_count": latest.get("query_count", 0) if latest else 0,
        "head2head_domains": latest.get("domains", []) if latest else [],
        "head2head_sampling_strategy": latest.get("sampling_strategy") if latest else None,
        "heuristic_success": heuristic_summary.get("num_success", 0),
        "learned_success": learned_summary.get("num_success", 0),
        "quality_delta": latest.get("quality_delta", 0.0) if latest else 0.0,
        "macro_quality_delta": latest.get("macro_quality_delta", 0.0) if latest else 0.0,
        "blocked_by_preflight": latest.get("blocked_by_preflight", False) if latest else False,
        "missing_backends": latest.get("missing_backends", []) if latest else [],
        "heuristic_ready": heuristic_preflight.get("is_ready", True),
        "learned_ready": learned_preflight.get("is_ready", True),
        "key_findings": findings,
    }


def write_markdown(brief: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Research Brief",
        "",
        f"- Created at: {brief.get('created_at', '')}",
        f"- Benchmark size: {brief.get('benchmark_size', 0)} queries across {brief.get('benchmark_domain_count', 0)} domains",
        f"- Domain imbalance ratio: {brief.get('benchmark_domain_imbalance_ratio', 'N/A')}",
        f"- Recommended policy artifact: `{brief.get('recommended_policy_artifact', 'N/A')}`",
        f"- Recommended artifact eval accuracy: `{brief.get('recommended_policy_eval_accuracy', 'N/A')}`",
        "",
        "## Executive Summary",
        "",
    ]

    if brief.get("blocked_by_preflight"):
        lines.append(
            f"- Current learned-vs-heuristic head-to-head is blocked by missing backend configuration: `{', '.join(brief.get('missing_backends', []))}`."
        )
    else:
        lines.append(
            f"- Latest head-to-head shows quality_delta={brief.get('quality_delta', 0.0):+.4f} and macro_quality_delta={brief.get('macro_quality_delta', 0.0):+.4f}."
        )
    lines.append(
        f"- Latest balanced head-to-head sample covers {brief.get('head2head_query_count', 0)} queries from domains: `{brief.get('head2head_domains', [])}`."
    )
    lines.append(
        f"- Research readiness audit currently recommends `{brief.get('recommended_policy_artifact', 'N/A')}` as the strongest learned policy candidate."
    )

    lines.extend(["", "## Evidence", ""])
    for finding in brief.get("key_findings", []):
        lines.append(f"- {finding}")

    lines.extend(["", "## Next Actions", ""])
    if brief.get("blocked_by_preflight"):
        lines.append("- Provide live backend env configuration, then rerun balanced learned-vs-heuristic head-to-head.")
    lines.append("- Keep using balanced-domain sampling for small query-level comparisons.")
    lines.append("- Prefer domain-macro quality as a primary comparison metric alongside overall average.")
    lines.append("- Expand toward cross-fold cache-first evaluation once real search-cache JSONL artifacts are available.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a concise research brief from indexed outputs")
    parser.add_argument("--index_json", type=str, default="outputs/research_index/research_output_index.json")
    parser.add_argument("--output_dir", type=str, default="outputs/research_brief")
    args = parser.parse_args()

    index_path = PROJECT_ROOT / args.index_json
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index_payload = _load_json(index_path)
    brief = build_brief(index_payload)

    json_path = output_dir / "research_brief.json"
    md_path = output_dir / "research_brief.md"
    json_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(brief, md_path)

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
