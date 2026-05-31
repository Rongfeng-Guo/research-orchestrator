#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a presentation-friendly research dashboard from current output artifacts.

Usage:
  python scripts/render_research_dashboard.py
  python scripts/render_research_dashboard.py --output_dir outputs/research_dashboard
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


def _ascii_bar(value: int, max_value: int, width: int = 24) -> str:
    if max_value <= 0:
        return ""
    filled = max(1, round(width * value / max_value)) if value > 0 else 0
    return "#" * filled


def _build_domain_table(distribution: dict[str, Any]) -> list[dict[str, Any]]:
    if not distribution:
        return []

    normalized: list[tuple[str, int]] = []
    for domain, count in distribution.items():
        try:
            normalized.append((str(domain), int(count)))
        except (TypeError, ValueError):
            continue

    normalized.sort(key=lambda item: (-item[1], item[0]))
    max_count = normalized[0][1] if normalized else 0
    return [
        {"domain": domain, "count": count, "bar": _ascii_bar(count, max_count)}
        for domain, count in normalized
    ]


def _build_policy_leaderboard(audit_payload: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = (audit_payload.get("policy_artifacts", {}) or {}).get("artifacts", [])
    rows: list[dict[str, Any]] = []
    if not isinstance(artifacts, list):
        return rows

    for item in artifacts:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "artifact": item.get("artifact"),
                "eval_accuracy": float(item.get("eval_accuracy", 0.0) or 0.0),
                "train_accuracy": float(item.get("train_accuracy", 0.0) or 0.0),
                "num_examples": int(item.get("num_examples", 0) or 0),
                "browser_examples": int(item.get("browser_examples", 0) or 0),
                "label_imbalance_ratio": float(item.get("label_imbalance_ratio", 0.0) or 0.0),
            }
        )

    rows.sort(key=lambda item: (-item["eval_accuracy"], -item["num_examples"], str(item["artifact"])))
    return rows


def _build_run_registry(index_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in index_payload.get("artifacts", []):
        if item.get("artifact_type") != "policy_head2head":
            continue
        rows.append(
            {
                "label": item.get("label"),
                "created_at": item.get("created_at"),
                "query_count": int(item.get("query_count", 0) or 0),
                "sampling_strategy": item.get("sampling_strategy") or "front_slice",
                "blocked_by_preflight": bool(item.get("blocked_by_preflight")),
                "missing_backends": list(item.get("missing_backends", []) or []),
                "quality_delta": float(item.get("quality_delta", 0.0) or 0.0),
                "macro_quality_delta": float(item.get("macro_quality_delta", 0.0) or 0.0),
                "domains": list(item.get("domains", []) or []),
            }
        )
    rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return rows


def build_dashboard(
    brief_payload: dict[str, Any],
    index_payload: dict[str, Any],
    audit_payload: dict[str, Any],
) -> dict[str, Any]:
    benchmark = audit_payload.get("benchmark", {}) if isinstance(audit_payload.get("benchmark"), dict) else {}
    strongest = (
        (audit_payload.get("policy_artifacts", {}) or {}).get("strongest_eval_artifact", {})
        if isinstance(audit_payload.get("policy_artifacts"), dict)
        else {}
    )
    latest_run = next((item for item in index_payload.get("artifacts", []) if item.get("artifact_type") == "policy_head2head"), {})

    domain_table = _build_domain_table(benchmark.get("domain_distribution", {}) or {})
    leaderboard = _build_policy_leaderboard(audit_payload)
    run_registry = _build_run_registry(index_payload)

    blocked = bool(brief_payload.get("blocked_by_preflight"))
    missing_backends = list(brief_payload.get("missing_backends", []) or [])
    findings = list(brief_payload.get("key_findings", []) or [])

    blocker_summary = (
        f"Live learned-vs-heuristic evaluation is blocked by missing backend configuration: {', '.join(missing_backends)}."
        if blocked and missing_backends
        else "Latest learned-vs-heuristic evaluation is runnable or already executed."
    )

    next_actions = []
    if blocked:
        next_actions.append("Add the missing backend configuration, then rerun the balanced head-to-head benchmark.")
    next_actions.append("Continue using balanced-domain sampling for small head-to-head comparisons.")
    next_actions.append("Report domain-macro quality alongside overall quality to offset benchmark imbalance.")
    next_actions.append("Expand cross-fold cache-first evaluation once stable JSONL search-cache artifacts are available.")

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "headline_metrics": {
            "benchmark_size": int(brief_payload.get("benchmark_size", 0) or 0),
            "benchmark_domain_count": int(brief_payload.get("benchmark_domain_count", 0) or 0),
            "benchmark_domain_imbalance_ratio": brief_payload.get("benchmark_domain_imbalance_ratio"),
            "recommended_policy_artifact": strongest.get("artifact") or brief_payload.get("recommended_policy_artifact"),
            "recommended_policy_eval_accuracy": strongest.get("eval_accuracy", brief_payload.get("recommended_policy_eval_accuracy")),
            "latest_head2head_label": brief_payload.get("head2head_label"),
            "latest_head2head_query_count": int(brief_payload.get("head2head_query_count", 0) or 0),
            "latest_quality_delta": float(brief_payload.get("quality_delta", 0.0) or 0.0),
            "latest_macro_quality_delta": float(brief_payload.get("macro_quality_delta", 0.0) or 0.0),
        },
        "blockers": {
            "blocked_by_preflight": blocked,
            "missing_backends": missing_backends,
            "summary": blocker_summary,
        },
        "benchmark_domain_table": domain_table,
        "policy_leaderboard": leaderboard,
        "run_registry": run_registry,
        "latest_run_domains": list(latest_run.get("domains", []) or brief_payload.get("head2head_domains", []) or []),
        "key_findings": findings,
        "next_actions": next_actions,
    }


def write_markdown(dashboard: dict[str, Any], output_path: Path) -> None:
    headline = dashboard.get("headline_metrics", {})
    blockers = dashboard.get("blockers", {})
    lines = [
        "# Research Dashboard",
        "",
        f"- Created at: {dashboard.get('created_at', '')}",
        f"- Benchmark: {headline.get('benchmark_size', 0)} queries / {headline.get('benchmark_domain_count', 0)} domains / imbalance {headline.get('benchmark_domain_imbalance_ratio', 'N/A')}",
        f"- Recommended policy: `{headline.get('recommended_policy_artifact', 'N/A')}` (eval_accuracy={headline.get('recommended_policy_eval_accuracy', 'N/A')})",
        f"- Latest head-to-head: `{headline.get('latest_head2head_label', 'N/A')}` with {headline.get('latest_head2head_query_count', 0)} queries",
        f"- Latest deltas: quality={headline.get('latest_quality_delta', 0.0):+.4f}, macro={headline.get('latest_macro_quality_delta', 0.0):+.4f}",
        "",
        "## Blockers",
        "",
        f"- {blockers.get('summary', 'No blockers recorded.')}",
        "",
        "## Benchmark Domain Distribution",
        "",
        "| domain | count | relative_load |",
        "|---|---:|---|",
    ]

    for row in dashboard.get("benchmark_domain_table", []):
        lines.append(f"| {row.get('domain', '')} | {row.get('count', 0)} | `{row.get('bar', '')}` |")

    lines.extend(["", "## Policy Artifact Leaderboard", "", "| artifact | eval_acc | train_acc | examples | browser_examples | imbalance |", "|---|---:|---:|---:|---:|---:|"])
    for row in dashboard.get("policy_leaderboard", []):
        lines.append(
            f"| {row.get('artifact', '')} | {row.get('eval_accuracy', 0.0):.4f} | {row.get('train_accuracy', 0.0):.4f} | "
            f"{row.get('num_examples', 0)} | {row.get('browser_examples', 0)} | {row.get('label_imbalance_ratio', 0.0):.3f} |"
        )

    lines.extend(["", "## Head-to-Head Run Registry", "", "| label | created_at | queries | sampling | blocked | quality_delta | macro_delta | missing_backends |", "|---|---|---:|---|---:|---:|---:|---|"])
    for row in dashboard.get("run_registry", []):
        lines.append(
            f"| {row.get('label', '')} | {row.get('created_at', '')} | {row.get('query_count', 0)} | {row.get('sampling_strategy', '')} | "
            f"{'yes' if row.get('blocked_by_preflight') else 'no'} | {row.get('quality_delta', 0.0):+.4f} | "
            f"{row.get('macro_quality_delta', 0.0):+.4f} | {','.join(row.get('missing_backends', [])) or '-'} |"
        )

    lines.extend(["", "## Key Findings", ""])
    for finding in dashboard.get("key_findings", []):
        lines.append(f"- {finding}")

    lines.extend(["", "## Next Actions", ""])
    for action in dashboard.get("next_actions", []):
        lines.append(f"- {action}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a research dashboard from indexed outputs")
    parser.add_argument("--brief_json", type=str, default="outputs/research_brief/research_brief.json")
    parser.add_argument("--index_json", type=str, default="outputs/research_index/research_output_index.json")
    parser.add_argument("--audit_json", type=str, default="outputs/research_audit/research_readiness_audit.json")
    parser.add_argument("--output_dir", type=str, default="outputs/research_dashboard")
    args = parser.parse_args()

    brief_payload = _load_json(PROJECT_ROOT / args.brief_json)
    index_payload = _load_json(PROJECT_ROOT / args.index_json)
    audit_payload = _load_json(PROJECT_ROOT / args.audit_json)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard = build_dashboard(brief_payload, index_payload, audit_payload)
    json_path = output_dir / "research_dashboard.json"
    md_path = output_dir / "research_dashboard.md"
    json_path.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(dashboard, md_path)

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
