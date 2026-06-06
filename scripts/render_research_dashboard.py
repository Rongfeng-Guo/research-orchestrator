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
import html
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
        artifact_type = item.get("artifact_type")
        if artifact_type not in {"policy_head2head", "policy_head2head_cv"}:
            continue
        sampling_strategy = item.get("sampling_strategy")
        if not sampling_strategy:
            sampling_strategy = "cross_fold" if artifact_type == "policy_head2head_cv" else "front_slice"
        rows.append(
            {
                "artifact_type": artifact_type,
                "label": item.get("label"),
                "created_at": item.get("created_at"),
                "query_count": int(item.get("query_count", 0) or 0),
                "fold_count": int(item.get("fold_count", 0) or 0),
                "sampling_strategy": sampling_strategy,
                "source_label": item.get("source_label"),
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
    latest_run = next(
        (
            item
            for item in index_payload.get("artifacts", [])
            if item.get("artifact_type") in {"policy_head2head", "policy_head2head_cv"}
        ),
        {},
    )

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
            "latest_cross_fold_label": brief_payload.get("cross_fold_label"),
            "latest_cross_fold_query_count": int(brief_payload.get("cross_fold_query_count", 0) or 0),
            "latest_cross_fold_count": int(brief_payload.get("cross_fold_count", 0) or 0),
            "latest_cross_fold_quality_delta": float(brief_payload.get("cross_fold_quality_delta", 0.0) or 0.0),
            "latest_cross_fold_macro_quality_delta": float(
                brief_payload.get("cross_fold_macro_quality_delta", 0.0) or 0.0
            ),
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
        f"- Latest cross-fold: `{headline.get('latest_cross_fold_label', 'N/A')}` with {headline.get('latest_cross_fold_query_count', 0)} queries / {headline.get('latest_cross_fold_count', 0)} folds",
        f"- Cross-fold deltas: quality={headline.get('latest_cross_fold_quality_delta', 0.0):+.4f}, macro={headline.get('latest_cross_fold_macro_quality_delta', 0.0):+.4f}",
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

    lines.extend(
        [
            "",
            "## Head-to-Head Run Registry",
            "",
            "| type | label | created_at | queries | folds | sampling | blocked | quality_delta | macro_delta | missing_backends |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in dashboard.get("run_registry", []):
        lines.append(
            f"| {row.get('artifact_type', '')} | {row.get('label', '')} | {row.get('created_at', '')} | "
            f"{row.get('query_count', 0)} | {row.get('fold_count', 0)} | {row.get('sampling_strategy', '')} | "
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


def write_html(dashboard: dict[str, Any], output_path: Path) -> None:
    headline = dashboard.get("headline_metrics", {})
    blockers = dashboard.get("blockers", {})
    domain_rows = []
    for row in dashboard.get("benchmark_domain_table", []):
        bar_width = min(100, int((len(str(row.get("bar", ""))) / 24) * 100)) if row.get("bar") else 0
        domain_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('domain', '')))}</td>"
            f"<td>{int(row.get('count', 0) or 0)}</td>"
            f"<td><div class='bar-track'><div class='bar-fill' style='width:{bar_width}%;'></div></div></td>"
            "</tr>"
        )

    leaderboard_rows = []
    for row in dashboard.get("policy_leaderboard", []):
        leaderboard_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(row.get('artifact', '')))}</code></td>"
            f"<td>{float(row.get('eval_accuracy', 0.0) or 0.0):.4f}</td>"
            f"<td>{float(row.get('train_accuracy', 0.0) or 0.0):.4f}</td>"
            f"<td>{int(row.get('num_examples', 0) or 0)}</td>"
            f"<td>{int(row.get('browser_examples', 0) or 0)}</td>"
            f"<td>{float(row.get('label_imbalance_ratio', 0.0) or 0.0):.3f}</td>"
            "</tr>"
        )

    run_rows = []
    for row in dashboard.get("run_registry", []):
        run_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('artifact_type', '')))}</td>"
            f"<td>{html.escape(str(row.get('label', '')))}</td>"
            f"<td>{html.escape(str(row.get('created_at', '')))}</td>"
            f"<td>{int(row.get('query_count', 0) or 0)}</td>"
            f"<td>{int(row.get('fold_count', 0) or 0)}</td>"
            f"<td>{html.escape(str(row.get('sampling_strategy', '')))}</td>"
            f"<td>{'yes' if row.get('blocked_by_preflight') else 'no'}</td>"
            f"<td>{float(row.get('quality_delta', 0.0) or 0.0):+.4f}</td>"
            f"<td>{float(row.get('macro_quality_delta', 0.0) or 0.0):+.4f}</td>"
            f"<td>{html.escape(', '.join(row.get('missing_backends', []) or []) or '-')}</td>"
            "</tr>"
        )

    findings_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in dashboard.get("key_findings", [])
    )
    actions_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in dashboard.get("next_actions", [])
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffdf8;
      --ink: #1f2a30;
      --muted: #5f6b73;
      --accent: #1b6b7a;
      --accent-soft: #cde8ea;
      --warn: #8a4b08;
      --warn-bg: #fff0d9;
      --grid: #d7d2c8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #efe2c8 0, transparent 32%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      display: grid;
      gap: 16px;
      grid-template-columns: 2fr 1fr;
      margin-bottom: 24px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(31, 42, 48, 0.08);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 14px 40px rgba(31, 42, 48, 0.08);
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 32px; line-height: 1.1; }}
    h2 {{ font-size: 18px; }}
    p, li {{ line-height: 1.55; }}
    .lede {{ color: var(--muted); max-width: 60ch; }}
    .kpis {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      margin-top: 18px;
    }}
    .kpi {{
      border: 1px solid var(--grid);
      border-radius: 14px;
      padding: 14px;
      background: #fff;
    }}
    .kpi-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; margin-top: 8px; }}
    .kpi-sub {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
    .warning {{
      background: var(--warn-bg);
      color: var(--warn);
      border: 1px solid rgba(138, 75, 8, 0.16);
    }}
    .grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }}
    .span-5 {{ grid-column: span 5; }}
    .span-7 {{ grid-column: span 7; }}
    .span-12 {{ grid-column: span 12; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--grid);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{
      font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
    .bar-track {{
      width: 100%;
      min-width: 120px;
      height: 10px;
      border-radius: 999px;
      background: #e6ecec;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #2f8f9d 0%, var(--accent) 100%);
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      margin: 4px 6px 0 0;
      font-size: 13px;
      font-weight: 600;
    }}
    @media (max-width: 860px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
      .span-5, .span-7, .span-12 {{ grid-column: span 1; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel">
        <h1>Research Dashboard</h1>
        <p class="lede">A single-page view of research readiness, policy artifact quality, and learned-vs-heuristic experiment status for Research Orchestrator.</p>
        <div class="kpis">
          <div class="kpi">
            <div class="kpi-label">Benchmark</div>
            <div class="kpi-value">{int(headline.get('benchmark_size', 0) or 0)}</div>
            <div class="kpi-sub">{int(headline.get('benchmark_domain_count', 0) or 0)} domains</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Imbalance</div>
            <div class="kpi-value">{html.escape(str(headline.get('benchmark_domain_imbalance_ratio', 'N/A')))}</div>
            <div class="kpi-sub">largest/smallest domain ratio</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Best Artifact</div>
            <div class="kpi-value">{float(headline.get('recommended_policy_eval_accuracy', 0.0) or 0.0):.4f}</div>
            <div class="kpi-sub"><code>{html.escape(str(headline.get('recommended_policy_artifact', 'N/A')))}</code></div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Latest Sample</div>
            <div class="kpi-value">{int(headline.get('latest_head2head_query_count', 0) or 0)}</div>
            <div class="kpi-sub"><code>{html.escape(str(headline.get('latest_head2head_label', 'N/A')))}</code></div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Cross-Fold</div>
            <div class="kpi-value">{int(headline.get('latest_cross_fold_count', 0) or 0)}</div>
            <div class="kpi-sub">{int(headline.get('latest_cross_fold_query_count', 0) or 0)} queries, delta {float(headline.get('latest_cross_fold_quality_delta', 0.0) or 0.0):+.4f}</div>
          </div>
        </div>
      </div>
      <div class="panel warning">
        <h2>Current Blocker</h2>
        <p>{html.escape(str(blockers.get('summary', 'No blockers recorded.')))}</p>
        <div>
          {''.join(f"<span class='pill'>{html.escape(str(item))}</span>" for item in dashboard.get("latest_run_domains", []))}
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="panel span-5">
        <h2>Benchmark Domain Distribution</h2>
        <table>
          <thead><tr><th>Domain</th><th>Count</th><th>Relative Load</th></tr></thead>
          <tbody>{''.join(domain_rows)}</tbody>
        </table>
      </div>
      <div class="panel span-7">
        <h2>Policy Artifact Leaderboard</h2>
        <table>
          <thead><tr><th>Artifact</th><th>Eval</th><th>Train</th><th>Examples</th><th>Browser</th><th>Imbalance</th></tr></thead>
          <tbody>{''.join(leaderboard_rows)}</tbody>
        </table>
      </div>
      <div class="panel span-12">
        <h2>Head-to-Head Run Registry</h2>
        <table>
          <thead><tr><th>Type</th><th>Label</th><th>Created</th><th>Queries</th><th>Folds</th><th>Sampling</th><th>Blocked</th><th>Quality</th><th>Macro</th><th>Missing Backends</th></tr></thead>
          <tbody>{''.join(run_rows)}</tbody>
        </table>
      </div>
      <div class="panel span-7">
        <h2>Key Findings</h2>
        <ul>{findings_html}</ul>
      </div>
      <div class="panel span-5">
        <h2>Next Actions</h2>
        <ul>{actions_html}</ul>
      </div>
    </section>
  </div>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


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
    html_path = output_dir / "research_dashboard.html"
    json_path.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(dashboard, md_path)
    write_html(dashboard, html_path)

    print(str(json_path))
    print(str(md_path))
    print(str(html_path))


if __name__ == "__main__":
    main()
