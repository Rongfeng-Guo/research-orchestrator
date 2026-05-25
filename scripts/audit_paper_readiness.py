#!/usr/bin/env python3
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


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _check(name: str, passed: bool, observed: Any, threshold: str, detail: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def evaluate_analysis_readiness(
    summary: dict[str, Any],
    *,
    min_success_queries: int = 8,
    min_browser_query_coverage: float = 0.25,
    min_nonzero_composite_ratio: float = 0.50,
    min_avg_citation_coverage: float = 0.05,
    min_avg_citation_density: float = 0.20,
    min_avg_citation_quality_score: float = 0.20,
    min_real_source_query_ratio: float = 0.80,
    max_mock_source_query_ratio: float = 0.0,
    min_avg_real_sources_per_query: float = 2.0,
    min_relevant_source_query_ratio: float = 0.80,
    min_avg_relevant_sources_per_query: float = 2.0,
    min_avg_source_relevance: float = 0.15,
) -> list[dict[str, Any]]:
    """Evaluate whether one cache-analysis summary is paper-ready."""
    checks = [
        _check(
            "success_query_count",
            _safe_int(summary.get("num_success")) >= min_success_queries,
            _safe_int(summary.get("num_success")),
            f">= {min_success_queries}",
        ),
        _check(
            "browser_query_coverage",
            _safe_float(summary.get("browser_query_coverage")) >= min_browser_query_coverage,
            round(_safe_float(summary.get("browser_query_coverage")), 4),
            f">= {min_browser_query_coverage:.4f}",
        ),
        _check(
            "nonzero_composite_query_ratio",
            _safe_float(summary.get("nonzero_composite_query_ratio")) >= min_nonzero_composite_ratio,
            round(_safe_float(summary.get("nonzero_composite_query_ratio")), 4),
            f">= {min_nonzero_composite_ratio:.4f}",
        ),
        _check(
            "avg_citation_coverage",
            _safe_float(summary.get("avg_citation_coverage")) >= min_avg_citation_coverage,
            round(_safe_float(summary.get("avg_citation_coverage")), 4),
            f">= {min_avg_citation_coverage:.4f}",
        ),
        _check(
            "avg_citation_density",
            _safe_float(summary.get("avg_citation_density")) >= min_avg_citation_density,
            round(_safe_float(summary.get("avg_citation_density")), 4),
            f">= {min_avg_citation_density:.4f}",
        ),
        _check(
            "avg_citation_quality_score",
            _safe_float(summary.get("avg_citation_quality_score")) >= min_avg_citation_quality_score,
            round(_safe_float(summary.get("avg_citation_quality_score")), 4),
            f">= {min_avg_citation_quality_score:.4f}",
        ),
        _check(
            "real_source_query_ratio",
            _safe_float(summary.get("real_source_query_ratio")) >= min_real_source_query_ratio,
            round(_safe_float(summary.get("real_source_query_ratio")), 4),
            f">= {min_real_source_query_ratio:.4f}",
        ),
        _check(
            "mock_source_query_ratio",
            _safe_float(summary.get("mock_source_query_ratio")) <= max_mock_source_query_ratio,
            round(_safe_float(summary.get("mock_source_query_ratio")), 4),
            f"<= {max_mock_source_query_ratio:.4f}",
        ),
        _check(
            "avg_real_sources_per_query",
            _safe_float(summary.get("avg_real_sources_per_success_query")) >= min_avg_real_sources_per_query,
            round(_safe_float(summary.get("avg_real_sources_per_success_query")), 4),
            f">= {min_avg_real_sources_per_query:.4f}",
        ),
        _check(
            "relevant_source_query_ratio",
            _safe_float(summary.get("relevant_source_query_ratio")) >= min_relevant_source_query_ratio,
            round(_safe_float(summary.get("relevant_source_query_ratio")), 4),
            f">= {min_relevant_source_query_ratio:.4f}",
        ),
        _check(
            "avg_relevant_sources_per_query",
            _safe_float(summary.get("avg_relevant_sources_per_success_query")) >= min_avg_relevant_sources_per_query,
            round(_safe_float(summary.get("avg_relevant_sources_per_success_query")), 4),
            f">= {min_avg_relevant_sources_per_query:.4f}",
        ),
        _check(
            "avg_source_relevance",
            _safe_float(summary.get("avg_source_relevance")) >= min_avg_source_relevance,
            round(_safe_float(summary.get("avg_source_relevance")), 4),
            f">= {min_avg_source_relevance:.4f}",
        ),
    ]
    return checks


def evaluate_training_readiness(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not summary:
        return [
            _check(
                "training_summary_present",
                False,
                "missing",
                "required for learned-policy claim",
            )
        ]

    training_allowed = bool(summary.get("training_allowed", False))
    headline_ready = bool(summary.get("headline_claim_ready", False))
    status = str(summary.get("status", "") or "")
    return [
        _check(
            "training_allowed",
            training_allowed,
            training_allowed,
            "true",
            "; ".join(summary.get("training_readiness_failures", []) or []),
        ),
        _check(
            "headline_claim_ready",
            headline_ready,
            headline_ready,
            "true",
            "; ".join(summary.get("headline_readiness_failures", []) or []),
        ),
        _check(
            "training_status",
            status == "trained",
            status or "missing",
            "trained",
        ),
    ]


def recommend_next_actions(checks: list[dict[str, Any]]) -> list[str]:
    failed_names = {str(item.get("name", "")) for item in checks if not item.get("passed")}
    actions: list[str] = []
    if "mock_source_query_ratio" in failed_names or "real_source_query_ratio" in failed_names:
        actions.append("Run a real-evidence cache with a real search backend; mock evidence cannot support headline claims.")
    if "avg_real_sources_per_query" in failed_names:
        actions.append("Increase source inventory: preserve multiple search results and browse at least two independent real sources per query.")
    if (
        "relevant_source_query_ratio" in failed_names
        or "avg_relevant_sources_per_query" in failed_names
        or "avg_source_relevance" in failed_names
    ):
        actions.append("Improve search relevance: real URLs must match the query, not just be non-mock sources.")
    if "avg_citation_density" in failed_names or "avg_citation_coverage" in failed_names:
        actions.append("Improve final-report citation rendering so factual paragraphs receive inline citations.")
    if "success_query_count" in failed_names:
        actions.append("Collect a larger held-out split; fewer than the threshold queries is only a smoke test.")
    if "training_allowed" in failed_names or "training_status" in failed_names:
        actions.append("Do not compare learned vs heuristic policy until training gates pass on real evidence.")
    if not actions:
        actions.append("Proceed to held-out head-to-head and ablation; readiness gates passed.")
    return actions


def audit_readiness(
    analysis_summaries: list[dict[str, Any]],
    *,
    training_summary: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {}
    analysis_results: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    for idx, summary in enumerate(analysis_summaries, 1):
        checks = evaluate_analysis_readiness(summary, **thresholds)
        all_checks.extend(checks)
        analysis_results.append(
            {
                "index": idx,
                "input_paths": summary.get("input_paths", []),
                "num_success": summary.get("num_success", 0),
                "risk_flags": summary.get("risk_flags", []),
                "checks": checks,
                "status": "PASS" if all(item["passed"] for item in checks) else "BLOCKED",
            }
        )

    training_checks = evaluate_training_readiness(training_summary)
    all_checks.extend(training_checks)
    status = "PASS" if all(item["passed"] for item in all_checks) else "BLOCKED"
    return {
        "created_at": datetime.now().isoformat(),
        "status": status,
        "analysis_results": analysis_results,
        "training_checks": training_checks,
        "failed_checks": [item for item in all_checks if not item.get("passed")],
        "next_actions": recommend_next_actions(all_checks),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Paper Readiness Audit",
        "",
        f"- created_at: {audit.get('created_at', '')}",
        f"- status: {audit.get('status', 'BLOCKED')}",
        "",
        "## Analysis Checks",
        "",
    ]
    for result in audit.get("analysis_results", []):
        lines.extend(
            [
                f"### Analysis {result.get('index', '')}: {result.get('status', 'BLOCKED')}",
                "",
                f"- inputs: {', '.join(str(item) for item in result.get('input_paths', []))}",
                f"- num_success: {result.get('num_success', 0)}",
                f"- risk_flags: {', '.join(result.get('risk_flags', [])) or 'none'}",
                "",
                "| check | status | observed | threshold |",
                "|---|---|---:|---|",
            ]
        )
        for check in result.get("checks", []):
            status = "PASS" if check.get("passed") else "FAIL"
            lines.append(
                f"| {check.get('name', '')} | {status} | {check.get('observed', '')} | {check.get('threshold', '')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Training Checks",
            "",
            "| check | status | observed | threshold | detail |",
            "|---|---|---:|---|---|",
        ]
    )
    for check in audit.get("training_checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        lines.append(
            f"| {check.get('name', '')} | {status} | {check.get('observed', '')} | "
            f"{check.get('threshold', '')} | {check.get('detail', '')} |"
        )

    lines.extend(["", "## Next Actions", ""])
    for action in audit.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether experiment outputs are paper-ready.")
    parser.add_argument("--analysis-json", nargs="+", required=True, help="search-cache analysis JSON file(s)")
    parser.add_argument("--train-json", type=str, default=None, help="optional training iteration JSON")
    parser.add_argument("--output-dir", type=str, default="outputs/paper_readiness")
    parser.add_argument("--output-json", type=str, default=None, help="optional stable audit JSON path")
    parser.add_argument("--output-md", type=str, default=None, help="optional stable audit Markdown path")
    parser.add_argument("--min-success-queries", type=int, default=8)
    parser.add_argument("--min-browser-query-coverage", type=float, default=0.25)
    parser.add_argument("--min-nonzero-composite-ratio", type=float, default=0.50)
    parser.add_argument("--min-avg-citation-coverage", type=float, default=0.05)
    parser.add_argument("--min-avg-citation-density", type=float, default=0.20)
    parser.add_argument("--min-avg-citation-quality-score", type=float, default=0.20)
    parser.add_argument("--min-real-source-query-ratio", type=float, default=0.80)
    parser.add_argument("--max-mock-source-query-ratio", type=float, default=0.0)
    parser.add_argument("--min-avg-real-sources-per-query", type=float, default=2.0)
    parser.add_argument("--min-relevant-source-query-ratio", type=float, default=0.80)
    parser.add_argument("--min-avg-relevant-sources-per-query", type=float, default=2.0)
    parser.add_argument("--min-avg-source-relevance", type=float, default=0.15)
    args = parser.parse_args()

    thresholds = {
        "min_success_queries": args.min_success_queries,
        "min_browser_query_coverage": args.min_browser_query_coverage,
        "min_nonzero_composite_ratio": args.min_nonzero_composite_ratio,
        "min_avg_citation_coverage": args.min_avg_citation_coverage,
        "min_avg_citation_density": args.min_avg_citation_density,
        "min_avg_citation_quality_score": args.min_avg_citation_quality_score,
        "min_real_source_query_ratio": args.min_real_source_query_ratio,
        "max_mock_source_query_ratio": args.max_mock_source_query_ratio,
        "min_avg_real_sources_per_query": args.min_avg_real_sources_per_query,
        "min_relevant_source_query_ratio": args.min_relevant_source_query_ratio,
        "min_avg_relevant_sources_per_query": args.min_avg_relevant_sources_per_query,
        "min_avg_source_relevance": args.min_avg_source_relevance,
    }
    analyses = [_load_json(path) for path in args.analysis_json]
    training_summary = _load_json(args.train_json) if args.train_json else None
    audit = audit_readiness(analyses, training_summary=training_summary, thresholds=thresholds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(args.output_json) if args.output_json else output_dir / f"paper_readiness_audit_{timestamp}.json"
    md_path = Path(args.output_md) if args.output_md else output_dir / f"paper_readiness_audit_{timestamp}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(audit), encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(str(md_path))
    if audit.get("status") != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
