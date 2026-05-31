#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit repository-level research readiness signals.

The audit focuses on:
  - benchmark size and domain balance
  - search-policy artifact sample size and label balance
  - stale documentation claims about benchmark scale
  - suspicious one-record manifests under data/search_cache that look like test residue

Usage:
  python scripts/audit_research_readiness.py
  python scripts/audit_research_readiness.py --output_dir outputs/research_audit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_summary(project_root: Path) -> dict[str, Any]:
    from evaluation.benchmarks.research_bench import ResearchBench

    questions = ResearchBench.DEFAULT_QUESTIONS
    domain_counts = Counter(str(item.get("domain", "")) for item in questions)
    top_domain = max(domain_counts.values()) if domain_counts else 0
    bottom_domain = min(domain_counts.values()) if domain_counts else 0

    return {
        "num_questions": len(questions),
        "num_domains": len(domain_counts),
        "domain_distribution": dict(domain_counts),
        "largest_domain_size": top_domain,
        "smallest_domain_size": bottom_domain,
        "domain_imbalance_ratio": round(top_domain / bottom_domain, 3) if bottom_domain else None,
    }


def policy_artifact_summary(project_root: Path) -> dict[str, Any]:
    artifacts_dir = project_root / "artifacts"
    summaries: list[dict[str, Any]] = []

    for path in sorted(artifacts_dir.glob("search_policy*.json")):
        data = _load_json(path)
        stats = data.get("training_stats", {}) or {}
        label_distribution = stats.get("label_distribution", {}) or {}
        labels = [int(v) for v in label_distribution.values() if isinstance(v, (int, float))]
        min_label = min(labels) if labels else 0
        max_label = max(labels) if labels else 0
        summaries.append(
            {
                "artifact": path.name,
                "num_examples": stats.get("num_examples", 0),
                "eval_examples": stats.get("eval_examples", 0),
                "train_accuracy": stats.get("train_accuracy"),
                "eval_accuracy": stats.get("eval_accuracy"),
                "label_distribution": label_distribution,
                "label_imbalance_ratio": round(max_label / min_label, 3) if min_label else None,
                "browser_examples": int(label_distribution.get("browser", 0) or 0),
            }
        )

    strongest_eval = max(
        summaries,
        key=lambda item: (
            float(item.get("eval_accuracy") or 0.0),
            int(item.get("num_examples") or 0),
        ),
        default=None,
    )
    default_artifact = next((item for item in summaries if item["artifact"] == "search_policy.json"), None)

    return {
        "artifacts": summaries,
        "strongest_eval_artifact": strongest_eval,
        "default_artifact": default_artifact,
    }


def stale_claims_summary(project_root: Path) -> dict[str, Any]:
    targets = [
        project_root / "evaluation" / "benchmarks" / "research_bench.py",
        project_root / "docs" / "06-evaluation-and-experiments.md",
    ]
    findings: list[dict[str, Any]] = []

    for path in targets:
        text = path.read_text(encoding="utf-8")
        stale_markers = []
        if "20 道" in text:
            stale_markers.append("contains_20_questions_claim")
        if "35 道" in text:
            stale_markers.append("contains_35_questions_claim")
        findings.append({"path": str(path.relative_to(project_root)), "markers": stale_markers})

    return {"files": findings}


def search_cache_residue_summary(project_root: Path) -> dict[str, Any]:
    manifests_dir = project_root / "data" / "search_cache" / "citation_resurfaced"
    manifests: list[dict[str, Any]] = []

    for path in sorted(manifests_dir.glob("*.json")):
        data = _load_json(path)
        analysis_summary = data.get("analysis_summary", {}) or {}
        input_paths = data.get("input_paths", []) or []
        manifests.append(
            {
                "file": path.name,
                "num_records": data.get("num_records", 0),
                "num_changed": data.get("num_changed", 0),
                "risk_flags": analysis_summary.get("risk_flags", []) or [],
                "looks_like_test_output": any("pytest-" in str(item) for item in input_paths),
            }
        )

    suspicious = [
        item
        for item in manifests
        if item["num_records"] <= 1 and item["looks_like_test_output"]
    ]

    return {
        "num_manifests": len(manifests),
        "suspicious_test_like_manifests": suspicious,
    }


def build_findings(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []

    bench = summary["benchmark"]
    if bench["num_questions"] != 20:
        findings.append(
            f"ResearchBench 当前实际包含 {bench['num_questions']} 题，不是旧注释中的 20 题。"
        )
    if (bench.get("domain_imbalance_ratio") or 0) >= 4:
        findings.append(
            f"Benchmark 领域分布不均衡，最大/最小领域题量比约为 {bench['domain_imbalance_ratio']}。"
        )

    policy = summary["policy_artifacts"]
    default_artifact = policy.get("default_artifact") or {}
    strongest = policy.get("strongest_eval_artifact") or {}
    if int(default_artifact.get("num_examples") or 0) < 50:
        findings.append(
            f"默认 policy artifact `search_policy.json` 仅基于 {default_artifact.get('num_examples', 0)} 条样本训练。"
        )
    if int(default_artifact.get("browser_examples") or 0) <= 1:
        findings.append("默认 policy artifact 的 `browser` 标签样本极少，几乎无法支持稳定浏览决策。")
    if strongest and strongest.get("artifact") != "search_policy.json":
        findings.append(
            f"现有 artifact 中验证集表现最好的是 `{strongest['artifact']}`，优于默认产物。"
        )

    stale = summary["stale_claims"]["files"]
    if any("contains_20_questions_claim" in item["markers"] for item in stale):
        findings.append("仓库内仍存在与 benchmark 规模不一致的旧文案，影响论文与实验叙述一致性。")

    residue = summary["search_cache_residue"]
    if residue["suspicious_test_like_manifests"]:
        findings.append(
            f"`data/search_cache/citation_resurfaced` 下发现 {len(residue['suspicious_test_like_manifests'])} 个疑似测试残留 manifest。"
        )

    return findings


def analyze_repository_research_readiness(project_root: Path) -> dict[str, Any]:
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": benchmark_summary(project_root),
        "policy_artifacts": policy_artifact_summary(project_root),
        "stale_claims": stale_claims_summary(project_root),
        "search_cache_residue": search_cache_residue_summary(project_root),
    }
    summary["key_findings"] = build_findings(summary)
    return summary


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    bench = summary["benchmark"]
    strongest = summary["policy_artifacts"].get("strongest_eval_artifact") or {}
    default_artifact = summary["policy_artifacts"].get("default_artifact") or {}
    suspicious = summary["search_cache_residue"]["suspicious_test_like_manifests"]

    lines = [
        "# Research Readiness Audit",
        "",
        f"- Created at: {summary['created_at']}",
        f"- ResearchBench size: {bench['num_questions']}",
        f"- Domain count: {bench['num_domains']}",
        f"- Strongest policy artifact: {strongest.get('artifact', 'N/A')}",
        "",
        "## Key Findings",
        "",
    ]
    for item in summary["key_findings"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Benchmark",
            "",
            f"- Domain distribution: `{bench['domain_distribution']}`",
            f"- Domain imbalance ratio: `{bench['domain_imbalance_ratio']}`",
            "",
            "## Policy Artifacts",
            "",
            f"- Default artifact: `{default_artifact.get('artifact', 'N/A')}`",
            f"- Default num_examples: `{default_artifact.get('num_examples', 0)}`",
            f"- Default browser examples: `{default_artifact.get('browser_examples', 0)}`",
            f"- Strongest eval artifact: `{strongest.get('artifact', 'N/A')}`",
            f"- Strongest eval accuracy: `{strongest.get('eval_accuracy', 'N/A')}`",
            "",
            "## Search Cache Residue",
            "",
            f"- Suspicious test-like manifests: `{len(suspicious)}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit repository-level research readiness signals")
    parser.add_argument("--output_dir", type=str, default="outputs/research_audit", help="Directory for JSON/Markdown audit outputs")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze_repository_research_readiness(PROJECT_ROOT)
    json_path = output_dir / "research_readiness_audit.json"
    md_path = output_dir / "research_readiness_audit.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
