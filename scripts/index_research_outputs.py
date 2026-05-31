#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a lightweight index for research-facing output artifacts under outputs/.

Current artifact families:
  - research_audit
  - policy_head2head
  - policy_head2head cross-fold summaries (when present)

Usage:
  python scripts/index_research_outputs.py
  python scripts/index_research_outputs.py --outputs_dir outputs --output_dir outputs/research_index
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp_from_name(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})", path.name)
    return match.group(1) if match else ""


def _find_latest(pattern: str, directory: Path) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _index_research_audit(outputs_dir: Path) -> list[dict[str, Any]]:
    audit_dir = outputs_dir / "research_audit"
    json_path = audit_dir / "research_readiness_audit.json"
    md_path = audit_dir / "research_readiness_audit.md"
    if not json_path.exists():
        return []

    payload = _load_json(json_path)
    return [
        {
            "artifact_type": "research_audit",
            "label": "Research Readiness Audit",
            "created_at": payload.get("created_at", ""),
            "path_json": str(json_path),
            "path_md": str(md_path) if md_path.exists() else None,
            "highlights": payload.get("key_findings", []),
        }
    ]


def _index_head2head(outputs_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for directory in sorted(outputs_dir.glob("policy_head2head*")):
        if not directory.is_dir():
            continue
        latest_json = _find_latest("head2head_*.json", directory)
        latest_md = _find_latest("head2head_*.md", directory)
        if not latest_json:
            continue

        payload = _load_json(latest_json)
        runs = payload.get("runs", []) if isinstance(payload.get("runs"), list) else []
        by_mode = {str(run.get("mode", "")): run for run in runs}
        heuristic = by_mode.get("heuristic", {})
        learned = by_mode.get("learned", {})
        heuristic_summary = heuristic.get("summary", {}) if isinstance(heuristic.get("summary"), dict) else {}
        learned_summary = learned.get("summary", {}) if isinstance(learned.get("summary"), dict) else {}
        learned_preflight = learned.get("preflight", {}) if isinstance(learned.get("preflight"), dict) else {}
        heuristic_preflight = heuristic.get("preflight", {}) if isinstance(heuristic.get("preflight"), dict) else {}

        entries.append(
            {
                "artifact_type": "policy_head2head",
                "label": directory.name,
                "created_at": payload.get("created_at", "") or _timestamp_from_name(latest_json),
                "path_json": str(latest_json),
                "path_md": str(latest_md) if latest_md else None,
                "config_path": payload.get("config_path"),
                "query_count": payload.get("query_count", 0),
                "sampling_strategy": payload.get("sampling_strategy"),
                "heuristic_success": heuristic_summary.get("num_success", 0),
                "learned_success": learned_summary.get("num_success", 0),
                "quality_delta": round(
                    float(learned_summary.get("avg_composite_score", 0.0) or 0.0)
                    - float(heuristic_summary.get("avg_composite_score", 0.0) or 0.0),
                    4,
                ),
                "macro_quality_delta": round(
                    float(learned_summary.get("domain_macro_avg_composite_score", 0.0) or 0.0)
                    - float(heuristic_summary.get("domain_macro_avg_composite_score", 0.0) or 0.0),
                    4,
                ),
                "blocked_by_preflight": (not learned_preflight.get("is_ready", True)) or (not heuristic_preflight.get("is_ready", True)),
                "missing_backends": sorted(
                    {
                        *list(learned_preflight.get("missing_backends", []) or []),
                        *list(heuristic_preflight.get("missing_backends", []) or []),
                    }
                ),
                "domains": sorted(
                    {
                        str(item.get("domain", "") or "")
                        for item in payload.get("queries", [])
                        if isinstance(item, dict)
                    }
                ),
            }
        )
    return entries


def build_index(outputs_dir: Path) -> dict[str, Any]:
    artifacts = []
    artifacts.extend(_index_research_audit(outputs_dir))
    artifacts.extend(_index_head2head(outputs_dir))
    artifacts.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outputs_dir": str(outputs_dir),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_markdown(index_payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Research Output Index",
        "",
        f"- Created at: {index_payload.get('created_at', '')}",
        f"- Artifact count: {index_payload.get('artifact_count', 0)}",
        "",
        "## Artifacts",
        "",
        "| type | label | created_at | query_count | blocked | highlights |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in index_payload.get("artifacts", []):
        highlights = []
        if item.get("artifact_type") == "research_audit":
            findings = item.get("highlights", []) or []
            highlights.append(f"{len(findings)} findings")
        if item.get("artifact_type") == "policy_head2head":
            highlights.append(f"quality_delta={item.get('quality_delta', 0.0):+.4f}")
            highlights.append(f"macro_delta={item.get('macro_quality_delta', 0.0):+.4f}")
            if item.get("missing_backends"):
                highlights.append("missing:" + ",".join(item["missing_backends"]))
        lines.append(
            f"| {item.get('artifact_type', '')} | {item.get('label', '')} | {item.get('created_at', '')} | "
            f"{int(item.get('query_count', 0) or 0)} | {'yes' if item.get('blocked_by_preflight') else 'no'} | "
            f"{'; '.join(highlights)} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index research-facing output artifacts")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="root outputs directory")
    parser.add_argument("--output_dir", type=str, default="outputs/research_index", help="directory for the generated index")
    args = parser.parse_args()

    outputs_dir = PROJECT_ROOT / args.outputs_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_index(outputs_dir)
    json_path = output_dir / "research_output_index.json"
    md_path = output_dir / "research_output_index.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
