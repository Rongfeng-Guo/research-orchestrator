#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refresh the full repository-level research reporting chain in one command.

This script rebuilds:
  1. research readiness audit
  2. research output index
  3. research brief
  4. research dashboard (JSON / Markdown / HTML)

Usage:
  python scripts/refresh_research_outputs.py
  python scripts/refresh_research_outputs.py --outputs_dir outputs
"""

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

from scripts.audit_research_readiness import analyze_repository_research_readiness
from scripts.audit_research_readiness import write_markdown as write_audit_markdown
from scripts.index_research_outputs import build_index
from scripts.index_research_outputs import write_markdown as write_index_markdown
from scripts.render_research_brief import build_brief
from scripts.render_research_brief import write_markdown as write_brief_markdown
from scripts.render_research_dashboard import build_dashboard, write_html, write_markdown as write_dashboard_markdown


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_research_outputs(
    project_root: Path,
    outputs_dir: Path,
    *,
    source_outputs_dir: Path | None = None,
) -> dict[str, Any]:
    audit_dir = outputs_dir / "research_audit"
    index_dir = outputs_dir / "research_index"
    brief_dir = outputs_dir / "research_brief"
    dashboard_dir = outputs_dir / "research_dashboard"
    source_outputs = source_outputs_dir or outputs_dir

    for directory in [audit_dir, index_dir, brief_dir, dashboard_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    audit_payload = analyze_repository_research_readiness(project_root)
    audit_json = audit_dir / "research_readiness_audit.json"
    audit_md = audit_dir / "research_readiness_audit.md"
    _write_json(audit_json, audit_payload)
    write_audit_markdown(audit_payload, audit_md)

    index_payload = build_index(source_outputs)
    index_json = index_dir / "research_output_index.json"
    index_md = index_dir / "research_output_index.md"
    _write_json(index_json, index_payload)
    write_index_markdown(index_payload, index_md)

    brief_payload = build_brief(index_payload)
    brief_json = brief_dir / "research_brief.json"
    brief_md = brief_dir / "research_brief.md"
    _write_json(brief_json, brief_payload)
    write_brief_markdown(brief_payload, brief_md)

    dashboard_payload = build_dashboard(brief_payload, index_payload, audit_payload)
    dashboard_json = dashboard_dir / "research_dashboard.json"
    dashboard_md = dashboard_dir / "research_dashboard.md"
    dashboard_html = dashboard_dir / "research_dashboard.html"
    _write_json(dashboard_json, dashboard_payload)
    write_dashboard_markdown(dashboard_payload, dashboard_md)
    write_html(dashboard_payload, dashboard_html)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "outputs_dir": str(outputs_dir),
        "source_outputs_dir": str(source_outputs),
        "generated": [
            {"artifact_type": "research_audit", "json": str(audit_json), "markdown": str(audit_md)},
            {"artifact_type": "research_index", "json": str(index_json), "markdown": str(index_md)},
            {"artifact_type": "research_brief", "json": str(brief_json), "markdown": str(brief_md)},
            {
                "artifact_type": "research_dashboard",
                "json": str(dashboard_json),
                "markdown": str(dashboard_md),
                "html": str(dashboard_html),
            },
        ],
        "headline": {
            "recommended_policy_artifact": brief_payload.get("recommended_policy_artifact"),
            "recommended_policy_eval_accuracy": brief_payload.get("recommended_policy_eval_accuracy"),
            "head2head_blocked_by_preflight": brief_payload.get("blocked_by_preflight"),
            "missing_backends": brief_payload.get("missing_backends", []),
        },
    }
    manifest_json = outputs_dir / "research_refresh_manifest.json"
    manifest["manifest_json"] = str(manifest_json)
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh repository-level research reporting artifacts")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="root outputs directory")
    parser.add_argument(
        "--source_outputs_dir",
        type=str,
        default=None,
        help="optional source outputs directory to index from when rendering into a separate destination",
    )
    args = parser.parse_args()

    outputs_dir = PROJECT_ROOT / args.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    source_outputs_dir = PROJECT_ROOT / args.source_outputs_dir if args.source_outputs_dir else None

    manifest = refresh_research_outputs(PROJECT_ROOT, outputs_dir, source_outputs_dir=source_outputs_dir)
    for item in manifest.get("generated", []):
        if item.get("json"):
            print(item["json"])
        if item.get("markdown"):
            print(item["markdown"])
        if item.get("html"):
            print(item["html"])
    print(manifest["manifest_json"])


if __name__ == "__main__":
    main()
