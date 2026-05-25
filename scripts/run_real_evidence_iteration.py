#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_search_cache import validate_real_evidence_preflight  # noqa: E402
from scripts.resurface_search_cache_citations import analysis_stem_for_output  # noqa: E402
from src.core.runner import load_config  # noqa: E402


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _command_to_display(command: list[str]) -> str:
    return " ".join(command)


def _build_cache_command(
    *,
    args: argparse.Namespace,
    config_path: str,
    queries_file: str,
    cache_dir: Path,
    output_file: Path,
    session_prefix: str,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/build_search_cache.py",
        "--config",
        config_path,
        "--queries_file",
        queries_file,
        "--output_dir",
        str(cache_dir),
        "--output_file",
        str(output_file),
        "--session_prefix",
        session_prefix,
        "--include_evaluation",
        str(args.include_evaluation).lower(),
        "--search-policy-mode",
        args.search_policy_mode,
    ]
    if args.allow_mock_evidence:
        command.append("--allow_mock_evidence")
    return command


def _resurface_command(
    *,
    cache_file: Path,
    resurfaced_file: Path,
    resurfaced_dir: Path,
    analysis_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/resurface_search_cache_citations.py",
        "--inputs",
        str(cache_file),
        "--output-dir",
        str(resurfaced_dir),
        "--output-file",
        str(resurfaced_file),
        "--analysis-output-dir",
        str(analysis_dir),
    ]


def build_iteration_plan(args: argparse.Namespace, *, timestamp: str | None = None) -> dict[str, Any]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = str(args.config)
    queries_file = str(args.queries_file)
    heldout_queries_file = getattr(args, "heldout_queries_file", None)
    session_prefix = args.session_prefix or f"real_evidence_{timestamp}"
    has_heldout = bool(heldout_queries_file)

    cache_dir = Path(args.cache_dir)
    resurfaced_dir = Path(args.resurfaced_dir)
    analysis_dir = Path(args.analysis_dir)
    train_dir = Path(args.train_dir)
    audit_dir = Path(args.audit_dir)
    artifacts_dir = Path(args.artifacts_dir)

    train_suffix = "_train" if has_heldout else ""
    cache_file = cache_dir / f"{session_prefix}{train_suffix}.jsonl"
    resurfaced_file = resurfaced_dir / f"{session_prefix}{train_suffix}_resurfaced.jsonl"
    analysis_json = analysis_dir / f"{analysis_stem_for_output(resurfaced_file, output_file_requested=True, timestamp=timestamp)}.json"
    train_summary_file = train_dir / f"{session_prefix}_train_summary.json"
    model_file = artifacts_dir / f"search_policy_{session_prefix}.json"
    audit_json = audit_dir / f"{session_prefix}_paper_readiness.json"
    audit_md = audit_dir / f"{session_prefix}_paper_readiness.md"

    build_command = _build_cache_command(
        args=args,
        config_path=config_path,
        queries_file=queries_file,
        cache_dir=cache_dir,
        output_file=cache_file,
        session_prefix=f"{session_prefix}{train_suffix}",
    )

    resurface_command = _resurface_command(
        cache_file=cache_file,
        resurfaced_file=resurfaced_file,
        resurfaced_dir=resurfaced_dir,
        analysis_dir=analysis_dir,
    )

    analysis_jsons = [analysis_json]
    commands = {
        "build_cache": build_command,
        "resurface_citations": resurface_command,
    }
    paths: dict[str, Any] = {
        "cache_file": str(cache_file),
        "resurfaced_file": str(resurfaced_file),
        "analysis_json": str(analysis_json),
        "analysis_jsons": [str(path) for path in analysis_jsons],
        "analysis_dir": str(analysis_dir),
        "train_dir": str(train_dir),
        "train_summary_file": str(train_summary_file),
        "model_file": str(model_file),
        "audit_dir": str(audit_dir),
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
    }

    if has_heldout:
        heldout_session_prefix = f"{session_prefix}_heldout"
        heldout_cache_file = cache_dir / f"{heldout_session_prefix}.jsonl"
        heldout_resurfaced_file = resurfaced_dir / f"{heldout_session_prefix}_resurfaced.jsonl"
        heldout_analysis_json = analysis_dir / (
            f"{analysis_stem_for_output(heldout_resurfaced_file, output_file_requested=True, timestamp=timestamp)}.json"
        )
        analysis_jsons.append(heldout_analysis_json)
        paths.update(
            {
                "heldout_queries_file": str(heldout_queries_file),
                "heldout_cache_file": str(heldout_cache_file),
                "heldout_resurfaced_file": str(heldout_resurfaced_file),
                "heldout_analysis_json": str(heldout_analysis_json),
                "analysis_jsons": [str(path) for path in analysis_jsons],
            }
        )
        commands["build_heldout_cache"] = _build_cache_command(
            args=args,
            config_path=config_path,
            queries_file=str(heldout_queries_file),
            cache_dir=cache_dir,
            output_file=heldout_cache_file,
            session_prefix=heldout_session_prefix,
        )
        commands["resurface_heldout_citations"] = _resurface_command(
            cache_file=heldout_cache_file,
            resurfaced_file=heldout_resurfaced_file,
            resurfaced_dir=resurfaced_dir,
            analysis_dir=analysis_dir,
        )

    train_command = [
        sys.executable,
        "scripts/train_search_policy_iteration.py",
        "--inputs",
        str(resurfaced_file),
        "--output-model",
        str(model_file),
        "--output-dir",
        str(train_dir),
        "--output-summary",
        str(train_summary_file),
    ]

    audit_command = [
        sys.executable,
        "scripts/audit_paper_readiness.py",
        "--analysis-json",
        *[str(path) for path in analysis_jsons],
        "--train-json",
        str(train_summary_file),
        "--output-dir",
        str(audit_dir),
        "--output-json",
        str(audit_json),
        "--output-md",
        str(audit_md),
    ]
    commands["train_policy"] = train_command
    commands["audit_paper_readiness"] = audit_command

    return {
        "created_at": datetime.now().isoformat(),
        "timestamp": timestamp,
        "config": config_path,
        "queries_file": queries_file,
        "heldout_queries_file": str(heldout_queries_file) if heldout_queries_file else None,
        "session_prefix": session_prefix,
        "paths": paths,
        "commands": commands,
        "display_commands": {name: _command_to_display(command) for name, command in commands.items()},
    }


def _run_step(name: str, command: list[str]) -> None:
    print(f"[real-evidence] running {name}: {_command_to_display(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a gated real-evidence cache -> citation resurfacing -> training iteration.",
    )
    parser.add_argument("--config", type=str, default="configs/real_evidence_smoke.yaml")
    parser.add_argument("--queries-file", type=str, default="data/queries/real_evidence_seed_20260516.jsonl")
    parser.add_argument(
        "--heldout-queries-file",
        type=str,
        default=None,
        help="Optional heldout JSONL. When set, the main queries file is treated as the train split.",
    )
    parser.add_argument("--cache-dir", type=str, default="data/search_cache/real_evidence_iterations")
    parser.add_argument("--resurfaced-dir", type=str, default="data/search_cache/real_evidence_resurfaced")
    parser.add_argument("--analysis-dir", type=str, default="outputs/search_cache_analysis_real_evidence")
    parser.add_argument("--train-dir", type=str, default="outputs/search_policy_iteration_real_evidence")
    parser.add_argument("--audit-dir", type=str, default="outputs/paper_readiness_real_evidence")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--session-prefix", type=str, default=None)
    parser.add_argument("--search-policy-mode", choices=["off", "heuristic", "learned"], default="heuristic")
    parser.add_argument("--include-evaluation", type=_parse_bool, default=True)
    parser.add_argument("--allow-mock-evidence", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    preflight_failures = validate_real_evidence_preflight(
        config,
        allow_mock_evidence=args.allow_mock_evidence,
    )
    plan = build_iteration_plan(args)
    plan["preflight_failures"] = preflight_failures
    plan["dry_run"] = bool(args.dry_run)
    plan["skip_train"] = bool(args.skip_train)
    plan["skip_audit"] = bool(args.skip_audit)

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if preflight_failures:
        plan_path = Path(args.analysis_dir) / f"real_evidence_iteration_blocked_{plan['timestamp']}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(
            "real-evidence preflight failed: "
            + "; ".join(preflight_failures)
            + f". Plan written to {plan_path}"
        )

    for path_key in [
        "cache_file",
        "resurfaced_file",
        "heldout_cache_file",
        "heldout_resurfaced_file",
        "model_file",
        "train_summary_file",
    ]:
        if path_key in plan["paths"]:
            Path(plan["paths"][path_key]).parent.mkdir(parents=True, exist_ok=True)
    Path(plan["paths"]["analysis_dir"]).mkdir(parents=True, exist_ok=True)
    Path(plan["paths"]["train_dir"]).mkdir(parents=True, exist_ok=True)
    Path(plan["paths"]["audit_dir"]).mkdir(parents=True, exist_ok=True)

    _run_step("build_cache", plan["commands"]["build_cache"])
    if "build_heldout_cache" in plan["commands"]:
        _run_step("build_heldout_cache", plan["commands"]["build_heldout_cache"])
    _run_step("resurface_citations", plan["commands"]["resurface_citations"])
    if "resurface_heldout_citations" in plan["commands"]:
        _run_step("resurface_heldout_citations", plan["commands"]["resurface_heldout_citations"])
    if not args.skip_train:
        _run_step("train_policy", plan["commands"]["train_policy"])
        if not args.skip_audit:
            _run_step("audit_paper_readiness", plan["commands"]["audit_paper_readiness"])

    plan_path = Path(args.analysis_dir) / f"real_evidence_iteration_plan_{plan['timestamp']}.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[real-evidence] plan written: {plan_path}")


if __name__ == "__main__":
    main()
