#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_search_cache.py
================================================================================
Build an offline search-cache dataset from Research Orchestrator runs.

The output is JSONL. Each line contains:
  - the original query and optional benchmark metadata
  - the final report and sources
  - structured policy trace / search cost / route stats
  - serialized task-level results for replay and analysis
  - optional benchmark evaluation

Usage:
  python scripts/build_search_cache.py --config configs/aliyun_smoke.yaml --num_questions 5
  python scripts/build_search_cache.py --query "Transformer architecture" --output_dir data/search_cache/dev
  python scripts/build_search_cache.py --queries_file data/my_queries.jsonl --include_evaluation false
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.metrics.search_policy_reward import SearchPolicyReward
from src.core.runner import initialize_modules, load_config, run_research, setup_logging
from src.evolution.collector import TrajectoryCollector


logger = logging.getLogger("build_search_cache")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def _load_queries_from_file(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"queries_file 未找到: {path}")

    if path.lower().endswith(".jsonl"):
        queries: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                queries.append(_normalize_query_item(item, line_no))
        return queries

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("queries_file 必须是 JSON list、JSONL，或包含 queries 字段的 JSON object")

    return [_normalize_query_item(item, idx + 1) for idx, item in enumerate(data)]


def _normalize_query_item(item: Any, fallback_index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "id": f"user_{fallback_index:04d}",
            "query": item,
        }
    if not isinstance(item, dict):
        raise ValueError(f"非法 query item: {item!r}")
    query = str(item.get("query", "") or "").strip()
    if not query:
        raise ValueError(f"query item 缺少 query 字段: {item!r}")
    normalized = dict(item)
    normalized.setdefault("id", f"user_{fallback_index:04d}")
    normalized["query"] = query
    return normalized


def apply_policy_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply optional runtime policy overrides from CLI arguments."""
    if args.evidence_policy_mode:
        evidence_cfg = dict(config.get("evidence_policy", {}) or {})
        evidence_cfg["mode"] = args.evidence_policy_mode
        if args.evidence_policy_model_path:
            evidence_cfg["model_path"] = args.evidence_policy_model_path
        config["evidence_policy"] = evidence_cfg

    if args.search_policy_mode:
        search_cfg = dict(config.get("search_policy", {}) or {})
        mode = str(args.search_policy_mode).lower().strip()
        search_cfg["mode"] = mode
        if mode == "off":
            search_cfg["enabled"] = False
        elif mode == "heuristic":
            search_cfg["enabled"] = True
            search_cfg["heuristic_fallback"] = True
        elif mode == "learned":
            search_cfg["enabled"] = True
            search_cfg["heuristic_fallback"] = False
        if args.search_policy_model_path:
            search_cfg["model_path"] = args.search_policy_model_path
        config["search_policy"] = search_cfg

    return config


def validate_real_evidence_preflight(
    config: dict[str, Any],
    *,
    allow_mock_evidence: bool = False,
) -> list[str]:
    """Return blocking reasons when a cache run would silently use mock evidence."""
    if allow_mock_evidence:
        return []

    failures: list[str] = []
    tools_cfg = config.get("tools", {}) if isinstance(config.get("tools", {}), dict) else {}
    search_cfg = tools_cfg.get("web_search", {}) if isinstance(tools_cfg.get("web_search", {}), dict) else {}
    if bool(search_cfg.get("mock_mode", True)):
        failures.append("tools.web_search.mock_mode=true")

    search_backend = os.getenv("SEARCH_BACKEND", "serpapi").lower().strip()
    required_key_by_backend = {
        "serpapi": "SERPAPI_KEY",
        "bing": "BING_SEARCH_KEY",
        "bocha": "BOCHA_API_KEY",
        "metaso": "METASO_API_KEY",
    }
    required_key = required_key_by_backend.get(search_backend)
    if required_key and not os.getenv(required_key):
        failures.append(f"missing_{required_key}_for_SEARCH_BACKEND={search_backend}")

    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    model_backends: set[str] = set()
    default_backend = str(model_cfg.get("backend", "") or "").strip().lower()
    if default_backend and default_backend != "auto":
        model_backends.add(default_backend)
    backend_mapping = model_cfg.get("backend_mapping", {})
    if isinstance(backend_mapping, dict):
        for backend in backend_mapping.values():
            backend_name = str(backend or "").strip().lower()
            if backend_name and backend_name != "auto":
                model_backends.add(backend_name)

    for backend in sorted(model_backends):
        prefix = backend.upper()
        if not os.getenv(f"{prefix}_API_KEY") and not os.getenv(f"{prefix}_BASE_URL"):
            failures.append(
                f"missing_{prefix}_API_KEY_or_{prefix}_BASE_URL_for_model_backend={backend}"
            )

    return failures


def _build_query_items(
    *,
    query: str | None,
    queries_file: str | None,
    benchmark: str,
    benchmark_data_path: str | None,
    domain: str | None,
    num_questions: int | None,
) -> tuple[list[dict[str, Any]], str]:
    if query:
        return ([{"id": "cli_query_0001", "query": query.strip()}], "cli_query")

    if queries_file:
        return (_load_queries_from_file(queries_file), f"file:{queries_file}")

    if benchmark != "research_bench":
        raise ValueError(f"暂不支持 benchmark={benchmark}")

    bench = ResearchBench(data_path=benchmark_data_path)
    items = bench.get_questions(domain=domain, n=num_questions)
    return (items, "research_bench")


def _serialize_agent_result(result: Any) -> dict[str, Any]:
    status = getattr(result, "status", "")
    if hasattr(status, "value"):
        status = status.value
    return {
        "task_id": getattr(result, "task_id", ""),
        "status": str(status),
        "output": getattr(result, "output", None),
        "trajectory": getattr(result, "trajectory", []),
        "action_log": getattr(result, "action_log", []),
        "token_usage": getattr(result, "token_usage", 0),
        "confidence": getattr(result, "confidence", 0.0),
        "metadata": getattr(result, "metadata", {}),
    }


def _flatten_task_actions(task_results: list[Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for result in task_results:
        action_log = getattr(result, "action_log", None)
        if isinstance(action_log, list) and action_log:
            flattened.extend(action_log)
        trajectory = getattr(result, "trajectory", None)
        if isinstance(trajectory, list) and trajectory:
            flattened.extend(trajectory)
    return flattened


def _summarize_reward_records(records: list[dict[str, Any]]) -> dict[str, float]:
    successful = [
        item for item in records
        if item.get("status") == "success"
        and isinstance(item.get("reward_breakdown"), dict)
    ]
    if not successful:
        return {
            "count": 0.0,
            "average_reward": 0.0,
            "average_reward_raw": 0.0,
            "average_quality_score": 0.0,
            "average_process_score": 0.0,
        }

    def _avg(field: str) -> float:
        values = [
            float(item["reward_breakdown"].get(field, 0.0))
            for item in successful
            if isinstance(item["reward_breakdown"].get(field), (int, float))
        ]
        return sum(values) / len(values) if values else 0.0

    return {
        "count": float(len(successful)),
        "average_reward": _avg("reward"),
        "average_reward_raw": _avg("reward_raw"),
        "average_quality_score": _avg("quality_score"),
        "average_process_score": _avg("process_score"),
    }


def _is_failed_report(report: Any, task_results: list[Any]) -> bool:
    content = str(getattr(report, "content", "") or "").strip()
    confidence = float(getattr(report, "confidence", 0.0) or 0.0)
    task_statuses: list[str] = []
    for result in task_results:
        status = getattr(result, "status", "")
        if hasattr(status, "value"):
            status = status.value
        normalized = str(status or "").strip().lower()
        if normalized:
            task_statuses.append(normalized)
    if "Research failed due to persistent errors" in content:
        return True
    if task_statuses and not any(status == "success" for status in task_statuses):
        return True
    if not task_results and confidence <= 0.0 and len(content) < 240:
        return True
    return False


def build_cache_records(
    *,
    query_items: list[dict[str, Any]],
    config: dict[str, Any],
    config_path: str,
    source_label: str,
    include_evaluation: bool = True,
    session_prefix: str = "search_cache",
    initialize_modules_fn: Callable[..., dict[str, Any]] = initialize_modules,
    run_research_fn: Callable[..., Any] = run_research,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    collector = TrajectoryCollector()
    bench = ResearchBench() if include_evaluation and source_label == "research_bench" else None

    success_count = 0
    failure_count = 0

    for idx, item in enumerate(query_items, 1):
        query = str(item["query"])
        session_id = f"{session_prefix}_{idx:04d}"
        cache_id = f"{session_prefix}_{idx:04d}"
        logger.info("[%s/%s] 运行 query: %s", idx, len(query_items), query[:100])

        try:
            modules = initialize_modules_fn(config, session_id=session_id)
            formatted_report, report = asyncio.run(
                run_research_fn(query, config, modules, return_report=True)
            )

            orchestrator = modules.get("orchestrator")
            task_results = list(getattr(orchestrator, "_results", []) or [])
            serialized_task_results = [_serialize_agent_result(r) for r in task_results]
            flattened_actions = _flatten_task_actions(task_results)

            if _is_failed_report(report, task_results):
                records.append(
                    {
                        "cache_id": cache_id,
                        "created_at": datetime.now().isoformat(),
                        "session_id": session_id,
                        "config_path": config_path,
                        "source_label": source_label,
                        "query_id": item.get("id", cache_id),
                        "domain": item.get("domain"),
                        "expected_topics": item.get("expected_topics"),
                        "ground_truth": item.get("ground_truth"),
                        "query": query,
                        "formatted_report": formatted_report,
                        "task_results": serialized_task_results,
                        "status": "failed",
                        "error": "orchestrator_failed_or_empty_report",
                    }
                )
                failure_count += 1
                continue

            collected = collector.collect(query, report, flattened_actions)
            record: dict[str, Any] = {
                "cache_id": cache_id,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
                "config_path": config_path,
                "source_label": source_label,
                "query_id": item.get("id", cache_id),
                "domain": item.get("domain"),
                "expected_topics": item.get("expected_topics"),
                "ground_truth": item.get("ground_truth"),
                "formatted_report": formatted_report,
                "task_results": serialized_task_results,
                "status": "success",
                **collected,
            }

            if bench is not None and item.get("id"):
                try:
                    record["evaluation"] = bench.evaluate_report(report, str(item["id"]))
                except Exception as eval_exc:
                    record["evaluation_error"] = f"{type(eval_exc).__name__}: {eval_exc}"

            record["reward_breakdown"] = SearchPolicyReward.breakdown(
                report_text=report.content,
                report_metadata=getattr(report, "metadata", {}) or {},
                report_sources=getattr(report, "sources", []) or [],
                evaluation_result=record.get("evaluation"),
                process_reward_trace=record.get("process_reward_trace", []),
                confidence=getattr(report, "confidence", 0.0),
            )
            record["reward"] = record["reward_breakdown"].get("reward", 0.0)

            records.append(record)
            success_count += 1
        except Exception as exc:
            logger.exception("构建 cache 失败: %s", query)
            records.append(
                {
                    "cache_id": cache_id,
                    "created_at": datetime.now().isoformat(),
                    "session_id": session_id,
                    "config_path": config_path,
                    "source_label": source_label,
                    "query_id": item.get("id", cache_id),
                    "domain": item.get("domain"),
                    "expected_topics": item.get("expected_topics"),
                    "ground_truth": item.get("ground_truth"),
                    "query": query,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failure_count += 1

    manifest = {
        "created_at": datetime.now().isoformat(),
        "config_path": config_path,
        "source_label": source_label,
        "num_queries": len(query_items),
        "num_success": success_count,
        "num_failed": failure_count,
        "include_evaluation": include_evaluation,
        "session_prefix": session_prefix,
        "reward_summary": _summarize_reward_records(records),
    }
    return records, manifest


def save_cache_records(
    *,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: str,
    output_file: str | None = None,
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_file or os.path.join(output_dir, f"search_cache_{timestamp}.jsonl")
    manifest_path = os.path.join(output_dir, f"search_cache_manifest_{timestamp}.json")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    manifest_to_save = dict(manifest)
    manifest_to_save["output_file"] = jsonl_path
    manifest_to_save["records_written"] = len(records)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_to_save, f, ensure_ascii=False, indent=2)

    return jsonl_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline search-cache dataset from Research Orchestrator runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/aliyun_smoke.yaml", help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="data/search_cache", help="输出目录")
    parser.add_argument("--output_file", type=str, default=None, help="输出 JSONL 文件路径")
    parser.add_argument("--query", type=str, default=None, help="单条 query")
    parser.add_argument("--queries_file", type=str, default=None, help="JSON/JSONL 查询文件")
    parser.add_argument("--benchmark", type=str, default="research_bench", help="benchmark 名称")
    parser.add_argument("--benchmark_data_path", type=str, default=None, help="自定义 benchmark 数据路径")
    parser.add_argument("--domain", type=str, default=None, help="按领域过滤 benchmark")
    parser.add_argument("--num_questions", type=int, default=5, help="采样题数")
    parser.add_argument("--include_evaluation", type=_parse_bool, default=True, help="是否附带 benchmark 评测")
    parser.add_argument("--session_prefix", type=str, default="search_cache", help="session/cache 前缀")
    parser.add_argument(
        "--allow_mock_evidence",
        action="store_true",
        help="允许使用 mock 搜索/浏览证据，仅用于 smoke 或管线调试，不应用于正式实验",
    )
    parser.add_argument(
        "--evidence-policy-mode",
        type=str,
        choices=["heuristic", "learned"],
        default=None,
        help="覆盖 evidence_policy.mode",
    )
    parser.add_argument(
        "--evidence-policy-model-path",
        type=str,
        default=None,
        help="覆盖 evidence_policy.model_path",
    )
    parser.add_argument(
        "--search-policy-mode",
        type=str,
        choices=["off", "heuristic", "learned"],
        default=None,
        help="覆盖 search_policy 运行模式",
    )
    parser.add_argument(
        "--search-policy-model-path",
        type=str,
        default=None,
        help="覆盖 search_policy.model_path",
    )
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)
    config = apply_policy_overrides(config, args)
    real_evidence_failures = validate_real_evidence_preflight(
        config,
        allow_mock_evidence=args.allow_mock_evidence,
    )
    if real_evidence_failures:
        raise SystemExit(
            "real-evidence preflight failed: "
            + "; ".join(real_evidence_failures)
            + ". Use --allow_mock_evidence only for smoke/debug runs."
        )

    query_items, source_label = _build_query_items(
        query=args.query,
        queries_file=args.queries_file,
        benchmark=args.benchmark,
        benchmark_data_path=args.benchmark_data_path,
        domain=args.domain,
        num_questions=args.num_questions,
    )

    records, manifest = build_cache_records(
        query_items=query_items,
        config=config,
        config_path=args.config,
        source_label=source_label,
        include_evaluation=args.include_evaluation,
        session_prefix=args.session_prefix,
    )
    jsonl_path, manifest_path = save_cache_records(
        records=records,
        manifest=manifest,
        output_dir=args.output_dir,
        output_file=args.output_file,
    )

    print(f"[search_cache] JSONL 已写入: {jsonl_path}")
    print(f"[search_cache] Manifest 已写入: {manifest_path}")
    print(
        "[search_cache] Summary: "
        f"{manifest['num_success']} success / {manifest['num_failed']} failed / {manifest['num_queries']} total"
    )


if __name__ == "__main__":
    main()
