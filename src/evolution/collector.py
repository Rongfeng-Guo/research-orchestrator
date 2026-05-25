"""
M6 自进化引擎 — Trajectory 收集器

TrajectoryCollector 负责将 DeepResearch Agent 的完整执行轨迹收集并转换为
veRL 训练所需的格式。它是 Solver（DeepResearch Agent）与训练框架之间的适配层。

设计决策：
1. 收集的内容包括：query、report、多轮交互轨迹、搜索次数、重规划次数等。
2. to_verl_format 方法复用项目一的 parquet 构建逻辑，输出标准格式。
3. 支持批量收集，便于后续构建训练数据集。
"""
from __future__ import annotations

import json
from typing import Any

from src.orchestrator.schemas import ResearchReport


__all__ = ["TrajectoryCollector"]


class TrajectoryCollector:
    """Trajectory 收集与格式转换器。

    Attributes:
        system_prompt: 可选的系统级 prompt，用于 veRL 数据格式。
    """

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def collect(
        self,
        query: str,
        report: ResearchReport,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """收集单次 DeepResearch 的完整轨迹。

        Args:
            query: 原始研究问题。
            report: 最终生成的研究报告。
            trajectories: 多轮交互轨迹列表，每轮包含 role/content/tool_calls 等。

        Returns:
            统一格式的 trajectory 字典，包含 veRL 所需的所有字段。
        """
        report_metadata = getattr(report, "metadata", {}) or {}
        structured_actions = (
            report_metadata.get("policy_trace", [])
            if isinstance(report_metadata, dict)
            and isinstance(report_metadata.get("policy_trace", []), list)
            and report_metadata.get("policy_trace")
            else self._extract_structured_actions(trajectories)
        )
        action_summary = self._summarize_actions(structured_actions)
        search_cost = (
            report_metadata.get("search_cost", {})
            if isinstance(report_metadata, dict)
            else {}
        )
        route_stats = (
            report_metadata.get("route_stats", {})
            if isinstance(report_metadata, dict)
            else {}
        )
        report_sources = getattr(report, "sources", []) or []
        if not isinstance(report_sources, list):
            report_sources = []
        trajectory_sources = self._extract_sources_from_trajectories(
            list(trajectories or []) + list(structured_actions or [])
        )
        sources = self._merge_sources(report_sources, trajectory_sources)
        evidence_snapshot = (
            report_metadata.get("evidence_snapshot", {})
            if isinstance(report_metadata, dict)
            else {}
        )
        evidence_transition_trace = self._normalize_transition_trace(
            report_metadata.get("evidence_transitions")
            or report_metadata.get("evidence_transition_trace", [])
            if isinstance(report_metadata, dict)
            else []
        )
        evidence_transition_summary = self._summarize_transition_trace(
            evidence_transition_trace
        )
        process_reward_trace = self._build_process_reward_trace(evidence_transition_trace)
        return {
            "query": query,
            "report_content": report.content,
            "sources": sources,
            "confidence": report.confidence,
            "num_searches": report.num_searches,
            "num_replan": report.num_replan,
            "adversarial_rounds": report.adversarial_rounds,
            "final_score": report.final_score,
            "report_metadata": report_metadata,
            "final_evidence_snapshot": evidence_snapshot,
            "evidence_transitions": evidence_transition_trace,
            "evidence_transition_trace": evidence_transition_trace,
            "evidence_transition_summary": evidence_transition_summary,
            "process_reward_trace": process_reward_trace,
            "trajectories": trajectories,
            "structured_actions": structured_actions,
            "action_summary": action_summary,
            "search_cost": search_cost,
            "route_stats": route_stats,
            # 元信息
            "trajectory_length": len(trajectories),
            "content_length": len(report.content),
            "source_count": len(sources),
            "transition_count": len(evidence_transition_trace),
        }

    def to_verl_format(self, data: dict[str, Any]) -> dict[str, Any]:
        """将收集的 trajectory 转换为 veRL 训练所需的 parquet 行格式。

        veRL 期望的字段（与项目一 scripts/11_build_grpo_parquet.py 对齐）：
        - prompt: list[dict] — 多轮对话格式，包含 system + user 初始 query
        - response: str — 模型的完整输出（report content）
        - trajectories: list[dict] — 多轮交互轨迹（observation, action pairs）
        - metadata: dict — 额外元信息

        Args:
            data: collect() 方法的输出。

        Returns:
            veRL 格式的字典，可直接写入 parquet。
        """
        query = data.get("query", "")
        trajectories = data.get("trajectories", [])
        report_content = data.get("report_content", "")

        # 构建 prompt 字段：system + user query
        prompt_messages: list[dict[str, str]] = []
        if self.system_prompt:
            prompt_messages.append({"role": "system", "content": self.system_prompt})
        prompt_messages.append({"role": "user", "content": query})

        # metadata 包含所有原始字段（去除大字段避免 parquet 膨胀）
        metadata = {
            "query": query,
            "num_searches": data.get("num_searches", 0),
            "num_replan": data.get("num_replan", 0),
            "adversarial_rounds": data.get("adversarial_rounds", 0),
            "final_score": data.get("final_score", 0.0),
            "report_metadata": data.get("report_metadata", {}),
            "final_evidence_snapshot": data.get("final_evidence_snapshot", {}),
            "evidence_transitions": data.get("evidence_transitions", []),
            "evidence_transition_trace": data.get("evidence_transition_trace", []),
            "evidence_transition_summary": data.get("evidence_transition_summary", {}),
            "process_reward_trace": data.get("process_reward_trace", []),
            "action_summary": data.get("action_summary", {}),
            "search_cost": data.get("search_cost", {}),
            "route_stats": data.get("route_stats", {}),
            "trajectory_length": data.get("trajectory_length", 0),
            "action_count": len(data.get("structured_actions", [])),
            "source_count": data.get("source_count", 0),
            "content_length": data.get("content_length", 0),
            "transition_count": data.get("transition_count", 0),
        }

        return {
            "prompt": prompt_messages,
            "response": report_content,
            "trajectories": trajectories,
            "evidence_transitions": data.get("evidence_transitions", []),
            "step_transitions": data.get("evidence_transition_trace", []),
            "process_reward_trace": data.get("process_reward_trace", []),
            "metadata": metadata,
        }

    def batch_to_verl(
        self, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """批量转换为 veRL 格式。

        Args:
            batch: collect() 输出列表。

        Returns:
            veRL 格式字典列表。
        """
        return [self.to_verl_format(item) for item in batch]

    def serialize(self, data: dict[str, Any]) -> str:
        """将 trajectory 序列化为 JSON 字符串（用于日志或持久化）。"""
        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_transition_trace(transitions: Any) -> list[dict[str, Any]]:
        """标准化 evidence transition trace，保持 JSON 友好。"""
        if not isinstance(transitions, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in transitions:
            if not isinstance(item, dict):
                continue
            before = dict(item.get("before", {})) if isinstance(item.get("before"), dict) else {}
            after = dict(item.get("after", {})) if isinstance(item.get("after"), dict) else {}
            if "evidence_graph_quality" not in before:
                before["evidence_graph_quality"] = TrajectoryCollector._quality_proxy(before)
            if "evidence_graph_quality" not in after:
                after["evidence_graph_quality"] = TrajectoryCollector._quality_proxy(after)
            normalized.append(
                {
                    "step": int(item.get("step", len(normalized) + 1) or len(normalized) + 1),
                    "task_id": str(item.get("task_id", "")),
                    "task_type": str(item.get("task_type", "")),
                    "status": str(item.get("status", "")),
                    "label_action": str(item.get("label_action", "")),
                    "policy_state_before": before,
                    "policy_state_after": after,
                    "before": before,
                    "after": after,
                    "delta": dict(item.get("delta", {})) if isinstance(item.get("delta"), dict) else {},
                }
            )
        return normalized

    @staticmethod
    def _build_process_reward_trace(transitions: list[dict[str, Any]]) -> list[float]:
        """从 evidence transitions 派生过程奖励，便于后续 reward shaping。"""
        rewards: list[float] = []
        for item in transitions:
            before = item.get("before", {})
            after = item.get("after", {})
            if not isinstance(before, dict) or not isinstance(after, dict):
                rewards.append(0.0)
                continue

            before_uncertainty = float(before.get("uncertainty", 1.0) or 1.0)
            after_uncertainty = float(after.get("uncertainty", before_uncertainty) or before_uncertainty)
            before_coverage = float(before.get("coverage_ratio", 0.0) or 0.0)
            after_coverage = float(after.get("coverage_ratio", before_coverage) or before_coverage)
            before_conflicts = float(before.get("open_conflicts", 0.0) or 0.0)
            after_conflicts = float(after.get("open_conflicts", before_conflicts) or before_conflicts)
            before_claims = float(before.get("claim_count", 0.0) or 0.0)
            after_claims = float(after.get("claim_count", before_claims) or before_claims)
            before_quality = float(before.get("evidence_graph_quality", 0.0) or 0.0)
            after_quality = float(after.get("evidence_graph_quality", before_quality) or before_quality)

            uncertainty_gain = max(0.0, before_uncertainty - after_uncertainty)
            coverage_gain = max(0.0, after_coverage - before_coverage)
            conflict_gain = max(0.0, before_conflicts - after_conflicts) / max(before_conflicts, 1.0)
            claim_gain = max(0.0, after_claims - before_claims) / max(before_claims, 1.0)
            quality_gain = max(0.0, after_quality - before_quality)

            reward = max(
                -1.0,
                min(
                    1.0,
                    0.40 * uncertainty_gain
                    + 0.25 * coverage_gain
                    + 0.15 * conflict_gain
                    + 0.10 * claim_gain
                    + 0.10 * quality_gain,
                ),
            )
            rewards.append(round(reward, 4))
        return rewards

    @staticmethod
    def _summarize_transition_trace(transitions: list[dict[str, Any]]) -> dict[str, Any]:
        """把 step-wise evidence transitions 压缩成训练友好的统计摘要。"""
        if not transitions:
            return {
                "num_transitions": 0,
                "coverage_gain": 0.0,
                "uncertainty_reduction": 0.0,
                "quality_gain": 0.0,
                "conflict_reduction": 0.0,
                "terminal_action": "",
            }

        first_before = transitions[0].get("before", {})
        last_after = transitions[-1].get("after", {})
        first_uncertainty = float(first_before.get("uncertainty", 1.0) or 1.0)
        last_uncertainty = float(last_after.get("uncertainty", first_uncertainty) or first_uncertainty)
        first_coverage = float(first_before.get("coverage_ratio", 0.0) or 0.0)
        last_coverage = float(last_after.get("coverage_ratio", first_coverage) or first_coverage)
        first_quality = float(first_before.get("evidence_graph_quality", 0.0) or 0.0)
        last_quality = float(last_after.get("evidence_graph_quality", first_quality) or first_quality)
        first_conflicts = float(first_before.get("open_conflicts", 0.0) or 0.0)
        last_conflicts = float(last_after.get("open_conflicts", first_conflicts) or first_conflicts)

        return {
            "num_transitions": len(transitions),
            "coverage_gain": round(last_coverage - first_coverage, 4),
            "uncertainty_reduction": round(first_uncertainty - last_uncertainty, 4),
            "quality_gain": round(last_quality - first_quality, 4),
            "conflict_reduction": round(first_conflicts - last_conflicts, 4),
            "terminal_action": str(last_after.get("recommended_action", "") or ""),
        }

    @staticmethod
    def _quality_proxy(state: dict[str, Any]) -> float:
        """在 transition 只有 summary 时，构造一个轻量 evidence-quality 代理分数。"""
        if not isinstance(state, dict) or not state:
            return 0.0

        claim_count = max(float(state.get("claim_count", 0.0) or 0.0), 1.0)
        support_ratio = min(
            1.0,
            float(state.get("support_edge_count", 0.0) or 0.0) / claim_count,
        )
        contradiction_ratio = min(
            1.0,
            float(state.get("open_conflicts", 0.0) or 0.0) / claim_count,
        )
        trust = max(0.0, min(1.0, float(state.get("avg_source_trust", 0.0) or 0.0)))
        coverage = max(0.0, min(1.0, float(state.get("coverage_ratio", 0.0) or 0.0)))
        strength = max(0.0, min(1.0, float(state.get("evidence_strength", 0.0) or 0.0)))
        return round(
            max(
                0.0,
                min(
                    1.0,
                    0.28 * support_ratio
                    + 0.24 * trust
                    + 0.24 * strength
                    + 0.14 * coverage
                    + 0.10 * (1.0 - contradiction_ratio),
                ),
            ),
            4,
        )

    @staticmethod
    def _extract_structured_actions(
        trajectories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """从原始轨迹中提取结构化 action records。

        支持两种来源：
        1. 轨迹元素已包含 action 语义（action_type/tool_name 等）
        2. 传统 role=tool 的轨迹（向后兼容）
        """
        actions: list[dict[str, Any]] = []
        for idx, step in enumerate(trajectories):
            if not isinstance(step, dict):
                continue
            if step.get("action_type"):
                actions.append(dict(step))
                continue
            if step.get("role") == "tool":
                result = step.get("result", {})
                result_count = 0
                top_urls: list[str] = []
                if isinstance(result, dict):
                    if isinstance(result.get("results"), list):
                        result_count = len(result["results"])
                        for item in result["results"][:5]:
                            if isinstance(item, dict) and isinstance(item.get("url"), str):
                                top_urls.append(item["url"])
                    elif isinstance(result.get("papers"), list):
                        result_count = len(result["papers"])
                        for item in result["papers"][:5]:
                            if isinstance(item, dict):
                                url = item.get("pdf_url") or item.get("url")
                                if isinstance(url, str):
                                    top_urls.append(url)

                actions.append(
                    {
                        "turn": step.get("turn", idx),
                        "action_type": "tool_call",
                        "tool_name": step.get("name"),
                        "tool_call_id": step.get("tool_call_id"),
                        "backend": result.get("backend") if isinstance(result, dict) else None,
                        "result_count": result_count,
                        "top_urls": top_urls,
                        "latency_ms": step.get("latency_ms"),
                        "error": step.get("error"),
                    }
                )
        return actions

    @staticmethod
    def _merge_sources(*source_lists: list[Any]) -> list[dict[str, Any]]:
        """Merge source inventories while preserving first-seen metadata."""
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for sources in source_lists:
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url", "") or source.get("pdf_url", "") or "").strip()
                if not url:
                    continue
                key = url.rstrip("/").lower()
                if key in seen:
                    continue
                seen.add(key)
                item = dict(source)
                item["url"] = url
                item.pop("pdf_url", None)
                merged.append(item)
        return merged

    @staticmethod
    def _extract_sources_from_trajectories(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Recover source inventory from raw tool results and structured action logs."""
        sources: list[dict[str, Any]] = []

        def add_source(url: Any, *, title: Any = "", snippet: Any = "", backend: Any = "", task_id: Any = "", rank: Any = None) -> None:
            url_text = str(url or "").strip()
            if not url_text:
                return
            item: dict[str, Any] = {
                "url": url_text,
                "title": str(title or ""),
                "snippet": str(snippet or ""),
            }
            if backend:
                item["backend"] = str(backend)
            if task_id:
                item["task_id"] = str(task_id)
            if rank is not None:
                item["rank"] = rank
            sources.append(item)

        for step in trajectories or []:
            if not isinstance(step, dict):
                continue

            result = step.get("result", {})
            backend = ""
            if isinstance(result, dict):
                backend = str(result.get("backend", "") or result.get("source", "") or "")
                if isinstance(result.get("results"), list):
                    for item in result["results"]:
                        if not isinstance(item, dict):
                            continue
                        add_source(
                            item.get("url"),
                            title=item.get("title", ""),
                            snippet=item.get("snippet", ""),
                            backend=item.get("backend", backend),
                            task_id=step.get("task_id", ""),
                            rank=item.get("rank"),
                        )
                if isinstance(result.get("papers"), list):
                    for item in result["papers"]:
                        if not isinstance(item, dict):
                            continue
                        add_source(
                            item.get("pdf_url") or item.get("url"),
                            title=item.get("title", ""),
                            snippet=item.get("summary", "") or item.get("snippet", ""),
                            backend=item.get("backend", backend),
                            task_id=step.get("task_id", ""),
                        )

            action_backend = str(step.get("backend", "") or backend or "")
            task_id = step.get("task_id", "")
            top_urls = step.get("top_urls", [])
            if isinstance(top_urls, list):
                for rank, url in enumerate(top_urls, 1):
                    add_source(url, backend=action_backend, task_id=task_id, rank=rank)

            tool_args = step.get("tool_args", {})
            if isinstance(tool_args, dict) and str(step.get("tool_name", "") or "") == "browser":
                add_source(tool_args.get("url"), backend=action_backend or "browser", task_id=task_id)

        return TrajectoryCollector._merge_sources(sources)

    @staticmethod
    def _summarize_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
        """生成轻量 action 统计，供 reward 和离线分析使用。"""
        tool_calls = [a for a in actions if a.get("action_type") == "tool_call"]
        total_latency = sum(
            int(a.get("latency_ms", 0))
            for a in tool_calls
            if isinstance(a.get("latency_ms", 0), int | float)
        )
        tool_breakdown: dict[str, int] = {}
        for action in tool_calls:
            name = str(action.get("tool_name") or "unknown")
            tool_breakdown[name] = tool_breakdown.get(name, 0) + 1

        return {
            "num_actions": len(actions),
            "num_tool_calls": len(tool_calls),
            "tool_breakdown": tool_breakdown,
            "total_latency_ms": total_latency,
            "avg_latency_ms": (total_latency / len(tool_calls)) if tool_calls else 0.0,
        }
