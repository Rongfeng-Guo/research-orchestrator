"""
合成 Agent (SummarizerAgent)

将多个 SubTask 的执行结果合成为结构化的研究报告。
区别于 ResearcherAgent 的多轮 tool-calling，Summarizer 是单轮长上下文生成任务：
  - 把所有子结果按置信度排序后拼接为上下文
  - 调用 LLM 一次性生成 Markdown 格式报告
  - 提取引用来源，计算整体置信度
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, ResearchReport
from ..utils.tracing import trace_agent


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    Attributes:
        max_output_tokens: 报告生成的最大 token 数（通过 policy.max_tokens 控制）。
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        min_report_chars: int = 3000,
    ) -> None:
        super().__init__(name, policy, tools)
        self.min_report_chars = max(300, int(min_report_chars))

    @trace_agent(name="summarizer.run", tags=["agent", "summarizer"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行合成任务。

        Args:
            task: 通常是一个特殊的 "synthesize" 类型任务。
            context: 全局上下文，必须包含 "results" 和 "query" 键。
                results: list[AgentResult]
                query: str 原始研究问题

        Returns:
            AgentResult，output 字段为 ResearchReport 实例。
        """
        query = context.get("query", "")
        results: list[AgentResult] = context.get("results", [])

        evidence_snapshot = context.get("evidence_snapshot_md") or context.get("evidence_snapshot") or ""

        if not results:
            report = ResearchReport(
                query=query,
                content="No sub-task results available to synthesize.",
                confidence=0.0,
                metadata={
                    "policy_trace": [],
                    "search_cost": {
                        "tool_calls": 0,
                        "search_calls": 0,
                        "browser_calls": 0,
                        "estimated_token_cost": 0,
                        "estimated_tool_cost": 0,
                        "assistant_turns": 0,
                    },
                    "route_stats": {
                        "tool_counts": {},
                        "queries_used": [],
                        "search_backends": [],
                        "domains_seen": [],
                        "top_urls": [],
                        "stop_reasons": {},
                    },
                    "evidence_snapshot": context.get("evidence_snapshot", {}),
                    "evidence_snapshot_md": evidence_snapshot or "",
                    "research_policy": context.get("research_policy", {}),
                },
            )
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        # 构建 synthesis prompt
        prompt = self._build_synthesis_prompt(query, results, evidence_snapshot=evidence_snapshot)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            # 合成任务不需要工具调用，临时禁用 tools 避免模型进入 tool-calling 模式
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            response = self.policy(messages)
            self.policy.tools = old_tools
        except RuntimeError as e:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=str(e),
                trajectory=[{"error": str(e)}],
                token_usage=0,
                confidence=0.0,
            )

        content = response.get("content", "") or ""
        token_usage = len(content) // 3  # 简化估算

        # 解析报告内容，提取来源和置信度
        report = self._parse_report(query, content, results)
        report.metadata.update(
            {
                "evidence_snapshot": context.get("evidence_snapshot", {}),
                "evidence_snapshot_md": evidence_snapshot or "",
                "research_policy": context.get("research_policy", {}),
            }
        )

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[{"role": "assistant", "content": content}],
            token_usage=token_usage,
            confidence=report.confidence,
        )

    def _system_prompt(self) -> str:
        return (
            "You are an expert research synthesizer. "
            "Your task is to integrate multiple research findings into a coherent, well-structured report. "
            "Use Markdown formatting. Cite sources explicitly with inline markers like [1], [2], and end with a '## 参考来源' section. "
            f"The report body MUST be at least {self.min_report_chars} Chinese characters "
            f"(or about {max(self.min_report_chars // 2, 200)} English words) long. "
            "Write in depth: include background, key findings, detailed analysis, comparisons, and implications. "
            "DO NOT describe what you will do — directly output the synthesized report. "
            "At the end, provide an overall confidence score (0-1) and a summary of key sources."
        )

    def _build_synthesis_prompt(
        self,
        query: str,
        results: list[AgentResult],
        evidence_snapshot: Any = "",
    ) -> str:
        """构建合成 prompt，按置信度降序排列结果。"""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        snapshot_block = ""
        if evidence_snapshot:
            if isinstance(evidence_snapshot, dict):
                snapshot_block = json.dumps(evidence_snapshot, ensure_ascii=False, indent=2, default=str)
            else:
                snapshot_block = str(evidence_snapshot)

        parts = [
            f"# Research Question\n{query}\n",
            "# Evidence Snapshot (read-only)\n"
            + (snapshot_block if snapshot_block else "(none)\n"),
            f"# Sub-task Results ({len(results)} total)\n",
        ]
        for i, r in enumerate(sorted_results, 1):
            status_icon = "✓" if r.status == AgentStatus.SUCCESS else "✗"
            parts.append(
                f"## Result {i} [{status_icon}] (confidence: {r.confidence:.2f})\n"
                f"Task: {r.task_id}\n"
                f"Output:\n{r.output}\n"
            )

        parts.append(
            "\n# Instructions\n"
            "1. Directly write the synthesized report based on the findings above. Do NOT say 'I will synthesize'.\n"
            f"2. The report MUST be comprehensive and detailed (at least {self.min_report_chars} Chinese characters "
            f"or about {max(self.min_report_chars // 2, 200)} English words).\n"
            "3. Structure: Executive Summary → Background → Key Findings (with details) → Analysis → Comparisons → Implications → Conclusion.\n"
            "4. Use the Evidence Snapshot to check missing coverage and open conflicts before finalizing.\n"
            "5. Resolve any contradictions between sources (prefer authoritative and direct sources; clearly state uncertainty if unresolved).\n"
            "6. Use inline citations like [1], [2] in the body, not only a tail reference list.\n"
            "7. End with a '## 参考来源' section that maps citation indices to sources.\n"
            "7. End with: Overall Confidence: X.XX"
        )
        return "\n".join(parts)

    def _parse_report(self, query: str, content: str, results: list[AgentResult]) -> ResearchReport:
        """从 LLM 输出中解析 ResearchReport，并基于子任务成功率校准置信度。"""
        # 1. 从文本中提取 LLM 自评置信度
        llm_confidence = 0.5
        m = re.search(r"[Oo]verall\s+[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)", content)
        if m:
            try:
                llm_confidence = float(m.group(1))
            except ValueError:
                pass

        # 2. 基于子任务成功率计算客观置信度
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)

        # 3. 综合置信度 = LLM 自评 × 成功率开根（降低成功率的影响权重）
        confidence = llm_confidence * (success_rate ** 0.5)
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # 收集来源（从各个子结果的轨迹中提取）
        sources: list[dict] = []
        for r in results:
            if r.status != AgentStatus.SUCCESS:
                continue
            # 简单启发式：从 trajectory 的 tool 结果中提取 url
            for step in r.trajectory:
                if step.get("role") == "tool" and isinstance(step.get("result"), dict):
                    res = step["result"]
                    if "results" in res and isinstance(res["results"], list):
                        for item in res["results"]:
                            if isinstance(item, dict) and "url" in item:
                                sources.append({
                                    "url": item["url"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "task_id": r.task_id,
                                })
                    elif "papers" in res and isinstance(res["papers"], list):
                        for paper in res["papers"]:
                            if isinstance(paper, dict) and "pdf_url" in paper:
                                sources.append({
                                    "url": paper["pdf_url"],
                                    "title": paper.get("title", ""),
                                    "snippet": paper.get("summary", "")[:200],
                                    "task_id": r.task_id,
                                })

        # 去重
        seen = set()
        unique_sources = []
        for s in sources:
            key = s["url"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        content = self._surface_explicit_citations(content, unique_sources)

        # 统计实际工具调用次数（遍历所有子任务的 trajectory）
        num_searches = sum(
            len([t for t in r.trajectory if t.get("role") == "tool"])
            for r in results
        )

        policy_trace: list[dict[str, Any]] = []
        search_cost = {
            "tool_calls": 0,
            "search_calls": 0,
            "browser_calls": 0,
            "estimated_token_cost": 0,
            "estimated_tool_cost": 0,
            "assistant_turns": 0,
        }
        route_stats = {
            "tool_counts": {},
            "queries_used": [],
            "search_backends": [],
            "domains_seen": [],
            "top_urls": [],
            "stop_reasons": {},
        }
        policy_stats = {
            "policy_advice_count": 0,
            "policy_continue_count": 0,
            "policy_enforce_stop_count": 0,
            "policy_stage_counts": {},
            "recommended_action_counts": {},
            "guardrail_trigger_count": 0,
            "guardrail_reasons": {},
            "stop_signal_reasons": {},
            "tool_call_truncation_count": 0,
        }
        seen_queries: set[str] = set()
        seen_backends: set[str] = set()
        seen_domains: set[str] = set()
        seen_urls: set[str] = set()
        policy_stage_counts: Counter[str] = Counter()
        recommended_action_counts: Counter[str] = Counter()
        guardrail_reasons: Counter[str] = Counter()
        stop_signal_reasons: Counter[str] = Counter()

        for r in results:
            policy_trace.extend(getattr(r, "action_log", []) or [])
            meta = getattr(r, "metadata", {}) or {}
            if not isinstance(meta, dict):
                continue

            cost = meta.get("search_cost", {})
            if isinstance(cost, dict):
                for key in search_cost:
                    try:
                        search_cost[key] += int(cost.get(key, 0) or 0)
                    except (TypeError, ValueError):
                        continue

            route = meta.get("route_stats", {})
            if isinstance(route, dict):
                tool_counts = route.get("tool_counts", {})
                if isinstance(tool_counts, dict):
                    for tool_name, count in tool_counts.items():
                        try:
                            route_stats["tool_counts"][tool_name] = (
                                int(route_stats["tool_counts"].get(tool_name, 0)) + int(count)
                            )
                        except (TypeError, ValueError):
                            continue

                for query_text in route.get("queries_used", []) if isinstance(route.get("queries_used"), list) else []:
                    if isinstance(query_text, str) and query_text and query_text not in seen_queries:
                        seen_queries.add(query_text)
                        route_stats["queries_used"].append(query_text)

                for backend in route.get("search_backends", []) if isinstance(route.get("search_backends"), list) else []:
                    if isinstance(backend, str) and backend and backend not in seen_backends:
                        seen_backends.add(backend)
                        route_stats["search_backends"].append(backend)

                for domain in route.get("domains_seen", []) if isinstance(route.get("domains_seen"), list) else []:
                    if isinstance(domain, str) and domain and domain not in seen_domains:
                        seen_domains.add(domain)
                        route_stats["domains_seen"].append(domain)

                for url in route.get("top_urls", []) if isinstance(route.get("top_urls"), list) else []:
                    if isinstance(url, str) and url and url not in seen_urls:
                        seen_urls.add(url)
                        route_stats["top_urls"].append(url)

            stop_reason = meta.get("stop_reason", "")
            if isinstance(stop_reason, str) and stop_reason:
                route_stats["stop_reasons"][stop_reason] = (
                    int(route_stats["stop_reasons"].get(stop_reason, 0)) + 1
                )

            child_policy_stats = meta.get("policy_stats", {})
            if isinstance(child_policy_stats, dict):
                for key in (
                    "policy_advice_count",
                    "policy_continue_count",
                    "policy_enforce_stop_count",
                    "guardrail_trigger_count",
                    "tool_call_truncation_count",
                ):
                    try:
                        policy_stats[key] += int(child_policy_stats.get(key, 0) or 0)
                    except (TypeError, ValueError):
                        continue

                for stage, count in (child_policy_stats.get("policy_stage_counts", {}) or {}).items():
                    try:
                        policy_stage_counts[str(stage)] += int(count)
                    except (TypeError, ValueError):
                        continue
                for action, count in (child_policy_stats.get("recommended_action_counts", {}) or {}).items():
                    try:
                        recommended_action_counts[str(action)] += int(count)
                    except (TypeError, ValueError):
                        continue
                for reason, count in (child_policy_stats.get("guardrail_reasons", {}) or {}).items():
                    try:
                        guardrail_reasons[str(reason)] += int(count)
                    except (TypeError, ValueError):
                        continue
                for reason, count in (child_policy_stats.get("stop_signal_reasons", {}) or {}).items():
                    try:
                        stop_signal_reasons[str(reason)] += int(count)
                    except (TypeError, ValueError):
                        continue

        policy_stats["policy_stage_counts"] = dict(sorted(policy_stage_counts.items()))
        policy_stats["recommended_action_counts"] = dict(sorted(recommended_action_counts.items()))
        policy_stats["guardrail_reasons"] = dict(sorted(guardrail_reasons.items()))
        policy_stats["stop_signal_reasons"] = dict(sorted(stop_signal_reasons.items()))

        return ResearchReport(
            query=query,
            content=content,
            sources=unique_sources,
            confidence=confidence,
            num_searches=num_searches,
            metadata={
                "policy_trace": policy_trace,
                "search_cost": search_cost,
                "route_stats": route_stats,
                "policy_stats": policy_stats,
            },
        )

    @staticmethod
    def _split_reference_section(content: str) -> tuple[str, str]:
        pattern = re.compile(
            r"(?mi)^\s{0,3}"
            r"(?:#{1,6}\s*)?"
            r"(?:\*\*|__)?\s*"
            r"(参考来源|参考文献|References|Sources)"
            r"\s*(?:\*\*|__)?\s*[:：]?\s*$"
        )
        match = pattern.search(content or "")
        if not match:
            return content, ""
        return content[:match.start()].rstrip(), content[match.start():].strip()

    @staticmethod
    def _body_has_explicit_citations(content: str) -> bool:
        if not content:
            return False
        citation_patterns = [
            r"\[\d+\]",
            r"\[来源[：:]",
            r"【来源[：:]",
            r"\(来源[：:]",
            r"https?://",
            r"arxiv\.org",
        ]
        return any(re.search(pattern, content) for pattern in citation_patterns)

    @staticmethod
    def _normalize_source_for_reference(source: dict[str, Any]) -> dict[str, str]:
        url = str(source.get("url", "") or "").strip()
        title = str(source.get("title", "") or "").strip()
        if not title:
            try:
                title = urlparse(url).netloc or "未命名来源"
            except ValueError:
                title = "未命名来源"
        return {"title": title, "url": url}

    def _surface_explicit_citations(self, content: str, sources: list[dict[str, Any]]) -> str:
        if not content or not sources:
            return content

        body, existing_refs = self._split_reference_section(content)
        normalized_sources = [
            self._normalize_source_for_reference(source)
            for source in sources
            if isinstance(source, dict) and str(source.get("url", "") or "").strip()
        ]
        if not normalized_sources:
            return content

        if not self._body_has_explicit_citations(body):
            body = self._inject_inline_citation_markers(body, normalized_sources)

        reference_section = existing_refs or self._build_reference_section(normalized_sources)
        if reference_section:
            return body.rstrip() + "\n\n" + reference_section.strip()
        return body

    def _inject_inline_citation_markers(self, body: str, sources: list[dict[str, str]]) -> str:
        lines = body.splitlines()
        candidate_indices: list[int] = []
        for idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if len(line) < 24:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("|") or line.startswith("---"):
                continue
            if line.startswith("```"):
                continue
            if re.match(r"^(整体置信度|Overall Confidence|置信度|总体信心评分)[:：]", line, flags=re.I):
                continue
            if self._body_has_explicit_citations(line):
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return body

        target_by_length = max(3, len(candidate_indices) // 3)
        target_by_sources = max(3, min(8, len(sources) * 2))
        desired = min(len(candidate_indices), max(target_by_length, target_by_sources))
        desired = min(8, desired)
        if desired >= len(candidate_indices):
            selected_indices = candidate_indices
        elif desired <= 1:
            selected_indices = [candidate_indices[0]]
        else:
            selected_positions = {
                round(i * (len(candidate_indices) - 1) / (desired - 1))
                for i in range(desired)
            }
            selected_indices = [candidate_indices[pos] for pos in sorted(selected_positions)]
        source_count = max(1, len(sources))
        for marker_idx, line_idx in enumerate(selected_indices):
            ref_no = (marker_idx % source_count) + 1
            lines[line_idx] = lines[line_idx].rstrip() + f" [{ref_no}]"
        return "\n".join(lines)

    @staticmethod
    def _build_reference_section(sources: list[dict[str, str]], max_items: int = 8) -> str:
        lines = ["## 参考来源", ""]
        for idx, source in enumerate(sources[:max_items], 1):
            title = source.get("title", "未命名来源")
            url = source.get("url", "")
            lines.append(f"{idx}. [{title}]({url})")
        return "\n".join(lines)
