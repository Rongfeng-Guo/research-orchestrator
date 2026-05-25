"""
研究员 Agent (ResearcherAgent)

执行搜索和分析类 SubTask，实现多轮 tool-calling 循环。
设计为项目一 ToolAgentLoop 的简化版：
  - 单 trajectory，无批处理
  - 支持 7 种工具：web_search, arxiv_reader, code_sandbox, browser,
    file_reader, calculator, notepad
  - 通过 VLLMPolicy 进行 LLM 调用
  - 工具结果回写后自动继续，直到模型不再调用工具或达到 max_turns
"""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import re
import time
from typing import Any
from urllib.parse import urlparse

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus
from ..utils.tracing import trace_agent


__all__ = ["ResearcherAgent"]

_QUERY_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")


class ResearcherAgent(BaseAgent):
    """研究员 Agent：负责搜索、分析、验证类任务。

    可用工具（7 个）：
      - web_search:   网页搜索，返回标题/链接/摘要
      - browser:      网页阅读器，打开 URL 提取正文
      - arxiv_reader: ArXiv 论文元数据检索
      - file_reader:  本地文件阅读（.txt/.md/.pdf/.csv/.json/.docx）
      - code_sandbox: Python 代码沙箱执行
      - calculator:   轻量数学计算（比沙箱更快更安全）
      - notepad:      草稿笔记（记录中间结论/待办/搜索策略）

    Attributes:
        max_turns: 最大交互轮数，防止无限循环。
        tool_map: 工具名称到工具实例的映射。
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        max_turns: int = 10,
        max_tool_calls_per_turn: int = 2,
        max_tool_calls_per_task: int | None = None,
        search_policy: Any | None = None,
    ) -> None:
        super().__init__(name, policy, tools)
        self.max_turns = max_turns
        self.max_tool_calls_per_turn = max(max_tool_calls_per_turn, 1)
        self.max_tool_calls_per_task = max_tool_calls_per_task
        self.tool_map: dict[str, Any] = {t.name: t for t in (tools or [])}
        self.search_policy = search_policy

    @trace_agent(name="researcher.run", tags=["agent", "researcher"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行 Researcher 任务。

        流程:
          1. 构建初始 system + user messages
          2. 循环调用 policy，解析 tool_calls
          3. 执行工具，将结果追加为 tool message
          4. 直到无 tool_calls 或达到 max_turns
        """
        trajectory: list[dict] = []
        action_log: list[dict[str, Any]] = []
        total_tokens: int = 0
        executed_tool_calls = 0
        tool_budget_exhausted_prompts = 0
        search_policy_state = (
            self.search_policy.new_state()
            if self.search_policy is not None and getattr(self.search_policy, "is_available", False)
            else None
        )

        # 构建任务描述
        task_desc = self._build_task_prompt(task, context)

        # 查询可行性判断：如果任务明显无法通过网络搜索获得答案，直接基于已知信息分析
        if self._is_non_searchable(task, context):
            messages = [
                {"role": "system", "content": self._system_prompt_direct_analysis()},
                {"role": "user", "content": task_desc},
            ]
            try:
                response = self.policy(messages)
                content = response.get("content", "") or ""
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=[{"role": "assistant", "content": content}],
                    action_log=[{
                        "task_id": task.task_id,
                        "turn": 0,
                        "action_type": "direct_analysis",
                        "query_text": context.get("query", ""),
                        "tool_name": None,
                        "latency_ms": None,
                        "result_count": 0,
                        "estimated_token_cost": len(content) // 3,
                        "stop_reason": "non_searchable_direct_analysis",
                    }],
                    token_usage=len(content) // 3,
                    confidence=self._extract_confidence(content),
                    metadata=self._build_result_metadata(
                        action_log=action_log + [{
                            "task_id": task.task_id,
                            "turn": 0,
                            "action_type": "direct_analysis",
                            "query_text": context.get("query", ""),
                            "tool_name": None,
                            "latency_ms": None,
                            "result_count": 0,
                            "estimated_token_cost": len(content) // 3,
                            "stop_reason": "non_searchable_direct_analysis",
                        }],
                        total_tokens=len(content) // 3,
                        stop_reason="non_searchable_direct_analysis",
                    ),
                )
            except Exception as e:
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"Direct analysis failed: {e}",
                    trajectory=[{"error": str(e)}],
                    action_log=[{
                        "task_id": task.task_id,
                        "turn": 0,
                        "action_type": "direct_analysis",
                        "error": str(e),
                        "stop_reason": "direct_analysis_error",
                    }],
                    token_usage=0,
                    confidence=0.0,
                    metadata=self._build_result_metadata(
                        action_log=[{
                            "task_id": task.task_id,
                            "turn": 0,
                            "action_type": "direct_analysis",
                            "error": str(e),
                            "stop_reason": "direct_analysis_error",
                        }],
                        total_tokens=0,
                        stop_reason="direct_analysis_error",
                    ),
                )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task_desc},
        ]

        # 若 policy 支持 tool 设置，则注册可用工具
        if hasattr(self.policy, "set_tools") and self.tools:
            schemas = [t.get_openai_tool_schema() for t in self.tools]
            self.policy.set_tools(schemas)

        # 根据任务类型确定 fallback 工具
        desc_lower = (task.description or "").lower()
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        fallback_tool = "arxiv_reader" if any(kw in desc_lower for kw in academic_keywords) else "web_search"
        
        for turn in range(self.max_turns):
            # Fallback: if last turn had no tool_calls, force a search instruction
            if turn > 0 and messages and messages[-1].get("role") == "assistant":
                last_tool_calls = messages[-1].get("tool_calls", [])
                if not last_tool_calls:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You did not use any tools. "
                            f"You MUST call the '{fallback_tool}' tool now to search for information. "
                            f"Do not write a summary without searching first."
                        ),
                    })

            try:
                # 使用线程池执行同步 policy，避免阻塞 asyncio 事件循环
                response = await asyncio.to_thread(self.policy, messages)
            except RuntimeError as e:
                # 上下文长度超限等致命错误
                trajectory.append({"turn": turn, "error": str(e)})
                action_log.append({
                    "task_id": task.task_id,
                    "turn": turn,
                    "action_type": "policy_call",
                    "error": str(e),
                    "stop_reason": "policy_runtime_error",
                })
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=str(e),
                    trajectory=trajectory,
                    action_log=action_log,
                    token_usage=total_tokens,
                    confidence=0.0,
                    metadata=self._build_result_metadata(
                        action_log=action_log,
                        total_tokens=total_tokens,
                        stop_reason="policy_runtime_error",
                    ),
                )

            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls", []) or []
            original_tool_call_count = len(tool_calls)
            remaining_tool_budget = None
            if self.max_tool_calls_per_task is not None:
                remaining_tool_budget = max(int(self.max_tool_calls_per_task) - executed_tool_calls, 0)
            allowed_tool_calls = self.max_tool_calls_per_turn
            if remaining_tool_budget is not None:
                allowed_tool_calls = min(allowed_tool_calls, remaining_tool_budget)
            if allowed_tool_calls <= 0:
                tool_calls = []
            elif original_tool_call_count > allowed_tool_calls:
                tool_calls = tool_calls[:allowed_tool_calls]

            trajectory.append({
                "turn": turn,
                "role": "assistant",
                "content": content,
                "tool_calls": [dict(tc) for tc in tool_calls],
            })
            assistant_action = {
                "task_id": task.task_id,
                "turn": turn,
                "action_type": "assistant_response",
                "tool_calls_count": len(tool_calls),
                "original_tool_calls_count": original_tool_call_count,
                "estimated_token_cost": len(content) // 3,
            }
            action_log.append(assistant_action)
            if original_tool_call_count > len(tool_calls):
                action_log.append({
                    "task_id": task.task_id,
                    "turn": turn,
                    "action_type": "tool_calls_truncated",
                    "original_tool_calls_count": original_tool_call_count,
                    "kept_tool_calls_count": len(tool_calls),
                    "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
                    "max_tool_calls_per_task": self.max_tool_calls_per_task,
                })
            if search_policy_state is not None:
                self.search_policy.observe_action(search_policy_state, assistant_action)

            # 估算 token（简化：字符数 / 3）
            total_tokens += len(json.dumps(messages, ensure_ascii=False)) // 3

            # 无工具调用 → 任务完成
            if not tool_calls:
                if original_tool_call_count > 0 and tool_budget_exhausted_prompts == 0 and turn < self.max_turns - 1:
                    tool_budget_exhausted_prompts += 1
                    messages.append({"role": "assistant", "content": content or "[Tool calls omitted by budget guard.]"})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Tool-call budget is exhausted. Do NOT call more tools. "
                            "Write the final summary now using the evidence already gathered, and explicitly state any evidence gaps."
                        ),
                    })
                    continue
                if search_policy_state is not None:
                    recommendation = self.search_policy.recommend(
                        search_policy_state,
                        preferred_search_tool=fallback_tool,
                    )
                    recommendation = self._apply_search_policy_guardrails(
                        recommendation,
                        task=task,
                        context=context,
                        stage="assistant_no_tool",
                        preferred_search_tool=fallback_tool,
                    )
                    action_log.append(
                        self._build_search_policy_log(
                            task_id=task.task_id,
                            turn=turn,
                            recommendation=recommendation,
                            stage="assistant_no_tool",
                        )
                    )
                    if recommendation.get("can_continue") and turn < self.max_turns - 1:
                        messages.append(self._assistant_message_from_response(response))
                        messages.append({
                            "role": "user",
                            "content": recommendation["guidance"],
                        })
                        continue
                # B方案：检测 LLM 回复是否包含明显的工具失败说明
                if self._is_tool_failure_explanation(content):
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=content,
                        trajectory=trajectory,
                        action_log=action_log + [{
                            "task_id": task.task_id,
                            "turn": turn,
                            "action_type": "stop",
                            "stop_reason": "tool_failure_explanation",
                        }],
                        token_usage=total_tokens,
                        confidence=0.0,
                        metadata=self._build_result_metadata(
                            action_log=action_log + [{
                                "task_id": task.task_id,
                                "turn": turn,
                                "action_type": "stop",
                                "stop_reason": "tool_failure_explanation",
                            }],
                            total_tokens=total_tokens,
                            stop_reason="tool_failure_explanation",
                        ),
                    )
                confidence = self._extract_confidence(content)
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=trajectory,
                    action_log=action_log + [{
                        "task_id": task.task_id,
                        "turn": turn,
                        "action_type": "stop",
                        "stop_reason": "model_finished_no_tool_calls",
                    }],
                    token_usage=total_tokens,
                    confidence=confidence,
                    metadata=self._build_result_metadata(
                        action_log=action_log + [{
                            "task_id": task.task_id,
                            "turn": turn,
                            "action_type": "stop",
                            "stop_reason": "model_finished_no_tool_calls",
                        }],
                        total_tokens=total_tokens,
                        stop_reason="model_finished_no_tool_calls",
                    ),
                )

            # 执行工具调用
            tool_results = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                original_query_text = str(args.get("query", "")) if isinstance(args, dict) else ""
                if tool_name == "web_search" and isinstance(args, dict):
                    args = self._augment_web_search_args(args, task=task, context=context)

                t0 = time.perf_counter()
                result = await self._execute_tool(tool_name, args)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                top_urls = self._extract_top_urls(result)
                result_count = self._extract_result_count(result)
                estimated_token_cost = (len(json.dumps(args, ensure_ascii=False)) + len(json.dumps(result, ensure_ascii=False, default=str))) // 3

                # B方案：检测工具返回结果是否包含 error 字段
                if isinstance(result, dict) and result.get("error"):
                    error_msg = result["error"]
                    trajectory.append({
                        "turn": turn,
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tool_name,
                        "error": error_msg,
                    })
                    action_log.append({
                        "task_id": task.task_id,
                        "turn": turn,
                        "action_type": "tool_call",
                        "tool_call_id": tc.get("id", ""),
                        "tool_name": tool_name,
                        "tool_args": args,
                        "query_text": str(args.get("query", "")),
                        "original_query_text": original_query_text,
                        "backend": self._extract_backend(result),
                        "latency_ms": latency_ms,
                        "result_count": result_count,
                        "top_urls": top_urls,
                        "estimated_token_cost": estimated_token_cost,
                        "error": error_msg,
                        "error_type": result.get("error_type"),
                    })
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=f"Tool '{tool_name}' failed: {error_msg}",
                        trajectory=trajectory,
                        action_log=action_log + [{
                            "task_id": task.task_id,
                            "turn": turn,
                            "action_type": "stop",
                            "stop_reason": "tool_error",
                            "tool_name": tool_name,
                        }],
                        token_usage=total_tokens,
                        confidence=0.0,
                        metadata=self._build_result_metadata(
                            action_log=action_log + [{
                                "task_id": task.task_id,
                                "turn": turn,
                                "action_type": "stop",
                                "stop_reason": "tool_error",
                                "tool_name": tool_name,
                            }],
                            total_tokens=total_tokens,
                            stop_reason="tool_error",
                        ),
                    )

                tool_results.append({
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "result": result,
                })
                trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "result": result,
                    "latency_ms": latency_ms,
                })
                tool_action = {
                    "task_id": task.task_id,
                    "turn": turn,
                    "action_type": "tool_call",
                    "tool_call_id": tc.get("id", ""),
                    "tool_name": tool_name,
                    "tool_args": args,
                    "query_text": str(args.get("query", "")),
                    "original_query_text": original_query_text,
                    "backend": self._extract_backend(result),
                    "latency_ms": latency_ms,
                    "result_count": result_count,
                    "top_urls": top_urls,
                    "estimated_token_cost": estimated_token_cost,
                }
                action_log.append(tool_action)
                executed_tool_calls += 1
                if search_policy_state is not None:
                    self.search_policy.observe_action(search_policy_state, tool_action)

            # 检测搜索结果是否全为空（工具返回了但无有效内容）
            all_empty = True
            for tr in tool_results:
                if tr["name"] == "web_search":
                    res = tr["result"]
                    if isinstance(res, dict) and res.get("results"):
                        for r in res["results"]:
                            if r.get("snippet", "").strip():
                                all_empty = False
                                break
            
            # 如果已搜索 2+ 轮或搜索结果全空，强制要求总结
            search_count = sum(1 for t in trajectory if t.get("role") == "tool" and t.get("name") == "web_search")
            force_summary = False
            force_summary_reason = ""
            policy_guidance_message = ""

            if search_policy_state is not None:
                recommendation = self.search_policy.recommend(
                    search_policy_state,
                    preferred_search_tool=fallback_tool,
                )
                recommendation = self._apply_search_policy_guardrails(
                    recommendation,
                    task=task,
                    context=context,
                    stage="post_tool",
                    preferred_search_tool=fallback_tool,
                )
                action_log.append(
                    self._build_search_policy_log(
                        task_id=task.task_id,
                        turn=turn,
                        recommendation=recommendation,
                        stage="post_tool",
                    )
                )
                if recommendation.get("enforce_stop"):
                    force_summary = True
                    force_summary_reason = "learned_policy_stop"
                elif recommendation.get("can_continue"):
                    policy_guidance_message = str(recommendation.get("guidance", "") or "")

            if search_count >= 2:
                force_summary = True
                force_summary_reason = "search_limit"
            if all_empty and tool_results:
                force_summary = True
                force_summary_reason = "empty_results"
            if force_summary:
                stop_signal_action = {
                    "task_id": task.task_id,
                    "turn": turn,
                    "action_type": "stop_signal",
                    "stop_reason": force_summary_reason,
                }
                action_log.append(stop_signal_action)
                if search_policy_state is not None:
                    self.search_policy.observe_action(search_policy_state, stop_signal_action)

            # 将 assistant message 和 tool results 追加到 messages
            messages.append(self._assistant_message_from_response(response))

            for tr in tool_results:
                msg_content = json.dumps(tr["result"], ensure_ascii=False, default=str)
                # 如果强制总结，给工具结果附加提示
                if force_summary:
                    msg_content += "\n\n[SYSTEM NOTICE] You have already searched enough. Write your final summary NOW. Do NOT call any more tools."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": msg_content,
                })
            if policy_guidance_message and not force_summary:
                messages.append({
                    "role": "user",
                    "content": policy_guidance_message,
                })

        # 达到 max_turns
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.TIMEOUT,
            output="Reached max_turns without final answer.",
            trajectory=trajectory,
            action_log=action_log + [{
                "task_id": task.task_id,
                "turn": self.max_turns,
                "action_type": "stop",
                "stop_reason": "max_turns_reached",
            }],
            token_usage=total_tokens,
            confidence=0.0,
            metadata=self._build_result_metadata(
                action_log=action_log + [{
                    "task_id": task.task_id,
                    "turn": self.max_turns,
                    "action_type": "stop",
                    "stop_reason": "max_turns_reached",
                }],
                total_tokens=total_tokens,
                stop_reason="max_turns_reached",
            ),
        )

    @staticmethod
    def _extract_backend(result: Any) -> str | None:
        if not isinstance(result, dict):
            return None
        backend = result.get("backend") or result.get("source")
        if isinstance(backend, str) and backend.strip():
            return backend.strip()
        return None

    @staticmethod
    def _extract_result_count(result: Any) -> int:
        if not isinstance(result, dict):
            return 0
        if isinstance(result.get("results"), list):
            return len(result.get("results", []))
        if isinstance(result.get("papers"), list):
            return len(result.get("papers", []))
        total = result.get("total")
        if isinstance(total, int):
            return total
        return 0

    @staticmethod
    def _extract_top_urls(result: Any, max_urls: int = 5) -> list[str]:
        if not isinstance(result, dict):
            return []
        urls: list[str] = []
        for item in result.get("results", []) if isinstance(result.get("results"), list) else []:
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
                    if len(urls) >= max_urls:
                        return urls
        for item in result.get("papers", []) if isinstance(result.get("papers"), list) else []:
            if isinstance(item, dict):
                url = item.get("pdf_url") or item.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)
                    if len(urls) >= max_urls:
                        return urls
        return urls

    @staticmethod
    def _assistant_message_from_response(response: dict[str, Any]) -> dict[str, Any]:
        assistant_msg = {
            "role": "assistant",
            "content": response.get("content", "") or "",
        }
        tool_calls = response.get("tool_calls", []) or []
        if tool_calls:
            assistant_msg["tool_calls"] = [dict(tc) for tc in tool_calls]
        if response.get("reasoning_content"):
            assistant_msg["reasoning_content"] = response["reasoning_content"]
        return assistant_msg

    def _apply_search_policy_guardrails(
        self,
        recommendation: dict[str, Any],
        *,
        task: SubTask,
        context: dict[str, Any],
        stage: str,
        preferred_search_tool: str,
    ) -> dict[str, Any]:
        if not isinstance(recommendation, dict):
            return recommendation

        adjusted = dict(recommendation)
        features = adjusted.get("features", {})
        if not isinstance(features, dict):
            features = {}
            adjusted["features"] = features

        label = str(adjusted.get("label", "") or "")
        browser_url = str(adjusted.get("browser_url", "") or "")
        tool_calls = float(features.get("tool_calls_so_far", 0.0) or 0.0)
        search_calls = float(features.get("search_calls_so_far", 0.0) or 0.0)
        browser_calls = float(features.get("browser_calls_so_far", 0.0) or 0.0)
        browser_available = float(features.get("browser_available", 0.0) or 0.0)
        recent_query_overlap = float(features.get("recent_query_overlap", 0.0) or 0.0)
        last_result_count = float(features.get("last_result_count", 0.0) or 0.0)
        budget_ratio = float(features.get("budget_ratio", 0.0) or 0.0)
        seed_urls = self._collect_primary_source_urls(task, context)
        has_seed_urls = bool(seed_urls)

        if stage == "assistant_no_tool" and tool_calls <= 0.0 and search_calls <= 0.0 and browser_calls <= 0.0:
            adjusted["label"] = "search"
            adjusted["can_continue"] = True
            adjusted["enforce_stop"] = False
            adjusted["guardrail_reason"] = "no_search_yet_force_search"
            adjusted["guidance"] = self._build_force_search_guidance(
                preferred_search_tool=preferred_search_tool,
                seed_urls=seed_urls,
            )
            return adjusted

        if (
            label == "stop"
            and has_seed_urls
            and search_calls >= 1.0
            and browser_calls <= 0.0
            and browser_available >= 0.5
            and browser_url
        ):
            adjusted["label"] = "browser"
            adjusted["can_continue"] = True
            adjusted["enforce_stop"] = False
            adjusted["guardrail_reason"] = "seeded_query_force_browser_before_stop"
            adjusted["guidance"] = (
                "[SEARCH POLICY] [GUARD] You have already retrieved candidate primary sources "
                "but have not opened any source page yet. Open the strongest page with the "
                f"'browser' tool now. Use this URL: {browser_url}"
            )
            return adjusted

        if (
            label in {"search", "browser"}
            and search_calls >= 2.0
            and browser_calls >= 1.0
            and last_result_count > 0.0
            and (recent_query_overlap >= 0.75 or budget_ratio >= 1.0)
        ):
            adjusted["label"] = "stop"
            adjusted["can_continue"] = False
            adjusted["enforce_stop"] = True
            adjusted["guardrail_reason"] = "repetitive_search_force_stop"
            adjusted["guidance"] = (
                "[SEARCH POLICY] [GUARD] Recent search steps are repeating with diminishing returns. "
                "Stop now and write the final answer using the evidence already gathered. "
                "Explicitly note any remaining evidence gaps instead of calling more tools."
            )
            return adjusted

        return adjusted

    @staticmethod
    def _build_force_search_guidance(
        *,
        preferred_search_tool: str,
        seed_urls: list[str],
    ) -> str:
        if seed_urls:
            return (
                "[SEARCH POLICY] [GUARD] You have not used any retrieval tool yet. "
                f"Run one focused search with the '{preferred_search_tool}' tool now, and prioritize these primary URLs: "
                + " ; ".join(seed_urls[:4])
            )
        return (
            "[SEARCH POLICY] [GUARD] You have not used any retrieval tool yet. "
            f"Run one focused search with the '{preferred_search_tool}' tool now before answering."
        )

    @staticmethod
    def _build_search_policy_log(
        *,
        task_id: str,
        turn: int,
        recommendation: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "turn": turn,
            "action_type": "policy_advice",
            "policy_stage": stage,
            "recommended_action": recommendation.get("label"),
            "policy_source": recommendation.get("source"),
            "confidence": recommendation.get("confidence", 0.0),
            "probabilities": recommendation.get("probabilities", {}),
            "feature_vector": recommendation.get("feature_vector", []),
            "features": recommendation.get("features", {}),
            "guidance": recommendation.get("guidance", ""),
            "can_continue": bool(recommendation.get("can_continue")),
            "enforce_stop": bool(recommendation.get("enforce_stop")),
            "guardrail_reason": recommendation.get("guardrail_reason"),
        }

    @staticmethod
    def _clean_seed_url(url: str) -> str:
        return (url or "").rstrip(".,;:!?)]}\"'")

    @classmethod
    def _extract_seed_urls(cls, text: str) -> list[str]:
        seen: set[str] = set()
        urls: list[str] = []
        for match in _QUERY_URL_PATTERN.findall(text or ""):
            url = cls._clean_seed_url(match)
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    @classmethod
    def _collect_primary_source_urls(cls, task: SubTask, context: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        candidates.extend(cls._extract_seed_urls(task.description or ""))
        query_text = context.get("query", "")
        if isinstance(query_text, str):
            candidates.extend(cls._extract_seed_urls(query_text))
        for hint in task.search_hints or []:
            if isinstance(hint, str):
                candidates.extend(cls._extract_seed_urls(hint))

        seen: set[str] = set()
        deduped: list[str] = []
        for url in candidates:
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return deduped

    @classmethod
    def _augment_web_search_args(
        cls,
        args: dict[str, Any],
        *,
        task: SubTask,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        query = str(args.get("query", "") or "").strip()
        if not query:
            return args
        if cls._extract_seed_urls(query):
            return args

        seed_urls = cls._collect_primary_source_urls(task, context)
        if not seed_urls:
            return args

        augmented = dict(args)
        augmented["query"] = (
            f"{query}. Prefer these primary sources first: {' ; '.join(seed_urls[:4])}"
        )
        return augmented

    def _system_prompt(self) -> str:
        return (
            "You are a meticulous research assistant. "
            "Your job is to gather and analyze information using the RIGHT tool for each task. "
            "\n\nAVAILABLE TOOLS:\n"
            "- web_search: General web search for news, market data, industry reports, current events. "
            "  Use this as the FIRST tool for most tasks.\n"
            "- arxiv_reader: Academic paper search (ArXiv / Semantic Scholar). "
            "  USE when the task involves: papers, publications, academic research, citation counts.\n"
            "- browser: Open a URL and extract full webpage text. "
            "  USE after web_search when search results are too short and you need to read the original article in depth.\n"
            "- code_sandbox: Execute Python code for calculations, data processing, simulations. "
            "  USE when the task requires: computing FLOPs, memory usage, statistical analysis, data transformation.\n"
            "- calculator: Quick math evaluation (+, -, *, /, sqrt, log, mean, etc.). "
            "  USE for simple calculations instead of code_sandbox.\n"
            "- notepad: Write/read intermediate notes to avoid forgetting findings during multi-step research. "
            "  USE to record key numbers, conclusions, or next search queries.\n"
            "- file_reader: Read local files (.txt, .md, .pdf, .csv, .json, .docx). "
            "  USE only when the task explicitly references a local file path.\n"
            "\nIMPORTANT RULES:\n"
            "1. You MUST use a tool to find factual information. Do NOT answer from your own knowledge.\n"
            "2. Choose the RIGHT tool based on the task type. You can use MULTIPLE tools in sequence.\n"
            "3. For most research tasks, START with web_search or arxiv_reader.\n"
            "4. If search results are too short, use browser to read the full article.\n"
            "5. If the task involves numbers/calculations, use calculator or code_sandbox.\n"
            "6. You may call tools AT MOST 2 times total. After that you MUST summarize.\n"
            "7. Only after gathering information, provide a concise summary with a confidence score (0-1).\n"
            "8. NEVER greet the user or ask what they want to search — just execute immediately.\n"
            "9. If you have already performed 2 tool calls, do NOT call more — write the final summary now."
        )

    def _system_prompt_direct_analysis(self) -> str:
        return (
            "You are a thoughtful analyst. "
            "The user has asked a question that cannot be answered by web search "
            "(e.g., analyzing a specific private individual, personal advice, or subjective judgment). "
            "Your job is to provide a reasoned analysis based ONLY on the information already provided in the context. "
            "Do NOT make up facts. Clearly state what is known, what can be reasonably inferred, and what remains unknown. "
            "End with a confidence score (0-1)."
        )

    def _is_non_searchable(self, task: SubTask, context: dict) -> bool:
        """启发式判断任务是否无法通过网络搜索获取答案。"""
        desc = (task.description or "").lower()
        query = context.get("query", "").lower()
        combined = desc + " " + query

        # 模式 1：分析/评价特定私人个体（姓名 + 描述性分析）
        if "朋友" in combined or "同学" in combined or "同事" in combined:
            if any(w in combined for w in ["分析", "评价", "是什么样", "性格", "人品"]):
                return True

        # 模式 2：主观建议类（基于个人情况）
        if any(w in combined for w in ["建议我", "我该怎么", "适合我吗", "要不要"]):
            if "朋友" in combined or "我" in query:
                return True

        # 模式 3：明显的个人隐私分析
        if "叫" in combined and any(w in combined for w in ["分析", "评价", "是什么样"]):
            return True

        return False

    def _build_task_prompt(self, task: SubTask, context: dict) -> str:
        """根据 SubTask 和全局上下文构建 user prompt。"""
        desc_lower = (task.description or "").lower()

        # Evidence snapshot (optional): surfaced by orchestrator/memory as a dict or markdown string.
        # This is intentionally explicit in the prompt so the agent can target missing coverage
        # and avoid re-doing already-supported facts.
        evidence_block = ""
        for k in ("evidence_snapshot_md", "evidence_snapshot", "evidence_snapshot_text"):
            if k in context and context.get(k):
                v = context.get(k)
                if isinstance(v, dict):
                    evidence_block = json.dumps(v, ensure_ascii=False, indent=2, default=str)
                else:
                    evidence_block = str(v)
                break
        
        # 智能工具推荐：根据任务描述关键词匹配
        tool_recommendations = []
        
        # 学术论文类
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        if any(kw in desc_lower for kw in academic_keywords):
            tool_recommendations.append("arxiv_reader")
        
        # 计算/数学类
        calc_keywords = ["计算", "flops", "显存", "内存", "参数量", "延迟", "成本", "公式", "数值", "统计", "数学", "公式", "推导"]
        if any(kw in desc_lower for kw in calc_keywords):
            tool_recommendations.append("calculator")
            tool_recommendations.append("code_sandbox")
        
        # 深度阅读类（需要读原文）
        browser_keywords = ["详细", "原文", "全文", "深度", "详细内容", "网页内容", "文章正文"]
        if any(kw in desc_lower for kw in browser_keywords):
            tool_recommendations.append("browser")
        
        # 文件类
        file_keywords = ["文件", "文档", "dataset", "数据集", "pdf", "csv", "json"]
        if any(kw in desc_lower for kw in file_keywords):
            tool_recommendations.append("file_reader")
        
        # 确定首选工具：学术论文类优先用 arxiv_reader，其他先用 web_search
        is_academic = "arxiv_reader" in tool_recommendations
        if is_academic:
            # 学术论文任务：arxiv_reader 优先，web_search 备选
            tool_recommendations = ["arxiv_reader"] + [t for t in tool_recommendations if t != "arxiv_reader"]
        elif not tool_recommendations:
            tool_recommendations.insert(0, "web_search")
        
        primary_tool = tool_recommendations[0]
        secondary_tools = tool_recommendations[1:]
        
        lines = [
            f"## Task: {task.description}",
            f"Type: {task.task_type.value}",
            f"Expected output: {task.expected_type}",
            "",
            f"## RECOMMENDED TOOLS (in priority order): {', '.join(tool_recommendations)}",
        ]

        if evidence_block:
            lines.extend([
                "",
                "## Evidence Snapshot (read-only)",
                evidence_block,
                "",
                "## Guidance From Evidence Snapshot",
                "1. If the snapshot lists missing terms/topics, prioritize searching for those specifically.",
                "2. If the snapshot shows open conflicts, prioritize retrieving authoritative sources to resolve them.",
                "3. Avoid repeating claims already strongly supported unless you are verifying contradictions.",
            ])
        
        if secondary_tools:
            lines.append(f"Start with '{primary_tool}'. If the task involves numbers/calculations, also use {', '.join(secondary_tools)}.")
        else:
            lines.append(f"Use '{primary_tool}' to gather information.")
        
        lines.extend([
            "",
            "## INSTRUCTIONS:",
            f"1. First, call the '{primary_tool}' tool with a relevant query to gather information.",
            "2. Review the results.",
            f"3. If needed, call '{primary_tool}' ONE MORE time with a refined query.",
            "   You may call tools AT MOST 2 times total. After the 2nd call, you MUST write the final summary.",
            "4. If search results are too short, you may use 'browser' to read the full article (counts as 1 tool call).",
            "5. If calculations are needed, use 'calculator' or 'code_sandbox' (counts as 1 tool call).",
            "6. Finally, summarize your findings in Chinese with a confidence score (0-1).",
            "7. DO NOT greet the user or ask clarifying questions — just execute immediately.",
            "8. IMPORTANT: Your query MUST directly address the task description.",
        ])
        if task.search_hints:
            lines.insert(1, f"Search hints (MUST use these as primary keywords): {', '.join(task.search_hints)}")
        if task.context_keys:
            ctx_parts = []
            for key in task.context_keys:
                if key in context:
                    ctx_parts.append(f"- {key}: {context[key]}")
            if ctx_parts:
                lines.append("\n## Context:")
                lines.extend(ctx_parts)
        return "\n".join(lines)

    async def _execute_tool(self, tool_name: str, args: dict) -> Any:
        """调用具体工具实例。"""
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        try:
            return await tool.execute(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _is_tool_failure_explanation(self, content: str) -> bool:
        """检测 LLM 回复是否是工具失败的解释说明而非真实研究结果。

        常见模式：额度用完、无法连接、无法搜索等。
        """
        if not content:
            return False
        c = content.lower()
        failure_keywords = [
            "无法通过", "无法执行", "无法使用", "无法获取", "无法访问",
            "额度已用尽", "配额已用完", "额度已用完", "搜索配额",
            "cannot search", "unable to search", "quota exceeded",
            "api key", "额度不足", "余额不足", "余额为", "余额：0",
            "网络错误", "连接失败", "无法连接到",
        ]
        return any(kw in c for kw in failure_keywords)

    def _extract_confidence(self, content: str) -> float:
        """从输出文本中尝试提取置信度分数。"""
        import re
        # 匹配 "Confidence: 0.85" 或 "置信度: 0.85"
        patterns = [
            r"[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)",
            r"置信度[:\s]+(0\.\d+|1\.0|1)",
        ]
        for pat in patterns:
            m = re.search(pat, content)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        # 默认中等置信度
        return 0.6

    def _build_result_metadata(
        self,
        action_log: list[dict[str, Any]],
        total_tokens: int,
        stop_reason: str,
    ) -> dict[str, Any]:
        tool_counts: dict[str, int] = {}
        queries_used: list[str] = []
        search_backends: set[str] = set()
        domains_seen: set[str] = set()
        top_urls: list[str] = []
        seen_queries: set[str] = set()
        seen_urls: set[str] = set()
        assistant_turns = 0
        estimated_tool_cost = 0
        policy_stage_counts: Counter[str] = Counter()
        recommended_action_counts: Counter[str] = Counter()
        guardrail_reasons: Counter[str] = Counter()
        stop_signal_reasons: Counter[str] = Counter()
        policy_advice_count = 0
        policy_continue_count = 0
        policy_enforce_stop_count = 0
        tool_call_truncation_count = 0

        for action in action_log:
            if action.get("action_type") == "assistant_response":
                assistant_turns += 1
            if action.get("action_type") == "tool_calls_truncated":
                tool_call_truncation_count += 1
                continue
            if action.get("action_type") == "stop_signal":
                reason = str(action.get("stop_reason", "") or "")
                if reason:
                    stop_signal_reasons[reason] += 1
                continue
            if action.get("action_type") == "policy_advice":
                policy_advice_count += 1
                stage = str(action.get("policy_stage", "") or "")
                if stage:
                    policy_stage_counts[stage] += 1
                recommended_action = str(action.get("recommended_action", "") or "")
                if recommended_action:
                    recommended_action_counts[recommended_action] += 1
                if bool(action.get("can_continue")):
                    policy_continue_count += 1
                if bool(action.get("enforce_stop")):
                    policy_enforce_stop_count += 1
                guardrail_reason = str(action.get("guardrail_reason", "") or "")
                if guardrail_reason:
                    guardrail_reasons[guardrail_reason] += 1
                continue
            if action.get("action_type") != "tool_call":
                continue
            tool_name = action.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                tool_counts[tool_name] = int(tool_counts.get(tool_name, 0)) + 1
            estimated_tool_cost += int(action.get("estimated_token_cost", 0) or 0)

            query_text = action.get("query_text")
            if isinstance(query_text, str) and query_text and query_text not in seen_queries:
                seen_queries.add(query_text)
                queries_used.append(query_text)

            backend = action.get("backend")
            if isinstance(backend, str) and backend:
                search_backends.add(backend)

            for url in action.get("top_urls", []) if isinstance(action.get("top_urls"), list) else []:
                if not isinstance(url, str) or not url:
                    continue
                if url not in seen_urls:
                    seen_urls.add(url)
                    top_urls.append(url)
                domain = self._domain_from_url(url)
                if domain:
                    domains_seen.add(domain)

        return {
            "stop_reason": stop_reason,
            "search_cost": {
                "tool_calls": sum(tool_counts.values()),
                "search_calls": int(tool_counts.get("web_search", 0)),
                "browser_calls": int(tool_counts.get("browser", 0)),
                "estimated_token_cost": total_tokens,
                "estimated_tool_cost": estimated_tool_cost,
                "assistant_turns": assistant_turns,
            },
            "route_stats": {
                "tool_counts": tool_counts,
                "queries_used": queries_used,
                "search_backends": sorted(search_backends),
                "domains_seen": sorted(domains_seen),
                "top_urls": top_urls[:10],
            },
            "policy_stats": {
                "policy_advice_count": policy_advice_count,
                "policy_continue_count": policy_continue_count,
                "policy_enforce_stop_count": policy_enforce_stop_count,
                "policy_stage_counts": dict(sorted(policy_stage_counts.items())),
                "recommended_action_counts": dict(sorted(recommended_action_counts.items())),
                "guardrail_trigger_count": sum(guardrail_reasons.values()),
                "guardrail_reasons": dict(sorted(guardrail_reasons.items())),
                "stop_signal_reasons": dict(sorted(stop_signal_reasons.items())),
                "tool_call_truncation_count": tool_call_truncation_count,
            },
        }

    @staticmethod
    def _domain_from_url(url: str) -> str:
        if not url:
            return ""
        try:
            return urlparse(url).netloc.lower()
        except ValueError:
            return ""
