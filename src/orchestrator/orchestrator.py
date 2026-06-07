"""
Research Orchestrator — 核心编排器 (M1: Multi-Agent Orchestrator)

9 状态状态机驱动的异步任务编排引擎：
  IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
  失败时进入 REPLANNING，最终可进入 FAILED。

设计亮点:
  - 自研 asyncio + DAG executor，不依赖 LangGraph/AutoGen
  - 拓扑排序后按层并发执行，Semaphore 控制最大并发度
  - 三级降级策略：单任务超时→标记继续；>50%失败→re-plan；全局超时→强制合成
  - 状态机用字典映射实现，便于扩展新状态和转换逻辑
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from .schemas import (
    OrchestratorState,
    SubTask,
    AgentResult,
    AgentStatus,
    ResearchReport,
    RunConfig,
    TaskType,
)
from .agent_pool import AgentPool
from ..planner.dag import DAG
from ..planner.planner import Planner, PlanParseError
from ..planner.budget_tracker import BudgetTracker
from ..utils.tracing import trace_chain

# M4: Memory Store 类型提示（延迟导入避免循环依赖）
SharedMemoryStore = Any


__all__ = ["Orchestrator"]


class Orchestrator:
    """Research Orchestrator 核心编排器。

    Attributes:
        planner: 自适应规划器，负责初始规划和增量重规划。
        agent_pool: Agent 对象池，管理 worker agent 生命周期。
        budget_tracker: Token 预算追踪器。
        memory_store: 全局共享内存，存储所有子任务结果和中间上下文。
        compressor: （预留）上下文压缩器接口。
    """

    def __init__(
        self,
        planner: Planner,
        agent_pool: AgentPool,
        budget_tracker: BudgetTracker | None = None,
        compressor: Any | None = None,
        adversarial_loop: Any | None = None,
        memory_store: Any | None = None,
        summarizer_policy: Any | None = None,
        summarizer_min_report_chars: int = 3000,
    ) -> None:
        self.planner = planner
        self.agent_pool = agent_pool
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.compressor = compressor
        self.adversarial_loop = adversarial_loop
        self.memory_store = memory_store
        self.summarizer_policy = summarizer_policy
        self.summarizer_min_report_chars = max(300, int(summarizer_min_report_chars))

        # 运行时状态（保留 dict 作为快速缓存，M4 提供持久化 + 语义检索）
        self._memory_store: dict[str, Any] = {}
        self._results: list[AgentResult] = []
        self._dag: DAG | None = None
        self._task_map: dict[str, SubTask] = {}
        self._current_state = OrchestratorState.IDLE
        self._query: str = ""
        self._config: RunConfig = RunConfig()
        self._start_time: float = 0.0
        self._replan_count: int = 0
        self._adversarial_count: int = 0

        # 状态机处理器映射
        self._state_handlers: dict[OrchestratorState, Callable[[], asyncio.Future[OrchestratorState]]] = {
            OrchestratorState.IDLE: self._on_idle,
            OrchestratorState.PLANNING: self._do_planning,
            OrchestratorState.DISPATCHING: self._do_dispatching,
            OrchestratorState.COLLECTING: self._do_collecting,
            OrchestratorState.SYNTHESIZING: self._do_synthesizing,
            OrchestratorState.ADVERSARIAL: self._do_adversarial,
            OrchestratorState.REPLANNING: self._do_replanning,
            OrchestratorState.DONE: self._on_done,
            OrchestratorState.FAILED: self._on_failed,
        }

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @trace_chain(name="orchestrator.run", tags=["m1", "orchestrator"])
    async def run(self, query: str, config: RunConfig | None = None) -> ResearchReport:
        """主入口：执行完整的研究流程。

        Args:
            query: 研究问题。
            config: 运行配置，默认使用 RunConfig()。

        Returns:
            ResearchReport: 最终研究报告。
        """
        self._query = query
        self._config = config or RunConfig()
        self._start_time = time.monotonic()
        self._replan_count = 0
        self._adversarial_count = 0
        self._memory_store.clear()
        self._results.clear()
        self._dag = None
        self._task_map.clear()
        self._current_state = OrchestratorState.IDLE

        # 状态机主循环
        while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
            # 全局超时检查
            if self._is_global_timeout():
                if self._current_state in (
                    OrchestratorState.COLLECTING,
                    OrchestratorState.SYNTHESIZING,
                    OrchestratorState.ADVERSARIAL,
                ):
                    # 强制合成：用已有结果生成报告
                    self._current_state = OrchestratorState.SYNTHESIZING
                else:
                    self._current_state = OrchestratorState.FAILED
                break

            handler = self._state_handlers.get(self._current_state)
            if handler is None:
                raise RuntimeError(f"Unknown state: {self._current_state}")

            next_state = await handler()
            self._current_state = next_state

            print(f"[Orchestrator] State transition: {self._current_state.value}")

        # 返回结果
        if self._current_state == OrchestratorState.DONE:
            # 最终报告应在 memory 中
            report = self._memory_store.get("final_report")
            if report is None:
                report = ResearchReport(query=query, content="Report generation failed unexpectedly.")
            if hasattr(report, "metadata") and isinstance(report.metadata, dict):
                report.metadata.setdefault("evidence_snapshot", self._memory_store.get("evidence_snapshot", {}))
                report.metadata.setdefault("evidence_snapshot_md", self._memory_store.get("evidence_snapshot_md", ""))
                report.metadata.setdefault(
                    "evidence_transitions",
                    self._memory_store.get("evidence_transition_trace", []),
                )
                report.metadata.setdefault(
                    "evidence_transition_trace",
                    self._memory_store.get("evidence_transition_trace", []),
                )
                report.metadata.setdefault("research_policy", self._memory_store.get("research_policy", {}))
            report.num_replan = self._replan_count
            report.adversarial_rounds = self._adversarial_count

            # M4: 将最终报告存入 SharedMemoryStore
            if self.memory_store is not None:
                try:
                    from src.memory.long_term import MemoryEntry
                    entry = MemoryEntry(
                        entry_id=f"final_report:{int(time.time())}",
                        claim=str(report.content)[:800],
                        source="orchestrator",
                        confidence=report.confidence,
                        agent_id="orchestrator",
                        timestamp=time.time(),
                        evidence_type="primary",
                        embedding=[],
                        topic=query[:50],
                        metadata={
                            "num_searches": report.num_searches,
                            "num_replan": report.num_replan,
                            "adversarial_rounds": report.adversarial_rounds,
                        },
                    )
                    self.memory_store.put(entry)
                    print(f"[M4] Final report stored to memory (confidence={report.confidence:.2f})")
                except Exception as e:
                    print(f"[M4] Failed to store final report: {e}")

            return report

        # FAILED 状态
        return ResearchReport(
            query=query,
            content="Research failed due to persistent errors or global timeout.",
            num_replan=self._replan_count,
            adversarial_rounds=self._adversarial_count,
        )

    # ------------------------------------------------------------------
    # 状态机处理器
    # ------------------------------------------------------------------

    async def _on_idle(self) -> OrchestratorState:
        """从 IDLE 自动进入 PLANNING。"""
        return OrchestratorState.PLANNING

    async def _do_planning(self) -> OrchestratorState:
        """调用 Planner 生成初始 DAG。

        失败时直接转入 FAILED（初始计划失败无法恢复）。
        """
        try:
            memory_ctx = self._build_memory_context()
            self._dag = self.planner.generate_plan(self._query, memory_ctx)
            # 从 planner 获取完整的 SubTask 信息（包括 description、search_hints 等）
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                # 降级：如果解析失败，使用占位符
                self._task_map = self._rebuild_task_map_from_dag()
        except PlanParseError as e:
            print(f"[Planning] Failed: {e}")
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Planning] Unexpected error: {e}")
            return OrchestratorState.FAILED

        n_tasks = len(self._dag)
        n_layers = len(self._dag.get_parallel_groups()) if self._dag else 0
        print(f"[Planning] [OK] DAG 生成完成: {n_tasks} 个子任务, {n_layers} 个执行层")
        # 打印子任务描述以便诊断
        for tid, task in self._task_map.items():
            print(f"[Planning]   {tid}: {task.description}")
        return OrchestratorState.DISPATCHING

    async def _do_dispatching(self) -> OrchestratorState:
        """拓扑排序 + 并发调度 sub-agents。

        核心逻辑:
          1. 获取并行执行层 (parallel groups)
          2. 每层内用 asyncio.gather + Semaphore 并发执行
          3. 每个 sub-task 设置单独超时 (asyncio.wait_for)
          4. 收集结果到 self._results
        """
        if self._dag is None or len(self._dag) == 0:
            return OrchestratorState.COLLECTING

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        parallel_groups = self._dag.get_parallel_groups()
        all_results: list[AgentResult] = []

        for layer_idx, group in enumerate(parallel_groups):
            print(f"[Dispatch] [RUN] Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (并行执行)")

            # 构建本层的 coroutine 列表
            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"SubTask '{task_id}' not found in task_map",
                        )

                    # 准备上下文：先执行依赖任务的结果
                    context = self._build_task_context(subtask)

                    # 获取 Agent
                    agent = await self.agent_pool.get_agent(subtask.task_type)
                    try:
                        # 设置单任务超时
                        result = await asyncio.wait_for(
                            agent.run(subtask, context),
                            timeout=subtask.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.TIMEOUT,
                            output=f"Task timed out after {subtask.timeout_seconds}s",
                        )
                    except Exception as e:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"Exception: {type(e).__name__}: {e}",
                        )
                    finally:
                        await self.agent_pool.release_agent(agent)

                    return result

            # 并发执行本层
            coros = [_run_one(tid) for tid in group]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)

            for lr in layer_results:
                if isinstance(lr, Exception):
                    # 将异常包装为 FAILED 结果
                    # 这种情况理论上不会发生（_run_one 内部已捕获），但保险起见
                    all_results.append(AgentResult(
                        task_id="unknown",
                        status=AgentStatus.FAILED,
                        output=f"Dispatch exception: {lr}",
                    ))
                else:
                    all_results.append(lr)

        self._results = all_results
        return OrchestratorState.COLLECTING

    async def _do_collecting(self) -> OrchestratorState:
        """收集结果，写入 memory，检查是否需要重规划。

        三级降级策略检查点:
          - 单任务超时/失败：已在 dispatch 层处理（标记状态，继续执行）
          - >50% 失败：触发 REPLANNING
          - 全局超时：由外层 run() 的循环检查处理
        """
        # 将结果写入运行时 memory dict
        for r in self._results:
            self._memory_store[f"result:{r.task_id}"] = r

        # M4: 将成功结果同步写入 SharedMemoryStore（持久化 + 向量索引）
        evidence_transition_trace: list[dict[str, Any]] = []
        previous_snapshot_summary: dict[str, Any] | None = None
        if self.memory_store is not None:
            try:
                initial_snapshot = self.memory_store.get_evidence_snapshot(self._query)
                previous_snapshot_summary = self._summarize_evidence_snapshot(initial_snapshot.to_dict())
            except Exception:
                previous_snapshot_summary = None

        if self.memory_store is not None:
            for idx, r in enumerate(self._results, 1):
                if r.status == AgentStatus.SUCCESS and r.output:
                    self._sync_result_to_memory_store(r)
                subtask = self._task_map.get(r.task_id)
                task_type = subtask.task_type.value if subtask is not None else "unknown"
                try:
                    current_snapshot = self.memory_store.get_evidence_snapshot(self._query)
                    current_snapshot_summary = self._summarize_evidence_snapshot(current_snapshot.to_dict())
                except Exception:
                    current_snapshot_summary = previous_snapshot_summary or {}

                if previous_snapshot_summary is not None:
                    evidence_transition_trace.append(
                        self._build_evidence_transition_record(
                            step=idx,
                            task_id=r.task_id,
                            task_type=task_type,
                            status=r.status.value,
                            before=previous_snapshot_summary,
                            after=current_snapshot_summary,
                        )
                    )
                previous_snapshot_summary = current_snapshot_summary

        if evidence_transition_trace:
            self._memory_store["evidence_transition_trace"] = evidence_transition_trace

        # M4: 生成当前证据状态快照，供重规划/合成/下游 Agent 读取
        if self.memory_store is not None:
            try:
                snapshot = self.memory_store.get_evidence_snapshot(self._query)
                snapshot_dict = snapshot.to_dict()
                snapshot_md = snapshot.to_markdown()
                self._memory_store["evidence_snapshot"] = snapshot_dict
                self._memory_store["evidence_snapshot_md"] = snapshot_md
                self._memory_store["evidence_snapshot_text"] = snapshot_md
                self._memory_store["research_policy"] = {
                    "recommended_action": snapshot.recommended_action,
                    "uncertainty": snapshot.uncertainty,
                    "coverage_ratio": snapshot.coverage_ratio,
                    "open_conflicts": len(snapshot.conflicts),
                    "missing_terms": snapshot.missing_terms,
                    "candidate_actions": snapshot.to_dict().get("candidate_actions", []),
                    "rationale": snapshot.rationale,
                }
                print(
                    "[M4] Evidence snapshot: "
                    f"action={snapshot.recommended_action}, "
                    f"uncertainty={snapshot.uncertainty:.2f}, "
                    f"coverage={snapshot.coverage_ratio:.2f}, "
                    f"conflicts={len(snapshot.conflicts)}"
                )
            except Exception as e:
                print(f"[M4] Failed to build evidence snapshot: {e}")

        success_count = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        total_count = len(self._results)
        fail_count = total_count - success_count
        status_icon = "[OK]" if success_count == total_count else "[WARN]"
        print(f"[Collect] {status_icon} 子任务完成: {success_count}/{total_count} 成功", end="")
        if fail_count > 0:
            print(f" ({fail_count} 失败)")
        else:
            print()

        # 检查是否需要重规划
        if self._should_replan(self._results):
            if self._replan_count < self._config.max_replan_rounds:
                self._replan_count += 1
                return OrchestratorState.REPLANNING
            else:
                print("[Collect] Max replan rounds reached, proceeding with partial results")
                # 超过最大重规划次数，继续合成（用已有结果）

        return OrchestratorState.SYNTHESIZING

    def _sync_result_to_memory_store(self, result: AgentResult) -> None:
        """将 AgentResult 同步到 M4 SharedMemoryStore。

        提取 output 中的关键 claim 作为记忆条目，支持后续语义检索。
        """
        try:
            # 延迟导入避免循环依赖
            from src.memory.long_term import MemoryEntry
            claim_text = str(result.output)[:500]  # 取前 500 字作为 claim
            entry = MemoryEntry(
                entry_id=result.task_id,
                claim=claim_text,
                source=f"task:{result.task_id}",
                confidence=getattr(result, "confidence", 0.5),
                agent_id=result.task_id,
                timestamp=time.time(),
                evidence_type="primary",
                embedding=[],  # SharedMemoryStore.put() 会自动生成 embedding
                topic=self._query[:50],
                metadata={
                    "status": result.status.value,
                    "token_usage": getattr(result, "token_usage", 0),
                },
            )
            self.memory_store.put(entry)
            print(f"[M4] Memory stored: {result.task_id} (claim={claim_text[:60]}...)")
        except Exception as e:
            print(f"[M4] Failed to store memory for {result.task_id}: {e}")

    async def _do_synthesizing(self) -> OrchestratorState:
        """调用 SummarizerAgent 合成研究报告。"""
        # 创建合成任务
        synth_task = SubTask(
            task_id="synthesize_final",
            task_type=TaskType.ANALYZE,  # 使用 ANALYZE 类型，实际由 SummarizerAgent 处理
            description="Synthesize all sub-task results into a final research report.",
            timeout_seconds=300,
        )

        context = {
            "query": self._query,
            "results": self._results,
        }
        if "evidence_snapshot" in self._memory_store:
            context["evidence_snapshot"] = self._memory_store["evidence_snapshot"]
        if "evidence_snapshot_md" in self._memory_store:
            context["evidence_snapshot_md"] = self._memory_store["evidence_snapshot_md"]
        if "evidence_snapshot_text" in self._memory_store:
            context["evidence_snapshot_text"] = self._memory_store["evidence_snapshot_text"]
        if "research_policy" in self._memory_store:
            context["research_policy"] = self._memory_store["research_policy"]

        agent = await self.agent_pool.get_agent(TaskType.ANALYZE)
        # 需要 SummarizerAgent，但 agent_pool 可能返回 ResearcherAgent
        # 这里我们通过类型检查或强制创建 SummarizerAgent
        from ..agents.summarizer import SummarizerAgent
        if not isinstance(agent, SummarizerAgent):
            # 优先使用配置的 summarizer_policy（更大的 max_tokens），fallback 到 agent.policy
            policy = self.summarizer_policy or agent.policy
            agent = SummarizerAgent(
                name="summarizer",
                policy=policy,
                tools=agent.tools,
                min_report_chars=self.summarizer_min_report_chars,
            )

        try:
            result = await asyncio.wait_for(
                agent.run(synth_task, context),
                timeout=synth_task.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.TIMEOUT,
                output="Synthesis timed out",
            )
        except Exception as e:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.FAILED,
                output=f"Synthesis error: {type(e).__name__}: {e}",
            )
        finally:
            await self.agent_pool.release_agent(agent)

        if result.status == AgentStatus.SUCCESS and isinstance(result.output, ResearchReport):
            if hasattr(result.output, "metadata") and isinstance(result.output.metadata, dict):
                result.output.metadata.setdefault("evidence_snapshot", self._memory_store.get("evidence_snapshot", {}))
                result.output.metadata.setdefault("evidence_snapshot_md", self._memory_store.get("evidence_snapshot_md", ""))
                result.output.metadata.setdefault(
                    "evidence_transitions",
                    self._memory_store.get("evidence_transition_trace", []),
                )
                result.output.metadata.setdefault(
                    "evidence_transition_trace",
                    self._memory_store.get("evidence_transition_trace", []),
                )
                result.output.metadata.setdefault("research_policy", self._memory_store.get("research_policy", {}))
            self._memory_store["final_report"] = result.output
        else:
            # 合成失败但已有结果，生成降级报告
            self._memory_store["final_report"] = ResearchReport(
                query=self._query,
                content=str(result.output) if result.output else "Synthesis failed.",
                confidence=0.0,
                num_searches=sum(
                    len([t for t in r.trajectory if t.get("role") == "tool"])
                    for r in self._results
                ),
                metadata={
                    "evidence_snapshot": self._memory_store.get("evidence_snapshot", {}),
                    "evidence_snapshot_md": self._memory_store.get("evidence_snapshot_md", ""),
                    "evidence_transitions": self._memory_store.get("evidence_transition_trace", []),
                    "evidence_transition_trace": self._memory_store.get("evidence_transition_trace", []),
                    "research_policy": self._memory_store.get("research_policy", {}),
                },
            )

        if self._config.enable_adversarial:
            print("[Synthesize] [OK] 报告合成完成，进入对抗优化")
            return OrchestratorState.ADVERSARIAL
        print("[Synthesize] [OK] 报告合成完成")
        return OrchestratorState.DONE

    async def _do_adversarial(self) -> OrchestratorState:
        """M5: Red-Blue 对抗降噪循环。

        调用 AdversarialLoop 对报告进行 challenge-verify 迭代优化。
        仅在报告置信度低于阈值时触发，避免资源浪费。
        """
        report = self._memory_store.get("final_report")
        if report is None:
            return OrchestratorState.DONE

        # 置信度足够高时跳过对抗
        if report.confidence >= 0.8:
            print("[Adversarial] [OK] 报告置信度已达标 (>=0.8)，跳过对抗优化")
            return OrchestratorState.DONE

        if self.adversarial_loop is None:
            print("[Adversarial] AdversarialLoop 未配置，跳过")
            return OrchestratorState.DONE

        try:
            print(f"[Adversarial] [RUN] 启动 Red-Blue 对抗优化 (当前置信度={report.confidence:.2f})")
            optimized_report, history = await self.adversarial_loop.run(report)
            self._memory_store["final_report"] = optimized_report
            self._adversarial_count += len(history)
            print(f"[Adversarial] [OK] 对抗优化完成: {len(history)} 轮, 最终置信度={optimized_report.confidence:.2f}")
        except Exception as e:
            print(f"[Adversarial] [FAIL] 对抗优化失败: {e}，使用原始报告")

        return OrchestratorState.DONE

    async def _do_replanning(self) -> OrchestratorState:
        """触发增量重规划。

        保留 confidence≥0.6 的成功结果，修改失败子问题。
        """
        failed_tasks = []
        for r in self._results:
            if r.status != AgentStatus.SUCCESS:
                st = self._task_map.get(r.task_id)
                if st:
                    failed_tasks.append(st)

        reason = self._build_failure_reason(self._results)
        print(f"[Replan] Round {self._replan_count}/{self._config.max_replan_rounds}. Failed tasks: {[t.task_id for t in failed_tasks]}")

        try:
            new_dag = self.planner.replan(
                query=self._query,
                failed_tasks=failed_tasks,
                existing_results=self._results,
                reason=reason,
            )
            self._dag = new_dag
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                self._task_map = self._rebuild_task_map_from_dag()
            # 清空上一轮结果（保留在 memory 中，新任务可通过 context_keys 引用）
            self._results = []
        except PlanParseError as e:
            print(f"[Replan] Failed: {e}")
            # 重规划失败，如果已有部分成功结果，尝试直接合成
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Replan] Unexpected error: {e}")
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED

        return OrchestratorState.DISPATCHING

    async def _on_done(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.DONE

    async def _on_failed(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.FAILED

    # ------------------------------------------------------------------
    # 决策逻辑
    # ------------------------------------------------------------------

    def _should_replan(self, results: list[AgentResult]) -> bool:
        """判断是否需要重规划。

        策略:
          - 失败率 > 50% 时触发
          - 或存在任何 TIMEOUT 且成功结果不足 30%
          - 或证据状态显示仍需 verify / expand_subquestion 且不确定性较高
        """
        if not results:
            return False
        total = len(results)
        failed = sum(1 for r in results if r.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT))
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)

        failure_rate = failed / total
        if failure_rate > 0.5:
            return True
        if success / total < 0.3 and failed > 0:
            return True

        policy = self._memory_store.get("research_policy")
        if isinstance(policy, dict):
            action = str(policy.get("recommended_action", ""))
            uncertainty = float(policy.get("uncertainty", 1.0) or 1.0)
            open_conflicts = int(policy.get("open_conflicts", 0) or 0)
            if action in {"verify", "expand_subquestion"} and uncertainty >= 0.45:
                return True
            if open_conflicts > 0 and uncertainty >= 0.50 and success > 0:
                return True
        return False

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _is_global_timeout(self) -> bool:
        """检查是否超过全局超时。"""
        elapsed = time.monotonic() - self._start_time
        return elapsed > self._config.global_timeout_seconds

    def _build_memory_context(self) -> str:
        """构建给 planner 的上下文摘要。

        优先使用 M4 SharedMemoryStore 的语义检索（如果已接入），
        否则回退到运行时 dict 遍历。
        """
        # M4: 语义检索相关记忆
        if self.memory_store is not None:
            try:
                ctx = self.memory_store.get_context_for_query(
                    self._query, max_tokens=2000
                )
                if ctx:
                    print(f"[M4] Retrieved {len(ctx)} chars of semantic memory context")
                    return ctx
            except Exception as e:
                print(f"[M4] Semantic memory query failed: {e}, falling back to dict")

        # 回退：运行时 dict 遍历
        parts = []
        if "evidence_snapshot_md" in self._memory_store:
            parts.append(self._memory_store["evidence_snapshot_md"])
        for key, value in self._memory_store.items():
            if key.startswith("result:"):
                continue
            parts.append(f"{key}: {str(value)[:200]}")

        # M3: 如果上下文过长，启用压缩
        if self.compressor is not None and parts:
            total_chars = sum(len(p) for p in parts)
            if total_chars > 6000:  # 约 2000 tokens 的启发式阈值
                try:
                    compressed = self.compressor.compress(
                        texts=parts,
                        query=self._query,
                        system_prompt_tokens=0,
                    )
                    print(f"[M3] Context compressed: {total_chars} → {sum(len(c) for c in compressed)} chars")
                    return "\n".join(compressed)
                except Exception as e:
                    print(f"[M3] Compression failed: {e}, using raw context")

        return "\n".join(parts) if parts else ""

    def _build_task_context(self, subtask: SubTask) -> dict:
        """为单个 SubTask 构建执行上下文。"""
        ctx = dict(self._memory_store)
        ctx["query"] = self._query
        # 注入依赖任务的结果
        for dep_id in subtask.dependencies:
            dep_key = f"result:{dep_id}"
            if dep_key in self._memory_store:
                ctx[f"dep:{dep_id}"] = self._memory_store[dep_key]
        return ctx

    def _build_failure_reason(self, results: list[AgentResult]) -> str:
        """分析失败原因，生成给 replanner 的描述。"""
        reasons = []
        timeout_count = sum(1 for r in results if r.status == AgentStatus.TIMEOUT)
        failed_count = sum(1 for r in results if r.status == AgentStatus.FAILED)
        if timeout_count > 0:
            reasons.append(f"{timeout_count} tasks timed out (may need simpler queries or longer timeout)")
        if failed_count > 0:
            reasons.append(f"{failed_count} tasks failed with errors")
        policy = self._memory_store.get("research_policy")
        if isinstance(policy, dict):
            missing_terms = policy.get("missing_terms") or []
            if missing_terms:
                reasons.append(f"evidence gaps: {', '.join(str(t) for t in missing_terms[:5])}")
            if policy.get("open_conflicts"):
                reasons.append(f"open conflicts: {policy.get('open_conflicts')}")
        return "; ".join(reasons) if reasons else "Unknown failure"

    def _summarize_evidence_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """压缩 snapshot，用于 step-wise transition trace。"""
        claims = snapshot.get("claims", []) if isinstance(snapshot, dict) else []
        supports = snapshot.get("support_edges", []) if isinstance(snapshot, dict) else []
        contradictions = snapshot.get("contradiction_edges", []) if isinstance(snapshot, dict) else []
        conflicts = snapshot.get("conflicts", []) if isinstance(snapshot, dict) else []
        trusts = snapshot.get("source_trusts", []) if isinstance(snapshot, dict) else []
        candidate_source = snapshot.get("candidate_actions", []) if isinstance(snapshot, dict) else []
        if not isinstance(candidate_source, list):
            candidate_source = []

        trust_scores = [
            float(item.get("trust_score", 0.0) or 0.0)
            for item in trusts
            if isinstance(item, dict)
        ]
        candidate_actions = []
        for item in candidate_source[:3]:
            if isinstance(item, dict):
                candidate_actions.append(
                    {
                        "name": str(item.get("name", "")),
                        "score": round(float(item.get("score", 0.0) or 0.0), 4),
                    }
                )

        return {
            "recommended_action": str(snapshot.get("recommended_action", "")) if isinstance(snapshot, dict) else "",
            "uncertainty": round(float(snapshot.get("uncertainty", 1.0) or 1.0), 4) if isinstance(snapshot, dict) else 1.0,
            "coverage_ratio": round(float(snapshot.get("coverage_ratio", 0.0) or 0.0), 4) if isinstance(snapshot, dict) else 0.0,
            "evidence_strength": round(float(snapshot.get("evidence_strength", 0.0) or 0.0), 4) if isinstance(snapshot, dict) else 0.0,
            "claim_count": len(claims),
            "support_edge_count": len(supports),
            "contradiction_edge_count": len(contradictions),
            "open_conflicts": len(conflicts),
            "avg_source_trust": round(sum(trust_scores) / len(trust_scores), 4) if trust_scores else 0.0,
            "missing_terms": list(snapshot.get("missing_terms", [])[:6]) if isinstance(snapshot, dict) else [],
            "candidate_actions": candidate_actions,
        }

    def _build_evidence_transition_record(
        self,
        *,
        step: int,
        task_id: str,
        task_type: str,
        status: str,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """构建 step-wise evidence transition，用于训练和评测。"""
        before_uncertainty = float(before.get("uncertainty", 1.0) or 1.0)
        after_uncertainty = float(after.get("uncertainty", before_uncertainty) or before_uncertainty)
        before_coverage = float(before.get("coverage_ratio", 0.0) or 0.0)
        after_coverage = float(after.get("coverage_ratio", before_coverage) or before_coverage)
        before_conflicts = int(before.get("open_conflicts", 0) or 0)
        after_conflicts = int(after.get("open_conflicts", before_conflicts) or before_conflicts)
        before_strength = float(before.get("evidence_strength", 0.0) or 0.0)
        after_strength = float(after.get("evidence_strength", before_strength) or before_strength)
        before_claims = int(before.get("claim_count", 0) or 0)
        after_claims = int(after.get("claim_count", before_claims) or before_claims)

        return {
            "step": step,
            "task_id": task_id,
            "task_type": task_type,
            "status": status,
            "label_action": str(before.get("recommended_action", "") or ""),
            "policy_state_before": before,
            "policy_state_after": after,
            "before": before,
            "after": after,
            "delta": {
                "uncertainty": round(after_uncertainty - before_uncertainty, 4),
                "coverage_ratio": round(after_coverage - before_coverage, 4),
                "open_conflicts": after_conflicts - before_conflicts,
                "evidence_strength": round(after_strength - before_strength, 4),
                "claim_count": after_claims - before_claims,
            },
        }

    def _rebuild_task_map_from_dag(self) -> dict[str, SubTask]:
        """从 DAG 重建 task_map（当缺少原始 SubTask 信息时使用占位符）。

        实际场景中，planner 应返回完整的 SubTask 列表；
        这里作为降级：为 DAG 中每个节点创建默认 SubTask。
        """
        if self._dag is None:
            return {}

        task_map: dict[str, SubTask] = {}
        for node_id in self._dag:
            deps = self._dag.get_dependencies(node_id)
            if node_id not in self._task_map:
                # 新建占位 SubTask
                task_map[node_id] = SubTask(
                    task_id=node_id,
                    task_type=TaskType.SEARCH,
                    description=f"Auto-generated task for {node_id}",
                    dependencies=deps,
                )
            else:
                # 保留已有信息，更新依赖
                old = self._task_map[node_id]
                task_map[node_id] = SubTask(
                    task_id=old.task_id,
                    task_type=old.task_type,
                    description=old.description,
                    dependencies=deps,
                    context_keys=old.context_keys,
                    timeout_seconds=old.timeout_seconds,
                    priority=old.priority,
                    expected_type=old.expected_type,
                    search_hints=old.search_hints,
                )
        return task_map
