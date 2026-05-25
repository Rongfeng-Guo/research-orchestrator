#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/runner.py
================================================================================
DeepResearch Agent 核心运行逻辑。

本模块包含初始化所有模块和执行完整研究流程的核心函数，
供 scripts/ 和 evaluation/ 统一调用，避免 evaluation/ 反向依赖 scripts/。

对外接口:
    - load_config(config_path) -> dict
    - initialize_modules(config) -> dict
    - run_research(query, config, modules, return_report=False) -> str | (str, ResearchReport)
    - save_report(report, query, output_dir) -> str
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 将项目根目录加入 sys.path，确保 src 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO") -> None:
    """配置全局日志格式与级别。"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """
    加载 YAML 配置文件。

    若未指定路径，默认加载 configs/default.yaml。
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ---------------------------------------------------------------------------
# Research Policy
# ---------------------------------------------------------------------------
def create_evidence_policy(config: dict):
    """Create the evidence-aware policy used by memory/orchestrator state updates."""
    logger = logging.getLogger("runner")

    from src.evidence import EvidenceAwareResearchPolicy, LearnedEvidenceAwareResearchPolicy

    policy_cfg = config.get("evidence_policy", {}) or {}
    mode = str(policy_cfg.get("mode", "heuristic") or "heuristic").strip().lower()
    stop_threshold = float(policy_cfg.get("stop_threshold", 0.32) or 0.32)
    search_threshold = float(policy_cfg.get("search_threshold", 0.22) or 0.22)
    expand_threshold = float(policy_cfg.get("expand_threshold", 0.45) or 0.45)

    base_kwargs = {
        "stop_threshold": stop_threshold,
        "search_threshold": search_threshold,
        "expand_threshold": expand_threshold,
    }

    if mode == "learned":
        model_path = policy_cfg.get("model_path")
        blend_weight = float(policy_cfg.get("blend_weight", 0.7) or 0.7)
        resolved_path: Path | None = None
        if model_path:
            raw_path = Path(str(model_path))
            resolved_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path

        if resolved_path and resolved_path.exists():
            logger.info("[Policy] Loaded learned evidence policy from %s", resolved_path)
            return LearnedEvidenceAwareResearchPolicy(
                model_path=resolved_path,
                blend_weight=blend_weight,
                **base_kwargs,
            )

        logger.warning(
            "[Policy] evidence_policy.mode=learned but model_path is missing or not found: %s. "
            "Falling back to heuristic policy.",
            resolved_path or model_path or "",
        )

    return EvidenceAwareResearchPolicy(**base_kwargs)


# ---------------------------------------------------------------------------
# Search Policy
# ---------------------------------------------------------------------------
def resolve_search_policy_mode(config: dict) -> str:
    """Resolve search policy mode with backward compatibility."""
    search_policy_cfg = config.get("search_policy", {}) or {}
    raw_mode = str(search_policy_cfg.get("mode", "") or "").strip().lower()
    if raw_mode in {"off", "heuristic", "learned"}:
        return raw_mode

    if raw_mode:
        logging.getLogger("runner").warning(
            "[SearchPolicy] 未知 mode=%s，回退到兼容模式推断。",
            raw_mode,
        )

    if not search_policy_cfg.get("enabled", False):
        return "off"
    if search_policy_cfg.get("heuristic_fallback", True) is False:
        return "learned"
    return "heuristic"


def create_search_policy(config: dict):
    """Create runtime search policy according to explicit mode."""
    logger = logging.getLogger("runner")
    mode = resolve_search_policy_mode(config)
    if mode == "off":
        return None

    from src.search_policy import LearnedSearchPolicy

    search_policy_cfg = config.get("search_policy", {}) or {}
    model_path = None
    if mode == "learned":
        model_path = search_policy_cfg.get("model_path", "artifacts/search_policy.json")

    heuristic_fallback = True if mode == "heuristic" else search_policy_cfg.get("heuristic_fallback", True)

    try:
        search_policy = LearnedSearchPolicy(
            model_path=model_path,
            continue_threshold=search_policy_cfg.get("continue_threshold", 0.58),
            stop_threshold=search_policy_cfg.get("stop_threshold", 0.66),
            heuristic_fallback=heuristic_fallback,
        )
        if not search_policy.is_available:
            logger.warning("[SearchPolicy] 已启用但不可用，已跳过 runtime policy: %s", model_path or "<heuristic>")
            return None
        if mode == "learned" and not search_policy.is_trained:
            logger.warning(
                "[SearchPolicy] mode=learned 但模型不可用，已回退到 heuristic: %s",
                model_path,
            )
        active_mode = "learned" if search_policy.is_trained else "heuristic"
        logger.info("[SearchPolicy] 已启用 %s runtime policy (%s)", active_mode, model_path or "no-model")
        return search_policy
    except Exception as exc:
        logger.warning("[SearchPolicy] 初始化失败，已跳过 runtime policy: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 工具工厂
# ---------------------------------------------------------------------------
def _create_tools_factory(config: dict):
    """创建工具工厂函数，返回 Agent 可用的工具列表。"""
    tools_cfg = config.get("tools", {})
    search_cfg = tools_cfg.get("web_search", {})
    mock_mode = search_cfg.get("mock_mode", True)

    # 如果配置里要求真实搜索，但当前没有对应 Key，则自动切换到 mock。
    # 这样单 LLM Key 也能完成完整链路验证，不会在工具初始化/调用阶段失败。
    if not mock_mode:
        from src.utils.env_config import get_env

        search_backend = get_env("SEARCH_BACKEND", "serpapi").lower().strip()
        required_key_by_backend = {
            "serpapi": "SERPAPI_KEY",
            "bing": "BING_SEARCH_KEY",
            "bocha": "BOCHA_API_KEY",
            "metaso": "METASO_API_KEY",
        }
        required_key = required_key_by_backend.get(search_backend)
        if required_key and not get_env(required_key):
            logger = logging.getLogger("runner")
            logger.warning(
                "搜索后端 %s 缺少 %s，自动切换为 mock_mode=True 以保证可运行。",
                search_backend,
                required_key,
            )
            mock_mode = True

    from src.tools import (
        WebSearchTool,
        MockWebSearchTool,
        ArxivReaderTool,
        BrowserTool,
        MockBrowserTool,
        FileReaderTool,
        CodeSandboxTool,
        CalculatorTool,
        NotepadTool,
    )

    tools = {}

    # 1. web_search
    if mock_mode:
        tools["web_search"] = MockWebSearchTool()
    else:
        tools["web_search"] = WebSearchTool()

    # 2. browser
    if mock_mode:
        tools["browser"] = MockBrowserTool()
    else:
        tools["browser"] = BrowserTool()

    # 3. arxiv_reader
    tools["arxiv_reader"] = ArxivReaderTool(use_mock=mock_mode)

    # 4. file_reader（不限制目录）
    tools["file_reader"] = FileReaderTool(allowed_base_dir=None)

    # 5. code_sandbox
    tools["code_sandbox"] = CodeSandboxTool(use_mock=mock_mode)

    # 6. calculator
    tools["calculator"] = CalculatorTool()

    # 7. notepad
    tools["notepad"] = NotepadTool()

    # 返回列表形式（AgentPool 和 Agent 构造函数需要 list）
    return list(tools.values())


# ---------------------------------------------------------------------------
# 模块初始化
# ---------------------------------------------------------------------------
def initialize_modules(config: dict, session_id: str = "") -> dict[str, Any]:
    """
    根据配置初始化所有核心模块。

    Args:
        config: 全局配置字典。
        session_id: 会话 ID，用于 memory store 的 session 隔离。

    返回一个包含各模块实例的字典。
    """
    logger = logging.getLogger("runner")
    logger.info("正在初始化核心模块...")

    modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 多后端 LLM 初始化（从 .env + configs/default.yaml 读取配置）
    # ------------------------------------------------------------------
    from src.models.model_router import ModelRouter

    model_cfg = config.get("model", {})
    default_backend = model_cfg.get("backend", "vllm")
    backend_mapping = model_cfg.get("backend_mapping", {})
    backend_sampling = model_cfg.get("backend_sampling", {})

    # 辅助函数：根据模块名获取采样参数覆盖
    def _get_sampling_kwargs(module_name: str, backend_name: str) -> dict:
        """合并后端全局默认 + 模块级覆盖参数。"""
        kwargs = {}
        # 1. 后端全局默认
        if backend_name in backend_sampling:
            kwargs.update(backend_sampling[backend_name])
        # 2. 模块级覆盖（优先级更高）
        module_overrides = backend_sampling.get("modules", {}).get(module_name, {})
        kwargs.update(module_overrides)
        return kwargs

    # 默认后端（所有模块共用）
    default_kwargs = _get_sampling_kwargs("default", default_backend)
    default_policy = ModelRouter.create_backend(default_backend, **default_kwargs)
    modules["default_policy"] = default_policy
    logger.info(f"[LLM] 默认后端已加载: {default_backend} ({default_kwargs})")

    # 多后端分工：不同模块用不同后端 + 不同采样参数
    for module_name, backend_name in backend_mapping.items():
        kwargs = _get_sampling_kwargs(module_name, backend_name)
        try:
            modules[f"{module_name}_policy"] = ModelRouter.create_backend(backend_name, **kwargs)
            logger.info(f"[LLM] {module_name} → 后端={backend_name}, 采样={kwargs}")
        except ValueError as e:
            # 允许只配置一个后端时继续运行，缺失的角色回退到默认后端。
            modules[f"{module_name}_policy"] = default_policy
            logger.warning(
                "[LLM] %s 后端=%s 未配置，已回退到默认后端=%s。原因: %s",
                module_name,
                backend_name,
                default_backend,
                e,
            )

    # 若未配置分工，所有模块回退到 default_policy
    # ------------------------------------------------------------------

    # M2: Adaptive Planner（Orchestrator 依赖 Planner，先初始化）
    from src.planner.planner import Planner
    from src.planner.budget_tracker import BudgetTracker

    planner_policy = modules.get("planner_policy", default_policy)
    budget_tracker = BudgetTracker()
    planner_cfg = config.get("planner", {})
    planner = Planner(
        policy=planner_policy,
        budget_tracker=budget_tracker,
        min_subtasks=planner_cfg.get("min_subtasks", 3),
        max_subtasks=planner_cfg.get("max_subtasks", 8),
    )
    modules["planner"] = planner
    logger.info("[M2] Planner 模块已初始化")

    # M3: Context Compressor
    from src.compressor.compressor import ContextCompressor

    compressor_policy = modules.get("compressor_policy", default_policy)
    compressor_cfg = config.get("compressor", {})
    compressor = ContextCompressor(
        llm_policy=compressor_policy,
        budget=compressor_cfg.get("max_context_length", 16000),
        output_reserve=compressor_cfg.get("output_reserve_tokens", 2048),
    )
    modules["compressor"] = compressor
    logger.info("[M3] Compressor 模块已初始化")

    # M4: Shared Memory Store
    from src.memory.memory_store import SharedMemoryStore

    memory_cfg = config.get("memory", {})
    evidence_policy = create_evidence_policy(config)
    memory_store = SharedMemoryStore(
        db_path=memory_cfg.get("db_path", "data/memory.db"),
        session_id=session_id,
        evidence_policy=evidence_policy,
    )
    modules["memory_store"] = memory_store
    logger.info(f"[M4] Memory Store 模块已初始化 (session={session_id})")

    # Tools（真实工具或 Mock 工具）
    tools_list = _create_tools_factory(config)
    modules["tools"] = tools_list
    logger.info(f"Tools 模块已初始化（共 {len(tools_list)} 个工具）")

    # Search policy runtime（默认关闭）
    search_policy = create_search_policy(config)
    modules["search_policy"] = search_policy

    # M5: Red-Blue Adversarial Loop（先创建，再注入 Orchestrator）
    from src.adversarial.loop import AdversarialLoop
    from src.adversarial.red_agent import RedAgent
    from src.adversarial.blue_agent import BlueAgent

    red_policy = modules.get("red_agent_policy", default_policy)
    blue_policy = modules.get("blue_agent_policy", default_policy)
    adversarial_cfg = config.get("adversarial", {})

    red_agent = RedAgent(policy=red_policy)
    blue_agent = BlueAgent(policy=blue_policy, tools=tools_list)
    adversarial_loop = AdversarialLoop(
        red_agent=red_agent,
        blue_agent=blue_agent,
        policy=modules.get("judge_policy", default_policy),
        max_rounds=adversarial_cfg.get("max_rounds", 3),
        score_threshold=adversarial_cfg.get("score_threshold", 8.0),
        delta_threshold=adversarial_cfg.get("delta_threshold", 0.3),
    )
    modules["adversarial"] = adversarial_loop
    logger.info("[M5] Adversarial 模块已初始化")

    # M1: Multi-Agent Orchestrator
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.agent_pool import AgentPool

    researcher_cfg = config.get("researcher", {})
    max_tool_calls_per_task = researcher_cfg.get(
        "max_tool_calls_per_task",
        planner_cfg.get("max_search_rounds_per_subagent", None),
    )
    agent_pool = AgentPool(
        policy_factory=lambda: modules.get("solver_policy", default_policy),
        tools_factory=lambda: list(modules["tools"]),
        search_policy=search_policy,
        max_turns=researcher_cfg.get("max_turns", 10),
        max_tool_calls_per_turn=researcher_cfg.get("max_tool_calls_per_turn", 2),
        max_tool_calls_per_task=max_tool_calls_per_task,
        max_idle=3,
    )
    modules["agent_pool"] = agent_pool

    orchestrator = Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        budget_tracker=budget_tracker,
        compressor=compressor,
        adversarial_loop=adversarial_loop,
        memory_store=memory_store,
        summarizer_policy=modules.get("summarizer_policy", default_policy),
        summarizer_min_report_chars=config.get("summarizer", {}).get("min_report_chars", 3000),
    )
    modules["orchestrator"] = orchestrator
    logger.info("[M1] Orchestrator 模块已初始化")

    # M6: Self-Evolution Engine（预留，默认禁用）
    if config.get("evolution", {}).get("enabled", False):
        logger.info("[M6] Evolution 模块已启用（预留接口）")
    else:
        logger.info("[M6] Evolution 模块已禁用")

    return modules


# ---------------------------------------------------------------------------
# 研究流程主函数
# ---------------------------------------------------------------------------
async def run_research(
    query: str,
    config: dict,
    modules: dict[str, Any],
    return_report: bool = False,
) -> str | tuple[str, ResearchReport]:
    """
    执行完整的研究流程。

    流程：
        1. Orchestrator 调用 Planner 拆解问题为子任务 DAG
        2. Orchestrator 调度 AgentPool 中的子 Agent 并行/串行执行
        3. 子 Agent 调用 Tools 检索信息并生成子报告
        4. Compressor 管理长上下文
        5. Memory 存储中间结果
        6. Adversarial Loop 对报告进行多轮对抗优化（若启用）
        7. 输出最终研究报告

    Args:
        query: 用户输入的研究问题。
        config: 全局配置字典。
        modules: 已初始化的模块实例字典。

    Args:
        return_report: 若为 True，则同时返回 `ResearchReport` 对象，便于批量评测和训练数据导出。

    Returns:
        默认返回最终研究报告文本（Markdown 格式）。
        若 `return_report=True`，则返回 `(final_report_text, report)`。
    """
    import asyncio

    logger = logging.getLogger("runner")
    logger.info(f"开始研究，查询: {query[:80]}...")

    start_time = time.time()

    # Step 1-3: Orchestrator 内部完成规划、调度、收集、合成
    orchestrator = modules["orchestrator"]
    from src.orchestrator.schemas import RunConfig

    run_cfg = RunConfig(
        max_concurrent=config.get("orchestrator", {}).get("max_concurrent", 5),
        global_timeout_seconds=config.get("orchestrator", {}).get("global_timeout_seconds", 600),
        max_replan_rounds=config.get("orchestrator", {}).get("max_replan_rounds", 3),
        max_sub_questions=config.get("orchestrator", {}).get("max_sub_questions", 8),
        enable_adversarial=config.get("adversarial", {}).get("enabled", True),
        enable_evolution=config.get("evolution", {}).get("enabled", False),
    )

    report = await orchestrator.run(query, config=run_cfg)
    logger.info(
        f"[Orchestrator] 报告生成完成 | 置信度={report.confidence:.2f} | "
        f"搜索轮数={report.num_searches} | 重规划={report.num_replan} | 对抗轮数={report.adversarial_rounds}"
    )

    # Step 4/5: 进化优化（如启用且已训练）
    if run_cfg.enable_evolution:
        logger.info("[Evolution] 进化优化已启用（预留接口）")
    else:
        logger.info("[Evolution] 进化优化已跳过")

    # 关闭 WebSearchTool 连接池
    from src.tools.web_search import WebSearchTool
    await WebSearchTool.close_session()

    elapsed = time.time() - start_time
    logger.info(f"研究完成，耗时: {elapsed:.2f} 秒")

    # 组装最终输出
    final_report = _format_report(report, elapsed)
    if return_report:
        return final_report, report
    return final_report


def _format_report(report, elapsed: float) -> str:
    """将 ResearchReport 格式化为 Markdown 文本。"""
    content = report.content or ""

    # 统一置信度：如果正文中有 LLM 自评的"整体置信度"，替换为实际计算值，避免不一致
    content = re.sub(
        r"(整体置信度|Overall Confidence|置信度|总体信心评分)[:：]\s*0?\.\d+",
        f"\\1: {report.confidence:.2f}",
        content,
        flags=re.I,
    )

    lines = [
        f"# 研究报告：{report.query}",
        "",
        "---",
        "",
        content,
        "",
        "---",
        "",
        "## 元信息",
        "",
        f"- **置信度**: {report.confidence:.2f}",
        f"- **搜索轮数**: {report.num_searches}",
        f"- **重规划次数**: {report.num_replan}",
        f"- **对抗轮数**: {report.adversarial_rounds}",
        f"- **总耗时**: {elapsed:.2f} 秒",
        "",
    ]

    has_reference_section = bool(
        re.search(
            r"(?mi)^\s{0,3}"
            r"(?:#{1,6}\s*)?"
            r"(?:\*\*|__)?\s*"
            r"(参考来源|参考文献|References|Sources)"
            r"\s*(?:\*\*|__)?\s*[:：]?\s*$",
            content,
        )
    )
    if report.sources and not has_reference_section:
        lines.append("## 参考来源")
        lines.append("")
        for i, src in enumerate(report.sources, 1):
            title = src.get("title", "未知标题")
            url = src.get("url", "")
            snippet = src.get("snippet", "")
            lines.append(f"{i}. [{title}]({url}) — {snippet}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 报告保存
# ---------------------------------------------------------------------------
def save_report(report: str, query: str, output_dir: str = "outputs/reports") -> str:
    """
    将研究报告保存到文件。

    文件名格式：report_YYYYMMDD_HHMMSS_<query前20字>.md
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    filename = f"report_{timestamp}_{safe_query}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath
