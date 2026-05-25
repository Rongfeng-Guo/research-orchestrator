# DeepResearch Agent 改进总结

日期：2026-05-12

这份文档总结本轮已经完成的所有关键改进，目标不是写成 changelog，而是把“这份仓库现在比原来强在哪里、已经补了哪些研究基础设施、后面还差什么”一次说清楚。

---

## 1. 总体结论

这一轮改进可以分成四层：

1. 把项目先低成本跑通，确认主链是活的。
2. 把“下一步做什么研究”从 brainstorm 收敛成明确主线。
3. 把运行轨迹从自由文本变成结构化搜索轨迹，为 agentic RL 做准备。
4. 把离线训练需要的 cache、评测、search policy metrics、reward shaping 接起来。

如果用一句话概括：

> 这套代码已经从“能跑的 deep research demo”推进成了“可以开始做 AI Search / Agentic RL 算法研究的平台雏形”。

---

## 2. 跑通与可运行性改进

### 2.1 新增低成本 smoke 配置

新增文件：

- `configs/aliyun_smoke.yaml`

作用：

- 用阿里 DashScope OpenAI-compatible 接口低成本跑通主链
- 降低任务数和报告长度，减少 token 消耗
- 默认使用 `mock_mode=true`，避免真实搜索 API 成本

关键配置：

- `model.backend = openai`
- `planner.min_subtasks = 2`
- `planner.max_subtasks = 3`
- `summarizer.min_report_chars = 800`
- `tools.web_search.mock_mode = true`
- `adversarial.enabled = false`

### 2.2 单次运行脚本兼容 Windows UTF-8 输出

涉及文件：

- `scripts/run_single.py`

改进点：

- 启动时显式把 `stdout / stderr` 切成 UTF-8
- 减少 Windows 环境下中文日志、Markdown 报告和模型输出的乱码风险

### 2.3 规划器和总结器增加更细粒度控制

涉及文件：

- `src/planner/planner.py`
- `src/core/runner.py`
- `src/orchestrator/orchestrator.py`
- `src/agents/summarizer.py`

改进点：

- Planner 支持 `min_subtasks / max_subtasks`
- Summarizer 支持 `min_report_chars`
- Runner 和 Orchestrator 把这些配置贯通到实际模块初始化和运行流程中

实际效果：

- smoke run 不再默认拆太多子任务
- 输出报告长度可控
- 更适合低预算调试和后续系统性实验

### 2.4 项目已完成一次真实 smoke run

已验证链路：

- `ModelRouter -> VLLMPolicy -> Orchestrator -> Planner -> ResearcherAgent -> SummarizerAgent -> Memory Store`

文档记录：

- `docs/CODEX_RUNBOOK.md`

已验证的结论：

- 主链闭环可运行
- SQLite memory 可写入
- 低成本配置下可从 query 跑到最终 Markdown 报告

---

## 3. 运行与代码认知文档补齐

### 3.1 运行手册

新增文件：

- `docs/CODEX_RUNBOOK.md`

内容覆盖：

- 实际如何跑通项目
- 本地环境与依赖判断
- `.env.local` 的运行思路
- smoke 配置为何这样设计
- 各核心模块职责
- 当前系统的已验证能力和已知限制

### 3.2 研究方向提案

新增文件：

- `idea.md`

核心判断：

- 不建议下一步继续堆更多 agent 或纯 UI
- 最值得押注的主线是：
  - `Budgeted Query Policy RL + Retriever Router`
  - 再往上接 `Summary-State RL`
  - 再接 `Evolving Rubrics Reward`

文档内容：

- 2024-2026 相关文献和系统调研
- 与当前仓库结构的匹配度
- 90 天研究路线
- 代表性工作和外链

### 3.3 正式研究执行计划

新增文件：

- `research_plan.md`

作用：

- 把 `idea.md` 的主线收敛成 4-8 周内可执行计划
- 明确代码落点、实验矩阵、验收标准、优先级和阶段目标

文档重点：

- Stage A：结构化搜索轨迹
- Stage B：search reward
- Stage C：Query Policy RL baseline
- Stage D：Retriever Router 和 Stop Policy
- Stage E：benchmark / ablation / 图表

---

## 4. 结构化搜索轨迹改造

这是本轮最关键的代码能力增强，因为它把后续 agentic RL 的“训练对象”准备出来了。

### 4.1 AgentResult 增加结构化动作日志

涉及文件：

- `src/orchestrator/schemas.py`

新增能力：

- `AgentResult.action_log`
- `AgentResult.metadata`

意义：

- 以前只有自由文本 `trajectory`
- 现在每个子任务结果不仅有 raw trajectory，还有 step-level action records 和聚合元信息

### 4.2 ResearcherAgent 记录 step-level search actions

涉及文件：

- `src/agents/researcher.py`

新增记录项包括：

- `action_type`
- `tool_name`
- `tool_args`
- `query_text`
- `backend`
- `latency_ms`
- `result_count`
- `top_urls`
- `estimated_token_cost`
- `error`
- `error_type`
- `stop_reason`

覆盖的行为类型：

- `assistant_response`
- `tool_call`
- `stop`
- `stop_signal`
- `direct_analysis`

实际效果：

- 可以从一次子任务运行中恢复“搜了什么、点了什么、为什么停”
- 为后续 router、stop policy、offline reward、cached search 提供原始监督数据

### 4.3 Summarizer 聚合全局 policy trace

涉及文件：

- `src/agents/summarizer.py`

新增聚合字段写入 `ResearchReport.metadata`：

- `policy_trace`
- `search_cost`
- `route_stats`

其中 `search_cost` 包括：

- `tool_calls`
- `search_calls`
- `browser_calls`
- `estimated_token_cost`
- `estimated_tool_cost`
- `assistant_turns`

其中 `route_stats` 包括：

- `tool_counts`
- `queries_used`
- `search_backends`
- `domains_seen`
- `top_urls`
- `stop_reasons`

意义：

- 最终报告对象本身已经携带搜索过程摘要
- 后续评测和训练不需要额外回头解析中间日志

### 4.4 Collector 优先消费结构化轨迹

涉及文件：

- `src/evolution/collector.py`

改进点：

- 优先使用 `report.metadata.policy_trace`
- 同时携带：
  - `search_cost`
  - `route_stats`
  - `structured_actions`
  - `action_summary`
  - `process_reward_trace`

意义：

- veRL / offline dataset 适配层不再依赖松散轨迹
- 训练样本已经有结构化动作统计

---

## 5. 搜索工具输出标准化

涉及文件：

- `src/tools/web_search.py`

### 5.1 统一 mock 和真实后端的返回结构

在保持原有字段兼容的前提下，统一补充：

- 顶层字段：
  - `backend`
  - `source`
  - `cost`
  - `error_type`
- 每条结果字段：
  - `rank`
  - `domain`
  - `snippet_length`
  - `backend`
  - `source`

### 5.2 新增轻量成本视图

`cost` 中包含：

- `request_count`
- `requested_top_n`
- `returned_results`
- `latency_ms`

意义：

- 后续做 search budget / route cost / stop policy 时不需要再临时推断

### 5.3 新增错误分类

当前会把搜索错误粗分为：

- `auth_error`
- `rate_limit`
- `backend_server_error`
- `network_error`
- `unknown_error`

意义：

- 为 reward penalty 和鲁棒性分析提供基础标签

---

## 6. 离线 Search Cache 数据集构建

涉及文件：

- `scripts/build_search_cache.py`

### 6.1 新增最小可用 search cache 构建脚本

支持三种输入方式：

- `--query`
- `--queries_file`
- `ResearchBench` 采样

输出格式：

- JSONL
- manifest JSON

每条记录包含：

- 原始 query 和 benchmark 元信息
- 最终报告内容和来源
- `policy_trace`
- `search_cost`
- `route_stats`
- `process_reward_trace`
- task-level serialized results
- 可选 benchmark 评测

### 6.2 Search cache 已经适配当前结构化轨迹

脚本会把缓存样本组织成后续可直接训练/分析的离线记录，而不是仅仅保存报告文本。

意义：

- 为 offline policy learning、bandit/ranking、reward ablation 提供数据来源
- 后续不需要每次都重新消耗 LLM/搜索 API

---

## 7. Search Policy 评测指标

涉及文件：

- `evaluation/metrics/search_policy.py`
- `evaluation/metrics/__init__.py`
- `evaluation/benchmarks/research_bench.py`

### 7.1 新增 SearchPolicyMetrics

基于 `policy_trace / search_cost / route_stats` 计算：

- `successful_tool_call_ratio`
- `query_diversity`
- `source_diversity`
- `citation_grounding`
- `stop_efficiency`
- `budget_efficiency`
- `search_policy_score`

### 7.2 接入 ResearchBench

`ResearchBench.evaluate_report()` 现在除了原有：

- factual
- citation
- logic
- comprehensiveness
- evidence metrics

还会输出：

- `search_policy_metrics`

并且在有结构化 trace 时，会把：

- `search_policy_score`

纳入 `metrics`，作为综合指标的一部分。

### 7.3 batch_evaluate 支持 search policy 汇总

`ResearchBench.batch_evaluate()` 现在会额外返回：

- `average_search_policy_metrics`
- `by_domain_search_policy_metrics`

意义：

- 后续可以直接比较不同策略、不同 reward、不同 router 在搜索行为层面的差异

---

## 8. 离线 Reward Shaping

涉及文件：

- `evaluation/metrics/search_policy_reward.py`
- `evaluation/metrics/__init__.py`
- `scripts/build_search_cache.py`

### 8.1 新增 SearchPolicyReward

这个模块把以下信号组合成可训练 reward：

- final answer quality proxy
- `search_policy_score`
- `citation_grounding`
- `budget_efficiency`
- `stop_efficiency`
- `process_reward_trace`
- confidence proxy

输出：

- `reward_raw`
- `reward`

其中：

- `reward_raw` 在 `[0, 1]`
- `reward` 映射到 `[-1, 1]`

### 8.2 Search cache 样本自动带 reward

`build_search_cache.py` 现在每条成功记录会新增：

- `reward_breakdown`
- `reward`

manifest 还会汇总：

- `average_reward`
- `average_reward_raw`
- `average_quality_score`
- `average_process_score`

意义：

- 离线数据集已经不只是“记录了轨迹”
- 而是“记录了轨迹 + 记录了训练信号”

---

## 9. 验证与测试

### 9.1 本地无 API 验证

已经完成的本地验证包括：

- `python -m py_compile ...`
  - 多次覆盖改动文件的语法检查
- `LOCAL_CHECK_OK`
  - 验证 `MockWebSearchTool`、`SummarizerAgent`、`TrajectoryCollector`
- `RESEARCHER_LOCAL_CHECK_OK`
  - 验证 `ResearcherAgent.run()` 会真实生成 `action_log / search_cost / route_stats`
- `SEARCH_CACHE_LOCAL_CHECK_OK`
  - 验证 `build_search_cache.py` 能在 fake runner 下写出 JSONL
- `SEARCH_POLICY_EVAL_OK`
  - 验证 `ResearchBench.evaluate_report()` / `batch_evaluate()` 的新 search policy 指标
- `SEARCH_POLICY_REWARD_OK`
  - 验证 search cache 样本会写出 `reward_breakdown / reward`

### 9.2 真实链路 smoke run

已实际跑过的核心命令：

```bash
python scripts/run_single.py --query "Transformer architecture" --config configs/aliyun_smoke.yaml --output_dir outputs/smoke --session_id aliyun_smoke
```

已验证：

- 主链可跑
- 最终报告能正常落盘
- 低成本配置可用

---

## 10. 本轮新增/更新的重要文件

### 运行与研究文档

- `docs/CODEX_RUNBOOK.md`
- `idea.md`
- `research_plan.md`

### 低成本运行配置

- `configs/aliyun_smoke.yaml`

### 结构化轨迹与搜索输出

- `src/orchestrator/schemas.py`
- `src/agents/researcher.py`
- `src/agents/summarizer.py`
- `src/evolution/collector.py`
- `src/tools/web_search.py`

### 离线数据与评测

- `scripts/build_search_cache.py`
- `evaluation/metrics/search_policy.py`
- `evaluation/metrics/search_policy_reward.py`
- `evaluation/metrics/__init__.py`
- `evaluation/benchmarks/research_bench.py`

---

## 11. 这轮改进之后，仓库能力有什么变化

改进前更像：

- 能跑通的 deep research system
- 有 planner、memory、adversarial、evolution 外壳
- 但搜索过程对训练来说仍然太粗

改进后更像：

- 有真实运行手册
- 有研究方向文档
- 有正式实验计划
- 有结构化搜索动作日志
- 有离线 search cache 数据集构建能力
- 有 search policy 评测指标
- 有 search policy reward shaping

也就是说，现在已经具备以下研究前置条件：

1. 能记录搜索行为
2. 能缓存搜索样本
3. 能评估搜索行为
4. 能给搜索行为打 reward

这四件事做完，后面再做 `Query Policy RL + Retriever Router` 就不是空谈了。

---

## 12. 当前还没做完的关键缺口

虽然基础设施已经补了很多，但下面这些还没真正完成：

### 12.1 还没有最小 offline trainer

当前还缺：

- `src/search_policy/train_policy.py`

需要做的事：

- 从 search cache 读取样本
- 抽 `(state, action, reward)` 或 route/stop label
- 先做轻量 baseline

### 12.2 还没有 learned router / learned stop policy

当前有：

- 结构化数据
- 指标
- reward

但还没有真正学出来的：

- route selector
- stop policy
- query rewrite policy

### 12.3 veRL/GRPO 仍未接到完整训练闭环

`src/evolution/engine.py` 仍然更像编排框架，而不是已经打通的 RL trainer。

---

## 13. 建议的下一步

如果继续推进，最合理的顺序是：

1. 实现最小 offline trainer
   - 先不碰重型 online RL
   - 先做 bandit / ranking / classifier baseline
2. 做 learned router / stop policy
   - 最先学会“什么时候搜、什么时候停”
3. 跑一轮 head-to-head offline 对比
   - 启发式策略 vs learned strategy
4. 再考虑是否进入 Tree-GRPO / summary-state RL / evolving rubrics

如果只选一个下一步，我建议：

> **先实现 `src/search_policy/train_policy.py`，把当前 search cache + reward 真正变成一个可训练 baseline。**

