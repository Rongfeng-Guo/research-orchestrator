# Evidence-Aware Research Policy

更新时间：2026-05-12

## 为什么要加这一层

原项目的主流程已经覆盖了规划、执行、记忆、对抗和评测，但控制逻辑仍然偏向：

`plan once -> execute -> summarize -> repair`

这类结构的上限通常是“把报告写得更稳”，但很难回答更关键的问题：

- 当前证据够不够？
- 还缺哪些子主题？
- 哪些冲突值得再查？
- 现在该继续搜、扩展、验证，还是可以收敛？

因此我们新增了一层 **evidence-aware research policy**，把系统从“固定 DAG 执行器”往“证据驱动决策器”推进。

## 当前实现

### 1. Evidence Snapshot

在 `SharedMemoryStore` 中，系统会基于当前 query 做一次证据状态汇总，得到：

- `nodes`: 当前最相关的证据条目
- `claims`: claim 级结构化节点
- `support_edges`: claim 与 source 的支持边
- `contradiction_edges`: claim 之间的冲突边
- `source_trusts`: source trust 估计
- `conflicts`: 与这些条目相关的未解决矛盾
- `focus_terms`: query 中抽取的关注词
- `covered_terms` / `missing_terms`: 现有证据覆盖了什么、缺了什么
- `evidence_strength`: 证据强度
- `uncertainty`: 当前不确定性
- `candidate_actions`: 候选动作排序
- `recommended_action`: 当前最优动作

### 2. Candidate Action Ranking

当前 policy 不再只输出一个“建议动作”，而是对下面这些动作做排序：

- `search`
- `retrieve_more`
- `verify`
- `expand_subquestion`
- `stop`

这个排序是确定性的，后续可以直接替换成 learned policy，而不必修改上下游数据结构。

### 3. Orchestrator Integration

在收集子任务结果后，orchestrator 会把 evidence snapshot 写入运行时状态，并将其传给：

- `ResearcherAgent`
- `SummarizerAgent`
- replan decision
- `ResearchReport.metadata`

同时，orchestrator 还会记录 `evidence_transition_trace`，把每个子任务带来的证据状态变化显式保存下来，包括：

- `uncertainty` 的变化
- `coverage_ratio` 的变化
- `open_conflicts` 的变化
- `evidence_strength` 的变化

这让后续训练不再只有终局分数，而是可以做 step-wise reward shaping。

### 4. Evaluation & Trajectory Export

评测和训练出口现在也能消费这层状态：

- `run_research(..., return_report=True)` 会同时返回 Markdown 报告和 `ResearchReport`
- `evaluation/benchmarks/research_bench.py` 可以接收 `report.metadata`
- 规则评测现在会聚合 `claim/support/contradiction/source trust` 指标
- `TrajectoryCollector` 会导出 `evidence_transition_trace` 和 `process_reward_trace`
- `SelfEvolutionEngine` 写 parquet 时会保留这些过程信号，方便后续做离线学习和 replay
- `SelfEvolutionEngine` 还会额外导出 `evidence_transition_dataset.jsonl`
- 每个 JSONL row 都是一个 `state -> action -> reward` 样本，包含固定维度 `feature_vector`

这样 evidence 状态不再只是日志，而是实际影响下一步行为。

### 5. Transition Dataset -> Learned Policy

当前选择的方向不是直接把整个 Agent 替换成 RL policy，而是先把最关键的局部控制问题学出来：

`当前证据状态下，下一步该 search / retrieve_more / verify / expand_subquestion / stop 哪一个？`

具体做法：

- 继续保留现有 heuristic policy，作为冷启动和线上 fallback
- orchestrator 导出 step-wise `evidence_transition_trace`
- `src/evolution/evidence_dataset.py` 把 transition flatten 成监督学习样本
- `scripts/train_evidence_policy.py` 聚合多个 round 的 JSONL，训练 `LearnedEvidenceAwareResearchPolicy`
- learned policy 在推理时不会重写整条流程，而是对 heuristic 生成的 `candidate_actions` 做 learned rerank

这样做的好处是：

- 风险低：没模型或模型很差时，直接回退 heuristic
- 数据效率高：不需要等终局 reward，step-wise transition 本身就能监督
- 接口稳定：上游下游仍然只看 `EvidenceSnapshot` / `candidate_actions`
- 后续容易升级：可以从线性模型平滑替换成 contextual bandit、offline RL 或 graph policy

### 6. 训练入口

```bash
# 先跑 evolution，round 目录会自动生成 evidence_transition_dataset.jsonl
python scripts/run_evolution.py --config configs/evolution/grpo_online.yaml --num_rounds 3

# 再训练 learned evidence policy
python scripts/train_evidence_policy.py \
  --input evolution_output \
  --output-model artifacts/evidence_policy.json
```

训练脚本当前会：

- 递归收集 `evidence_transition_dataset.jsonl`
- 训练一个轻量线性多分类 action ranker
- 用 reward 做样本加权，优先拟合高收益 transition
- 输出 train accuracy、weighted accuracy 和动作分布
- 产出一个 JSON 模型文件，供 `LearnedEvidenceAwareResearchPolicy.load(...)` 直接加载

## 现在的作用

- 让 prompt 更聚焦缺口，而不是重复已有结论
- 让合成阶段显式处理 open conflicts
- 让重规划不只看失败率，还看证据状态
- 为以后把 rule-based policy 替换成 learned policy 提供统一接口

## 目前的限制

现在的证据状态仍然是 heuristic 版本，主要依赖：

- 语义相似度
- 关键词覆盖
- 冲突记录
- 简单分数合成

这已经比纯 DAG 更像“研究决策”，但还不是完整的 evidence graph learning。

## 下一步

1. 把当前线性 `LearnedEvidenceAwareResearchPolicy` 升级成 reward-aware contextual bandit。
2. 把 `MemoryEntry` 的 `claim` 进一步结构化为 `claim / support / source / temporal validity`。
3. 在 evidence graph 上引入 graph encoder，而不只用 hand-crafted summary feature。
4. 把 benchmark adapter 进一步统一成可直接消费 `report.metadata` 的格式。
5. 从 imitation + reward weighting 继续推进到 offline RL / policy improvement。

## 近期改动记录

- 2026-05-12
  - 新增 `src/evidence/policy.py`
  - `SharedMemoryStore` 增加 `get_evidence_snapshot()`
  - orchestrator 收集阶段写入 evidence snapshot
  - researcher / summarizer prompt 接入 snapshot
  - `ResearchReport.metadata` 透传 evidence snapshot / policy state / transition trace
  - snapshot 增加 claim/support/contradiction/source trust 图层
  - `evaluation/metrics/rule_based.py` 增加 `evidence_graph_quality`、`evidence_transition_quality`
  - `evaluation/benchmarks/research_bench.py` 接入 claim/edge 级指标与 batch 聚合
  - `src/evolution/collector.py` 导出 `evidence_transition_trace` / `process_reward_trace`
  - `src/evolution/engine.py` 导出 `evidence_transition_dataset.jsonl`
  - 新增 `src/evidence/learned_policy.py`
  - 新增 `scripts/train_evidence_policy.py`
  - `src/evolution/engine.py` 和 `src/evolution/symbolic_learning.py` 读取过程信号
  - 新增 `tests/test_evidence_policy.py`
