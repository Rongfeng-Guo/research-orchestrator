# Research Agenda (2026-05-31)

本文档基于仓库当前代码、artifact 和数据目录的静态审计，总结下一阶段最值得推进的研究问题。

## 当前结论

- `ResearchBench` 实际已包含 35 题，覆盖 11 个领域，但题量分布不均，科技题显著多于长尾领域。
- 默认 `artifacts/search_policy.json` 训练样本只有 20 条，且 `browser` 标签仅 1 条，不适合作为当前主推 policy。
- 现有 artifact 中，`search_policy_20260515_natural_train_v1.json` 在样本规模和验证集精度上都明显更像候选主模型。
- `data/search_cache/citation_resurfaced/` 下存在多份单记录、疑似由 pytest 生成的 manifest，不能直接当作正式实验产物引用。

## 最值得做的研究问题

### 1. Search Policy 是否真的能稳定优于 heuristic

关键问题：

- 当前 learned policy 的提升，是来自真实泛化，还是来自小样本和切分方式带来的偶然性。

建议实验：

- 以 `search_policy_20260515_natural_train_v1.json` 为 learned 候选，与 heuristic 做 head-to-head。
- 按 query 级别切分，而不是 step 级别切分。
- 报告总体结果外，再按领域汇报 `科技/医疗/金融/长尾领域` 的分组结果。

主要指标：

- composite score
- citation coverage / citation grounding
- source relevance
- browser usage coverage
- average tool calls per query

### 2. Browser 决策是否被低估

关键问题：

- 默认 policy artifact 几乎没有 `browser` 监督，导致浏览动作可能被系统性低估。

建议实验：

- 从真实 search cache 中筛出 browser-positive query 子集。
- 构建 “search-only” vs “search+browser available” 的对照分析。
- 单独报告 browser 触发时的 citation grounding、source relevance、latency 成本变化。

主要指标：

- browser trigger rate
- relevant source count
- grounding render gap
- elapsed time / cost proxy

### 3. Benchmark 领域不均衡是否在误导总体结论

关键问题：

- 如果科技题占比过高，总体分数可能主要反映模型在科技问题上的表现，而不是研究系统的通用性。

建议实验：

- 统一按领域做 macro-average，而不是只看 overall average。
- 为 `传媒` 等长尾领域补题，至少让每个领域达到 3-4 题。
- 在论文或 README 中区分 “overall score” 与 “balanced domain score”。

主要指标：

- domain macro-average
- domain variance
- per-domain confidence interval

## 近期工程配套建议

- 把 `audit_research_readiness.py` 产出的 JSON/Markdown 作为研究记录基线。
- 清理 `data/` 下疑似测试残留的 manifest，避免后续分析误读。
- 在后续实验文档中明确写出：默认 artifact 与推荐 artifact 是否一致。

## 推荐执行顺序

1. 先用 `audit_research_readiness.py` 固化当前仓库状态。
2. 再跑 learned vs heuristic 的 query-level head-to-head。
3. 然后补 benchmark 长尾领域题目，重算 balanced domain score。
