# 评测体系与实验脚本

本文档介绍仓库中的评测体系与实验脚本。

## 1. 评测范围

评测层覆盖以下内容:

- benchmark 运行
- 模块消融
- baseline 对比
- 统计分析

## 2. 四个主要脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/run_eval.py` | 标准 benchmark 评测 |
| `scripts/run_ablation.py` | 消融实验 |
| `scripts/run_benchmark.py` | 单轮 LLM vs Agent 对比 |
| `scripts/run_all_experiments.py` | 一键跑完整实验集 |

## 3. `run_eval.py`：标准评测入口

它支持两个 benchmark：

- `research_bench`
- `hotpotqa`

### `research_bench`

流程是：

1. 从 `ResearchBench` 取题。
2. 对每道题运行 `run_research()`。
3. 用规则指标评估生成报告。
4. 聚合成 `EvaluationReport`。

### `hotpotqa`

流程是：

1. 取多跳问答题。
2. 运行 `run_research()`。
3. 从报告中抽取预测答案。
4. 计算 EM / F1 / pass@1 等指标。

## 4. `ResearchBench`：自建深度研究题集

### 4.1 数据集结构

每道题会提供：

- `query`
- `expected_topics`
- `ground_truth`
- `domain`

### 4.2 题量

代码里这套 benchmark 实际包含 35 道题。

代码中的 `DEFAULT_QUESTIONS` 数量为 35。

### 4.3 覆盖领域

包括但不限于：

- 科技
- 医疗
- 金融
- 教育
- 法律
- 能源
- 消费
- 汽车
- 游戏
- 传媒
- 交叉领域

## 5. 规则指标是怎么打分的

`ResearchBench.evaluate_report()` 会调用规则指标，主要包括：

- fact accuracy
- semantic fact accuracy
- hallucination rate
- citation coverage
- logical consistency
- comprehensiveness

最后再做 composite score。

## 6. Judge 指标与规则指标

- 规则指标: 面向可重复计算的程序化评估
- Judge 指标: 面向模型评分的补充评估

`evaluation/metrics/composite.py` 提供了统一入口，把：

- 规则分
- Judge 分

组合成一个综合分。

## 7. `run_ablation.py`：消融实验

它支持两种模式。

### 7.1 模块消融

典型配置包括：

- full
- no_adversarial
- no_compressor
- no_memory
- no_evolution

### 7.2 对抗轮数消融

轮次数设置包括:

- 0 轮
- 1 轮
- 2 轮
- 3 轮

### 7.3 统计方法

脚本会计算：

- paired bootstrap 95% CI
- p-value
- Cohen's d

## 8. `run_benchmark.py`：单轮 LLM vs Agent

对比双方包括:

- baseline：单轮 LLM 直接回答
- agent：完整 DeepResearch 流程

然后再用 Judge 做 head-to-head 评分。

## 9. `run_all_experiments.py`：一键实验总控

这个脚本把多个实验串起来一次跑完：

- 模块消融
- 对抗轮数消融
- 标准评测集
- 多领域对比
- Agent vs LLM
- HotpotQA
- Judge 深度评分

## 10. 输出结果

评测和实验脚本会输出：

- JSON 结果文件
- Markdown 摘要
- 各实验子目录

常见输出目录包括:

- `outputs/evaluation/`
- `outputs/experiments/`

## 11. 补充说明

- 评测脚本与主流程共用同一运行内核
- 同时支持规则指标和 Judge 指标
- 包含 benchmark、ablation、baseline 对照和统计显著性分析
