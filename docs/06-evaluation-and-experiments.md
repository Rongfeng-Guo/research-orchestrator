# 评测体系与实验脚本

这个仓库一个很强的地方在于：它不只提供“运行系统”的入口，还把“怎么评这个系统”也写进了代码。

## 1. 评测层解决什么问题

如果没有评测层，这个项目最多只能回答：

- “它能不能跑起来？”

有了评测层之后，才能回答：

- “它比单轮 LLM 好多少？”
- “哪些模块真的有用？”
- “不同领域下表现是否稳定？”
- “对抗环到底有没有带来提升？”

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

### 4.1 设计目的

`ResearchBench` 不是为了评短答案，而是为了评研究报告。

每道题会提供：

- `query`
- `expected_topics`
- `ground_truth`
- `domain`

### 4.2 实际题量

代码里这套 benchmark 实际包含 35 道题。

需要注意：

- 文件头部有“20 道题”的描述
- 但真实 `DEFAULT_QUESTIONS` 数量已经是 35

所以应以代码内容为准，而不是注释里的旧描述。

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

## 6. Judge 指标与规则指标的关系

总体关系可以理解为：

- 规则指标：更稳定、便宜、可重复
- Judge 指标：更接近人工审稿，但更贵，也更依赖模型本身

`evaluation/metrics/composite.py` 提供了统一入口，把：

- 规则分
- Judge 分

组合成一个综合分。

## 7. `run_ablation.py`：消融实验

这个脚本回答的问题是：

> “如果去掉某个模块，系统会退化多少？”

它支持两种模式。

### 7.1 模块消融

典型配置包括：

- full
- no_adversarial
- no_compressor
- no_memory
- no_evolution

### 7.2 对抗轮数消融

对比：

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

这个脚本专门回答：

> “整套 Agent 流程，相比直接问模型，值不值得？”

对比双方是：

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

## 10. 输出结果通常长什么样

评测和实验脚本会输出：

- JSON 结果文件
- Markdown 摘要
- 各实验子目录

比如：

- `outputs/evaluation/`
- `outputs/experiments/`

## 11. 评测层的优点

- 和主流程共用同一运行内核，不是另写一套 demo evaluator。
- 同时支持规则指标和 Judge 指标。
- 有 benchmark，也有 ablation，也有 baseline 对照。
- 统计思路比很多 demo 仓库更完整。

## 12. 评测层的现实局限

### 规则指标局限

- 对长文质量只能近似判断
- 对复杂论证的质量把握有限

### Judge 局限

- Judge 自身也是模型，存在偏好和波动
- Judge 结果不是绝对真值

### HotpotQA 适配局限

- HotpotQA 本质是 QA 基准
- 这里把它改成“研究报告式运行”，评测口径会带折中

## 13. 一句话总结

这个项目的评测层不是装饰品。

它已经把：

- benchmark
- ablation
- baseline 对照
- 统计显著性

这些在研究系统里很关键的环节都做进来了。
