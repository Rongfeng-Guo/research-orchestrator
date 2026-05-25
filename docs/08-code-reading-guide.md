# 源码阅读指南

这篇文档不是介绍功能，而是告诉你“如果你现在要读源码，怎么读效率最高”。

## 1. 最短阅读路线

如果你时间很少，只看下面 6 个文件：

1. `scripts/run_single.py`
2. `src/core/runner.py`
3. `src/orchestrator/orchestrator.py`
4. `src/planner/planner.py`
5. `src/agents/researcher.py`
6. `src/agents/summarizer.py`

## 2. 第一轮：先看主线，不要急着看工具细节

建议第一轮阅读顺序：

### `scripts/run_single.py`

关注：

- 命令行参数
- 怎么调用 runner
- 报告最后如何保存

### `src/core/runner.py`

关注：

- `initialize_modules()`
- `run_research()`

### `src/orchestrator/orchestrator.py`

关注：

- 状态机
- 调度逻辑
- 什么时候 replan
- 什么时候 synthesize
- 什么时候进入 adversarial

## 3. 第二轮：看任务是怎么被拆和被执行的

### `src/planner/planner.py`

带着这些问题读：

- Planner prompt 长什么样
- 输出要求是什么
- JSON 解析如何容错
- DAG 是怎么构建的

### `src/planner/dag.py`

带着这些问题读：

- 为什么能并发
- DAG 分层怎么做

### `src/agents/researcher.py`

带着这些问题读：

- 工具是怎么注册给模型的
- 消息历史怎么构造
- 什么时候停止工具调用

## 4. 第三轮：看报告是怎么成形的

### `src/agents/summarizer.py`

关注：

- 子结果如何拼到 prompt
- 置信度怎么计算
- 来源怎么抽取

### `src/core/runner.py` 中的 `_format_report()`

关注：

- 最终 Markdown 结构
- 元信息如何附加

## 5. 第四轮：看增强模块

### `src/memory/memory_store.py`

适合回答：

- 结果如何持久化
- 去重和冲突检测怎么做

### `src/compressor/compressor.py`

适合回答：

- 超长上下文时准备怎么压

### `src/adversarial/loop.py`

适合回答：

- Red/Blue 对抗什么时候开始
- 怎么判断结束

## 6. 第五轮：看外设与外部依赖

### `src/models/model_router.py`

关注：

- 后端怎么切
- 模块怎么绑定不同模型

### `src/models/vllm_policy.py`

关注：

- 为什么消息要清洗
- 上下文截断怎么做
- tool call 怎么解析

### `src/tools/`

建议按下面顺序看：

1. `web_search.py`
2. `browser.py`
3. `arxiv_reader.py`
4. `file_reader.py`
5. `calculator.py`
6. `code_sandbox.py`
7. `notepad.py`

## 7. 第六轮：看评测层

### `scripts/run_eval.py`

回答：

- 标准评测怎么跑

### `evaluation/benchmarks/research_bench.py`

回答：

- 题集长什么样
- 规则指标依赖什么 ground truth

### `scripts/run_ablation.py`

回答：

- 模块消融怎么组织
- 对抗轮数消融怎么组织

### `scripts/run_benchmark.py`

回答：

- 如何比较 Agent 和单轮 LLM

## 8. 遇到问题时，应该去哪看

### 规划结果很奇怪

优先看：

- `src/planner/planner.py`
- Planner prompt

### 工具总是没被调用

优先看：

- `src/agents/researcher.py`
- `src/models/vllm_policy.py`

### 报告很短或质量差

优先看：

- `src/agents/summarizer.py`
- `ResearcherAgent` 的工具调用上限

### 记忆不起作用

优先看：

- `src/memory/memory_store.py`
- `Orchestrator._build_memory_context()`

### 对抗环没触发

优先看：

- `RunConfig.enable_adversarial`
- `Orchestrator._do_adversarial()`

## 9. 阅读时不要忽略的实现真相

- `search/analyze/verify` 并不是三个不同 worker 实现。
- `ResearcherAgent` 被 prompt 限制为最多 2 次工具调用。
- 压缩模块实现完整，但在主流程中的接入还有限。
- 自进化模块更多是扩展方向而非当前主线能力。
