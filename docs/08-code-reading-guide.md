# 源码阅读指南

本文档整理源码阅读顺序和相关代码入口。

## 1. 最短阅读路线

核心阅读文件:

1. `scripts/run_single.py`
2. `src/core/runner.py`
3. `src/orchestrator/orchestrator.py`
4. `src/planner/planner.py`
5. `src/agents/researcher.py`
6. `src/agents/summarizer.py`

## 2. 第一轮：主流程

第一轮阅读顺序:

### `scripts/run_single.py`

重点:

- 命令行参数
- 怎么调用 runner
- 报告最后如何保存

### `src/core/runner.py`

重点:

- `initialize_modules()`
- `run_research()`

### `src/orchestrator/orchestrator.py`

重点:

- 状态机
- 调度逻辑
- 什么时候 replan
- 什么时候 synthesize
- 什么时候进入 adversarial

## 3. 第二轮：任务拆解与执行

### `src/planner/planner.py`

重点:

- Planner prompt 长什么样
- 输出要求是什么
- JSON 解析如何容错
- DAG 是怎么构建的

### `src/planner/dag.py`

重点:

- 为什么能并发
- DAG 分层怎么做

### `src/agents/researcher.py`

重点:

- 工具是怎么注册给模型的
- 消息历史怎么构造
- 什么时候停止工具调用

## 4. 第三轮：报告生成

### `src/agents/summarizer.py`

重点:

- 子结果如何拼到 prompt
- 置信度怎么计算
- 来源怎么抽取

### `src/core/runner.py` 中的 `_format_report()`

重点:

- 最终 Markdown 结构
- 元信息如何附加

## 5. 第四轮：增强模块

### `src/memory/memory_store.py`

重点:

- 结果如何持久化
- 去重和冲突检测怎么做

### `src/compressor/compressor.py`

重点:

- 超长上下文时准备怎么压

### `src/adversarial/loop.py`

重点:

- Red/Blue 对抗什么时候开始
- 怎么判断结束

## 6. 第五轮：模型与工具

### `src/models/model_router.py`

重点:

- 后端怎么切
- 模块怎么绑定不同模型

### `src/models/vllm_policy.py`

重点:

- 为什么消息要清洗
- 上下文截断怎么做
- tool call 怎么解析

### `src/tools/`

推荐顺序:

1. `web_search.py`
2. `browser.py`
3. `arxiv_reader.py`
4. `file_reader.py`
5. `calculator.py`
6. `code_sandbox.py`
7. `notepad.py`

## 7. 第六轮：评测层

### `scripts/run_eval.py`

重点:

- 标准评测怎么跑

### `evaluation/benchmarks/research_bench.py`

重点:

- 题集长什么样
- 规则指标依赖什么 ground truth

### `scripts/run_ablation.py`

重点:

- 模块消融怎么组织
- 对抗轮数消融怎么组织

### `scripts/run_benchmark.py`

重点:

- 如何比较 Agent 和单轮 LLM

## 8. 常见排查路径

### 规划结果很奇怪

查看:

- `src/planner/planner.py`
- Planner prompt

### 工具总是没被调用

查看:

- `src/agents/researcher.py`
- `src/models/vllm_policy.py`

### 报告很短或质量差

查看:

- `src/agents/summarizer.py`
- `ResearcherAgent` 的工具调用上限

### 记忆不起作用

查看:

- `src/memory/memory_store.py`
- `Orchestrator._build_memory_context()`

### 对抗环没触发

查看:

- `RunConfig.enable_adversarial`
- `Orchestrator._do_adversarial()`

## 9. 补充说明

- `search/analyze/verify` 并不是三个不同 worker 实现。
- `ResearcherAgent` 被 prompt 限制为最多 2 次工具调用。
- 压缩模块实现完整，但在主流程中的接入还有限。
- 自进化模块更多是扩展方向而非当前主线能力。
