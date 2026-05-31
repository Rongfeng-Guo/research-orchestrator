# 项目总览

## 1. 项目简介

Research Orchestrator 面向复杂研究任务自动化，围绕规划、执行、记忆、合成、修复和评测构建完整流程。

复杂研究任务通常具有以下特点:

- 问题范围较宽，包含多个子问题
- 需要多次检索和交叉验证
- 需要结构化整理结果
- 需要保留中间证据和过程信息

项目的主流程包括:

1. 规划: 把研究问题拆成多个可执行子任务
2. 调度: 按依赖关系并发执行子任务
3. 执行: 调用搜索、浏览、论文查询、计算等工具
4. 记忆: 保存子任务结果，供后续任务复用
5. 合成: 汇总多个子结果并生成研究报告
6. 修复: 在需要时进入 Red/Blue 对抗修复
7. 评测: 使用 benchmark、规则指标和 Judge 进行评估

## 2. 核心模块

项目的核心价值来自一条完整链路:

- `Planner` 负责把问题转成 DAG
- `Orchestrator` 负责状态机和并发调度
- `ResearcherAgent` 负责搜索、分析和验证类任务
- `SummarizerAgent` 负责最终报告生成
- `SharedMemoryStore` 负责跨任务记忆
- `EvidenceAwareResearchPolicy` 负责提供研究策略信号
- `ContextCompressor` 负责上下文控制
- `AdversarialLoop` 负责报告修复
- `evaluation/` 负责实验和量化评估

## 3. 输入与输出

### 输入

典型输入是自然语言研究问题，例如:

- “2024-2025 年大模型 Agent 技术趋势与落地案例研究”
- “比较 GPT-4o、Claude 3.5 Sonnet、DeepSeek-V3 的推理能力差异”
- “分析中国新能源车渗透率提升对燃油车产业链的冲击”

### 输出

系统输出一份 Markdown 研究报告，内容包括:

- 报告正文
- 元信息
- 置信度
- 搜索轮数
- 重规划次数
- 对抗轮数
- 来源链接列表

相关格式化逻辑位于 `src/core/runner.py` 的 `_format_report()`。

## 4. 模块状态

| 模块 | 设计目标 | 当前状态 |
| --- | --- | --- |
| M1 Orchestrator | 多任务编排与调度 | 已实现 |
| M2 Planner | 用 LLM 生成 DAG | 已实现 |
| M3 Compressor | 控制长上下文膨胀 | 已实现 |
| M4 Memory Store | 跨任务共享记忆与语义检索 | 已实现 |
| M5 Adversarial Loop | Red/Blue 报告修复 | 已实现 |
| M6 Evolution Engine | 自进化与训练闭环 | 已接入，部分仍为原型 |

## 5. 当前架构特点

项目的运行特征主要包括:

- 任务被拆解为多个子任务
- 子任务可按 DAG 层级并发执行
- 不同阶段可以使用不同模型后端
- 搜索、分析、验证类任务由统一的研究 worker 负责执行
- 研究报告生成由独立的 summarizer 完成

## 6. 项目定位

这个项目适合作为:

- 结构完整的 deep research Agent 原型系统
- 可运行、可扩展、可做实验的研究平台
- 后续继续扩展工具、策略、评测和界面的基础设施

如果你想继续了解一次 query 的完整运行路径，请阅读 [02-runtime-flow.md](./02-runtime-flow.md)。
