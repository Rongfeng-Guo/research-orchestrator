# DeepResearch Agent 文档总览

这套 `docs/` 不是对 README 的简单搬运，而是面向“读源码、理清实现”的补充文档。

如果你已经知道这个项目的大方向，但希望进一步理解：

- 它到底是不是“多智能体”
- 一次 query 在代码里怎么流转
- 哪些模块是真正落地的，哪些还是原型接口
- 评测、配置、记忆、对抗修复分别是怎么接起来的

建议从这里开始。

## 推荐阅读顺序

1. [01-overview.md](./01-overview.md)
2. [02-runtime-flow.md](./02-runtime-flow.md)
3. [03-architecture-and-data-models.md](./03-architecture-and-data-models.md)
4. [04-agents-tools-and-models.md](./04-agents-tools-and-models.md)
5. [05-memory-compression-and-adversarial.md](./05-memory-compression-and-adversarial.md)
6. [06-evaluation-and-experiments.md](./06-evaluation-and-experiments.md)
7. [07-configuration-and-environment.md](./07-configuration-and-environment.md)
8. [08-code-reading-guide.md](./08-code-reading-guide.md)
9. [09-limitations-and-observations.md](./09-limitations-and-observations.md)
10. [10-directory-walkthrough.md](./10-directory-walkthrough.md)

## 文档地图

| 文件 | 重点 |
| --- | --- |
| [01-overview.md](./01-overview.md) | 项目在解决什么问题，当前实现属于什么形态 |
| [02-runtime-flow.md](./02-runtime-flow.md) | 一次 query 从命令行进入到最终报告输出的完整路径 |
| [03-architecture-and-data-models.md](./03-architecture-and-data-models.md) | 系统分层、模块关系、关键数据结构 |
| [04-agents-tools-and-models.md](./04-agents-tools-and-models.md) | Agent、工具层、模型路由和策略封装 |
| [05-memory-compression-and-adversarial.md](./05-memory-compression-and-adversarial.md) | 记忆、压缩、Red/Blue 对抗修复三块质量控制逻辑 |
| [06-evaluation-and-experiments.md](./06-evaluation-and-experiments.md) | 标准评测、消融实验、benchmark、批量实验 |
| [07-configuration-and-environment.md](./07-configuration-and-environment.md) | `default.yaml`、`.env`、后端切换与工具配置 |
| [08-code-reading-guide.md](./08-code-reading-guide.md) | 适合按源码顺序阅读的路线图 |
| [09-limitations-and-observations.md](./09-limitations-and-observations.md) | 当前实现的局限、偏差和容易误读的地方 |
| [10-directory-walkthrough.md](./10-directory-walkthrough.md) | 仓库目录与关键文件的逐层导览 |

## 补充附录

| 文件 | 说明 |
| --- | --- |
| [evidence_aware_policy.md](./evidence_aware_policy.md) | `src/evidence/` 这一层的补充说明，聚焦 Evidence Snapshot 与 research policy |
| [CODEX_RUNBOOK.md](./CODEX_RUNBOOK.md) | 当前仓库在本机 smoke run 的环境记录和操作注意事项，偏运行附录 |
| [PROGRESS_2026-05-14.md](./PROGRESS_2026-05-14.md) | 2026-05-14 的最新进度记录，说明主线判断、search policy 在线接入现状与下一步 |
| [PROGRESS_2026-05-15.md](./PROGRESS_2026-05-15.md) | 2026-05-15 的继续推进记录，包含 cache 分析、受控 browser 增广、v2 训练、head-to-head 与两层门控 |
| [PROGRESS_2026-05-20.md](./PROGRESS_2026-05-20.md) | 2026-05-20 的 real-evidence 推进记录，补上 query bank、heldout runner、source relevance gate 和 OpenCode/Bing HTML 探针 |
| [PROGRESS_2026-05-21.md](./PROGRESS_2026-05-21.md) | 2026-05-21 的继续推进记录，补上显式官方 URL seeded retrieval、probe query 升级与论文草稿落地 |
| [PROGRESS_2026-05-23.md](./PROGRESS_2026-05-23.md) | 2026-05-23 的继续推进记录，补上 8-query readiness、4-fold live heldout head-to-head，以及 learned policy 仍然 mixed 的真实结论 |
| [RESEARCH_POSITIONING_2026-05-14.md](./RESEARCH_POSITIONING_2026-05-14.md) | 研究定位与实验设计说明，回答“我们在做什么实验、相关工作、新意边界、目标结果和不该乱做什么”，并补充 2026-05-15 的相关工作与新意收窄判断 |
| [EXPERIMENT_PROTOCOL_2026-05-14.md](./EXPERIMENT_PROTOCOL_2026-05-14.md) | 下一轮实验协议，明确模式 bug、旧结果可信度边界、重跑基线和接下来只允许改什么 |
| [GOAL_LADDER_2026-05-15.md](./GOAL_LADDER_2026-05-15.md) | 目标分层说明，明确当前更像研究原型项目还是论文种子，并补充 v2 训练后为什么仍然不是 CCF-A story |
| [PAPER_DRAFT_2026-05-21.md](./PAPER_DRAFT_2026-05-21.md) | 面向投稿叙事的第一版论文骨架，先把问题、方法、实验协议、当前结果与剩余缺口写实收敛 |

## 一分钟理解这个项目

这个项目的核心目标不是“问答”，而是“研究”。

用户给出一个复杂问题后，系统会：

1. 把问题拆成多个子任务。
2. 用 DAG 描述子任务依赖关系。
3. 按拓扑层次并发执行这些子任务。
4. 每个子任务由 `ResearcherAgent` 调用搜索、网页阅读、论文检索、计算等工具完成。
5. 子任务结果写入共享记忆。
6. 最后由 `SummarizerAgent` 合成为研究报告。
7. 如果置信度不够，再进入 Red/Blue 对抗修复流程。

最终产出不是一句回答，而是一份 Markdown 报告，附带置信度、搜索轮数、重规划次数、对抗轮数和来源列表。

当前代码里还额外接入了一层 evidence-aware policy：

- `SharedMemoryStore` 可以生成 `EvidenceSnapshot`
- `Orchestrator` 会把 `recommended_action`、`missing_terms` 等研究策略信号写入运行时上下文
- `ResearcherAgent` 和 `SummarizerAgent` 会消费这些信号
- replan 判断也不再只看失败率

## 你会在这些文档里看到什么

这些文档会刻意区分两层：

- 设计意图：README 和模块命名所表达的目标
- 当前实现：代码里真正发生的事情

例如：

- 项目名义上是“多智能体”，但当前 `search`、`analyze`、`verify` 三类任务都复用了同一个 `ResearcherAgent`
- 项目名义上有 M6 自进化，但训练环节目前仍是 placeholder
- `tools.code_sandbox.enabled: false` 出现在配置里，但工具工厂仍会直接创建 `CodeSandboxTool`
- `ResearchBench` 的类注释还写着 20 道题，但实际内置题目数是 35
- 当前仓库还存在 `src/evidence/`，说明系统已经开始从“固定 DAG 执行器”往“证据驱动研究策略”扩展

这些实现细节会决定你应该怎样理解它：

- 它更像一个研究型 Agent 框架，而不是已经产品化的系统
- 它的强项是“结构清晰、可配置、适合实验”
- 它的短板是“某些能力仍然靠 prompt 和启发式，部分模块尚未彻底闭环”

## 适合什么场景

- 阅读源码并准备答辩或面试讲解
- 在现有框架上继续扩展工具、评测或 UI
- 做 Agent 系统课程项目或研究原型
- 理解“规划 -> 执行 -> 记忆 -> 合成 -> 修复 -> 评测”的完整链路

## 不适合期待什么

- 不要把它当成一个已经非常稳的生产级 deep research 平台
- 不要默认 README 中的每一条表述都和当前实现一一对应
- 不要默认 `evolution/` 已经接好完整训练闭环

如果你的目标是快速搞清楚主流程，请先读 [02-runtime-flow.md](./02-runtime-flow.md)。
