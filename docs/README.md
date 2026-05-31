# Research Orchestrator 文档总览

这套 `docs/` 包含项目主流程、架构、配置、评测和目录结构说明。

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

| 文件 | 内容 |
| --- | --- |
| [01-overview.md](./01-overview.md) | 项目概览、输入输出和核心模块 |
| [02-runtime-flow.md](./02-runtime-flow.md) | 一次 query 的完整运行路径 |
| [03-architecture-and-data-models.md](./03-architecture-and-data-models.md) | 系统分层、模块关系、关键数据结构 |
| [04-agents-tools-and-models.md](./04-agents-tools-and-models.md) | Agent、工具层、模型路由与策略封装 |
| [05-memory-compression-and-adversarial.md](./05-memory-compression-and-adversarial.md) | 记忆、压缩与对抗修复 |
| [06-evaluation-and-experiments.md](./06-evaluation-and-experiments.md) | benchmark、ablation、head-to-head 与实验脚本 |
| [07-configuration-and-environment.md](./07-configuration-and-environment.md) | 配置结构、环境变量与后端切换 |
| [08-code-reading-guide.md](./08-code-reading-guide.md) | 按源码阅读项目的推荐路径 |
| [09-limitations-and-observations.md](./09-limitations-and-observations.md) | 模块说明、实现细节与阅读提示 |
| [10-directory-walkthrough.md](./10-directory-walkthrough.md) | 仓库目录逐层导览 |

## 补充文档

| 文件 | 说明 |
| --- | --- |
| [evidence_aware_policy.md](./evidence_aware_policy.md) | `src/evidence/` 相关设计与运行时信号 |
| [EXPERIMENT_PROTOCOL_2026-05-14.md](./EXPERIMENT_PROTOCOL_2026-05-14.md) | 实验协议与执行要求 |
| [RESEARCH_POSITIONING_2026-05-14.md](./RESEARCH_POSITIONING_2026-05-14.md) | 研究定位与实验设计说明 |
| [GOAL_LADDER_2026-05-15.md](./GOAL_LADDER_2026-05-15.md) | 项目目标分层说明 |

## Additional Notes

补充记录类文档位于 `docs/` 目录下，可按文件名查阅：

- `CODEX_RUNBOOK.md`
- `CODEX_IMPROVEMENTS_SUMMARY.md`
- `PAPER_DRAFT_2026-05-21.md`
- `PROGRESS_2026-05-14.md`
- `PROGRESS_2026-05-15.md`
- `PROGRESS_2026-05-16.md`
- `PROGRESS_2026-05-20.md`
- `PROGRESS_2026-05-21.md`
- `PROGRESS_2026-05-23.md`
- `PROGRESS_2026-05-25.md`

## 项目流程

用户输入一个复杂研究问题后，系统会:

1. 生成 DAG 子任务图
2. 并发执行搜索、浏览、分析类任务
3. 将中间结果写入共享记忆
4. 汇总为结构化研究报告
5. 在需要时执行对抗修复与实验评测

最终输出是一份 Markdown 报告，并附带置信度、搜索轮数、重规划次数、对抗轮数和来源列表。

## 文档用途

- 阅读源码并理解 research orchestrator workflow
- 在现有框架上扩展工具、评测或界面
- 作为 Agent 系统课程项目参考
- 复用现有实验脚本进行 benchmark 和策略研究
