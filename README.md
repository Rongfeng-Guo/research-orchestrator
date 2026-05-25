# DeepResearch Agent

从复杂查询到结构化研究报告的研究型 Agent 原型。

这个仓库围绕 deep research workflow 构建，覆盖从问题拆解到报告生成的一整套流程:

- 先把复杂问题拆成子任务
- 再按依赖并发执行搜索、阅读、分析
- 把中间证据写入共享记忆
- 最后合成为带引用的 Markdown 报告
- 必要时再进入对抗修复和实验评估

项目定位:

> 一个可运行、可扩展、可做实验的 deep research agent research prototype。

它已经具备完整主流程和较强的实验框架，适合作为研究原型与实验平台。

## Project Overview

这个仓库整体定位为研究原型与实验平台。

当前包含的主要能力:

- 复杂 query -> DAG plan -> 并发执行 -> 报告合成的主流程
- 搜索、网页读取、论文读取、计算、文件读取等工具层
- 共享记忆、上下文压缩、Red/Blue 对抗修复、模型路由
- `ResearchBench`、HotpotQA 适配、ablation、head-to-head、paper-readiness audit 等评测脚本
- search policy、evidence-aware policy 与相关训练接口

## What This Repo Does

主线流程如下:

1. `Planner` 把用户问题拆成带依赖关系的 DAG 子任务图
2. `Orchestrator` 按拓扑层级并发调度子任务
3. `ResearcherAgent` 调用搜索、浏览器、ArXiv、计算器等工具完成子任务
4. `SharedMemoryStore` 保存中间证据、上下文和过程信号
5. `SummarizerAgent` 把子任务结果合成为研究报告
6. 如果报告置信度不足，进入 Red/Blue adversarial refinement
7. `evaluation/` 下的脚本对系统做 benchmark、ablation 和对照实验

最终产出是一份带引用、带元信息的 Markdown 研究报告。

## Architecture

核心模块可以概括为 6 个层次:

| Module | Responsibility | Current status |
| --- | --- | --- |
| `orchestrator/` | 状态机、DAG 调度、并发执行 | 已实现 |
| `planner/` | 复杂问题拆解与重规划 | 已实现 |
| `agents/` | research worker 与 summarizer | 已实现 |
| `memory/` + `compressor/` | 共享记忆与上下文控制 | 已实现 |
| `adversarial/` | Red/Blue 报告修复 | 已实现 |
| `search_policy/` + `evidence/` + `evolution/` | 证据驱动策略与训练探索 | 已接入，部分仍为原型 |

当前架构说明:

- 这个项目名义上是多智能体系统，但当前代码中 `search` / `analyze` / `verify` 三类任务主要仍由同一个 `ResearcherAgent` 执行。
- 当前整体是“编排器 + 通用研究 worker + 合成器”的架构。

## Main Features

- 自研 `asyncio` + DAG 编排器，不依赖 LangGraph / AutoGen
- 复杂 query 的规划、并发执行和重规划机制
- 搜索、浏览器、论文检索、文件读取、代码沙箱、计算器、notepad 工具链
- 共享记忆、语义压缩、证据快照和 research policy 信号
- Red/Blue 对抗修复，用于低置信度报告的后处理
- 多模型后端路由，可切换 DeepSeek / MiMo / vLLM / OpenAI
- 自建 benchmark、规则指标、LLM-as-Judge、head-to-head、ablation、paper-readiness audit

## Repository Layout

```text
deepresearch-agent-main/
├── configs/         # YAML 配置
├── src/             # 核心源码
│   ├── orchestrator/
│   ├── planner/
│   ├── agents/
│   ├── tools/
│   ├── memory/
│   ├── compressor/
│   ├── adversarial/
│   ├── evidence/
│   ├── search_policy/
│   └── evolution/
├── scripts/         # 运行与实验入口
├── evaluation/      # benchmark、metrics、reporting
├── docs/            # 面向代码阅读和实验定位的补充文档
├── artifacts/       # 训练出的 policy 等产物
├── outputs/         # 已跑出的报告与实验结果
└── tests/           # 测试
```

## Quick Start

### 1. Environment

建议环境:

- Python `3.11`
- 一个可用的 LLM API key
- 可选搜索 API key

安装:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

环境变量配置可参考:

- `.env.template`
- `.env.tools.template`
- [docs/07-configuration-and-environment.md](docs/07-configuration-and-environment.md)

注意:

- 如果真实搜索后端缺失对应 key，运行时会在部分路径上自动回退到 mock 模式，以保证主流程可 smoke run。

### 2. Run a Single Research Task

```bash
python scripts/run_single.py \
  --query "2024-2025年大模型Agent技术趋势与落地案例研究" \
  --config configs/default.yaml
```

### 3. Run Interactive REPL

```bash
python scripts/run_repl.py
```

### 4. Run a Minimal Evaluation

```bash
python scripts/run_eval.py \
  --benchmark research_bench \
  --num_questions 2 \
  --config configs/aliyun_smoke.yaml
```

## Recommended Reading Order

如果你是第一次看这个仓库，建议按下面顺序读:

1. [docs/01-overview.md](docs/01-overview.md)
2. [docs/02-runtime-flow.md](docs/02-runtime-flow.md)
3. [docs/03-architecture-and-data-models.md](docs/03-architecture-and-data-models.md)
4. [docs/06-evaluation-and-experiments.md](docs/06-evaluation-and-experiments.md)
5. [docs/09-limitations-and-observations.md](docs/09-limitations-and-observations.md)

如果你想了解当前实验到底做到哪里了，重点看:

- [docs/PROGRESS_2026-05-23.md](docs/PROGRESS_2026-05-23.md)
- [docs/PAPER_DRAFT_2026-05-21.md](docs/PAPER_DRAFT_2026-05-21.md)
- [docs/GOAL_LADDER_2026-05-15.md](docs/GOAL_LADDER_2026-05-15.md)

## Evaluation and Experiments

这个仓库包含完整的实验与评测框架，重点覆盖以下方向:

- 证据质量与引用质量评估
- 来源相关性与 primary evidence 检索
- search policy 与 evidence-aware policy 的训练和运行时接入
- benchmark、ablation、head-to-head 与 paper-readiness audit

仓库中已经提供:

- evidence-readiness 相关评测接口和审计流程
- source quality / citation hygiene 的分析脚本和实验产物
- learned policy、heuristic policy 与运行时 policy 接入代码

这个仓库可用于:

- 研究项目
- agentic search / deep research 的实验平台
- 后续论文或扩展工作的基础设施

## Implementation Notes

实现细节和模块说明可参考:

- [docs/09-limitations-and-observations.md](docs/09-limitations-and-observations.md)

## Tests

仓库包含较完整的测试集合，示例:

```bash
pytest tests/test_policy_head2head_cv.py -q
pytest tests/test_search_policy.py -q
pytest tests/test_run_eval_hotpotqa.py -q
```

## License

MIT
