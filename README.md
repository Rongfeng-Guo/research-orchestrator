# Research Orchestrator

[![Python CI](https://github.com/Rongfeng-Guo/research-orchestrator/actions/workflows/python-ci.yml/badge.svg)](https://github.com/Rongfeng-Guo/research-orchestrator/actions/workflows/python-ci.yml)
[![Packaging Check](https://github.com/Rongfeng-Guo/research-orchestrator/actions/workflows/packaging-check.yml/badge.svg)](https://github.com/Rongfeng-Guo/research-orchestrator/actions/workflows/packaging-check.yml)

面向复杂研究任务的结构化研究工作流实现。

该仓库提供从问题拆解、任务调度、工具调用、共享记忆到报告生成的完整流程。

## Overview

主要能力:

- 复杂 query -> DAG plan -> 并发执行 -> 报告合成
- 搜索、网页读取、论文读取、计算、文件读取等工具层
- 共享记忆、上下文压缩、Red/Blue 对抗修复、模型路由
- `ResearchBench`、HotpotQA 适配、ablation、head-to-head、paper-readiness audit 等评测脚本
- search policy、evidence-aware policy 与相关训练接口

## Workflow

主流程:

1. `Planner` 把用户问题拆成带依赖关系的 DAG 子任务图
2. `Orchestrator` 按拓扑层级并发调度子任务
3. `ResearcherAgent` 调用搜索、浏览器、ArXiv、计算器等工具完成子任务
4. `SharedMemoryStore` 保存中间证据、上下文和过程信号
5. `SummarizerAgent` 把子任务结果合成为研究报告
6. 在需要时进入 Red/Blue adversarial refinement
7. `evaluation/` 下的脚本对系统做 benchmark、ablation 和对照实验

输出结果为带引用和元信息的 Markdown 研究报告。

## Architecture

核心模块:

| Module | Responsibility | Current status |
| --- | --- | --- |
| `orchestrator/` | 状态机、DAG 调度、并发执行 | 已实现 |
| `planner/` | 复杂问题拆解与重规划 | 已实现 |
| `agents/` | research worker 与 summarizer | 已实现 |
| `memory/` + `compressor/` | 共享记忆与上下文控制 | 已实现 |
| `adversarial/` | Red/Blue 报告修复 | 已实现 |
| `search_policy/` + `evidence/` + `evolution/` | 证据驱动策略与训练探索 | 已接入，部分仍为原型 |

- `search` / `analyze` / `verify` 三类任务当前由同一个 `ResearcherAgent` 执行
- 执行结构由编排器、研究 worker 和合成器组成

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

Requirements:

- Python `3.11`
- 一个可用的 LLM API key
- 可选搜索 API key

安装:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment templates:

- `.env.template`
- `.env.tools.template`
- [docs/07-configuration-and-environment.md](docs/07-configuration-and-environment.md)

If the configured live search backend is unavailable, some execution paths fall back to mock mode.

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

Documentation:

1. [docs/01-overview.md](docs/01-overview.md)
2. [docs/02-runtime-flow.md](docs/02-runtime-flow.md)
3. [docs/03-architecture-and-data-models.md](docs/03-architecture-and-data-models.md)
4. [docs/06-evaluation-and-experiments.md](docs/06-evaluation-and-experiments.md)
5. [docs/09-limitations-and-observations.md](docs/09-limitations-and-observations.md)

Additional documents:

- [docs/PROGRESS_2026-05-23.md](docs/PROGRESS_2026-05-23.md)
- [docs/PAPER_DRAFT_2026-05-21.md](docs/PAPER_DRAFT_2026-05-21.md)
- [docs/GOAL_LADDER_2026-05-15.md](docs/GOAL_LADDER_2026-05-15.md)

## Evaluation and Experiments

实验与评测框架覆盖以下方向:

- 证据质量与引用质量评估
- 来源相关性与 primary evidence 检索
- search policy 与 evidence-aware policy 的训练和运行时接入
- benchmark、ablation、head-to-head 与 paper-readiness audit

Included:

- evidence-readiness 相关评测接口和审计流程
- source quality / citation hygiene 的分析脚本和实验产物
- learned policy、heuristic policy 与运行时 policy 接入代码

## Implementation Notes

Further details:

- [docs/09-limitations-and-observations.md](docs/09-limitations-and-observations.md)

## Tests

Example test commands:

```bash
pytest tests/test_policy_head2head_cv.py -q
pytest tests/test_search_policy.py -q
pytest tests/test_run_eval_hotpotqa.py -q
```

## Development

- GitHub Actions 会在 `push` 到 `main` 和 `pull_request` 时运行全量测试
- 可手动触发 `Packaging Check` 验证源码分发与 wheel 构建
- 本地开发建议先运行 `pytest -q`，再执行需要的脚本或评测入口

## License

MIT
