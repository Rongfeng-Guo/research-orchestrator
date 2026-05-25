# DeepResearch Agent

从复杂查询到结构化研究报告的研究型 Agent 原型。

这个仓库的目标不是做一个单轮问答 demo，而是探索一条更完整的 deep research 流程:

- 先把复杂问题拆成子任务
- 再按依赖并发执行搜索、阅读、分析
- 把中间证据写入共享记忆
- 最后合成为带引用的 Markdown 报告
- 必要时再进入对抗修复和实验评估

当前最合适的定位是:

> 一个可运行、可扩展、可做实验的 deep research agent research prototype。

它已经具备完整主流程和较强的实验框架，但并不是已经收口的生产系统，也不是已经完成全部论文结论的终版项目。

## Project Status

这个仓库现在适合公开为个人项目，前提是对外表述要准确。

当前已经成立的部分:

- 复杂 query -> DAG plan -> 并发执行 -> 报告合成的主流程已闭环
- 搜索、网页读取、论文读取、计算、文件读取等工具层已接入
- 共享记忆、上下文压缩、Red/Blue 对抗修复、模型路由已实现
- `ResearchBench`、HotpotQA 适配、ablation、head-to-head、paper-readiness audit 等评测脚本已接入
- real-evidence fast benchmark 上的 evidence-readiness / source-quality 主线已经得到正向信号

当前仍在研究中的部分:

- learned search policy 相对 heuristic 的优势还不稳定
- 一些结果在 fold-level 上仍然 mixed，不能写成稳定 superiority claim
- `evolution/` 和部分 learned-policy 训练链路仍偏研究原型
- 该项目更适合作为研究框架和实验平台，而不是现成产品

如果你只想知道一句话结论:

> 这个项目已经足够作为 GitHub 上的完整个人项目公开，但应把它表述为研究原型，而不是“所有实验都已收尾的论文系统”。

## What This Repo Does

主线流程如下:

1. `Planner` 把用户问题拆成带依赖关系的 DAG 子任务图
2. `Orchestrator` 按拓扑层级并发调度子任务
3. `ResearcherAgent` 调用搜索、浏览器、ArXiv、计算器等工具完成子任务
4. `SharedMemoryStore` 保存中间证据、上下文和过程信号
5. `SummarizerAgent` 把子任务结果合成为研究报告
6. 如果报告置信度不足，进入 Red/Blue adversarial refinement
7. `evaluation/` 下的脚本对系统做 benchmark、ablation 和对照实验

最终产出不是一句回答，而是一份带引用、带元信息的 Markdown 研究报告。

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

需要特别说明的一点:

- 这个项目名义上是多智能体系统，但当前代码中 `search` / `analyze` / `verify` 三类任务主要仍由同一个 `ResearcherAgent` 执行。
- 因此它更准确地说是“编排器 + 通用研究 worker + 合成器”的架构，而不是很多高度异构角色完全分工的 agent 平台。

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

## Experimental Positioning

这个仓库的实验主线，已经从“能不能生成一份看起来像样的报告”收窄到更具体的问题:

- 证据是否真实
- 来源是否相关
- retrieval 是否能命中 primary evidence
- learned policy 是否真的改善 quality-cost frontier

截至当前版本，最诚实的总结是:

- evidence-readiness 主线已经有正结果
- source quality / citation hygiene 已经有比较清晰的实验接口和审计流程
- learned policy 的 aggregate signal 略正，但还不能写成稳定优于 heuristic

这也是为什么本仓库更适合作为:

- 研究型个人项目
- agentic search / deep research 的实验平台
- 后续论文或扩展工作的基础设施

而不是作为:

- 完整收官的论文结论仓库
- 开箱即用的生产级 research product

## Limitations

当前最重要的限制包括:

- 多角色 agent 的行为差异还不算大
- 一些高级模块更像扩展插槽，而不是彻底闭环的基础设施
- 部分实验依赖具体搜索后端的稳定性
- public benchmark 接入存在任务范式适配问题
- 文档中的设计目标不总是等于代码中的当前实现

更详细的限制和容易误读之处见:

- [docs/09-limitations-and-observations.md](docs/09-limitations-and-observations.md)

## Suggested GitHub Framing

如果你打算把它作为个人项目公开，建议在仓库简介或简历里用类似描述:

> Built a research-oriented deep research agent framework with planning, DAG orchestration, tool-using workers, shared memory, evidence-aware evaluation, and paper-readiness auditing.

或者中文:

> 一个面向复杂研究任务的 Agent 原型系统，支持问题拆解、并发执行、证据管理、报告生成和实验评测，重点探索 evidence-aware deep research workflow。

避免写成:

- “已证明 learned policy 明显优于 baseline”
- “已完成论文级实验收尾”
- “生产级 deep research 平台”

## Tests

仓库包含较完整的测试集合，示例:

```bash
pytest tests/test_policy_head2head_cv.py -q
pytest tests/test_search_policy.py -q
pytest tests/test_run_eval_hotpotqa.py -q
```

## License

MIT
