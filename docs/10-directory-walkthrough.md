# 目录逐项说明

这一篇按目录来讲仓库，不讲抽象概念，只讲“每个目录里放了什么、为什么存在”。

## 1. 仓库根目录

### `README.md`

仓库级概览文档，负责讲愿景、模块划分和运行方式。

### `pyproject.toml`

项目元信息与依赖声明，包含：

- 包名
- Python 版本要求
- 依赖
- 可选依赖
- 命令行脚本入口

### `requirements.txt`

更直接的依赖安装列表。

### `.env.template`

后端和工具的环境变量模板。

### `.env.tools.template`

更聚焦工具层的环境变量模板。

## 2. `configs/`

这是全局配置中心。

### `configs/default.yaml`

最重要的配置文件，决定：

- 模型路由
- 编排参数
- 记忆参数
- 压缩参数
- 对抗参数
- 演化参数
- 工具参数

### 其他子目录

- `configs/agents/`
- `configs/planner/`
- `configs/tools/`
- `configs/evolution/`

它们更多承载细分模块配置和实验性配置。

## 3. `src/`

核心源码目录。

### `src/core/`

系统运行壳层。

主要文件：

- `runner.py`
- `judge.py`
- `ablation.py`

### `src/orchestrator/`

系统的调度中枢。

主要文件：

- `orchestrator.py`
- `agent_pool.py`
- `schemas.py`

### `src/planner/`

任务拆解层。

主要文件：

- `planner.py`
- `dag.py`
- `budget_tracker.py`

### `src/agents/`

执行层 Agent。

主要文件：

- `base_agent.py`
- `researcher.py`
- `summarizer.py`

### `src/tools/`

工具层。

主要文件：

- `web_search.py`
- `browser.py`
- `arxiv_reader.py`
- `file_reader.py`
- `calculator.py`
- `code_sandbox.py`
- `notepad.py`

### `src/memory/`

记忆层。

### `src/evidence/`

证据策略层。

主要文件：

- `policy.py`

它负责把共享记忆进一步整理成 `EvidenceSnapshot` 和候选研究动作，是当前仓库里比较新的能力层。

### `src/compressor/`

上下文压缩层。

### `src/adversarial/`

对抗优化层。

### `src/models/`

模型适配层。

### `src/evolution/`

自进化层，当前更像原型性扩展方向。

### `src/utils/`

工具函数和基础设施。

## 4. `evaluation/`

评测层源码。

### `evaluation/benchmarks/`

放 benchmark 定义：

- `research_bench.py`
- `hotpotqa.py`

### `evaluation/metrics/`

放规则指标、Judge 指标、统计指标和综合指标。

### `evaluation/report.py`

统一评测报告封装。

## 5. `scripts/`

命令行脚本入口目录。

### 用户最常用

- `run_single.py`
- `run_repl.py`

### 研究和实验常用

- `run_eval.py`
- `run_ablation.py`
- `run_benchmark.py`
- `run_all_experiments.py`
- `run_judge.py`
- `run_evolution.py`

## 6. `tests/`

测试和演示脚本目录。

主要文件：

- `validate_env.py`
- `demo.py`
- `continuous_test.py`

注意这里不完全是传统单元测试组织方式，也包含演示和流程验证脚本。

## 7. `outputs/`

运行产出目录。

你会在这里看到：

- 研究报告
- 实验汇总
- benchmark 结果

## 8. 怎么用目录结构反推系统设计

从目录结构能看出，作者把系统拆成了三大面：

### 运行主线

- `scripts/`
- `src/core/`
- `src/orchestrator/`
- `src/agents/`

### 质量增强

- `src/memory/`
- `src/evidence/`
- `src/compressor/`
- `src/adversarial/`

### 研究实验

- `evaluation/`
- `scripts/run_*`
- `src/evolution/`
