# 评测体系与实验脚本

本文档介绍仓库中的评测、策略对照和研究报告产物链路。当前评测层既覆盖传统 benchmark，也覆盖 search policy / evidence policy 的训练、head-to-head 对照、cross-fold held-out 验证，以及可发布的 research reporting 快照。

## 1. 评测范围

评测层覆盖以下内容:

- benchmark 运行: `ResearchBench`、HotpotQA
- 模块消融: memory、compressor、adversarial、evolution 等组件开关
- baseline 对比: 单轮 LLM vs 完整 Agent 流程
- policy 对照: search policy / evidence policy 的 off、heuristic、learned 模式
- cache-first 验证: 从已缓存的真实搜索轨迹构造训练集、分折和 held-out 题组
- reporting: readiness audit、output index、brief、dashboard 与 refresh manifest

## 2. 主要脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/run_eval.py` | 标准 benchmark 评测入口 |
| `scripts/run_ablation.py` | 模块与对抗轮数消融 |
| `scripts/run_benchmark.py` | 单轮 LLM vs Agent 对比 |
| `scripts/run_all_experiments.py` | 一键运行完整实验集 |
| `scripts/run_policy_head2head.py` | policy off / heuristic / learned 对照 |
| `scripts/run_policy_head2head_cv.py` | 从 search cache 构造 query-level cross-fold 对照 |
| `scripts/build_search_cache.py` | 从运行结果生成离线 search-cache 数据集 |
| `scripts/analyze_search_cache.py` | 汇总 search-cache 质量与来源信号 |
| `scripts/refresh_research_outputs.py` | 一键刷新 audit、index、brief、dashboard 和 manifest |

## 3. 标准 benchmark

`scripts/run_eval.py` 支持两个 benchmark:

- `research_bench`
- `hotpotqa`

最小运行示例:

```bash
python scripts/run_eval.py \
  --benchmark research_bench \
  --num_questions 2 \
  --config configs/aliyun_smoke.yaml
```

### `research_bench`

流程:

1. 从 `ResearchBench` 取题。
2. 对每道题运行 `run_research()`。
3. 用规则指标评估生成报告。
4. 聚合成 `EvaluationReport`。

每道题会提供:

- `query`
- `expected_topics`
- `ground_truth`
- `domain`

代码中的 `DEFAULT_QUESTIONS` 覆盖 35 道题，领域包括科技、医疗、金融、教育、法律、能源、消费、汽车、游戏、传媒和交叉领域。

### `hotpotqa`

流程:

1. 取多跳问答题。
2. 运行 `run_research()`。
3. 从报告中抽取预测答案。
4. 计算 EM、F1、pass@1 等指标。

## 4. 指标体系

`ResearchBench.evaluate_report()` 会调用规则指标，主要包括:

- factual accuracy
- semantic fact accuracy
- hallucination rate
- citation coverage
- logical consistency
- comprehensiveness

最后会汇总为 composite score。

`evaluation/metrics/composite.py` 提供规则指标和 Judge 指标的组合入口。规则指标用于可重复的程序化评估，Judge 指标用于更接近人工偏好的补充评分。

## 5. 消融与基线对比

`scripts/run_ablation.py` 支持两类消融:

- 模块消融: `full`、`no_adversarial`、`no_compressor`、`no_memory`、`no_evolution`
- 对抗轮数消融: 0、1、2、3 轮

脚本会计算 paired bootstrap 95% CI、p-value 和 Cohen's d。

`scripts/run_benchmark.py` 对比:

- `baseline`: 单轮 LLM 直接回答
- `agent`: 完整 Research Orchestrator 流程

随后再用 Judge 做 head-to-head 评分。

`scripts/run_all_experiments.py` 会串联模块消融、对抗轮数消融、标准评测集、多领域对比、Agent vs LLM、HotpotQA 和 Judge 深度评分。

## 6. Policy Head-to-Head

`scripts/run_policy_head2head.py` 用同一批题对比 policy 模式:

- `off`
- `heuristic`
- `learned`

示例:

```bash
python scripts/run_policy_head2head.py \
  --config configs/aliyun_smoke.yaml \
  --num_questions 5 \
  --modes off,heuristic,learned \
  --search-policy-model-path artifacts/search_policy.json \
  --output_dir outputs/policy_head2head
```

输出会包含 JSON 和 Markdown 摘要，便于比较各模式的质量、引用和工具调用信号。

## 7. Cross-Fold Policy 验证

`scripts/run_policy_head2head_cv.py` 用已有 search cache 构造 query-level folds:

1. 从 JSONL cache 中筛出目标 `source_label`。
2. 按 query 和 domain 构造 deterministic folds。
3. 对每折训练一个 search policy。
4. 可选执行 held-out head-to-head。
5. 聚合 `cv_summary.json` 和 Markdown 报告。

仅构造 folds 和训练模型:

```bash
python scripts/run_policy_head2head_cv.py \
  --config configs/real_evidence_fast.yaml \
  --cache-inputs outputs/search_cache \
  --source-label natural_browser \
  --output-dir outputs/policy_head2head_cv \
  --num-folds 4
```

执行 held-out 对照时增加:

```bash
python scripts/run_policy_head2head_cv.py \
  --config configs/real_evidence_fast.yaml \
  --cache-inputs outputs/search_cache \
  --source-label natural_browser \
  --output-dir outputs/policy_head2head_cv \
  --num-folds 4 \
  --run-head2head
```

如果 readiness 检查失败，脚本会在摘要中记录 preflight blocker。只有在明确接受小样本或弱信号训练时才使用 `--force-train`。

## 8. Search Cache 与 Evidence 数据

`scripts/build_search_cache.py`、`scripts/analyze_search_cache.py`、`scripts/resurface_search_cache_citations.py` 和 `scripts/split_search_cache_dataset.py` 用于把运行输出整理为可复用的 policy 数据。

常见流程:

```bash
python scripts/build_search_cache.py \
  --config configs/aliyun_smoke.yaml \
  --num_questions 5 \
  --output_dir outputs/search_cache \
  --output_file outputs/search_cache/search_cache.jsonl

python scripts/analyze_search_cache.py \
  --inputs outputs/search_cache/search_cache.jsonl
```

cache-first 流程的目标是让 policy 训练和 cross-fold 验证尽量复用已落盘的真实检索轨迹，减少不可重复的 live API 波动。

## 9. Research Reporting 链路

仓库提供一键刷新脚本:

```bash
python scripts/refresh_research_outputs.py \
  --outputs_dir outputs
```

该命令会生成:

- `outputs/research_audit/research_readiness_audit.json`
- `outputs/research_index/research_output_index.json`
- `outputs/research_brief/research_brief.json`
- `outputs/research_dashboard/research_dashboard.json`
- `outputs/research_dashboard/research_dashboard.md`
- `outputs/research_dashboard/research_dashboard.html`
- `outputs/research_refresh_manifest.json`

如果需要从一个目录读取历史实验产物、写入另一个目录，可以使用:

```bash
python scripts/refresh_research_outputs.py \
  --source_outputs_dir outputs \
  --outputs_dir outputs_snapshot
```

报告链路会索引 policy artifacts、head-to-head 结果、cross-fold `cv_summary.json`、readiness audit 和 dashboard headline。GitHub Actions 中的 `Research Report` 与 `Research Snapshot` 工作流也复用这条链路。

## 10. 输出目录

常见输出目录包括:

- `outputs/evaluation/`
- `outputs/experiments/`
- `outputs/policy_head2head/`
- `outputs/policy_head2head_cv/`
- `outputs/research_audit/`
- `outputs/research_index/`
- `outputs/research_brief/`
- `outputs/research_dashboard/`

`outputs/` 通常是本地或 CI 产物目录，不应依赖其中的本机临时文件来通过测试。需要稳定测试数据时，应使用 fixture 或显式构造的临时目录。

## 11. 本地验证建议

开发评测或 reporting 相关代码时，建议至少运行:

```bash
pytest tests/test_policy_head2head.py tests/test_policy_head2head_cv.py -q
pytest tests/test_index_research_outputs.py tests/test_render_research_brief.py tests/test_render_research_dashboard.py tests/test_refresh_research_outputs.py -q
python scripts/refresh_research_outputs.py --outputs_dir outputs_local
```

提交前仍建议运行完整测试:

```bash
pytest -q
python -m compileall src scripts evaluation
```
