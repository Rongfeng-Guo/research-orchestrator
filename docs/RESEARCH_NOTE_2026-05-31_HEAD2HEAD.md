# Research Note: Learned vs Heuristic Head-to-Head (2026-05-31)

本次尝试的目标是验证：

- `artifacts/search_policy_20260515_natural_train_v1.json`

相对于 heuristic search policy，是否在 query-level head-to-head 中带来更好的质量/成本权衡。

## 实验设置

命令：

```bash
python scripts/run_policy_head2head.py \
  --config configs/real_evidence_fast.yaml \
  --num_questions 3 \
  --modes heuristic,learned \
  --search-policy-model-path artifacts/search_policy_20260515_natural_train_v1.json \
  --output_dir outputs/policy_head2head_natural_train_v1
```

抽样题目：

- `tech_001`
- `tech_002`
- `med_001`

## 结果

本次实验未进入 live execution，原因不是 policy 本身，而是运行前置条件不满足。

预检结果：

- `configs/real_evidence_fast.yaml` 需要 `openai` backend
- 当前仓库 clone 目录下不存在 `.env` 或 `.env.local`
- `openai` backend 对应的 `*_API_KEY` / `*_BASE_URL` 未配置

因此：

- `heuristic`：`0/3` 成功
- `learned`：`0/3` 成功

输出文件：

- `outputs/policy_head2head_natural_train_v1/head2head_20260531_164625.json`
- `outputs/policy_head2head_natural_train_v1/head2head_20260531_164625.md`

## 这次实验的价值

虽然没有拿到质量对比结果，但这次尝试暴露了一个真实问题：

- 现有 head-to-head 实验入口过去会在每个 query 上重复失败，不利于研究复现
- 现在脚本已经支持 backend preflight，并把阻塞原因写入 JSON/Markdown 产物

这使得后续实验可以区分：

- `experiment blocked by environment`
- `experiment executed but policy underperformed`

这是两类完全不同的研究结论。

## 下一步

要得到真正可比较的 learned vs heuristic 结果，需要先满足以下条件之一：

1. 在仓库根目录提供 `.env.local`，配置 `OPENAI_API_KEY` 和/或 `OPENAI_BASE_URL`
2. 改用已配置的 backend，并同步调整 `configs/real_evidence_fast.yaml`
3. 准备可用的 search-cache JSONL，走 cache-first 的 cross-fold evaluation 路径

在当前仓库状态下，更推荐的优先顺序是：

1. 先补 live backend 配置
2. 再跑 3-5 题的小规模 head-to-head smoke
3. 通过后扩到 query-level cross-fold evaluation
