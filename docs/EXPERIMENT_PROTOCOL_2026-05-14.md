# DeepResearch Agent 下一轮实验协议

日期：2026-05-14

这份文档不是进度汇报，而是把接下来该怎么做实验写死，避免继续“边改边试、很难归因”。

---

## 1. 当前唯一合理的主问题

截至 2026-05-14，我们现在真正要回答的问题只有一个：

> **在 long-form deep research 场景里，learned search policy 能否在不牺牲 citation grounding 的前提下，改善质量-成本 frontier？**

这里的 `search policy` 暂时只指：

- `search`
- `browser`
- `stop`

不包括：

- retriever route selection
- query rewrite
- summary-state RL
- full browser RL

---

## 2. 当前已经知道的事实

### 2.1 能力层面

当前仓库已经具备：

- `policy_trace`
- `search_cache`
- offline `train_policy`
- runtime `search_policy`
- `heuristic / learned` CLI 对照

所以现在不缺“实验基础设施”，缺的是：

- 更可信的数据
- 更干净的对照
- 更清楚的成功标准

### 2.2 可信度层面

今天发现了一个关键问题：

> 之前 CLI 的 `--search-policy-mode heuristic` 存在模式漏洞，只要本地已有 `artifacts/search_policy.json`，实际运行就可能加载 learned model。

这意味着：

- 之前部分 `heuristic vs learned` 数字不能直接当研究结论；
- 旧结果最多只能算“历史运行记录”；
- 修复模式语义之后重新跑的对照，才是当前可信基线。

### 2.3 修复后最小可信对照

本轮修复后，重新运行了：

```bash
python scripts/run_policy_head2head.py \
  --config configs/aliyun_smoke.yaml \
  --num_questions 2 \
  --modes heuristic,learned \
  --search-policy-model-path artifacts/search_policy.json \
  --output_dir outputs/policy_head2head_20260514_fixed
```

输出：

- `outputs/policy_head2head_20260514_fixed/head2head_20260514_153412.md`
- `outputs/policy_head2head_20260514_fixed/head2head_20260514_153412.json`

结果摘要：

- `heuristic`
  - `avg_composite = 0.494`
  - `avg_factual = 0.225`
  - `avg_citation = 0.000`
  - `avg_policy_score = 0.325`
  - `avg_tool_calls = 6.0`
  - `avg_browser_calls = 3.0`
  - `avg_tokens = 12025`
- `learned`
  - `avg_composite = 0.512`
  - `avg_factual = 0.225`
  - `avg_citation = 0.000`
  - `avg_policy_score = 0.448`
  - `avg_tool_calls = 6.5`
  - `avg_browser_calls = 0.0`
  - `avg_tokens = 14587`

当前最重要的 insight 不是 `+0.019` 的 composite，而是：

> **learned policy 目前主要学到的是“少开 browser，多继续 search”，但这还没有带来 citation gain，反而增加了 token cost。**

这说明当前 learned behavior 更像：

- search-loop preference

而不是：

- evidence-grounded browsing strategy

---

## 3. 当前假设应该怎么收窄

不要同时追四个假设。当前只保留两个。

### H1

> learned policy 能减少无效动作，而不是单纯减少动作。

验证信号：

- `search_policy_score` 上升
- `avg_tool_calls` 不明显恶化
- `avg_tokens` 不明显恶化

### H2

> learned policy 学到的不是“更早停”或“只会重复 search”，而是更好的 evidence gathering。

验证信号：

- `citation_coverage` 不下降
- `browser_calls` 不是被完全压到零
- `factual_accuracy` 至少不下降

如果 H1 成立而 H2 不成立，那么当前结果仍然不够像研究结论，只能说明：

> 我们学到了一种动作偏好，而不是学到了一种更好的研究策略。

---

## 4. 下一轮只允许改一件事

下一轮不允许同时改 reward、route、rewrite、summary state。

只允许做下面这一件事：

> **扩充 `search_cache`，尤其补足 `browser` 类样本，然后在完全相同配置下重训并复跑 head-to-head。**

原因：

1. 当前训练数据极小，且 `browser` 类稀缺。
2. 修复后最小对照显示 learned 几乎把 `browser` 压没了。
3. 如果不先补数据，就无法判断：
   - 是 reward 有问题；
   - 还是模型根本没见过足够多的 browser 正例。

---

## 5. 下一轮实验模板

### Round A: 扩充数据，不改算法

固定不变：

- `configs/aliyun_smoke.yaml`
- `planner / summarizer / tool` 配置
- reward 定义
- evaluation 口径

唯一动作：

- 生成更大的 `search_cache`

目标不是“大”，而是“足够看出动作分布”：

- 先到 `8-12` 个 query
- 先观察 `search / browser / stop` 分布
- 如果 `browser` 仍接近 `0-1`，不要急着训第二版论文结论

### Round B: 只重训一次

固定不变：

- 模型结构
- 训练脚本
- 学习率级别

允许调整：

- 训练输入数据

目标：

- 看 `browser` 类是否开始被学到

### Round C: 只跑 `heuristic vs learned`

固定不变：

- 同一题集
- 同一 config
- 同一模型后端

输出必须包含：

- `avg_composite`
- `avg_factual`
- `avg_citation`
- `avg_search_policy_score`
- `avg_tool_calls`
- `avg_browser_calls`
- `avg_tokens`
- `avg_elapsed_seconds`

---

## 6. 什么结果才算值得继续

### Go

下面两类结果，任意一类出现，都值得继续：

1. `avg_composite` 提升，且 `citation_coverage` 不下降。
2. 质量基本持平，但 `tool_calls / tokens / time` 明显下降。

### No-Go

下面这些结果即使表面上“分高了”，也不要继续吹：

1. `search_policy_score` 提升，但 `citation_coverage` 持续为零或明显下降。
2. learned 把 `browser` 基本压成零。
3. learned 只是把 `search` 调用变多，`tokens` 和 `time` 一起上升。
4. 只在 `2-3` 个 query 上波动，没有稳定趋势。

---

## 7. 当前最应该防的误判

### 误判 1

“composite 涨了，所以 learned 更好。”

不对。因为当前 composite 可能还没有足够惩罚 citation failure。

### 误判 2

“tool 数少了，所以策略更聪明。”

不对。少动作可能只是更早停，或者跳过了该开的网页。

### 误判 3

“search policy score 涨了，所以研究问题解决了。”

不对。`search_policy_score` 本身可能偏向过程收敛，而不是证据质量。

---

## 8. 外部有效性边界

当前我们主要在仓库内置 `ResearchBench` 上做规则评测。

这足够支持工程迭代，但还不够支持外部研究 claim。原因是：

1. `ResearchBench` 是内部题集，不是社区通用 benchmark。
2. 当前指标以 rule-based evaluation 为主。
3. citation 为零时，很多“质量提升”结论其实很脆弱。

在相关工作层面，至少要意识到外部评价环境已经在往更开放的 deep research benchmark 走，例如：

- `DeepResearch Bench`: https://arxiv.org/abs/2506.11763
- `DeepResearchGym`: https://arxiv.org/abs/2506.11957
- `BrowseComp`: https://arxiv.org/abs/2504.12516

这不意味着现在就去复现它们，而是意味着：

> **在只用内部题集时，我们最多说“在当前仓库设定下有效”，不能直接说“对 deep research 普遍有效”。**

---

## 9. 当前阶段最诚实的表述

截至 2026-05-14，当前最诚实的结论应该是：

> 我们已经把 long-form deep research 的局部搜索控制问题做成了一个可运行、可训练、可对照的实验对象；但当前 learned policy 还没有证明自己学到的是更好的 citation-grounded research behavior。

这句话比“我们已经做出了新方法”更准确，也更有研究价值。
