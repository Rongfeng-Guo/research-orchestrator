# Paper Draft v0

日期：2026-05-21

工作标题：

**Toward Paper-Grade Deep Research Agents: Evidence Integrity Gates, Source Relevance Audits, and Official-Source-Aware Retrieval**

## 1. 摘要草稿

Deep research agents often appear strong in demos while failing on the part that matters most for publishable claims: whether their cited evidence is real, relevant, and reproducible. We study this gap in a research-agent codebase evolved from a public open-source project into a paper-oriented experimental system. Our current system adds three layers of experimental hygiene on top of a multi-step research agent pipeline: evidence-integrity auditing, source-relevance auditing, and official-source-aware retrieval that can prioritize explicit primary URLs given in the query. We also build a real-evidence query bank with train/heldout splits and a paper-readiness audit that prevents overclaiming from visually plausible but weakly grounded runs. Initial real-search probes show that citation rendering and mock-source filtering can already work well, but keyless HTML search backends remain too unstable to support a headline claim. These results narrow the core research problem: paper-grade deep research is bottlenecked less by post-hoc citation formatting than by reliable retrieval of relevant primary evidence. We release the benchmark artifacts, audits, and runner interfaces needed to reproduce this conclusion and to support the next stage of controlled train/heldout experiments once a reliable search backend is available.

## 2. 要解决的问题

我们现在真正想回答的不是：

- “这个 agent 会不会写报告”

而是：

- “它给出的证据是不是真的”
- “这些来源和 query 是否真的相关”
- “实验结果是否足够干净，能支撑论文级结论”

这条线和单纯做一个 deep research demo 的区别在于：

1. 不把漂亮报告误当成高质量证据；
2. 不把真实 URL 误当成相关来源；
3. 不把少量跑通的 case 误当成可发表主结论。

## 3. 方法草稿

### 3.1 Base system

系统仍然保留原始项目的主结构：

- planner 负责拆任务；
- researcher 负责多轮工具调用；
- shared memory 保存中间证据；
- summarizer 汇总报告；
- red/blue repair 处理低置信度输出。

### 3.2 Evidence integrity layer

我们在搜索缓存和分析层显式区分：

- mock source
- real source
- cited source
- citation density
- citation quality

对应目标是先确保“被引用的东西是真来源”，而不是凭空 hallucinate。

### 3.3 Source relevance layer

我们新增 source relevance gate，判断来源是否和 query、expected topics、实际使用查询词存在足够重合。核心思想很简单：

- 真来源不等于好来源；
- 如果 URL 来自真实网站但主题跑偏，仍不能支撑 headline claim。

### 3.4 Official-source-aware retrieval

我们进一步补了一层工具能力：

- 如果 query 自带显式官方 URL，则搜索工具先将这些 URL 作为 primary seeds；
- HTML 页面会抽取 title、description、首段文本；
- PDF 或其他非 HTML 内容，也会保留为最小可用 seeded source；
- 然后与搜索后端结果做 URL 去重和重排。

这一层不是为了绕开搜索，而是为了测试：

> 当用户已经明确给出 primary-source hint 时，系统能否真正消费这部分结构化先验。

一个后来才暴露、但很关键的实现细节是：

- 仅在 `web_search` 工具层支持 seeded retrieval 还不够；
- 如果 agent 在调用工具前，把原始 query 压缩成不带 URL 的短搜索词，那么 primary-source hint 仍然会在执行层丢失。

因此我们又补了 agent-side propagation：

- 从原始任务描述和上下文抽取显式 URL；
- 当模型发出的 `web_search` query 丢失这些 URL 时，自动将其补回工具参数；
- 同时记录 `original_query_text` 与最终执行的 `query_text`，把“模型原始意图”和“最终执行检索”显式区分开。

## 4. 数据与协议草稿

### 4.1 Real-evidence query bank

我们已构建：

- `data/queries/real_evidence_bank_20260520.jsonl`

特点：

- 40 条 query
- 10 个领域
- 更强调真实世界争议、政策、产业和技术 claims

### 4.2 Train / heldout split

已稳定切分为：

- train 30
- heldout 10

并由 `scripts/run_real_evidence_iteration.py` 支持双路运行。

### 4.3 Fast probe

我们另外保留 4 条 `real_evidence_fast_probe_20260520.jsonl` 作为轻量验证集，用于检查：

- 检索是否能先碰到官方来源；
- citation 与 relevance gate 是否改善；
- 工具预算约束下是否仍能保有最基本的 grounding。

## 5. 当前结果草稿

### 5.1 已经成立的部分

从 2026-05-20 的 real-search probe 看：

- `real_source_query_ratio = 1.0`
- `mock_source_query_ratio = 0.0`
- `avg_citation_quality_score = 0.78`

这说明：

- citation resurfacing 和 source inventory 已经不再只是 demo 级别；
- 系统能够在部分 case 中产出真实来源、真实引用、真实链接。

### 5.2 尚未成立的部分

同一轮 probe 里：

- `avg_source_relevance = 0.10`
- fast probe 里甚至只有 `0.05`

这意味着：

- 真来源很多，但相关来源不够；
- 系统更像“拿到了很多真的网页”，而不是“拿到了真正回答问题的证据”。

但 seeded + agent propagation 的单题验证也给出了一个更细的结论：

- 一旦 primary-source hint 真正进入完整 agent loop，
- 官方来源可以稳定进入 `top_urls` 和最终 `sources`；
- `relevant_source_query_ratio` 能从 `0.0` 抬到 `1.0`（单题 smoke）；
- `avg_relevant_sources_per_success_query` 能达到 `2.0`（单题 smoke）。

这说明当前问题并不是“official-source-aware retrieval 完全无效”，而是：

1. 它先前没有真正贯穿 agent 层；
2. 即便贯穿了，单跳预算仍不足以读取正文条款；
3. 真正的条款级 evidence extraction 还依赖 `search + browser` 两跳链路。

但在 2026-05-21 的后续两跳 smoke 中，我们又得到了一条更有分量的中间结论：

- 当显式 primary URLs 被真正传到工具层；
- 当 runtime policy 能把 agent 从 seeded search 引到 official HTML page；
- 当 browser 实际读取正文而不是停在 snippet；

则单题 smoke 已经可以达到：

- `headline_claim_ready = true`

在 4 条 fast probe 的状态修正版上，3 条成功样本同样满足：

- `browser_query_coverage = 1.0`
- `relevant_source_query_ratio = 1.0`
- `avg_source_relevance = 0.2083`
- `avg_relevant_sources_per_success_query = 2.6667`
- `headline_claim_ready = true`

此时 paper-readiness 仍然 blocked，但阻塞原因已经明显收缩为：

- success query 数量太少，尚未达到正式训练/比较门槛；
- 而不再是 relevance / citation / browser coverage 本身不过关。

这条结果很重要，因为它改变了研究叙事的重点：

> 先前我们担心的是“方法是不是根本无效”；  
> 现在更合理的判断是“方法在 smoke 级别已经显示可行，但还没有足够样本支撑正式 headline claim”。

后续修复又进一步强化了这个结论。

首先，browser 侧补上了对 `content-encoding: br` 页面的稳健处理：

- 默认避免请求 brotli 编码；
- 若站点仍返回压缩 payload，则回退到原始字节抓取并手动解压。

这解决了 GitHub 官方研究博客这类关键 primary source 先前会把 `fast_ai_code_001` 直接打成失败的问题。

其次，planner 侧增加了两个防抖机制：

- prompt 明确禁止把完整 URL 复制进 `search_hints`；
- 若规划输出因长 URL 被截断而不可解析，则自动退回一个确定性的单 `search` 任务。

这意味着 fast probe 的失败不再轻易来自“planner 输出格式抖动”，而更集中地反映 retrieval / evidence 本身的问题。

在这两处修复后，4 条两跳 seeded-browser fast probe 的合成结果达到了：

- `num_success = 4`
- `browser_query_coverage = 1.0`
- `relevant_source_query_ratio = 1.0`
- `avg_source_relevance = 0.2646`
- `avg_relevant_sources_per_success_query = 3.25`
- `avg_citation_quality_score = 0.8229`
- `headline_claim_ready = true`

而 paper-readiness audit 仍然阻塞的唯一核心原因是：

- `success_queries_below_threshold(4 < 8)`

这使当前最诚实、也最有力的论文叙事变成：

> 在 explicit primary URLs、agent-side propagation、official-source-aware seeded retrieval、以及 `search + browser` 两跳链路全部打通后，headline-level evidence quality 已可在 smoke 集上达到门槛；当前剩余主阻塞主要是样本量，而不是方法本身失效。

### 5.3 当前最诚实的结论

当前最可写进论文的一句话不是：

> 我们已经做出了强于现有方法的 deep research agent。

而是：

> 我们构建了一套把 deep research demo 变成 paper-grade experiment 所必需的 evidence hygiene pipeline，并用真实检索探针表明，当前主要瓶颈是高相关 primary evidence retrieval，而不是 citation rendering。

## 6. 实验表与图的预留位

### Table 1

系统版本对比：

- original demo pipeline
- + evidence integrity gate
- + source relevance gate
- + official-source-aware retrieval

指标建议：

- success rate
- real source ratio
- relevant source ratio
- citation coverage
- citation density
- paper readiness

### Table 2

检索后端对比：

- mock
- bing_html
- duckduckgo
- reliable API backend future run

### Figure 1

一次 query 的 evidence flow：

- query
- search / browser
- source inventory
- citation resurfacing
- relevance audit
- paper-readiness gate

### Figure 2

不同后端下：

- citation quality 可达
- relevance gate 未过

用来说明“当前真正卡住的是 retrieval relevance”。

## 7. 目前离 CCF-A 还差什么

最关键的不是“再多写一点方法描述”，而是以下几项实验事实仍未补齐：

1. 需要可靠搜索 API，而不是只靠 keyless HTML fallback。
2. 需要在 40 条 real-evidence bank 上跑出干净的 train/heldout 结果。
3. 需要 relevance gate 稳定达线，而不是只在个别 case 看起来不错。
4. 需要把 primary-source-aware search 与 browser follow-up 联通，而不是停在“只确认官方 URL 存在”。
5. 需要 learned policy 与 heuristic policy 的正式 heldout head-to-head。
6. 需要把成本、工具调用预算和质量提升放在同一个实验表里。

除此之外，还有一个很实际的运行环境问题：

- 浏览器端 PDF 提取能力的代码已经补齐；
- 但当前环境缺少 `pdfplumber` / `PyPDF2`；
- 因此 White House 这类官方 PDF 仍然受环境依赖约束。

不过这并没有阻止当前 smoke 结果成立，因为两跳链路已经能优先利用 EUR-Lex 这类 official HTML sources。

所以这份草稿的定位必须非常克制：

- 现在是 paper scaffold；
- 不是 ready-to-submit manuscript；
- 更不是已经达到 CCF-A 说服力的定稿。

## 8. 接下来怎么把这份 draft 写实

下一轮一旦具备可靠搜索 API，优先补：

1. 升级后 fast probe 的 seeded retrieval 结果；
2. 40 条 query bank 的 train/heldout 审计；
3. heuristic vs learned 的 heldout 表格；
4. error analysis：
   - irrelevant real URLs
   - secondary-source domination
   - browser under-use
   - budget truncation side effects

到那一步，这份文档就可以逐步从“草稿”变成真正的论文正文。

## 9. 2026-05-23 补充：8-query readiness 已过，但 no-leak superiority claim 仍未成立

到 2026-05-23，为了把“4-query smoke 过线”推进成更像论文的实验事实，我们又完成了三步：

1. 再补 4 条 explicit-primary-source、HTML-friendly fast probe，形成统一的 8-query benchmark：
   - `data/queries/real_evidence_fast_probe_8_20260522.jsonl`
2. 在合并后的 all-data 8-query cache 上完成正式训练门与 paper-readiness 审计
3. 为 query-level no-leak evaluation 补齐 custom-query head-to-head fallback 与 cross-fold runner

这使得我们终于能把“headline quality 是否达线”与“learned policy 是否真正泛化”拆成两个不同层次的问题。

### 9.1 已经成立的新结果

在 unified 8-query fast benchmark 上：

- `num_success = 8`
- `browser_query_coverage = 1.0`
- `real_source_query_ratio = 1.0`
- `relevant_source_query_ratio = 1.0`
- `avg_source_relevance = 0.3160`
- `avg_relevant_sources_per_success_query = 3.5`
- `avg_citation_quality_score = 0.8246`
- `training_allowed = true`
- `headline_claim_ready = true`
- `paper_readiness.status = PASS`

这意味着一件此前还不能诚实声称的事现在已经可以写进论文草稿：

> 在 explicit primary URLs、agent-side propagation、official-source-aware seeded retrieval、以及 `search + browser` 两跳链路全部打通后，一个小型但真实的 8-query benchmark 已经可以通过 formal evidence-readiness gates。

这和 earlier smoke result 的区别很关键：

- 先前只是“个别 query 或 4-query smoke 看起来可行”
- 现在则是“在一个统一 benchmark 上，正式 gate 已被满足”

### 9.2 仍然不能成立的主张

即便如此，我们仍然**不能**在当前阶段声称：

> learned search policy 已经稳定优于 heuristic baseline

原因有两层。

第一层，是 no-leak heldout 证据仍然太少。

在首个 2-query custom heldout smoke 中：

- heuristic
  - `avg_composite_score = 0.5622`
  - `avg_estimated_token_cost = 41835`
- learned
  - `avg_composite_score = 0.5798`
  - `avg_estimated_token_cost = 38704`

聚合均值对 learned 有利，但逐 query 结果是 mixed：

- `fast_ai_law_001` 上 learned 更好
- `fast_energy_001` 上 heuristic 更好

第二层，是 query-level cross-fold train-side signal 也并不单向。

我们新增了：

- `scripts/run_policy_head2head_cv.py`

并已将当前 8-query benchmark 预切成 4 个 query-level folds。每个 fold 的 train split 都满足：

- `headline_claim_ready = true`

但 step-level validation 相对 heuristic 的比较仍是 mixed：

- fold 1：持平
- fold 2：learned 低于 heuristic
- fold 3：learned 高于 heuristic
- fold 4：learned 低于 heuristic

因此，当前最稳妥的论文表述必须收敛成：

> 我们已经把一个 deep research demo 清理成了可通过 evidence-readiness gates 的 real-evidence benchmark scaffold；但 learned-vs-heuristic superiority 仍需更大 no-leak heldout 与 cross-fold live evaluation 才能成立。

### 9.3 这对论文叙事意味着什么

这组更新实际上让论文故事更健康，而不是更弱。

我们现在可以更清楚地区分三件事：

1. **Evidence hygiene contribution 已成立**
   - source reality
   - source relevance
   - browser grounding
   - paper-readiness gating
2. **Official-source-aware retrieval 在 benchmark 级别已显示可行**
   - 不再只是单题修复
3. **Policy learning claim 仍未闭环**
   - 需要更大、真正 no-leak 的 heldout evidence

这比“把所有改进都压成一个大而模糊的正结论”更像可发表工作，因为它把贡献和未解问题边界都写清楚了。

### 9.4 下一轮最有价值的实验

在当前基础上，最值得继续做的不是再堆方法细节，而是：

1. 用现有 cross-fold runner 跑完 live heldout comparison
2. 把 fast benchmark 从 8 条扩到 12-16 条 explicit-primary-source queries
3. 在更大的 no-leak heldout 上重新审查：
   - learned vs heuristic quality
   - token cost
   - browser usage
   - failure modes by domain

只有到那一步，论文才有资格从：

- “paper-grade benchmark scaffold + strong smoke evidence”

走向：

- “paper-grade benchmark + defensible policy-learning result”

## 10. 2026-05-23 再补充：first cross-fold live heldout 结果如何改变叙事

在补齐 query-level cross-fold runner 之后，我们又继续把当前 8-query benchmark 的 **4-fold live heldout** 真正跑完了：

- `outputs/policy_head2head_cv_fast_probe8_live/cv_summary.json`
- `outputs/policy_head2head_cv_fast_probe8_live/cv_summary.md`

这一步很关键，因为它第一次让当前 8 条 fast benchmark 中的每一条 query 都在 no-leak 条件下被作为 heldout 评估一次，而不是只看一个 2-query split。

### 10.1 聚合结果：learned 有轻微正信号，而且更省

在 8 条 heldout query 的 cross-fold 聚合上：

- heuristic
  - `avg_composite_score = 0.5575`
  - `avg_citation_coverage = 0.2944`
  - `avg_estimated_token_cost = 29231.6`
  - `avg_tool_calls = 7.50`
  - `avg_elapsed_seconds = 84.87`
- learned
  - `avg_composite_score = 0.5628`
  - `avg_citation_coverage = 0.2944`
  - `avg_estimated_token_cost = 20162.6`
  - `avg_tool_calls = 5.38`
  - `avg_elapsed_seconds = 59.64`

换句话说：

- `quality_delta = +0.005`
- `citation_delta ≈ 0`
- `token_delta = -9069`
- `tool_delta = -2.12`
- `elapsed_delta = -25.23s`

这是当前第一次让我们可以比较有把握地写出：

> learned policy 在 no-leak live heldout 上已经出现了轻微的 aggregate quality gain，同时显著降低了平均 token 成本、工具调用数与耗时。

### 10.2 但为什么这还不够写成 superiority claim

原因在于 fold-level variance 仍然偏大。

4 个 folds 的 heldout 对比并不是单向一致：

- fold 01：成本下降，但质量下降
- fold 02：质量下降且成本上升
- fold 03：质量上升，成本基本持平
- fold 04：质量上升且成本大幅下降

因此，虽然 aggregate signal 已从“完全 unclear”推进到“slightly positive”，我们仍然不能把结果写成：

> learned policy stable outperforms heuristic

更准确的说法应当是：

> learned policy now shows an encouraging aggregate efficiency-quality tradeoff, but fold-level variance remains too large to support a stable superiority claim.

### 10.3 这如何改变论文定位

这次 live cross-fold 结果带来的变化，并不是让论文一下子从 scaffold 跳到定稿，而是让结论更细、更可信：

1. **Evidence-readiness claim 已站稳**
   - 8-query benchmark 已通过 formal gates
2. **Policy-learning claim 从“纯 smoke”升级为“有 aggregate signal 的 early result”**
   - 不再只是单个 2-query heldout
3. **但 superiority 仍未闭环**
   - 仍需要更大 heldout 与更多 cross-fold observations

这意味着论文当前最适合的表述已经从：

- “我们有一个看起来可行的 learned policy”

更新为：

- “我们有一个通过 evidence gates 的 real-evidence benchmark scaffold，并在 first cross-fold live heldout 上观察到 learned policy 的轻微正向 aggregate signal，但尚未达到稳定 superiority 的证据门槛。”

### 10.4 真正还差什么

如果要把这篇工作推进到更接近 CCF-A 说服力的版本，接下来最值得投入的仍然是：

1. 把 fast benchmark 从 8 条继续扩到 12-16 条 explicit-primary-source heldout queries
2. 在更大的 no-leak cross-fold live heldout 上复核 learned 的 aggregate gain 是否仍成立
3. 做更细的 error analysis，尤其针对：
   - premature stopping
   - under-browsing on legal / energy queries
   - cost-heavy failures in fold 02
   - reward mismatch between composite quality and search-policy score

只有当这些问题都被补齐后，当前这条线才更可能从：

- “可信的 benchmark / protocol 论文”

走向：

- “可信的 benchmark / protocol + learned policy improvement 论文”

## 11. 2026-05-23 再补一层：为什么先修 agent-side guardrails，而不是继续调模型

在 first cross-fold live heldout 之后，我们做的下一步不是立刻继续扩数据或重训模型，而是先在 agent 接入层补了一层 **search-policy guardrails**：

- `src/agents/researcher.py`

原因很直接：当前 observed variance 里，有一部分并不像“模型学得不够好”，而更像是缺少最低限度的执行护栏。

具体来说，fold-level failure mode 已经足够清楚：

1. 有的 query 还没真正检索就停了
2. 有的 query 已经拿到 browser 证据，却还在做高重叠的后续检索
3. 有的 explicit-primary-source query 在 stop 前甚至没打开过任何 primary page

这些错误如果直接计入 learned-vs-heuristic 比较，会混淆两个层次的问题：

- policy representation / training 是否有效
- runtime execution hygiene 是否足够稳定

因此我们先补了三条保守护栏：

1. **no-search-yet force search**
   - 若还没有任何 search/browser 行为，不允许直接结束
2. **seeded-query force browser before stop**
   - 若 query 带显式官方 URL，且已检索到候选页面但尚未打开原文，则 stop 前必须至少读一次源页
3. **repetitive-search force stop**
   - 若已经发生 `search + browser`，且后续查询高度重复或预算明显上升，则停止继续搜，直接总结

这一步的意义不是“让 learned policy 看起来更好”，而是把实验定义得更合理：

> 在 explicit-primary-source real-evidence benchmark 上，系统至少应满足最基本的 execution hygiene，之后再比较 learned 与 heuristic 的策略差异。

从论文叙事上看，这个修正很重要，因为它让我们对贡献边界的表述更严谨：

- learned policy 负责改善 route/stop 决策
- agent-side guardrails 负责防止显然不合理的执行失真

二者并不冲突，反而更接近真实系统论文的写法：  
**policy learning on top of explicit execution-safety constraints**。

当前这层修正已经通过全量回归：

- `106 passed in 7.77s`

因此下一轮更有价值的实验，不是再改一轮 reward，而是：

1. 在同一 8-query cross-fold live heldout 上重跑
2. 检查 fold 01 的 premature-stop 是否下降
3. 检查 fold 02 的 over-search 是否下降
4. 再决定 learned aggregate gain 到底是在变稳，还是仍然只是偶然波动

### 9.5 live cross-fold 补充结果

2026-05-23，现有 8-query benchmark 的 4-fold live heldout comparison 已经跑完：

- `outputs/policy_head2head_cv_fast_probe8_live/cv_summary.json`
- `outputs/policy_head2head_cv_fast_probe8_live/cv_summary.md`

聚合结果是：

| mode | success | avg_composite | avg_citation | avg_tokens | avg_tool_calls |
|---|---:|---:|---:|---:|---:|
| heuristic | 8/8 | 0.557 | 0.294 | 29231.6 | 7.50 |
| learned | 8/8 | 0.563 | 0.294 | 20162.6 | 5.38 |

也就是说，learned policy 在聚合均值上略高，并且明显更省 token：

- `quality_delta = +0.005`
- `token_delta = -9069.0`
- `tool_delta = -2.12`

但 fold-level 结果仍然 mixed：

- fold 01：learned 降成本但也降质量和 citation coverage
- fold 02：learned 质量更低且成本更高
- fold 03：learned 质量和 citation coverage 更高，但成本略高
- fold 04：learned 质量和 citation coverage 更高，且大幅降成本

因此这轮 live CV 结果应该写成：

> Learned policy shows a promising cost-quality signal on the 8-query benchmark, but the current no-leak evidence is still mixed and does not justify a stable superiority claim over the heuristic policy.

这条补充让论文叙事更清楚：当前已经可以主张 evidence-readiness scaffold 成立，但 learned policy 的 headline claim 还需要扩展到 12-16 条以上、更稳定的 no-leak heldout。

## 10. 公开 Benchmark Sanity Check：HotpotQA

为了避免只在自建 fast probe 上得出结论，我们接入了公开 HotpotQA `dev_distractor` 作为外部 sanity check。

数据：

- `data/public_benchmarks/hotpotqa/hotpot_dev_distractor_v1.json`
- source: `http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json`

这组实验不应该被写成 deep research 主实验，因为 HotpotQA 是短答案、多跳 QA benchmark，不是 long-form evidence research benchmark。但它很适合回答一个诊断问题：

> 当前系统接公开 benchmark 失败时，是模型不会读 context，还是 deep research 编排方式不适合短答案 QA？

我们因此加入了两种 HotpotQA 模式：

1. `agent`
   - 走完整 planner / researcher / summarizer 链路；
   - 更接近当前 deep research agent。
2. `context_qa`
   - 不调用外部搜索；
   - 只基于 HotpotQA 提供的 context 直接回答；
   - 作为 context-grounded QA baseline。

固定 `seed=42` 的 10 条 `context_qa` smoke 结果：

- `exact_match = 0.9`
- `f1 = 0.9`
- `pass@1 = 0.9`
- `gold_entity_coverage = 1.0`

同 seed 下，完整 agent 的 2 条 smoke 结果：

- `exact_match = 0.5`
- `f1 = 0.5667`
- `pass@1 = 0.5`
- `gold_entity_coverage = 1.0`

随后我们用固定 `seed=42` 跑了 50 条 public HotpotQA 子样本，对比 closed-book 与 context-grounded baseline：

| mode | n | EM | F1 | pass@1 | gold entity coverage |
|---|---:|---:|---:|---:|---:|
| `closed_book_qa` | 50 | 0.200 | 0.297 | 0.200 | 0.398 |
| `context_qa` | 50 | 0.600 | 0.669 | 0.600 | 0.878 |
| `context_agent` | 50 | 0.660 | 0.798 | 0.660 | 0.895 |
| `context_agent_v9` | 200 | 0.765 | 0.865 | 0.765 | 0.892 |

这个结果支持一个外部 benchmark 诊断结论：

> Provided evidence context substantially improves multi-hop QA performance, but our current agent orchestration is not optimized for short-answer QA scoring.

The intermediate `context_agent` mode adds a lightweight evidence-notes step before answer extraction. Its improvement over `context_qa` suggests that explicit evidence structuring can help even on a public short-answer benchmark:

- EM: `+0.060`
- F1: `+0.129`
- gold entity coverage: `+0.017`

它也给出了一个明确的边界：这还不是正式 SOTA claim。HotpotQA 官方 leaderboard 的 distractor setting 顶部方法 Beam Retrieval 的 Answer EM/F1 约为 `72.69 / 85.04`。我们的 `context_agent_v9` 在固定 `seed=42` 的 200 条 dev-distractor 子样本上达到 `76.5 / 86.5`，数值上超过这个 leaderboard 条目，但它不是 blind test server 结果，也没有 supporting-fact / joint metrics。因此，这部分最多可以写成 strong external diagnostic result，而不是正式 SOTA 对比表。

After adding supporting-fact and joint metrics, the boundary becomes even clearer. On a fixed 100-example dev-distractor subset with indexed context and supporting-fact extraction:

| n | Answer EM | Answer F1 | SP EM | SP F1 | Joint EM | Joint F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.750 | 0.835 | 0.230 | 0.604 | 0.140 | 0.516 |

Thus, the system is much stronger as an answer extractor than as a supporting-fact selector. Any SOTA-oriented claim must therefore address supporting-fact selection and joint scoring, not only answer EM/F1.

这个结果说明：

- 公开 HotpotQA 数据和 context-grounded QA baseline 已经可用；
- 当前 LLM 在给定 context 下可以完成多跳 QA；
- 完整 deep research agent 的主要问题不是“完全找不到答案”，而是会把短答案 QA 任务转成报告式 research workflow；
- 这种 workflow 会覆盖答案实体，但常输出完整句子或报告段落，和 HotpotQA 的 EM/pass@1 标准不匹配。

因此，HotpotQA 在论文中的合理定位是外部诊断实验，而不是主 claim：

> It demonstrates that the system's evidence layer can consume public benchmark context, while also exposing a mismatch between long-form research orchestration and short-answer QA evaluation.

后续如果要把公开 benchmark 写得更硬，需要实现 agent-side context-grounded execution：

- 将 dataset context 放入受控 evidence pool；
- 禁止或弱化外部 search；
- 要求 subtask 只引用 context；
- 在 summarizer 后增加 answer extraction；
- 再跑 50-100 条 HotpotQA，并报告 agent vs context_qa 的差距。
