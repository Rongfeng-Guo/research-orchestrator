# DeepResearch Agent 研究定位与实验设计记录

日期：2026-05-14

这份文档整理研究定位、实验范围和设计说明，覆盖以下内容：

- 我们现在到底在做什么实验？
- 这件事和已有工作相比，新意可能在哪里？
- 什么结果值得继续投入，什么结果只是“做了个 baseline”？
- 接下来哪些实验是合理推进，哪些是“乱做实验”？

---

## 1. 一句话结论

当前最合理的研究定位不是：

- “我们做了一个更复杂的 deep research agent”

而是：

> **我们在研究：能否在长报告 deep research 场景里，把搜索动作本身做成一个 budget-aware、citation-aware、可学习的策略，从而在质量-成本 frontier 上优于启发式基线。**

这里的关键词是：

- `budget-aware`
- `search policy`
- `route / stop`
- `long-form deep research`
- `quality-cost frontier`

如果这条线成立，它更像一篇“AI search / agentic RL / deep research policy learning”的论文起点，而不是一次普通功能迭代。

---

## 2. 我们现在在做什么实验

### 2.1 当前实验的最小定义

当前实验不是 end-to-end 浏览器 RL，也不是多模态 web agent。

当前最小实验单元是：

1. 运行 deep research agent。
2. 记录结构化 `policy_trace`。
3. 从真实运行里构建 `search_cache`。
4. 把轨迹展平成 step-level 样本。
5. 训练一个最小的 learned `search / browser / stop` policy。
6. 在线接入 `ResearcherAgent`。
7. 比较 `heuristic` 与 `learned` 的：
   - 报告质量
   - citation 指标
   - 搜索步数
   - 工具调用数
   - 运行成本 / 时间

也就是说，我们现在做的不是“让 agent 更像人”，而是非常具体地研究：

> **在长报告 deep research 中，什么时候继续搜、什么时候停、什么时候该点开页面，这个局部控制问题能不能学出来。**

### 2.2 这不是随机实验

当前实验并不是拍脑袋选的，而是由仓库约束和研究空白共同决定的：

1. 当前仓库已经有：
   - `web_search`
   - `browser`
   - `policy_trace`
   - `reward_breakdown`
   - `search_cache`
   - `run_eval`

2. 当前仓库还没有：
   - 真实浏览器级 RL 训练环境
   - 大规模在线 trainer 闭环
   - 高质量 route labels

3. 所以最合理的下一步不是直接冲最重的 RL，而是：
   - 先把结构化搜索决策学起来；
   - 先验证 learned policy 是否真的改变质量-成本关系；
   - 再决定值不值得进入更重的 route/query policy 或 online RL。

---

## 3. 这件事算“别人没做过的新研究”吗

最短答案：

> **广义上不是全新方向；狭义上有可能做出一个有新意、且可发表的切入点。**

更具体一点：

### 3.1 什么不新

下面这些说法都不能作为新意主张：

- “我们让 LLM 会搜索了”
- “我们用 RL 优化搜索”
- “我们做了 deep research agent”
- “我们做了 browser / stop / search 决策”

原因很简单：这几个方向在 2025-2026 已经有明确相关工作。

### 3.2 什么可能是新意

如果要形成更可信的 novelty，必须把 claim 收窄到下面这种层级：

1. **任务设定的新意**
   - 不是短答案 QA；
   - 而是长报告 deep research。

2. **优化目标的新意**
   - 不是只追求最终答对率；
   - 而是显式优化 `quality / citation / cost / stopping` 的 frontier。

3. **动作空间的新意**
   - 不是纯 token-level CoT；
   - 而是结构化的 `search / browser / stop / route / rewrite` 决策。

4. **训练方式的新意**
   - 不一定一开始就做 online RL；
   - 可以先证明 offline-from-cache 也能稳定改进长报告 agent 的局部搜索策略。

5. **评测主张的新意**
   - 不是只看 EM/F1；
   - 而是看 long-form report 下的 citation faithfulness、tool budget、search policy behavior。

### 3.3 最诚实的定位

当前阶段最诚实的说法应该是：

> 我们不是在发明“search-RL”这个方向，而是在 deep research 这个更长时程、更强调引用和成本的 setting 下，寻找一个还没有被做扎实的、足够聚焦的策略学习切口。

这类定位通常比“这是从未有人做过的全新问题”更可信。

---

## 4. 和已有工作的关系

下面不是完整文献综述，而是最影响当前实验设计的工作地图。

### 4.1 Search-R1

- 论文：Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- 时间：2025-03-12（arXiv 初版）
- 链接：https://arxiv.org/abs/2503.09516

这篇工作的核心点是：

- 搜索不是纯 inference-time prompt 技巧；
- 模型可以通过 RL 学“何时搜、搜什么”；
- 目标更偏 search-enabled reasoning。

对我们的启发：

- 支持“search action 可以学”这个基本前提。

它和我们当前方向的差别在于：

- 它更偏 QA / reasoning benchmark；
- 我们更偏 long-form deep research report；
- 我们还要显式关注 citation 与 cost。

### 4.2 ReSearch

- 论文：ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning
- 时间：2025-03-25（arXiv 初版）
- 链接：https://arxiv.org/abs/2503.19470

这篇工作进一步说明：

- 可以不依赖 reasoning step 的监督标签；
- 直接通过 RL 学 reasoning-search interleaving；
- 搜索操作可以成为推理链的一部分。

对我们的启发：

- 说明“search 决策学习”不必从重监督开始；
- 对我们当前的 offline cache + reward 方向是支持性的。

### 4.3 DeepResearcher

- 论文：DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments
- 时间：2025-04-04（arXiv 初版）
- 链接：https://arxiv.org/abs/2504.03160

这篇工作很重要，因为它明确强调：

- 真实 web 环境和固定 RAG 环境不是一回事；
- deep research agent 需要应对开放网页的噪声和不确定性；
- end-to-end RL 在 real-world environments 中是关键。

对我们的启发：

- 说明最终方向应该尽量往真实环境靠；
- 但也提醒我们：如果现在仓库没有 browser-RL 环境，就不该假装自己已经在做那件最重的事。

### 4.4 WebAgent-R1

- 论文：WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning
- 时间：2025-05-22（arXiv 初版）
- 链接：https://arxiv.org/abs/2505.16421

这篇工作证明：

- 多轮 web interaction 的 RL 是可行的；
- test-time interaction scaling 也很重要。

但它更偏：

- web environment task completion；
- GUI / page interaction；
- success/failure 型任务奖励。

而我们当前更偏：

- long-form report synthesis；
- report grounding；
- citation / budget / stopping。

### 4.5 BrowseComp

- OpenAI 页面：BrowseComp: a benchmark for browsing agents
- 页面日期：2025-04-10
- 页面链接：https://openai.com/index/browsecomp/
- 论文链接：https://arxiv.org/abs/2504.12516

BrowseComp 的重要性在于：

- 它证明浏览任务不能靠简单问答 benchmark 代表；
- 需要 persistence、creativity、多跳信息查找；
- 还说明 test-time compute 和 browsing training 很重要。

但 BrowseComp 主要是：

- 短答案 browsing benchmark；
- 不是 long-form deep research report benchmark。

所以它更多是对“为什么搜索预算和 browsing 过程值得研究”的支持，而不是我们最终的直接任务定义。

### 4.6 SAGE

- 论文：SAGE: Benchmarking and Improving Retrieval for Deep Research Agents
- 时间：2026-02-05（arXiv 初版）
- 链接：https://arxiv.org/abs/2602.05975

这篇工作对我们尤其关键，因为它指出：

- 在 deep research retrieval 里，agent 生成的 sub-query 往往是关键词型；
- 在这个 setting 下，BM25 显著强于某些 LLM retriever；
- 问题不只是 retriever 谁更强，而是 query style 与 route 的匹配。

这直接支持我们当前主线里的一个核心判断：

> **route 选择和 query style 联动，本身就是研究对象。**

### 4.7 DR Tulu

- 论文：DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research
- 时间：2025-11-24（arXiv 初版）
- 链接：https://arxiv.org/abs/2511.19399

这篇工作说明：

- open-ended long-form deep research 不能只靠 RLVR；
- reward 需要 evolving rubrics；
- 长报告训练要关注引用、覆盖、信息新发现，而不是只看终局可验证答案。

它对我们的启发是：

- 当前 reward 设计不能只看最终 composite；
- 如果 learned policy 提升了 search policy score 却伤害 citation coverage，这不是小事，而是 reward 失配信号。

### 4.8 ReSum / WebWeaver

- ReSum：ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization
  - 时间：2025-09-16
  - 链接：https://arxiv.org/abs/2509.13313
- WebWeaver：WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for Open-Ended Deep Research
  - 时间：2025-09-16
  - 链接：https://arxiv.org/abs/2509.13312

这两篇工作告诉我们：

- 长时程搜索的难点不只是 search policy；
- summary state、dynamic outline、memory-grounded synthesis 也会决定最后表现。

它们对当前实验的意义不是“马上去复现”，而是：

- 帮我们划清阶段边界；
- 当前先做 `Query Policy RL + Retriever Router`；
- 后续若这条线成立，再往 `Summary-State RL` 扩展更合理。

### 4.9 DeepResearch Bench / DeepResearchGym

- DeepResearch Bench
  - 时间：2025-06-12
  - 链接：https://arxiv.org/abs/2506.11763
- DeepResearchGym
  - 时间：2025-06-13
  - 链接：https://arxiv.org/abs/2506.11957

这两类工作的重要性不在于“它们和我们的方法完全一样”，而在于它们提醒我们：

- open-ended deep research 需要更像真实研究任务的 benchmark；
- 只在内部题集上跑出一点分数，不足以支撑强 claim；
- process evaluation、source grounding、long-horizon behavior 正在成为评测重点。

对我们当前实验最直接的约束是：

> **当前仓库内置 `ResearchBench` 可以支持迭代，但还不能直接替代更公开、更开放的 deep research benchmark。**

---

## 5. 这条研究线真正应该追求什么结果

### 5.1 不应该追求的结果

下面这些结果即使做出来，也不构成强研究结论：

1. learned policy 在 2-3 道题上偶然赢了。
2. search policy score 提升，但 citation faithfulness 下降很多。
3. 工具调用变少了，但报告质量明显掉了。
4. 只证明“模型会更早停”，却不能说明“停得更合理”。

### 5.2 应该追求的结果

第一阶段最值得追求的结果只有两类：

1. **相同预算下，质量更高**
   - 相同或更少 tool calls / token cost；
   - 更高 composite / factual / citation 指标。

2. **相同质量下，成本更低**
   - 保持相近报告质量；
   - 显著减少 search steps / tool calls / elapsed time。

这两类结果共同对应一个更像论文摘要的 claim：

> 在长报告 deep research 场景中，budget-aware learned search policy 能在 quality-cost frontier 上稳定优于 heuristic baseline。

### 5.3 当前最有价值的单个核心 claim

如果只选一个最值得争取的主结论，我建议是：

> **在 long-form deep research setting 中，learned stop / route policy 能在不降低 citation faithfulness 的前提下，减少搜索步数并提升最终报告质量。**

原因是：

- 这个 claim 足够聚焦；
- 和已有 search-RL 工作能形成区分；
- 又直接对接用户最关心的“质量 vs 成本”问题。

---

## 6. 当前小样本结果该怎么解读

现在已有的 smoke 结果显示：

- learned policy 提升了：
  - `average_composite`
  - `search_policy_score`
  - `factual_accuracy`
- learned policy 同时降低了：
  - `citation_coverage`
- learned policy 还带来了：
  - 更少的 tool calls / action counts
  - 但更长的平均耗时

这组现象本身很有信息量。

### 6.1 可能的正向解读

这说明 learned policy 可能已经学到：

- 更积极地压缩无效搜索；
- 更倾向于提前停在“看起来足够”的状态；
- 至少在当前 reward 下，能更强地优化局部搜索行为。

### 6.2 更重要的负向解读

如果 citation coverage 下滑，就要高度警惕：

- 当前 policy 学到的是“更快停”，不是“更好搜”；
- 当前 reward 或 label construction 可能对 citation grounding 约束不够；
- 现在的 `search / stop` 目标函数可能过于偏向 efficiency。

这正是有 insight 的地方：

> **如果 learned policy 的第一反应是更快停，而不是更好引文，那说明当前 reward 比起“研究质量”，更在奖励“动作收敛”。**

这不是失败，而是下一轮实验设计最重要的反馈。

---

## 7. 接下来哪些实验值得做，哪些不值得做

### 7.1 值得做的下一步

1. 扩大 `search_cache`，优先补齐 `browser` 样本。
2. 把对照从 `2` 题 smoke 提到一个更稳的小 benchmark。
3. 单独跟踪下面几类指标：
   - tool call 数
   - stop 触发位置
   - citation grounding
   - citation coverage
   - factual accuracy
4. 明确分析 learned policy 的失败模式：
   - 是不是过早停止？
   - 是不是 query 变多但来源没有变多？
   - 是不是 `browser` 根本没学起来？

### 7.2 现在不值得做的事

1. 直接上重型 browser RL。
2. 一开始就引入多模态 browsing。
3. 在当前样本很小、citation 还不稳时，直接宣称“我们超越了 heuristic”。
4. 同时改 reward、route、query rewrite、summary state 四件事。
5. 只基于内部 `ResearchBench` 就对外宣称“对 deep research 普遍有效”。

原因很简单：

- 变量太多，无法归因；
- 成本太高，实验会失控；
- 论文故事线会变糊。

---

## 8. 当前最合理的论文故事线

如果后面结果做出来，我认为最像样的一条论文叙事应该是：

### 题目方向

`Budgeted Search Policy Learning for Long-form Deep Research`

或者更完整一点：

`Budgeted, Citation-Aware Search Policy Learning for Long-form Deep Research Agents`

### 问题

- 现有 search-RL 工作更多关注 QA 或 web task completion；
- 现有 deep research 工作更关注 full-stack agent 或 reward；
- 但 long-form deep research 里的局部搜索控制问题：
  - 何时搜
  - 何时停
  - 何时展开页面
  - 何时切换检索路由

还没有被充分单独建模和评估。

### 方法

- 结构化 `policy_trace`
- offline `search_cache`
- step-level `search / browser / stop / route` policy
- quality-cost-citation reward / evaluation

### 贡献

1. 给 deep research agent 提供一个可学习的 budget-aware search control layer。
2. 提出一套能同时看质量、引用、成本的评测方式。
3. 证明 learned policy 在 long-form setting 下能改善 quality-cost frontier。

### 风险

如果最终只能证明：

- 动作数变少；
- 但 citation 没变好甚至变差；

那这个故事线就不够强，需要继续补 reward / route 设计。

---

## 9. 最终判断

最重要的判断不是“我们是不是第一个做 search policy 的人”，而是：

> **我们是否能把 deep research 里的局部搜索控制问题，做成一个足够清晰、可重复、可评估、且和已有工作有边界差异的研究对象。**

当前答案是：

- 有机会；
- 但前提是不要把“做了 learned baseline”误当成“已经有论文新意”。

更具体地说：

- **现在这条线值得继续做；**
- **但只有在“质量-成本-citation”三者关系被说清楚之后，它才像研究，而不是功能实验。**

---

## 参考工作

- Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. arXiv, 2025-03-12. https://arxiv.org/abs/2503.09516
- ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning. arXiv, 2025-03-25. https://arxiv.org/abs/2503.19470
- DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments. arXiv, 2025-04-04. https://arxiv.org/abs/2504.03160
- BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents. arXiv, 2025-04-16. https://arxiv.org/abs/2504.12516
- BrowseComp: a benchmark for browsing agents. OpenAI, 2025-04-10. https://openai.com/index/browsecomp/
- WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning. arXiv, 2025-05-22. https://arxiv.org/abs/2505.16421
- ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization. arXiv, 2025-09-16. https://arxiv.org/abs/2509.13313
- WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for Open-Ended Deep Research. arXiv, 2025-09-16. https://arxiv.org/abs/2509.13312
- DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research. arXiv, 2025-11-24. https://arxiv.org/abs/2511.19399
- SAGE: Benchmarking and Improving Retrieval for Deep Research Agents. arXiv, 2026-02-05. https://arxiv.org/abs/2602.05975

---

## 10. 2026-05-15 补充判断：这条线的新意边界需要继续收窄

过去一天的新实验让一个判断更清楚了：

- 受控 `browser-positive` 数据能修复 learned policy 的行为塌缩；
- 但它还没有自动转化成 citation gain 或明显质量优势。

这意味着当前最值得守住的新意，不是：

- “我们也做了搜索策略学习”

而是：

> **在 long-form deep research 中，把局部搜索控制做成一个 citation-aware、budget-aware、可门控训练的独立研究对象。**

这里真正可能有价值的点，已经收窄成三件事：

1. `search / browser / stop` 的局部控制，而不是全栈 agent 万能改进。
2. `bootstrap-ready` 和 `headline-ready` 两层实验卫生，而不是“只要能训就算结果”。
3. long-form report 的引用与成本约束，而不是短答案 QA 上的 search-RL 复现。

---

## 11. 相关工作更新后，当前最合理的对位方式

结合最近这批相关工作，可以更清楚地看到我们应该站在哪个缝隙里。

### 11.1 不能主张的东西

下面这些说法现在都很难成立：

1. “搜索动作可学习”本身是新意。
2. “deep research agent 需要真实环境和长期轨迹”本身是新意。
3. “需要更像研究任务的 benchmark”本身是新意。

因为这些方向已经分别被 Search-R1 / ReSearch、DeepResearcher、BrowseComp、DeepResearch Bench 一类工作覆盖。

### 11.2 还可能成立的东西

如果要保住研究空间，更合理的 claim 应该是：

1. **局部搜索控制而非全栈系统**
   - 避免和 full-stack deep research agent 直接正面撞车

2. **citation-aware quality-cost frontier**
   - 避免退化成普通 search-RL

3. **训练门控与 failure-mode honesty**
   - 把“什么数据只适合 bootstrap，什么数据才有资格支撑 claim”写成程序，而不是凭感觉判断

4. **针对 deep research 的 route / browse 行为分析**
   - 不是只看最终分数
   - 而是看 policy collapse、browser suppression、citation grounding 等过程现象

### 11.3 这条线为什么还值得继续

因为最新相关工作其实在反过来证明一件事：

- 社区正在快速补齐 deep research 的 benchmark、framework、retrieval、real-world RL；
- 但这也意味着，如果我们不把问题收得足够窄，就会马上掉进“已有工作已经做过大框架”的区域。

所以当前最合理的坚持不是做得更大，而是做得更尖：

> **把局部搜索策略学清楚，证明它和 citation / cost / stopping 的关系，再决定是否扩到 retriever router 或 summary state。**

---

## 12. 最新参考工作补充

- DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents. arXiv, 2025-06-12. https://arxiv.org/abs/2506.11763
- DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Report. arXiv, 2026-01-13. https://arxiv.org/abs/2601.08536
- OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis. arXiv, 2026-03-17. https://arxiv.org/abs/2603.20278
- AgentIR: Reasoning-Aware Retrival for Deep Research Agents. arXiv, 2026-03-04. https://arxiv.org/abs/2603.04384
