# 记忆、压缩与对抗优化

本文档介绍共享记忆、上下文压缩和对抗优化模块。

## 1. 共享记忆：`SharedMemoryStore`

### 1.1 作用

在一个多子任务研究流程里，问题不只是“搜到信息”，还包括：

- 如何让后续步骤复用前面的发现
- 如何避免重复写入同一类信息
- 如何识别相互矛盾的结论

### 1.2 两层结构

`SharedMemoryStore` 包含两层结构:

- 底层：`LongTermMemory`，使用 SQLite 持久化
- 上层：内存 embedding 索引，做快速语义相似检索

### 1.3 写入流程

写入一条记忆时，大致要经过：

1. 垃圾内容过滤
2. 自动生成 embedding
3. 去重检测
4. 矛盾检测
5. 持久化
6. 更新内存索引

### 1.4 去重逻辑

当前实现中，若相似度高于阈值，就视为重复：

- 重复时保留置信度更高的版本

### 1.5 矛盾检测

如果两条内容：

- 语义上比较接近
- 但在词面上有明显相反含义

就会被记为冲突。

当前矛盾检测主要依赖：

- 否定词
- 反义词集合
- 相似度阈值

### 1.6 查询流程

`get_context_for_query()` 会：

1. 对当前 query 做相似检索。
2. 结合相似度、置信度、时间衰减做综合排序。
3. 在 token 预算内拼接成文本上下文。

这个返回值会被 Planner 用作历史记忆输入。

### 1.7 Evidence Snapshot

当前 `SharedMemoryStore` 还集成了 `EvidenceAwareResearchPolicy`。

`SharedMemoryStore` 会生成 `EvidenceSnapshot`，其中包含：

- 证据节点
- claim 级结构
- 支持边和冲突边
- source trust 估计
- `covered_terms` / `missing_terms`
- `evidence_strength`
- `uncertainty`
- `candidate_actions`
- `recommended_action`

## 2. 运行时内存与长期记忆

Orchestrator 自己也有一个 `_memory_store` 字典。

系统中包含两类记忆:

### 运行时 `_memory_store`

- 本次运行内部使用
- 保存 task result 和 final report
- 读写非常直接

### `SharedMemoryStore`

- 跨运行持久化
- 可做语义检索
- 更适合长期上下文

### `EvidenceSnapshot`

- 基于记忆计算得到的证据状态摘要
- 用于 replan、worker prompt 和 summarizer prompt

## 3. 上下文压缩：`ContextCompressor`

### 3.1 作用

压缩模块用于控制长上下文中的信息密度和 token 开销。

### 3.2 三层压缩策略

压缩器定义了三层策略：

- L1：相关性过滤
- L2：关键句提取
- L3：LLM 摘要

### 3.3 L1：相关性过滤

做法：

- 用 embedding 计算文本与 query 的相似度
- 低相似文本直接丢弃

### 3.4 L2：关键句提取

做法：

- 对文本做 extractive compression
- 只保留最相关句子

### 3.5 L3：LLM 摘要

做法：

- 先对单文档做摘要
- 再把多个摘要聚合

### 3.6 主流程中的接入位置

压缩模块当前主要在 `Orchestrator._build_memory_context()` 中被调用，用于控制提供给 `Planner` 的历史记忆上下文长度。

## 4. 对抗优化：`AdversarialLoop`

### 4.1 目标

对抗环用于检查报告中的事实、逻辑、来源和覆盖问题。

### 4.2 组成

- `RedAgent`：负责攻击和挑错
- `BlueAgent`：负责修复
- `AdversarialLoop`：负责控制轮次、终止条件和历史记录

### 4.3 RedAgent 的五个维度

它会分别从以下维度挑错：

- factual
- hallucination
- logical
- source credibility
- coverage

### 4.4 BlueAgent 的三种修复方式

- 原地修正
- 补充搜索后修正
- 删除高风险内容

### 4.5 启动条件

主流程里有一个外部门槛：

- 只有报告置信度低于 `0.8` 才进入对抗环

而在对抗环内部，还有自己的终止条件，例如：

- 分数达到阈值
- 相邻轮次变化收敛
- 达到最大轮数

### 4.6 震荡检测

- 如果某个已修复的问题在后续轮次再次出现
- 系统判定进入震荡
- 提前终止

## 5. 证据策略层如何影响主流程

当前 orchestrator 已经不只根据失败率做决策，还会参考 `research_policy`：

- 在收集阶段生成 evidence snapshot
- 将 `recommended_action` 和 `missing_terms` 写入运行时状态
- 构建后续 task context 时注入这些信号
- replan 原因分析也会参考 evidence gaps
- synthesizer 会把 evidence snapshot 作为只读背景纳入 prompt

## 6. 实现说明

### 记忆层

- embedding 检索和去重已经落地
- 冲突判定采用启发式方法

### 压缩层

- 模块本身比较完整
- 当前主流程中的接入范围有限

### 对抗层

- 可以运行
- 采用基于 prompt 的文本级修复

## 7. 模块分工

- 记忆层负责“保留什么”
- 证据策略层负责“下一步该做什么”
- 压缩层负责“舍弃什么”
- 对抗层负责“修正什么”

这些模块共同参与中间证据组织、上下文控制和报告修复。
