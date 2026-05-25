# Agent、工具与模型层

## 1. Agent 层的角色划分

当前仓库中的 Agent 主要分成四类：

- `BaseAgent`
- `ResearcherAgent`
- `SummarizerAgent`
- `RedAgent` / `BlueAgent`

## 2. `BaseAgent`：统一抽象

`BaseAgent` 很薄，只定义了三件事：

- `name`
- `policy`
- `tools`

以及统一的 `run(task, context)` 接口。

它的价值在于：

- 统一生命周期管理
- 方便 `AgentPool` 做对象复用
- 方便后续替换不同 agent 实现

## 3. `ResearcherAgent`：真正干活的 worker

### 3.1 它负责什么

`ResearcherAgent` 是当前系统中最重要的执行体，负责：

- 搜索型任务
- 分析型任务
- 验证型任务

虽然任务类型不同，但核心执行逻辑是同一套。

### 3.2 它怎么工作

执行流程大致是：

1. 构造任务 prompt。
2. 给模型注册可调用工具。
3. 调模型。
4. 如果模型发出 tool calls，就执行工具。
5. 把工具结果回填到消息历史。
6. 继续让模型决定是否继续调用工具或输出总结。

### 3.3 它有哪些控制策略

当前实现里有很多约束：

- 如果任务看起来不适合搜索，会走“直接分析”分支。
- 如果模型没有主动调用工具，会再提醒一次必须调用工具。
- 如果任务明显偏学术，会优先推荐 `arxiv_reader`。
- 如果结果全空或已经搜了两轮，会强制要求总结。
- 如果上下文里有 `evidence_snapshot_md` 或 `research_policy`，这些内容也会进入任务 prompt，用来提示当前证据缺口和建议动作。

### 3.4 最重要的现实限制

Prompt 明确写了：

- 一个子任务最多只能调用 2 次工具。

这意味着：

- 它不是“无限探索型”代理。
- 它更像“有限搜索预算下的研究助理”。

## 4. `SummarizerAgent`：最终报告合成器

它和 `ResearcherAgent` 最大的区别是：

- 不做多轮工具调用
- 不再搜信息
- 直接吃所有子结果，生成一份长报告

它的主要任务：

- 把所有子结果按置信度排序
- 构造长上下文 prompt
- 输出结构化 Markdown
- 抽取来源
- 根据子任务成功率校准整体置信度
- 在当前实现里，还会显式读取 Evidence Snapshot，关注 missing coverage 和 open conflicts

## 5. `RedAgent` 与 `BlueAgent`

### `RedAgent`

负责从五个维度挑错：

- factual
- hallucination
- logical
- source credibility
- coverage

### `BlueAgent`

根据 verdict 做三类修复：

- 原地修正
- 补充搜索后修正
- 删除高风险内容

它们真正组成一个质量后处理回路，而不是研究主执行体。

## 6. `AgentPool`：对象池，而不是智能调度器

`AgentPool` 的职责很朴素：

- 延迟创建 agent
- 复用空闲 agent
- 在 agent 被污染或截断后丢弃

它不做复杂负载均衡，而只是一个生命周期容器。

## 7. 工具层概览

当前内置工具包括：

| 工具 | 作用 | 典型用途 |
| --- | --- | --- |
| `web_search` | 搜索网页结果 | 新闻、行业、一般事实 |
| `browser` | 打开 URL 抽取正文 | 深读搜索命中的文章 |
| `arxiv_reader` | 查论文元数据 | 学术论文、引用、摘要 |
| `file_reader` | 读取本地文件 | 文档、PDF、CSV、JSON |
| `calculator` | 安全数学表达式求值 | 快速数值计算 |
| `code_sandbox` | 受限 Python 执行 | 稍复杂的程序化计算 |
| `notepad` | 个人笔记本 | 记录中间结论、待办、策略 |

## 8. 各工具的关键实现特点

### `web_search`

- 支持多个后端：SerpAPI、Bing、博查、秘塔。
- 可通过 `.env` 切换。
- 有 mock 模式。

### `browser`

- 用 `aiohttp` 抓网页。
- 用 BeautifulSoup 抽正文。
- 不支持复杂 JS 渲染站点。

### `arxiv_reader`

- 不只支持 ArXiv。
- 还支持 Semantic Scholar 和 OpenAlex。
- 这让它在中文网络环境下更实用。

### `file_reader`

- 支持 `.txt`、`.md`、`.pdf`、`.csv`、`.json`、`.docx`
- 可以限制只读某个目录
- 不会执行文件里的任何代码

### `calculator`

- 只允许安全 AST
- 面向“快速数值表达式”
- 目的就是替代不必要的 `code_sandbox`

### `code_sandbox`

- 当前是受限实现，不是完整 Docker 沙箱
- 默认不允许 `import`
- 更像“可控表达式执行器”

### `notepad`

- 纯内存
- 非持久化
- 面向单 agent 的临时草稿

## 9. `notepad` 和 `memory_store` 的区别

### `notepad`

- 单 agent 私有草稿
- 临时记录
- 不做去重和冲突检测

### `SharedMemoryStore`

- 跨 agent 共享
- SQLite 持久化
- 做 embedding 检索
- 做去重和矛盾检测

你可以把它们理解成：

- `notepad` 是草稿纸
- `memory_store` 是知识库
- `evidence_snapshot` 是从知识库提炼出来的“当前研究决策面板”

## 10. 模型层：为什么叫 `VLLMPolicy`

虽然类名叫 `VLLMPolicy`，但它不是只能接 vLLM。

实际上它是一个：

- OpenAI 兼容 API 的统一封装器

它支持：

- 本地 vLLM
- OpenAI
- DeepSeek
- 小米 MiMo
- 任何兼容 chat completions 的后端

## 11. `ModelRouter`：按模块路由模型

`ModelRouter` 负责：

- 读 `.env`
- 根据后端名字构造 `VLLMPolicy`
- 做缓存复用

同时配置层允许不同模块用不同后端，比如：

- planner 用强结构化模型
- solver 用强推理模型
- red/blue/judge 用更稳定或更便宜的模型

## 12. `VLLMPolicy` 的实际职责

它不仅仅是一个 API client，还做了几件关键事情：

- 清洗 message 格式
- 合并连续角色消息
- 维护工具调用消息格式
- 主动截断长上下文
- 兼容工具调用的回退解析
- 把异常分成“上下文超限”和“其他错误”

换句话说：

> 这个类承担了不少“与模型接口打交道的脏活”

## 13. 当前实现的几个事实

- 研究 worker 主要是一个通用 `ResearcherAgent`。
- 工具调用能力比角色分工更重要。
- 报告质量很大程度依赖 `SummarizerAgent` 和后处理对抗环。
- 模型层实际上承担了不少稳定性工作。
