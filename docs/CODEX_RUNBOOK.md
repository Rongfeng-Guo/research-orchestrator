# DeepResearch Agent 运行与代码认知手册

本文档记录了我在当前仓库中的实际跑通方式、环境配置、关键代码结构、运行链路、已知限制和后续建议。目标不是重复 README，而是把“这份代码现在到底怎么用、为什么这样用、哪里会踩坑”一次讲清楚。

## 1. 当前结论

截至 2026-05-12，本仓库已经在当前机器上完成一次真实的 smoke run，满足以下条件：

- LLM 后端：阿里 DashScope OpenAI-compatible 接口
- 模型：`qwen-flash`
- 运行入口：`scripts/run_single.py`
- 配置文件：`configs/aliyun_smoke.yaml`
- 环境文件：`.env.local`
- 工具模式：`mock_mode=true`
- 已验证输出：
  - `outputs/smoke/report_20260512_205416_Transformer_architec.md`
  - `outputs/smoke/run_20260512_205325.log`

这次验证证明：

- 模型路由 `ModelRouter -> VLLMPolicy -> OpenAI client` 是可用的
- Planner / Orchestrator / AgentPool / ResearcherAgent / SummarizerAgent 主链路可闭环
- Memory Store 能正常写入 SQLite
- 低成本 smoke 配置下，项目可以从规划、子任务执行、报告合成一路跑到最终产物落盘

这次验证没有证明的内容：

- 真实搜索 API 模式可用
- 长时间 REPL 会话稳定性
- 批量实验脚本在当前环境可完整跑完
- evaluation 子系统依赖、数据集和统计报告已全部可用

## 2. 我实际怎么跑通的

### 2.1 本地环境

当前 Python 环境：

- `Python 3.12.3`
- `openai` 已安装
- `torch` 已安装
- `beautifulsoup4` 已安装
- `aiohttp` 在当前环境中可导入

注意：

- `sentence-transformers` 没有安装，但仓库内 `src/memory/embedder.py` 有 deterministic fallback，所以不会阻塞主链启动
- `datasets`、`transformers`、`langsmith` 没有作为本次 smoke run 的硬依赖使用

### 2.2 本地环境变量

我创建了一个仓库本地的 `.env.local`，作用如下：

- 关闭 LangSmith tracing，避免缺少 `langsmith` 时产生额外干扰
- 把默认后端切到 `openai`
- 把 OpenAI-compatible base URL 指向阿里 DashScope 的 compatible-mode
- 把模型指定为 `qwen-flash`
- 把 ArXiv 阅读器默认切到 `openalex`

说明：

- `.env.local` 已被 `.gitignore` 忽略，不会进入版本库
- 文档中不记录明文 key
- 当前这份 `.env.local` 是“本机可跑”的运行态配置，不是通用模板

### 2.3 我新增的低成本 smoke 配置

文件：`configs/aliyun_smoke.yaml`

设计目标：

- 只验证主链能否跑通
- 尽量省 token
- 尽量避免额外搜索 API 成本
- 避免默认配置的高任务数和超长报告输出

关键策略：

- 所有模块统一走 `openai` 后端
- LLM 使用 `qwen-flash`
- `planner.min_subtasks=2`，`planner.max_subtasks=3`
- `orchestrator.max_replan_rounds=0`
- `adversarial.enabled=false`
- `summarizer.min_report_chars=800`
- `tools.web_search.mock_mode=true`
- 输出目录写入 `outputs/`

### 2.4 实际执行命令

我实际执行的命令：

```bash
python scripts/run_single.py --query "Transformer architecture" --config configs/aliyun_smoke.yaml --output_dir outputs/smoke --session_id aliyun_smoke
```

执行结果：

- 成功退出，`Exit code: 0`
- 全链路耗时约 41 秒
- 规划了 3 个子任务
- 生成了最终 Markdown 研究报告
- SQLite memory 正常写入

### 2.5 为什么我先用 smoke 模式

原因很直接：

- 你给的是 LLM API key，没有提供搜索 API key
- 当前仓库默认真实搜索依赖 `SerpAPI / Bing / 博查 / 秘塔`
- 默认配置下 Planner 和 Summarizer 都比较“重”，会比必要值多耗 token

因此我先把“LLM 编排主链是否可用”验证出来，再决定是否值得继续扩成真实搜索模式。这是成本最省、风险最小的路径。

## 3. 我对代码结构的理解

下面按“主调用链”和“模块职责”来讲。

### 3.1 顶层调用链

最小运行入口是：

1. `scripts/run_single.py`
2. `src/core/runner.py`
3. `src/models/model_router.py`
4. `src/orchestrator/orchestrator.py`
5. `src/planner/planner.py`
6. `src/agents/researcher.py`
7. `src/agents/summarizer.py`
8. `src/memory/memory_store.py`
9. `src/tools/*`

整体流程：

1. 脚本读取 CLI 参数并加载 YAML 配置
2. `runner.initialize_modules()` 初始化模型、Planner、Compressor、Memory、Tools、Adversarial、AgentPool、Orchestrator
3. `runner.run_research()` 把 query 交给 `Orchestrator.run()`
4. Orchestrator 先调用 Planner 生成 DAG 子任务
5. AgentPool 逐层调度 ResearcherAgent 执行子任务
6. 每个子任务内部由 LLM 驱动工具调用循环
7. 子结果写入内存，成功率足够则进入合成
8. SummarizerAgent 一次性合成最终报告
9. 如果启用了 adversarial，则进入 Red-Blue 优化
10. 最终报告格式化后写入 Markdown 文件

### 3.2 `src/core/runner.py`

这是“组装工厂”和“公共执行入口”。

我对它的理解：

- `load_config()` 只负责读取单个 YAML 文件，不做多文件 merge
- `_create_tools_factory()` 根据配置决定生成真实工具还是 Mock 工具
- `initialize_modules()` 是整个项目最关键的初始化函数
- `run_research()` 是运行主线，不负责 CLI，只负责业务执行
- `save_report()` 负责把最终 Markdown 落盘

需要注意：

- 配置文件缺任何字段时，多数模块会用代码里的默认值补齐
- 但模型后端相关的环境变量缺失时，初始化会直接失败

### 3.3 `src/models/model_router.py`

这是“多后端 LLM 路由层”。

它本质上做的事：

- 读取 `.env` / `.env.local`
- 按后端前缀组装连接信息
- 用这些信息构造 `VLLMPolicy`

支持的思路是“名称约定”：

- `DEEPSEEK_*`
- `OPENAI_*`
- `MIMO_*`
- `VLLM_*`
- 以及任意自定义 `{PREFIX}_API_KEY`

重要结论：

- 这里并不强依赖某一家模型服务商
- 只要目标服务实现 OpenAI-compatible chat completion，就能接进去
- 所以阿里 DashScope compatible-mode 可以直接复用 `openai` 这套配置名

### 3.4 `src/models/vllm_policy.py`

这是模型请求的统一封装层。

它负责：

- 清洗 message 格式
- 合并连续消息
- 主动做长上下文截断
- 发起 OpenAI-compatible 请求
- 解析 tool calls
- 在出错时做统一回退

我的理解：

- 名字叫 `VLLMPolicy`，但实际上不只服务于 vLLM
- 它是整个仓库里真正意义上的“LLM client abstraction”
- 当前项目对 OpenAI-compatible 的耦合主要都在这里

### 3.5 `src/orchestrator/orchestrator.py`

这是系统的状态机调度核心。

内部状态：

- `idle`
- `planning`
- `dispatching`
- `collecting`
- `synthesizing`
- `adversarial`
- `replanning`
- `done`
- `failed`

它的职责不是做推理，而是做“流程控制”：

- 规划 DAG
- 逐层并发执行
- 汇总结果
- 判断是否重规划
- 触发合成
- 选择是否进入对抗优化

我的理解：

- 这是一个典型的任务编排器，不依赖 LangGraph/AutoGen
- 每个子任务是否成功，直接影响后续是否重规划或直接合成
- `RunConfig.max_sub_questions` 只是运行态配置，不会自动写回 Planner prompt，原始代码里这一点并没有真正收紧首次规划规模

### 3.6 `src/planner/planner.py`

Planner 负责把 query 变成结构化 DAG。

关键行为：

- 用 LLM 生成 `sub_tasks`
- 解析 JSON
- 反序列化成 `SubTask`
- 构建依赖图
- 支持失败后增量重规划

我对它的判断：

- 功能是完整的
- 但 prompt 驱动很重，输出规模和质量强依赖模型能力
- 原始版本首次规划会偏激进，容易增加成本

我这次做的改动：

- 给 Planner 加了 `min_subtasks` / `max_subtasks` 配置能力
- 让首次规划的 prompt 真正尊重配置，而不是硬编码激进扩展

### 3.7 `src/agents/researcher.py`

ResearcherAgent 是执行子任务的 worker。

它的核心模式是：

- 给 LLM system prompt
- 注册 tools
- 让模型决定是否调用工具
- 执行工具结果后回写
- 最多循环若干轮后结束

重要行为：

- 学术类任务倾向 `arxiv_reader`
- 普通事实类任务默认优先 `web_search`
- 工具调用最多 2 轮，之后强制总结
- 如果工具返回 `error`，任务会判失败

我的理解：

- 这是典型的 tool-calling agent loop
- 成功率非常依赖搜索工具是否真能返回有效结果
- 在真实模式下，如果搜索 API 没有配，很多任务会快速失败

### 3.8 `src/agents/summarizer.py`

SummarizerAgent 负责最终报告合成。

输入：

- 原始 query
- 所有子任务结果

输出：

- `ResearchReport`

它的工作方式是：

- 把所有子结果按置信度排序
- 拼进一个长 prompt
- 单轮调用 LLM 生成最终 Markdown 报告
- 从工具轨迹里提取 sources
- 用“LLM 自评 * 子任务成功率”粗略校准最终置信度

我的判断：

- 这是整个仓库里最消耗输出 token 的一个点
- 原始版本固定要求 3000+ 中文字，明显更适合正式报告，不适合低成本冒烟验证

我这次做的改动：

- 给 `SummarizerAgent` 加了 `min_report_chars` 配置能力
- 让报告长度下限可调

### 3.9 `src/memory/memory_store.py` 与 `src/memory/long_term.py`

这一层是“共享记忆 + SQLite 持久化”。

结构是两层：

- `LongTermMemory`: 纯 SQLite CRUD
- `SharedMemoryStore`: 在 SQLite 之上加 embedding、去重、冲突检测、语义检索

我对它的理解：

- 它不是简单日志存储，而是带语义检索意图的 memory 层
- 即使 `sentence-transformers` 缺失，也不会卡死，只是退化为 deterministic random embedding
- 对 REPL 会话继承比较关键

### 3.10 `src/tools/*`

当前仓库提供的工具主要有：

- `web_search`
- `browser`
- `arxiv_reader`
- `file_reader`
- `code_sandbox`
- `calculator`
- `notepad`

其中最关键的是前三个：

- `web_search`: 负责找到候选来源
- `browser`: 负责读网页正文
- `arxiv_reader`: 负责学术元数据

我的观察：

- 工具体系设计是完整的
- 但真实运行最卡的不是 LLM，而是搜索 API 的可用性
- 当前 `runner._create_tools_factory()` 把 `mock_mode` 绑定得比较粗，一开 mock，`web_search / browser / arxiv_reader / code_sandbox` 都会一起偏向 mock/简化模式

## 4. 配置与入口脚本认知

### 4.1 主要运行入口

#### `scripts/run_single.py`

用途：

- 单条 query 研究
- 适合验证主链是否可跑
- 会自动保存日志和报告

这是我本次实际使用的入口。

#### `scripts/run_repl.py`

用途：

- 交互式连续提问
- 支持会话隔离与 session 继承
- 模块只初始化一次，更适合长期会话

适合场景：

- 持续研究一个主题
- 想把记忆库积累起来

#### `scripts/run_all_experiments.py`

用途：

- 一键拉起批量实验
- 汇总模块消融、标准评测、benchmark、judge 等结果

适合场景：

- 做研究评测
- 跑面试/论文展示材料

不适合首次验证，因为太重。

### 4.2 README 与代码的几个不一致点

我已经确认的文档偏差：

- README 写的是复制 `.env.example`，实际仓库里是 `.env.template`
- README 默认叙述以 DeepSeek / MiMo 为主，但代码底层其实更通用，只要是 OpenAI-compatible 就能接
- README 的“快速开始”更像理想路径，不是当前机器上最省成本的验证路径

## 5. 本次我做过的代码修改

### 5.1 新增低成本配置

新增文件：

- `configs/aliyun_smoke.yaml`

作用：

- 用阿里兼容接口快速验证主链
- 减少首次规划和最终合成的 token 开销

### 5.2 Planner 支持可配置子任务规模

修改文件：

- `src/planner/planner.py`
- `src/core/runner.py`

作用：

- 不再把首次规划规模硬编码死
- 可以在配置里显式指定 `planner.min_subtasks` 和 `planner.max_subtasks`

### 5.3 Summarizer 支持可配置最小报告长度

修改文件：

- `src/agents/summarizer.py`
- `src/orchestrator/orchestrator.py`
- `src/core/runner.py`

作用：

- 冒烟验证时可以生成更短报告
- 正式研究时仍可把长度拉高

### 5.4 Windows 输出编码保护

修改文件：

- `scripts/run_single.py`

作用：

- 启动时把 stdout / stderr 切成 UTF-8
- 降低 Windows 控制台因为模型输出特殊字符或 emoji 导致崩溃的概率

## 6. 当前已知限制与风险

### 6.1 真实搜索依赖没有消除

这是最现实的限制。

如果你把 `mock_mode` 切回 `false`，则 `web_search` 会进入真实模式，而真实模式依赖以下之一：

- `SERPAPI_KEY`
- `BING_SEARCH_KEY`
- `BOCHA_API_KEY`
- `METASO_API_KEY`

如果这些都没有，很多子任务会因为搜索失败直接报错。

### 6.2 默认配置成本偏高

默认配置对正式研究友好，但对首次验证不友好，原因包括：

- 模块多后端映射复杂
- Planner 容易生成较多子任务
- Summarizer 默认追求长报告
- Adversarial 默认开启

### 6.3 Mock 开关耦合较粗

当前 runner 的逻辑不是“只 mock 搜索”，而是：

- `web_search`
- `browser`
- `arxiv_reader`
- `code_sandbox`

会一起受到 `mock_mode` 的影响或间接影响。

这意味着：

- smoke 模式很稳
- 但 smoke 模式并不代表真实检索链路也已验证完成

### 6.4 Embedding 目前处于 fallback 模式

因为未安装 `sentence-transformers`，当前 memory embedding 使用 deterministic random fallback。

影响：

- 不阻塞运行
- 但语义检索质量不是真实 embedding 水平

### 6.5 评测与实验脚本很重

`scripts/run_all_experiments.py` 本质上是一个“全量实验 orchestrator”，不是快速验证脚本。

如果直接用它做首次测试，风险包括：

- API 花费高
- 运行时间长
- 一旦中途某个实验失败，定位困难

## 7. 如果你想继续升级成“真实研究模式”

建议顺序如下：

1. 保留 `.env.local` 里的 DashScope 兼容 LLM 配置
2. 额外补一个真实搜索 API key
3. 基于 `configs/aliyun_smoke.yaml` 复制一份新配置，例如 `configs/aliyun_real.yaml`
4. 把 `tools.web_search.mock_mode` 改成 `false`
5. 视预算把：
   - `planner.max_subtasks`
   - `summarizer.min_report_chars`
   - `adversarial.enabled`
   调高
6. 先用 `run_single.py` 验证，再考虑 REPL，再考虑批量实验

我建议至少先准备一个搜索后端：

- 国内优先：博查 / 秘塔
- 海外或通用环境：SerpAPI / Bing

## 8. 推荐运行方式

### 8.1 最稳的验证方式

```bash
python scripts/run_single.py --query "Transformer architecture" --config configs/aliyun_smoke.yaml --output_dir outputs/smoke --session_id aliyun_smoke
```

用途：

- 验证主链是否正常
- 验证阿里兼容接口是否正常
- 验证报告生成和落盘是否正常

### 8.2 会话式继续问

```bash
python scripts/run_repl.py --config configs/aliyun_smoke.yaml --session_id aliyun_smoke
```

用途：

- 复用本次 session memory
- 继续追问同主题

注意：

- 这仍然是 smoke 配置，不是真实搜索配置

### 8.3 切换到真实搜索前先做的事

- 准备搜索 API key
- 决定是不是需要 adversarial
- 决定是否要安装 `sentence-transformers`
- 决定报告长度是否需要恢复到正式水平

## 9. 这份代码在我看来“本质上是什么”

如果只用一句话概括：

> 这不是一个单纯的“问答脚本”，而是一套以 OpenAI-compatible LLM 为核心、通过 Planner + Orchestrator + Tool Agent Loop + Memory + Summarizer 组装出来的深度研究流水线。

它的优点：

- 架构完整
- 模块边界清楚
- 有 mock/real 双模式
- 适合继续扩展

它现在最欠缺的地方：

- 首次运行默认体验不够保守
- 对搜索 API 的依赖仍然比较硬
- README 没把“最低成本可跑路径”讲清楚

## 10. 我建议你后面优先做的三件事

### 建议 1：补一个真实搜索配置

这是从 smoke 模式迈向真实研究模式的关键一步。

### 建议 2：给 README 增补一段“DashScope / Qwen 兼容接入说明”

因为底层代码已经支持，但文档没有把这条路线讲清楚。

### 建议 3：把 mock_mode 从“全局粗开关”拆成“按工具控制”

这样可以做到：

- 搜索 mock
- browser real
- arxiv real

对调试会更友好。

---

如果以后继续在这个仓库上开发，我会默认把这份文档当成当前的真实运行基线，而不是把 README 当成唯一事实来源。
