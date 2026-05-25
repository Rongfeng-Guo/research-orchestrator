# 配置系统与环境变量

项目配置分成两层：

- `configs/default.yaml`：模块策略和运行参数
- `.env` / `.env.local`：连接信息和密钥

## 1. 配置分层

- YAML 管理运行参数和模块策略
- `.env` 管理连接信息和密钥

## 2. 配置加载顺序

### YAML

- `load_config()` 默认读取 `configs/default.yaml`
- 也可以通过命令行 `--config` 指向自定义 YAML

### 环境变量

`env_config.py` 会按以下顺序加载：

1. `.env`
2. `.env.local`

后加载的 `.env.local` 会覆盖 `.env`。

## 3. `default.yaml` 的几个关键区块

### `system`

控制：

- 系统名
- 版本
- 日志级别
- 工作目录

### `model`

包含：

- 默认后端
- 后端采样参数
- 模块级采样覆盖
- 后端映射关系

### `orchestrator`

控制主流程调度参数：

- `max_concurrent`
- `global_timeout_seconds`
- `max_replan_rounds`
- `max_sub_questions`

### `planner`

控制规划器策略：

- 每个 sub-agent 最大搜索轮数
- 是否启用 replan
- 是否做完整性检查

### `compressor`

控制上下文压缩：

- 最大上下文长度
- 输出预留 token
- 各层阈值
- embedding 模型

### `memory`

控制共享记忆：

- 数据库路径
- 最大条目数
- 去重阈值
- 冲突阈值

### `adversarial`

控制 Red/Blue 环：

- 是否启用
- 最大轮数
- 分数阈值
- 收敛阈值

### `evolution`

控制自进化模块：

- 是否启用
- batch 大小
- GRPO 轮数
- 学习率

- 默认关闭

### `summarizer`

当前仓库里还可以通过单独的 `summarizer` 配置块控制一些报告生成约束，例如：

- `min_report_chars`

### `tools`

控制工具开关和部分超参数：

- `web_search.enabled`
- `web_search.mock_mode`
- `arxiv_reader.enabled`
- `code_sandbox.enabled`

## 4. 模型后端配置

`model` 配置块支持不同模块使用不同后端和不同采样参数。

## 5. `backend_mapping` 是怎么用的

在 `initialize_modules()` 里，系统会：

1. 先根据 `model.backend` 创建默认 policy。
2. 再根据 `backend_mapping` 为指定模块创建专用 policy。

常见映射包括：

- `solver`
- `planner`
- `summarizer`
- `judge`
- `red_agent`
- `blue_agent`
- `compressor`

## 6. `.env.template` 说明了哪些后端

模板里已经把主要后端和工具相关变量列出来了，包括：

- `DEFAULT_LLM_BACKEND`
- `DEEPSEEK_*`
- `VLLM_*`
- `OPENAI_*`
- `MIMO_*`
- `SEARCH_BACKEND`
- `SERPAPI_*`
- `BING_SEARCH_*`
- `BOCHA_*`
- `METASO_*`
- `ARXIV_READER_BACKEND`
- `SEMANTIC_SCHOLAR_API_KEY`
- `OPENALEX_EMAIL`
- `BROWSER_TIMEOUT`
- `CODE_SANDBOX_TIMEOUT`
- `FILE_READER_ALLOWED_BASE_DIR`

## 7. 工具相关配置

### `web_search.mock_mode`

这是一个很关键的开关。

当它开启时：

- `WebSearchTool` 会退化成 `MockWebSearchTool`
- `BrowserTool` 会退化成 `MockBrowserTool`
- `ArxivReaderTool` 和 `CodeSandboxTool` 也会走 mock 分支

### `SEARCH_BACKEND`

决定真实搜索走哪个后端。

### `ARXIV_READER_BACKEND`

决定论文工具走哪个学术数据源。

## 8. 常见配置项

### 模型切换

- `.env.local`
- `configs/default.yaml` 的 `backend_mapping`

### 运行速度与成本

- `max_concurrent`
- `global_timeout_seconds`
- 各模块 `max_tokens`
- adversarial 开关与轮数

### 流程调试

- `tools.web_search.mock_mode = true`

## 9. 配置说明

- 一些说明文字与当前代码可能存在轻微偏差，应以代码为准。
- `evolution` 提供配置接口和训练相关参数。
- 工具 `enabled` 字段在所有地方不一定都被严格消费，实际行为还要看 `initialize_modules()` 是否引用该配置。
- `configs/aliyun_smoke.yaml` 提供低成本链路验证配置。
