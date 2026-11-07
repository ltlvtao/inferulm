# 函数调用（OpenAI tools / tool_calls）落地（2026-11-07）

> 起点缺口的准确描述：旧实现的 `ChatMessage` 是 `{role: String, content: String}`，
> 请求体里出现 `tools`、`tool_calls`、`tool_choice`、`content: null`、
> `role: "tool"` 任一形态，serde 直接 422 —— **标准 OpenAI Agent 客户端连门都进不来**。
> 模板侧其实早就支持（`render_with(messages, extra)` 的额外顶层变量），只是 serve 没传。

## 一、这一刀做了什么

### 1. 请求 DTO 扩面（`wein-serve/openai.rs`）

| 字段 | 语义 |
|---|---|
| `messages[].content: Option<String>` | OpenAI 合同：带 `tool_calls` 的 assistant 消息 content 常为 null |
| `messages[].tool_calls` | 原样透传（模板读 `function.name` / `function.arguments`） |
| `messages[].tool_call_id` / `name` | 收到即接受（模板未消费，留作未来校验） |
| `messages[].reasoning_content` | 原样透传（Qwen 模板按它回放思维链） |
| `tools` | 必须是数组（坏形状 400），注入模板顶层 `tools` |
| `tool_choice` | `none`/`auto`/`required`/函数对象；**`none` ⇒ 不注入 tools 也不解析**，其它值 400 |
| `chat_template_kwargs` | vLLM 扩展：模板顶层 kwarg 直通（`enable_thinking` 等），非对象 400 |

模板钩子新增 `Tokenizer::render_chat_template_with(messages, extra)`（默认实现忽略
kwargs 转调旧钩子——4 处测试 stub 零改动，真模板在 cli 桥接覆盖）。

### 2. 模板侧三个必须修的细节（`wein-model/template.rs`）

字节级对拍 Python jinja2 3.1.6（HF 同款环境）后发现三处不能想当然：

1. **模板要 `tojson` 过滤器**（工具描述块 `<tools>{tool|tojson}</tools>` 与参数值），
   minijinja 不带 ⇒ 自己实现，且必须对齐 jinja2 的 `htmlsafe_json_dumps`：
   Python `json.dumps` 默认格式（`", "` / `": "` 分隔）、键排序、`ensure_ascii`
   （非 ASCII → `\uXXXX`，增补平面出代理对），再对 `< > & '` 做 HTML 安全替换。
   直接 `serde_json::to_string` 会差在分隔空格上（已由 ct7 抓出）。
2. **`arguments` 的 wire 形态是 JSON 字符串**，模板却按 mapping 消费
   （`tool_call.arguments|items`）—— 直接喂字符串渲染报
   "Can only get item pairs from a mapping"。新增 `normalize_tool_calls`
   在 `render_with` 入口把「可解析为 JSON 的字符串」换成解析结果（幂等，
   非字符串/解析失败原样保留）。
3. `content: Option` 序列化成 `null` 正好落进模板的 `content is none` 分支（空串）。

golden 用例：ct7（无 system 的 tools 块）/ ct8（tools + system 追加）/ ct9（历史
tool_calls + tool 响应）/ ct10（并行调用 + 两条 tool 响应合并进一个 user 块）/
ct11（`tools: []` 等价未传、normalize 幂等）—— 全部逐字节等于 jinja2。

### 3. 生成侧解析器（`wein-serve/tools.rs`，新增）

模型按模板要求吐 XML：`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>…`。
增量状态机把它切成 `Content` / `Call` 两类片：

- **跨 token 安全**：正文只放行「不再可能变成 `<tool_call>` 前缀」的部分（`holdback`
  按字节找最长候选后缀，注意 UTF-8 字符边界——首版在中文上 panic 过）。
- **不丢文本**：被 max_tokens 截断的未闭合调用，`finish()` 把起始标记回填、整体当正文。
- **参数定类型**：模板对 mapping/sequence 用 `tojson`、标量用 `str()`，所以按
  `tools` 声明还原（声明 `string` 的 `"007"` 保持字符串，`integer` 的 `1` 转数字，
  `object` 走 JSON 解析）。
- **退化形态**：body 直接是 `{"name":…,"arguments":…}` 也能解（模型漂移护栏）。
- 8 个单测（t1–t8）：整段/逐字节等价、悬空前缀、未闭合、纯正文、定类型、
  JSON body、多行参数值。

### 4. HTTP 两副皮（`wein-serve/api.rs`）

- 流式：正文片 → `delta.content`；调用片 → `delta.tool_calls[{index,id,type,function}]`；
  终止帧 `finish_reason: "tool_calls"`（仅当有调用且引擎正常停——`length`/`error` 不覆盖）。
- 非流式：同一状态机整段过一遍，`message.content` 剥掉标记（只有调用时给 `null`），
  `message.tool_calls` 不带 `index`（OpenAI 合同：index 是流式专用）。
- 开关：`WEIN_TOOL_PARSER=0` 全局关（回退旧行为：标记原样留在 content）。

## 二、验收

| 层 | 用例 | 结果 |
|---|---|---|
| 模板 | ct7-ct11（jinja2 字节对拍） | 5/5 绿（全套 11/11） |
| 解析器 | t1-t8 | 8/8 绿 |
| serve | ts1 非流式 / ts2 流式（帧序 + 标记零外泄）/ ts3 未给 tools 不解析 / ts4 完整 OpenAI 请求体不再 422 / ts5 `tool_choice:none` / ts6-ts7 坏形状 400 | 7/7 绿 |
| 回归 | wein-serve + wein-model 全量（含 24 个既有 serve 契约） | 全绿 |
| **真模型 E2E** | Qwen3.5-4B + `get_weather` 工具 | 见下 |

### 真模型端到端（`--ctx-len 2048 --kv-pages 256`）

1. **非流式**：`finish_reason: "tool_calls"`，`tool_calls[0].function = {"name":"get_weather",
   "arguments":"{\"city\":\"北京\",\"unit\":\"c\"}"}`，`content` 只剩推理正文。prompt 295 token
   （工具块 ≈ 100 token，与 jinja2 渲染一致）。
2. **流式**：4 帧——content 片 → tool_calls 片（index 0）→ `finish_reason: tool_calls` →
   `[DONE]`；**`<tool_call>` / `<function=` 一帧都没有**。
3. **多轮回灌**：assistant(tool_calls) + `role: "tool"` 结果 → 模型据此作答、
   `finish_reason: "stop"`、无多余调用（`<tool_response>` 块渲染正确）。

## 三、已知边界（下一刀的候选，不在本刀内）

1. **思维链没分离**：Qwen3.5 是思考模型，`content` 里含 `…</think>` 之前的推理文本。
   vLLM 用 `--reasoning-parser` 把它拆到 `reasoning_content` / `delta.reasoning_content`。
   本刀只做了「历史 reasoning_content 的输入侧」，输出侧未拆。
2. `content` 仍只收字符串（多模态 parts 数组 422）。
3. `tool_choice: required` 只当 auto（不强制）；`parallel_tool_calls: false` 未约束。
4. 采样器与工具调用互斥的前提未变；投机解码开启时解析器同样工作（都走同一文本流）。

## 四、性能

工具面只在 HTTP 侧加了一次 `Value` clone 与一个状态机（无 tools 时 `None`，
`dispatch` 直接走原路径）。battery 见 `/tmp/bat_tools.log`（本轮读数记入 MEMORY）。
