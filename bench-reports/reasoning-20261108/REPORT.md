# 输出侧思维链解析（`reasoning_content`）（2026-11-08）

> 上一刀的遗留：Qwen3.5 是思考模型，`content` 里混着 `</think>` 之前的推理文本。
> Agent 客户端要么把它当答案显示，要么自己正则切——vLLM 的解法是服务端
> `--reasoning-parser` 把推理拆进 `reasoning_content` / `delta.reasoning_content`。

## 一、语义（对齐 vLLM `deepseek_r1` / `qwen3`）

- 开思考时 Qwen 模板的**生成提示以 `<think>\n` 收尾** ⇒ 模型输出**没有起始标记**，
  只有第一次出现的 `</think>`。切分即 `text.split("</think>")` 的前后两段。
- 标记本身不外发（属格式）；之后若再出现 `</think>` 属正文。
- 流式：推理段 → `delta.reasoning_content`，正文段 → `delta.content`；
  **推理片必须全部先于正文片**（同一状态机天然保证）。
- 截断在推理中途（`length`）：已收到的全部进 `reasoning_content`，不丢字。
- 只有调用、没有正文时（`</think>` 后只剩空白）：`content` 为 `null`（OpenAI 合同）。

## 二、为什么必须"按模型/请求"开，而不能无条件切

没有思考能力的模型（或 `enable_thinking: false` 的请求）**永远不吐 `</think>`**，
无条件切分会把**全部正文误判成推理**、`content` 变空。两道闸：

1. `ServeState.reasoning`（默认 false）。CLI `--reasoning-parser auto|none|qwen3`
   （默认 `auto`）在启动时渲染一条探针消息：生成提示里最后一个 `<think>` **未闭合**
   ⇒ 开。启动行打 `+reasoning` 标记。
2. 请求级：`chat_template_kwargs.enable_thinking == false` ⇒ 本次不切分
   （此时提示里已是闭合的 `<think>\n\n</think>`）。

## 三、与工具调用的顺序（不能颠倒）

流水线：`文本流 → 思维链切分 → 正文段 → 工具调用解析`。

**先切思维链**是硬要求：模型在推理里经常"背诵"工具格式示例
（`<tool_call><function=…>`），若先过工具解析器，这些示例会变成**假调用**。
rs4 用例专门钉这一点：推理段中的示例调用被忽略，正文里的真调用照常解析。

## 四、实现

| 文件 | 内容 |
|---|---|
| `wein-serve/src/marker.rs`（新） | `safe_prefix(s, marker)`——流式安全切点（末尾可能是标记前缀时扣住，且切点必须是 UTF-8 字符边界；工具与思维链共用，避免两份易错逻辑分叉） |
| `wein-serve/src/reasoning.rs`（新） | `ReasoningSplitter::{feed, finish}` → `Seg::Reasoning/Content`；5 个单测（整段/任意切分粒度含逐字节/无标记/多标记/标记在开头） |
| `wein-serve/src/api.rs` | `seg_events()` 统一"切分片 → SSE 帧"；`dispatch(..., reasoning)`；非流式整段过同一状态机 |
| `wein-serve/src/lib.rs` | `ServeState.reasoning` 开关 |
| `wein-cli/src/main.rs` | `--reasoning-parser`（auto 探测模板）+ 启动行 `+reasoning` |

## 四·补、连带发现的 bench 缺陷（已修）

`wein-bench` 的流式计数只认 `delta.content`——推理被拆走后，L3q（`max_tokens=12`，
整段都在思考）**一帧 content 都没有**，battery 直接读出 `tps=0 / TTFT=null`。
测量客户端的语义应该是「一个数据帧 = 一个生成 token」，与它落在哪个字段无关：
改为 `content` 或 `reasoning_content` 非空即计（TTFT 取第一个此类帧）。
修后 L3q 手动复测回 **271.3 tps / ITL 12.91 / TTFT 41.0**（修前基线 269.4/12.28/43.7）。
> 教训：新增一个"输出字段"必须扫一遍所有**消费端**（这里是自研压测客户端），
> 否则基准会静默量出零。

## 五、验收

| 层 | 用例 | 结果 |
|---|---|---|
| 切分器 | r1-r5（含逐字节边界、多标记、无标记） | 5/5 |
| serve | rs1 关=旧行为逐字/rs2 非流式分离/rs3 流式帧序/rs4 推理里的格式示例不造假调用/rs5 `enable_thinking:false` 不切分/rs6 截断不丢字/rs7 只有调用 content=null | 7/7 |
| 回归 | wein-serve + wein-model 全量（含 tools ts1-ts7、模板 ct1-ct11） | 全绿 |

### 真模型端到端（Qwen3.5-4B，`--ctx-len 2048 --kv-pages 256`，启动行出现 `+reasoning`）

1. **思考开 + 工具**：`reasoning_content` = 完整推理（45 字），`content` = `null`，
   `tool_calls[0]` = `get_weather{"city":"北京","unit":"c"}`，`finish_reason` = `tool_calls`。
2. **流式**：帧序 `R×44 → C → T → F:tool_calls`（推理 44 片、正文 1 片、调用 1 片），
   `</think>` 零外泄。
3. **思考关**（`chat_template_kwargs.enable_thinking=false`）：无 `reasoning_content`，
   `content` = `null`（只有调用），工具参数与开思考时一致——模板两条分支都对。
4. **多轮回灌**（assistant `reasoning_content` + `content:null` + `tool_calls`，再跟
   `role:"tool"`）：模型据工具结果作答、`finish_reason: stop`、无多余调用。

## 五·补、battery（本轮 / 上一轮 / vLLM 同口径）

| 指标 | 本轮（reasoning） | 上一轮（tools） | vLLM（bf16 态、无 W8A8） |
|---|---|---|---|
| 图档满载 tps | **4526.9** | 4498.9 | 4330–4350 |
| 图档满载 ITL p50 | 19.74 | 19.89 | 19.11 |
| 图档满载 TTFT p50 / p99 | 46.7 / 188.9 | 48.4 / 188.0 | 144 |
| 轻载 c=4 图档 ITL | 11.21 | 11.64 | — |
| eager 满载 | 4180.7 @ 21.47 | 4290.5 @ 20.83 | — |
| L3q 真模型轻载 | 260.0 @ 12.39 | 269.4 @ 12.28 | — |

切分在 HTTP 侧（无 tools/无思考时零成本）；eager 那 2.4% 在既有抖动带内
（近四轮 eager 4122–4300）。`bench-reports/battery-20261108-*`。

## 六、边界

1. 只认 `</think>`：若某模型的模板把 `<think>` 放在**输出**侧，需要补"起始标记剥离"
   （本模型的模板把它放在提示尾，故未实现——不做投机设计）。
2. `content` 不做 trim（与 vLLM 逐字一致：`</think>` 后的 `\n\n` 保留在 content 里）；
   只在"仅调用"时按空白判空给 `null`。
3. 采样器/投机解码不感知此切分（都是同一文本流下游），已验证不影响。
