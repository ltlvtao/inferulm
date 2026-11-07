# Agent 场景端到端压测（2026-11-08）

> 前两刀把协议面补齐了（标准 DTO + `tools`/`tool_calls` + `reasoning_content`）。
> 这一刀换成**用真实形态量一遍**：多轮会话、每轮先调工具、工具结果灌回上下文、
> 上下文逐轮变长——这是 Agent 负载区别于"单发 chat"的全部要害（前缀复用能不能
> 吃到，决定 TTFT 与总吞吐）。
>
> 客户端用**官方 `openai` SDK**（3.41.0，Agent 框架的事实标准）：它自带 pydantic
> 严格校验，跑通即等于"标准客户端可用"；服务端两侧都按同一脚本、同一提示、同一
> 工具集跑，参数对齐。

## 一、仪器：`bench/agent_bench.py`

```
python bench/agent_bench.py --url http://127.0.0.1:8193 --model <id> \
    --sessions 16 --turns 6 --payload 256 --max-tokens 128 --json out.json
```

* **会话**：system(Agent 人格，要求必须先调工具) + 任务 → T 轮；每轮 = 一次流式
  `chat/completions`（带 2 个工具），模型给 `tool_calls` → 脚本回灌
  assistant(含 calls) + `role:"tool"` 结果（`--payload` 个伪 token，模拟日志/文档
  片段）→ 再追问"继续调查"。**上下文逐轮单调增长**（Agent 的真实增长形态）。
* **口径**：TTFT = 首个非空 delta（正文或推理）；ITL = 相邻 delta 间隔中位数；
  `tool_ok` = `finish_reason=="tool_calls"` 且参数是合法 JSON；前缀命中率 =
  `/metrics` 的命中 token 增量 / prompt token 增量（wein:
  `wein_prefix_hit_tokens_total`；vLLM: `vllm:prefix_cache_hits_total`）。
* **两侧度量差异必须记住**：vLLM 0.32 的推理增量字段叫 `delta.reasoning`（我们是
  `reasoning_content`），脚本两个都认；两侧的 chat 模板都会**丢弃**
  `reasoning_content` 回放（实测：带/不带渲染出的 prompt 长度逐字相同），所以基线
  是对齐的。

## 二、读数（8192 ctx，双方默认前缀缓存，bf16 状态、无 W8A8，greedy）

| 场景 | 侧 | turns/s | tok/s | 工具成功率 | TTFT 冷 p50 | TTFT 热 p50 | ITL p50 | 前缀命中率 |
|---|---|---|---|---|---|---|---|---|
| c=4×6 | vLLM 0.32 | **2.30** | 277 | 42% | 97.6 | **142.0** | **12.11** | **74.3%** |
| c=4×6 | wein (kv 64K) | 1.75 | 212 | 38% | 162.9 | 291.9 | 14.30 | 65.0% |
| c=16×6 | vLLM 0.32 | **5.72** | 678 | 51% | 278.4 | **148.5** | **15.00** | **70.7%** |
| c=16×6 | wein (kv 64K) | 1.55 | 183 | 46% | 162.9 | 4067.6 | 55.8 | 28.8% |

单个数字最刺眼的是 **c=16 热 TTFT 4.07s vs 148ms（27×）**、turns/s 1.55 vs 5.72。
逐轮看（wein, c=16）：TTFT 随轮次单调恶化 1.5s → 7.4s，ITL 稳在 56-63ms；vLLM
同场景 TTFT 稳在 130-190ms、ITL 12-16ms。**这不是 decode 慢，而是 prefill 队列越
堆越长**。

## 三、根因（三条，都有实测支撑）

### ① 前缀命中率 29% vs 71% —— 每轮多算 2.4× 的 prefill token
我们的 radix 命中必须落在**有状态快照槽**的块上（24 个线性层是递推状态，不能只靠
KV 重放），而快照是"**每请求每步末位置各一张**"：一步吃完整段 prompt 时，整段
prompt 只产生 1 张快照，位置在 chunk 末。于是：

* 满足条件时，命中 = 上一轮的 prompt（`seq=4488 → hit 3136`），下一轮新增的
  assistant 输出 + 工具结果必然重算——vLLM 连 **decode 生成的块**也带 hash 进树，
  这部分它不用重算；
* 16 会话 × 6 轮 = 96 个请求抢 64 个槽（每槽 25.2MB），LRU 一抖，连
  "system+tools" 这段**全军共享**的前缀也被淘汰 → 命中塌到个位数。

三次容量对照（c=16）：`kv512/slot64` 32.1%（但下面 ④ 说明这轮数据无效）→
`kv4096/slot64` 28.8% → `kv4096/slot128` 34.7%。**加槽救不回来**，说明病灶是
"每请求只落一张快照 + 快照粒度 = 整段 prompt"，不是槽数。

### ② prefill 单价 ~2×（这是长上下文那刀留下的已知项）
c=4 冷 TTFT 163ms vs vLLM 98ms；把每轮必算量（约 450 新增 token + 未命中段）按
两侧墙钟折算，我们每 token 约 30µs、vLLM 约 20µs——与 6009 冷那份账本的
"同 chunk 单价 1.4-2×"一致。

### ③ prefill 突发把 decode 挤成 56ms（vLLM 15ms）
c=16 的 ITL 从 14ms（c=4）涨到 56ms，而单 token 的 decode 成本没变——**同一步里
prefill 行与 decode 行抢同一个前向**，我们的 chunk 步对 decode 的伤害是 vLLM 的
3.7×。

### ④ 顺带抓到的真 bug：容量不足被报成 `finish_reason:"length"`（已修）
`kv512`（8K token 池）那两轮"跑得飞快"（c=16 6.45 turns/s、ITL 14.4ms）是假象：
80 个 turn 里有 **71 个拿到 0 个 token 且 `finish_reason:"length"`**——执行器的
`StepError::Capacity` 被映射成 `Length`。客户端（包括真实 Agent 框架）看到
"length + 空内容"只会理解为"被截断，重试短一点"，根本想不到是服务端装不下。
这跟当年 `assert!` 打死引擎是同一类错误：**错误语义错位 = 静默错误**。

修法：`FinishReason` 新增 `Capacity` 变体（与 `Length` 并列），serve 侧
`finish_reason:"error"` + `wein_finish:"capacity"`（OpenAI 协议没有 capacity 这个
值，用判别字段分档——与 `aborted`/`error` 同款取舍），`/metrics` 计入 failed。
端到端验证（池 64 页、4 个并发长请求）：3/4 报 `("error","capacity",0 tokens)`。

### ⑤ 一个坏工作点（容量账，不是慢）
`kv-pages 8192`（128K token KV）+ `snap-slots 128`（3.2GB 快照）→ 显存 24046/24564
MiB，TTFT 25s、ITL 137ms。**与 vLLM 在 `gpu-mem 0.9` 的坏态同构**：WSL 下 VRAM
打满会往宿主内存分页。容量必须按预算给，不能"越大越好"。

## 四、结论

* Agent 多会话是**我们目前最差的一类负载**：c=4 落后 1.3×，c=16 落后 3.7×。
  单发 chat 的满载我们已经反超 vLLM 4-5%，但这里的瓶颈完全在**组织形态**：
  前缀复用粒度（快照）与 prefill/decode 混流。
* 差距可分解为三块，方向明确：命中率 ×2.4（最大）、prefill 单价 ×1.5、
  decode 干扰 ×3.7。
* 本刀的产出：① 可复现的 Agent 压测仪器（openai SDK 真客户端）② 三条根因的
  实测账 ③ 一个真 bug 的修复（容量语义）④ 两个容量旋钮的坏工作点边界。

## 五、下一刀候选（按收益/风险排序）

1. **radix 按块 hash（含 decode 生成的块）+ 快照策略重做**：目标命中率 ≥ vLLM
   的 70%。要点：KV 块共享不需要快照（8 个 full-attn 层可无损复用），只有 24 个
   线性层需要状态——快照粒度可以粗（如 512/1024 token 一格），槽数按
   `KV 预算/会话数` 定；`system+tools` 这类全局共享前缀应常驻（pin 最低位置）。
2. **chunk 调度给 decode 让路**：prefill 步限行数/限时，避免 56ms 的 decode 饥饿。
3. 容量预算模型：KV/快照/状态池按 `--max-sessions × ctx` 反推，CLI 侧报"这套配置
   能装几个会话"，并把 ④⑤ 两类边界写成启动期检查。
