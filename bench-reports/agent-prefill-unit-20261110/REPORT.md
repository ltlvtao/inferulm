# Agent 热 turn prefill 单价（③）：radix 命中真相 + 增量单价账

日期：2026-11-10 · HEAD：226cf04+ · 口径：c=1 图档（--graph --ctx-len 32768 --kv-pages 4096 --chunk-size 2048，WEIN_DRAM=1）

## 1. 结论先行

1. **radix 前缀缓存无 bug**：此前"agent 热 turn 全量重 prefill（t=2048+608）"是误读——
   `t=2048` 不是全量，是**命中后的增量 prefill 第一步**。WEIN_RADIX_DEBUG 逐轮日志：
   | 请求 | seq | 命中 | 增量 prefill |
   |---|---|---|---|
   | warmup | 2048 | 0（reset 后） | t=2032+16 |
   | turn1 | 423 | 0（冷） | t=416+7 |
   | turn2 | 3074 | 416（system+tools 前缀） | t=2048+608 |
   | turn3 | 5729 | 3072（= turn2 prompt 端点快照） | t=2048+608 |
   | turn4 | ~8400 | 5856 | t=2048+592 |
2. **每轮增量 ~2656 token 是 agent_bench 本征形状**：上轮 assistant（reasoning ~150-200 tok
   + tool_call JSON ~80）+ 工具结果 filler 512 词（`word12345` ×512 ≈ 1300-1500 tok）+ 追问
   ~15 + 模板标记。prefix_hit_rate 0.69 与此一致（命中/prompt，如 5856/8400=0.697）。
3. **增量 2656 的单价账（wein GPU 170ms vs vLLM ~141ms）**：
   - gemm 118.7 vs ~118.5 —— **平价**（2048 步 86.7 + 608 步 32.0；vLLM 按 6009 冷 268ms
     线性折算 2656 token）
   - delta chunked 24.3 vs ~10 —— 差 14ms（最大单项；h/prep/o 换形态均已评估为贵）
   - 杂+两步切分开销 ~10 vs ~4
   - 非 GPU（调度/网络/首 token）：wein 53ms vs vLLM 57ms —— 相当
   - 客户端 ttft_hot：223.5 vs 197.7（+13%）
4. **两步切分是便宜的杠杆**：增量 2656 被 chunk 2048 切成 2048+608 两步，m=608 尾步
   gemm 效率低（52.6µs/tok vs 2048 步 42.3）。一步 2656 预期省 gemm ~7ms + 步开销 ~4ms。

## 2. chunk-size A/B（零代码，服务参数）

c=1 agent ×6 turns，3 发交错（服务口径同上，仅 --chunk-size 变）：

| chunk | ttft_hot 中位（3 发） | 合并分布（ms） |
|---|---|---|
| 2048 | 223.6 / 222.4 / 209.1 | [86, 190..259] |
| 4096 | 189.6 / 189.8 / 203.8 | [77, 77, 78, 78, 90, 190..233, 386, 392] |

**ttft_hot 222.4 → 189.8（−14.7%），反超 vLLM 197.7（−4%）**。机制：2656 增量一步吃下
（m=2656 的 gemm 形状效率 > 2048+608 两发；省一次步开销 + logits/norm 全套）。
chunk4096 的 77-90ms 点 = 短历史轮（增量一步 + 恢复点浅）直接进 60ms 档；
386/392 尾点 = 长历史轮（ctx ~11K）的偶发，中位不受影响。

### 2b. c=4 / c=8 混批（各 2 发交错）

| 并发 | chunk | turns/s | ttft_hot | itl | p95 | hit |
|---|---|---|---|---|---|---|
| c=4 | 2048 | 1.617/1.602 | 228/246 | 13.52/14.06 | 14.55/15.22 | 0.700/0.678 |
| c=4 | 4096 | 1.579/1.545 | **222/215** | **13.07/13.71** | 14.16/14.76 | 0.658/0.682 |
| c=8 | 2048 | 2.065/2.123 | 423/340 | 15.45/15.83 | 16.54/16.74 | 0.691/0.716 |
| c=8 | 4096 | 1.851/1.890 | 395/348 | 15.66/15.12 | 16.86/16.83 | 0.647/0.645 |

c=4：ttft/itl 双优。c=8：turns −10%，其中约一半是**生成轨迹漂移的口径污染**——
chunk 变 → prefill 分块变 → gemm m-shape 变 → ulp 级翻转 → reasoning 长度变 →
prompt_tokens 305K/295K → 316K/318K（+5-8% 工作量）；另一半是真实效应：混批下
预算闸（26ms est）把混批步切到同大小，但**纯 prefill 窗口**（多会话同时 prefill）
chunk4096 整步跑 2656+ token，阻塞 decode 会话（tok/s 272/296 → 249/249）。



## 3. 证据

- `/tmp/radixdbg/srv.log`（WEIN_RADIX_DEBUG+STEP_PROFILE full）—— match/retain/snap-fire 全链
- 步打印 t=/rows= 由 WEIN_STEP_PROFILE full 档输出（本刀新增的诊断字段 note_t）
- c=1 对照三发：wein ttft_hot 222.3/223.5/223.8（极稳）；vLLM 197.7（带
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3）

## 4. 决定

1. **chunk-size 默认保持 2048**（c=8 混批有真实回退）；低并发 agent 场景（c≤4）部署建议
   `--chunk-size 4096`：c=1 ttft_hot −14.7%（222→190，反超 vLLM 197.7），c=4 ttft/itl 双优。
   参数已存在，零代码。
2. **radix/快照链不动**——工作正常，"全量重 prefill"是误读（增量本征 2656/轮）。
3. 剩余差距的账（c=1：223.5→189.8 后 vs vLLM 197.7，已反超 4%）：delta chunked 24.3 vs
   ~10ms 是最大单项，但 h/prep/o 换形态均已评估为 1-2 周级且收益 ≤5ms（见 dn-gap 报告
   封刀结论）——不追。下一个数量级在 W8A8（用户约束暂缓）。
4. 诊断仪器入库：WEIN_STEP_PROFILE=full 档（全段列表+segsum+每步 t=/rows=），
   WEIN_RADIX_DEBUG 既有开关本刀重度使用（match/retain/snap-fire/锚窗口）。
   gemm_firstcall_probe（形状猜测不可靠）删除。

