# wein vs vLLM 终测收官报告（无 W8A8）

日期：2026-11-09 晚　wein HEAD=060281c　环境：RTX 4090 24G / WSL / 驱动 610.88
vLLM 0.32.0 + torch 2.13.0+cu130；同一客户端（wein-bench / longctx_probe.py /
agent_bench.py）、同一题面、成对交错 3 轮取中位（A 轴）；GPU 独占、代理全 unset、
每块前后查 Windows 侧 compute-apps。

口径：wein = 图档（--graph）默认配置（mix_cap 4096 + mix_budget 26ms/rows32，
DRAM 分层关）；vLLM = mem 0.85 / mbt 2048 / maxseqs 256，状态口径两档并列
（bf16ssm = --mamba-ssm-cache-dtype bfloat16 与 wein 同口径；f32 = 模型声明默认）。
**双方均无 W8A8/量化**（用户约束）。

## 1. A 轴：服务基准（final-vs-vllm-20261109-221849，3 轮交错中位）

### 短上下文（OpenAI 协议，wein-bench）

| 档 | wein | vLLM bf16ssm | vLLM f32(声明) | wein vs bf16 |
|---|---|---|---|---|
| 满载 c=100 tps | 4342.7 | **4424.6** | 3515.7 | −1.9% |
| 满载 ITL p50 | 20.57 | **19.22** | 25.38 | +7.0% |
| 满载 ITL p99 | **23.28** | 36.81 | 46.92 | **−37%** |
| 满载 TTFT p50 | 103.8 | 92.5 | 100.7 | +12% |
| 满载 TTFT p99 | **~114** | ~200 | ~211 | **−43%** |
| 中载 c=16 tps / ITL50 | 1127.9 / 13.30 | **1186.2 / 11.96** | — | −4.9% / +11% |
| 轻载 c=4 tps / ITL50 | 309.2 / 12.08 | 311.1 / 11.31 | — | −0.6% / +7% |
| 单流 c=1 tps / ITL50 | 87.9 / 10.78 | **92.7 / 10.25** | — | −5.2% / +5% |

### 长上下文（3009 / 6009 token prompt，32 token 生成，3 轮）

| 档 | wein | vLLM bf16ssm | 差 |
|---|---|---|---|
| 3009 冷 TTFT | 217.2-219.3 | **203.8-207.7** | +6.4% |
| 3009 热 TTFT | **17.8-18.9** | 30.2-32.2 | **−42%** |
| 3009 稳态 ITL | 10.94-11.14 | 10.33-10.52 | +6-7% |
| 6009 冷 TTFT | 419.7-422.9 | **395.7-401.2** | +5.5% |
| 6009 热 TTFT | **19.8-20.3** | 33.8-36.2 | **−43%** |
| 6009 稳态 ITL | 11.03-11.56 | 10.49-10.54 | +5-9% |

## 2. B 轴：Agent 多轮工具调用（agent_bench，6 轮×512B payload/会话，ctx 32768/4096 页）

**口径事故与修正**：vLLM 侧工具循环必须 `--enable-auto-tool-choice
--tool-call-parser qwen3_coder --reasoning-parser qwen3`（Qwen3.5 模板是
`<tool_call><function=…>` 函数式格式，hermes parser 解不出 ⇒ tool_ok=0、
会话循环退化、prompt 不增长——**此前存档的 vLLM agent 基线
（turns 1.259/hit 0.67/ttft_hot 31）即此残废题面，作废**）。wein 侧
WEIN_DRAM=1（长会话推荐位）+ 默认 mix_budget 26ms。

| 档 | wein（中位） | vLLM bf16ssm | 差 |
|---|---|---|---|
| c=4 turns/s | 1.359（1.351/1.374/1.359） | **1.613**（1.607/1.618） | −15.8% |
| c=4 ttft_hot ms | 284.7（284-292） | **250.8**（251/251） | +13% |
| c=4 ITL med/p95 | 15.01 / 18.03 | **12.50 / 13.66** | +20% / +32% |
| c=4 prefix_hit | **0.693**（0.675-0.697） | 0.665 | +4% |
| c=4 tool_ok / errors | 0.71-0.88 / 0 | 0.75-0.79 / 0 | 平 |
| c=8 turns/s | 1.41（1.380/1.435） | **2.460** | −42% |
| c=8 ttft_hot ms | 513（457-570） | **270.8** | +81% |
| c=8 ITL med/p95 | 17.84 / 20.40 | **13.98 / 15.89** | +28% / +28% |
| c=8 prefix_hit | 0.455 | **0.666** | −32% |

c=8 的 hit 塌是**容量问题**：8 会话×~42K token ≈ 336K 需求 vs wein 池
4096 页=65K token（+DRAM 快照层 64 槽）逐出频繁；vLLM mml 32768 下 KV 池
远大（mem 0.85 ≈ 12GB+ KV）。hit 塌 ⇒ 重算 ⇒ turns/ITL 连带。

## 3. 判读

**wein 赢的（服务质/尾部/成本）**：
- 满载 ITL p99 **23.3 vs 36.8（−37%）**、TTFT p99 **~114 vs ~200（−43%）**——
  mix_budget 闸门 + 图档的结果，SLO 视角显著更好。
- 长上下文热 TTFT **18-20 vs 30-36（−42%）**——radix 快照恢复快于 vLLM 的
  mamba 前缀缓存。
- f32 声明口径（vLLM 默认、真实用户默认配置）满载 tps **4343 vs 3516（+24%）**。
- 递推状态内存 bf16（vs vLLM 默认 f32 一半），且 32K ctx 常驻可用。

**vLLM 赢的（吞吐/单位延迟）**：
- 满载 tps −1.9%（4343 vs 4425）——接近平价；c=16 −4.9%、c=1 −5.2%。
- 长上下文冷 TTFT +5-7%（delta prefill 65.7ms 与 gemm 组织差距）；
  稳态 decode ITL +5-9%（gemm 8.34ms/步占 74%）。
- Agent c=4 turns −15.8%、ITL med +20%；c=8 −42%（容量受限，见 B 轴注）。

## 4. 结论

**wein（纯 Rust、单二进制、无量化）与 vLLM 0.32 在 RTX 4090 / Qwen3.5-4B 上
的总体位形：吞吐接近平价、服务质（尾延迟/热恢复）显著更优、单位延迟与
大并发 Agent 容量仍有差距。**

| 维度 | 判 |
|---|---|
| 满载吞吐（bf16 同口径） | −1.9%（4343 vs 4425），平价带内 |
| 满载吞吐（f32 声明口径=用户默认） | **+24%**（4343 vs 3516） |
| 满载尾延迟 p99 | **ITL −37%、TTFT −43%** |
| 长上下文热恢复 TTFT | **−42%**（radix 快照 vs mamba 前缀缓存） |
| 长上下文冷 TTFT | +5-7%（delta prefill 与 gemm 组织） |
| 中低并发 ITL / 稳态 decode | +5-7%（gemm 8.34ms/步=74% 是账面主项） |
| Agent c=4 | turns −16%、hit +4%；itl_med +20% 是主要差距 |
| Agent c=8 | turns −42%（KV 容量差，非 kernel 差） |

无 W8A8 前提下的收官判断：
1. **服务质轴（p99/TTFT/热恢复）已稳定领先**——调度闸门（mix_budget 26ms）
   + 图档 + radix 快照的组合是结构性优势，vLLM 同配置无对应机制。
2. **吞吐轴已被压到 −2% 以内**（f65bd8f 刀时 −15%）：js64 并行度刀 + 满载
   battery 4358 是历史最高。
3. **剩余差距三处，全部有名有姓**：①decode gemm 8.34ms/步（74%）——m-shape
   组织（vLLM 用 gemv+splitK 族，我们 cuBLASLt m=100 档 751GB/s 上限）；
   ②delta prefill 65.7 vs 22.7（BT=64 chunked 已收 3×，剩余在 h 链）；
   ③Agent 大并发 KV 容量（池 65K token vs vLLM ~300K+）——池扩容+准入联动。
   三处之外无系统性差距。
4. W8A8（对手 +24%）按用户约束未做，是双方都保留的下一张牌。
