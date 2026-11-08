# Agent c=16 的 ITL 52ms 归因（2026-11-08，诊断刀，生产代码零改动）

> 上一刀把命中率从 28.8% 拉到 37.7% 后，剩下的账是：**ITL 52.5ms vs vLLM 15.0ms**。
> 本刀原计划做"prefill 步限时/限行"，结果三条假设全被自己的数据打回，
> 但把 52ms 拆到了具体机制上——**下一刀的目标因此从"调度"改成"deltanet 小 span 阈值"**。

## 一、假设 1：纯 prefill 波步长太大（❌ 证伪）

`mix_cap`（默认 192）只在**批内存在 decode 行**时生效；Agent 会话异步轮转，批内多数时间
全是 prefill 行 ⇒ 单步吃满 `chunk_size`。于是加了 `prefill_cap`（纯 prefill 波的单步封顶，默认 1024）。

实测（c=16×6）：

| 配置 | turns/s | 热 TTFT | ITL p50 | 命中 |
|---|---|---|---|---|
| 基线（无 prefill_cap） | 2.00 | 2700 | 52.5 | 37.7% |
| prefill_cap = 1024 | 1.83 | 2806 | **56.6** | 40.8% |

**ITL 反而更差**。原因：纯 prefill 波里本来没有 decode 行在等，封顶只增加了步数
（每步固定开销 × 更多步）。**已回滚，生产零改动。**

## 二、假设 2：主机侧空洞（❌ 误判，我方测量有陷阱）

`WEIN_STEP_PROFILE=per` 显示 `step 2 wall=74.3 | attn=1.7 logits=1.4 ...`——最大段 1.7ms，
看起来 64ms 落在 kernel 之外。nsys 又显示请求期 239 次 `cuMemcpyHtoD_v2`（med 126µs）、
646 次 `cuMemAlloc_v2`（med 217µs），很像是"每 span 一次小分配 + 一次同步上传"。

两条都不成立：

1. **`prof_mark` 自己就是 85µs/次**（`qwen.rs:769-771` 源码注释自带实测），混合步 350-440 个边界
   ⇒ **插桩自己吃掉 30-40ms wall**，且这时间落在 kernel 空隙里、不进任何段。`wall - Σ段` 不可用。
2. **热路径的运行期分配为 0**：全树只有 `Gpu::alloc_from_slice` 能产生同步 `cuMemcpyHtoD`，
   它的运行期调用者只有 `mtp_draft_rope`（投机草稿，serve 侧不可达）。646 次 alloc / 429 次同步
   HtoD 是 **open 期权重逐张上传**（本 checkpoint 426 张）。

**测量纪律（新增）**：`WEIN_STEP_PROFILE=per` 的 `wall` 含插桩开销，只能看**段间相对比例**，
不能当绝对步长；要看主机账必须用报告里的 `host：上传/提交/读回` 三桶。

## 三、真正的定位：`mix_cap` 实验（✅ 机制确认）

直接改 `--mix-cap`（有 decode 行的步里 prefill 的配额）：

| mix_cap | ITL p50 | 热 TTFT | turns/s |
|---|---|---|---|
| 192（默认） | 52.5 | 2700 | 2.00 |
| **64** | **27.7（−48%）** | **16528** ⚠ | 0.89 |

结论：**ITL 由"有 decode 行的那些步里的 prefill 配额"直接决定**，且斜率极陡——
176 个 prefill token 换来 ~25ms 增量 ⇒ **≈280µs/token**。这比满载 gemm 边际（145µs）
和 6009 大块 prefill（85µs）都贵 2-3 倍。

TTFT 崩到 16.5s 说明不能靠压 `mix_cap` 解决（prefill 被饿死）。

## 四、280µs/token 从哪来：小 span 走不了 chunked（下一刀）

Agent 每轮的 prefill span ≈ **190 token**（新增 assistant 消息 + 工具结果 + 追问），而：

```rust
// crates/wein-exec/src/qwen.rs:236
const DN_CHUNK_MIN_SPAN: u32 = 256;
```

`cnt >= DN_CHUNK_MIN_SPAN` 才走 deltanet chunked（BT=64，≈10.8µs/token）；**190 token 的 span
落进最差区间**——只能走老路（vsplit 逐 token 串行状态更新，实测 **2.36µs/token/层**）：

190 × 24 层 × 2.36µs ≈ **10.8ms/步**，正是 176-token 增量里最贵的一块。

## 五、下一刀（已定，两件）

1. **`DN_CHUNK_MIN_SPAN` 256 → 64**（BT 的最小单位），让 Agent 的小 span 也走 chunked；
   风险：190-token span 的 chunked grid 可能欠铺 128 SM（3 个 BT=64 chunk × 行×vsplit），
   需实测；`WEIN_DN_CHUNK=0` 与容量不足的既有回退保持。
2. **`mix_cap` 与 `prefill_cap` 解耦**：单步封顶应按"**时间**"而不是 token 数——同一 token 数
   在 decode 行多时更贵。先取 `mix_cap` 128 + `prefill_cap` 独立值做二维小扫，找
   ITL ≤ 25ms 且 TTFT 不劣化的点。

## 六、验收基线（本刀读数，c=16×6，8192 ctx，kv6144，chunk2048）

| 指标 | 值 |
|---|---|
| turns/s / 热 TTFT / ITL p50 / 命中 | 2.00 / 2700ms / 52.5ms / 37.7% |
| 每步 token（metrics 反推） | 206 |
| 每步墙钟（wall/steps） | 50.0ms |
| vLLM 0.32 同脚本 | 5.72 turns/s / 148ms / 15.0ms / 71% |
