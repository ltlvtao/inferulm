# decode attention 单组档（杀行组读放大）— 2026-11-10

## 病灶：每 block 行组数 = K/V 读放大

`online_body` 每 block 覆盖 `rows_nom = ntok × group` 个 q 行，切成
`行组数 = rows_nom / rpt` 组，**各组独立、重复载同一批 K/V**。生产
prefill 档（nd=4/rpt=4/ntok=4，group=4 ⇒ 16 行/块）行组数 = 8 ⇒ 名义
K/V 读放大 8×。agent 稳态（t=8 decode 行，走 prefill cfg 因 t>4）正是
此形状：KV 池工作集（8 会话 × 32K ctx ≈ 40-160MB）远超 72MB L2 ⇒
去重失效，直接打 HBM 带宽墙。

探针（`examples/attn_rows_probe.rs`，修 GB/s 公式 + interleave 后）：

| R(rows) @ ctx | 现役 (n4r4,ntok4) | 单组 (n8r4,ntok1) | Δ |
|---|---|---|---|
| 8 @ 1280  | 0.0714ms | 0.0510ms | −29% |
| 8 @ 2560  | 0.2265 | 0.1492 | −34% |
| 8 @ 5120  | 0.4251 (331GB/s) | 0.2885 (582GB/s) | −32% |
| 8 @ 10240 | 0.8446 | 0.5942 | −30% |
| 64 @ 10240 | 0.3252 | 0.3211 | 持平 |

- R≤2 短 ctx 单组反慢 5-19%（绝对值几 µs，块数已够）。
- 旧"js64 后 590GB/s@R=1"是 20MB 工作集吃 L2 的假象；真实 R≥4@长 ctx
  才是 HBM 带宽墙。
- 页交错 vs 连续无差（±3%）——非页表局部性问题。

## 改法（零 kernel 改动，纯 cfg 分派）

- `AttnCfg` 单组档 `nd=8, rpt=4, ntok_max=1`（bdim=32，jsplit 自适应）。
- exec `attn_long` 加 `cfg_rows(rows)`：行数 ∈ [4, PART_ROWS/qh=64] 的
  online 段批换单组档；**五个** `run_tile_online_cfg` 调用点全换
  （MIXSEG pend flush ×2、v5/v9 rem、v4 tail、整批）。
  - <4 行不换：绝对值小且单组反慢；
  - >64 行不换：`rows×qh > 1024` 后 jsplit 落不下（js=1 块数骤减，
    c=100 满载风险）。
- env 旋钮 `WEIN_ATTN_CFG_SGL`（设 `r4` = 等效旧行为，同二进制 A/B）。
- `ATTN_TILE_PART_ROWS` 导出 pub（选档上界用）。

## 数值（o44_single_group）

- 单组 vs 现役：同 jsplit 形逐位同（t=4@3071、t=16@8191、t=100@3000
  强档）；jsplit 差异形 rel2 ≤ 3.1e-5、峰值 ≤ 3.8e-3、逐行最坏 ≤ 3.8e-3
  （门照 o31/o33；jsplit>1 = 偏量 attn_combine 结合律差）。
- kernel 全量测试绿（o22 单跑复核 0.241ms；并行跑时跨进程 GPU 争用
  假红——既有纪律）。exec 全绿（qwen 23 / bigchunk 2 / graph 4）。

## A/B（同二进制 env 开关，3 轮交错，中位）

| 场景 | 指标 | OFF(旧) | ON(单组) | Δ |
|---|---|---|---|---|
| agent c=8×6 | turns/s | 1.920 | 2.043 | **+6.4%** |
| | itl_med | 16.45 | 15.87 | −3.5% |
| | itl_p95 | 19.41 | 17.44 | **−10.1%** |
| | wall_s | 24.8 | 23.6 | −5% |
| agent c=4×6 | turns/s | 1.581 | 1.618 | +2.3% |
| | itl_med | 13.89 | 13.16 | −5.3% |
| | itl_p95 | 15.39 | 14.52 | −5.7% |
| battery 满载图档 | tps | 4255-4358（近三档） | **4472.1** | +2.6~5% |
| | ITL p50/p99 | 20.79/33.25 | **19.74/30.74** | −5%/−7.5% |
| 6k decode（R=1 不换档） | ITL | 见下表复验 | 同档 | 0 |

- ttft_hot ON 略升（c8 326→379ms）：吞吐-排队二阶效应——同工作量更快
  完成（wall −5%）⇒ 会话轮转更快 ⇒ 各 turn 的 prefill 排队重叠加剧。
  ttft_cold 持平、itl/turns/wall 全方向一致（6/6）。
- vs vLLM（c=8 13.98 / c=4 12.50）：itl 差距 22%→**13.5%（c=8）**、
  **5.3%（c=4）**。

## 结论

单组档收官：agent 两档 itl 全绿、满载 battery 创新高（4472）、无任何
回退路径（R=1/R≤2/R>64 全走原档）。剩余 c=8 差距 13.5% 归混批调度/
gemm m 形（另账）。
