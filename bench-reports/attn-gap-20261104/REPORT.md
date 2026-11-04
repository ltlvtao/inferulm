# 6009 prefill attention 差距解剖：wein v4（wmma）104.7ms vs vLLM FA2 19.27ms

日期：2026-11-04 · 结论先行：**差距不是"用没用 tensor core"（v4 用了），而是两个乘性因子：
KV 重读粒度 4× × 流水线质量 ~1.35×**。两边在各自的约束下都已接近自己的墙——
我们的墙是 L2 流量+潜伏期，他们的墙是 tensor core 本身。

---

## 1. 测量锚点（全部实测，无外推）

### vLLM 侧（bench-reports/vs-vllm-kernels-20261102/B-kernels.txt，6009 冷 prefill）

```
flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, false...>>
  n=24（3 chunk × 8 层） 总 19.267ms avg 802.8µs
```

- 模板参数实测含义（FA2 惯序）：**hd=256, kBlockM=64（Br=64 行/CTA）, kBlockN=64, 4 warps**
- 同 trace 里 prefill 段无 `flash_fwd_splitkv_combine_kernel`（decode 段才有）⇒ splits=1，
  splitkv 入口是 fork 的统一 varlen 路径，非真拆分
- chunked prefill 2048（--max-num-batched-tokens 2048）：3 步，每步 16 q 头 × 32/30 个
  64 行 CTA
- Python 侧（vllm_flash_attn/flash_attn_interface.py:420）：**pack_gqa=None** ⇒ GQA 不打包，
  逐 q 头 CTA——与我们同构，公平对比

### wein 侧（attn_v4_probe + 今天生产读数）

| 项 | 值 |
|---|---|
| 6009 单 span v4 计时 | **12.91 ms/层**（探针 10 发均摊）×8 = 103.3ms；生产段表 104.7ms |
| ptxas -v（sm_89） | **255 regs**、0 spill、8704B smem、`used 0 barriers` |
| 占用率 | 65536 regs/SM ÷ (255×32) = **8 warp/SM（12.5%）**，6016 CTA = 5.9 波 |
| 对照 v3(br64) | 24.84 ms/层（v4 已 1.92×） |
| 因果早停 | 生效——t=3000 单 span 3.79ms/层 ⇒ 若无早停需 4.75TB/s（超 L2 峰，不可能） |

## 2. 流量模型（两边都对得上账）

KV 每 token 每 kv 头 = K 512B + V 512B = 1KB。因果工作 U = Σ_r pos_r ≈ t²/2（两边同工，
他们 64 行瓦片取整多吃 ~5%）。

| | vLLM（Br=64，逐 q 头 CTA） | wein v4（Br=16，逐 q 头 warp） |
|---|---|---|
| 每层 KV 读量 | 16 qh × Σ(chunk/64) × ctx × 1KB ≈ **5.5 GB/层** | 16 qh × Σ(6009/16) × tile_end×16KB ≈ **18.1 GB/层** |
| 8 层总流量 | ~44 GB | ~145 GB |
| 有效带宽 | ~2.3 TB/s | **~1.40 TB/s** |
| tensor 利用率 | **~75% 峰值**（296 GFLOP/层 ÷ 2.41ms ≈ 123 TFLOPS） | **~14%**（296 GFLOP/层 ÷ 12.91ms ≈ 22.9 TFLOPS） |
| 时间 | 19.27ms | 104.7ms |

**5.43× = 流量比 3.3×（≈粒度 4× ÷ 工作差 1.05× × 模型斜率） × 带宽效率比 1.65×**

## 3. 两个因子的机理

### 3.1 粒度 4×：每 16 行就把整条 KV 流重读一遍

我们：block =（q 头, 16 行），每块独立扫自己的全部前缀。6009 行 ⇒ 376 个行片 × 16 q 头
= 6016 次"整条流"。同一 kvh 的流被 4 个 q 头块各读一遍、被 376/4=94 个行片反复读。
vLLM：64 行一 CTA，重读次数直接 ÷4。**这与 L2 命中与否无关——L2 只救 HBM，救不了
L2→SM 的流量本身**（24.6MB 工作集在 L2 里，但每字节仍要流到 SM 18.1GB/层）。

### 3.2 带宽效率 1.65×：8 warp/SM 的潜伏期受限 vs cp.async 流水

我们 v4 每 warp 每 tile 的串行链：
```
K 直接从 global 进 wmma fragment（同步 load，无预取）
→ 16 步 mma（S）
→ 掩码 + exp（ALU）
→ P 量化写 smem → bar.sync → 读回 fragment（2 道屏障）
→ 16 步 mma（O）
→ V^T staging（64 次 2B 散读 global + smem 散写）
```
255 regs 把占用钉死在 **8 warp/SM**——这个并行度藏不住 global load + 屏障潜伏期，
SM 发射槽大量空转。叠加 24.6MB 工作集开始跨 AD102 远 L2 分区（t=3000 时 12.3MB 还能
跑 2.38TB/s，t=6009 掉到 1.40——同 kernel 同代码路径，唯一变量是工作集翻倍）。

vLLM FA2（其 fork 未检出 CUDA 源，此段为 FA2 公开架构，模板参数为实测）：Q 块驻寄存器、
K/V 用 **cp.async 双缓冲** staging smem、softmax/rescale 与 mma 交错、4 warp 共享一个
64 行块的分摊。寄存器同样吃紧（O 累加 128 regs/thread，占用 ~12 warp/SM）但**没有
同步 global→fragment 的直读**，load 与 compute 真重叠 ⇒ 摸到 75% tensor 峰值。

## 4. 修复路径量化

| 步骤 | 预期 | 依据 |
|---|---|---|
| **v5 = 4 warp × 64 行共享 K/V**（流量 ÷4，一行 K/V 供 4 个行片-warp 复用） | 6009 单 span 104.7 → **30-45ms**；流量降到 4.5GB/层 ≈ vLLM 档 | 流量主导项直接 ÷4；mma 地板 1.79ms/层 |
| + K staging 进 smem（消灭 global→fragment 同步直读） | → **~20-25ms** | 潜伏期链缩短，带宽效率向 2.3TB/s 靠 |
| + cp.async/双缓冲（需新 intrinsic 旁路，同 shfl/smem 前例） | → **15-20ms ≈ vLLM 同工 19.27** | mma 75% 峰值 = 他们所在的位置 |
| 32K 场景（现 v3 173ms/步，ctx>8K 被 v4 门控拦住） | v5 流量 ÷4 对 HBM 同样成立 → **~45-60ms/步** | 超 L2 后流量即 HBM 流量 |

先做 v5 第一步即可把 6009 冷账的 attn 段从 104.7 打到 ~35ms，6009 冷 TTFT 预期
557 → ~490ms（对 vLLM 383-418 的差距从 1.32× 收到 ~1.17×）。

## 5. 为什么"换了 tensor core 还是慢"一句话版

tensor core 只保证乘加本身快；attention 的墙在**每字节 KV 要流多少次、流的时候
有没有人在等**。v4 把乘加换成了 mma，但每 16 行重读一遍 KV（4× 流量）且 8 warp/SM
藏不住访存潜伏期（1.65× 带宽效率）——两个乘子一乘，75% 峰值的对手就把 14% 峰值的
我们甩开 5.4×。
