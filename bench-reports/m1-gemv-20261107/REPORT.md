# 单流 c=1 −9%：m=1 直路 GEMV（`gemv_bf16`）接管 cuBLASLt

日期：2026-11-07　基线 HEAD：221467f　对照：vLLM（bf16 状态、无 W8A8）

## 1. 归因（nsys `--cuda-graph-trace=node`，c=1 图档，515 步中位）

图档单步 11.69ms 内 kernel busy 11.42 + gap 0.43（util 96.3%）。逐 kernel
（每步发数 / GPU 时间 / 单价）：

| kernel | 发/步 | ms/步 | µs/发 | 说明 |
|---|---|---|---|---|
| `cutlass Kernel2` ×3 档 | 184 | 8.592 | 43.5 | gate/up、down、qkv+o |
| `gemvx`（lm_head） | 1 | 1.406 | 1406 | 904 GB/s（好） |
| `gemvx`（k/v + a/b） | 64 | 0.306 | 4.8 | 92MB 只跑 301 GB/s |
| `splitKreduce_kernel` | 96 | 0.117 | 1.2 | **纯寄生发**（split-K 归约） |
| `add_norm_zc` | 33 | 0.242 | 7.33 | 1 block×256 线程 |
| `rms_norm_zc` | 48 | 0.197 | 4.11 | 32 层 + q/k norm 16 |
| `deltanet` | 24 | 0.154 | 6.43 | |
| 其余 12 类 | ~180 | 0.42 | 1-11 | conv/l2norm/rope/attn… |
| **合计** | **677** | **11.42** | | gap 0.43 |

账本 vs vLLM（10.31ms）：gemm 10.30 vs 9.90（+0.40）、非 gemm 1.11 vs 0.53
（+0.58）、gap 0.43 vs 0.21（+0.22）。**gemm 里可拆的两块病灶**：① 96 发
split-K 归约纯寄生；② k/v+a/b 这 64 发走 `gemvx` 只搬 92MB 却花 0.306ms。

## 2. 刀法：m=1 全部走自写 GEMV

`gemv_bf16`（warp-per-row，block=128=4 warp，每 lane 每轮 16B 两条 u64，
k 序 256 元素跨度）+ `gemv_bf16_row`（block-per-row，n≤256 时块数不足才用）。
接线点：`impl GemmOp for GemmLt::run_wt` 前置分派 `m == 1 && k % 8 == 0`，
门控 `WEIN_GEMV`（默认开，`=0` 回退）。m>1（满载/预填）路径**一字未动**。

- 算术：`warp_sum_f32` = 5 级 `shfl_down`，与 cuBLASLt 的 m=1 行内 k 升序
  累加**同序** ⇒ 实测 **rel2 = 0.000e0，位级同**（6 个真形状 n=32/1024/
  2560/4096/8192/9216）。这比容差档更强，是本刀敢默认开的基础。
- 发数：249 gemm（184 cutlass + 65 gemvx）+ 96 splitKreduce = **441 发 →
  247 发 gemv，split-K 归约 96 发归零**。整步 677 → 580 发。

## 3. 读数

**c=1 短上下文（图档，两轮交错）**

| | ITL p50 | ITL p99 | tps |
|---|---|---|---|
| 基线 221467f | 11.618 / 11.594 | 13.50 / 13.57 | 84.5 / 84.8 |
| gemv 默认开 | **11.143 / 11.134** | 13.51 / 14.31 | **87.2 / 87.0** |

配对差 **−0.46ms（−4.0%）**，两轮一致（−4.09% / −3.97%）。相对 vLLM
10.31 的差距 11.4% → **7.9%**。TTFT 不动（11.6→11.1 只走 decode 路径）。

**nsys 复核（`WEIN_GEMV=1`，515 步中位）**：busy 11.42 → **10.97**，
launches 677 → **580**，gap 0.43 → **0.39**；`splitKreduce_kernel` 绝迹；
gemm 段 10.30 → **10.10**（844 GB/s）。非 gemm 段 1.11 → 1.02。

## 4. 剩余差距的账（本刀之后）

| 项 | wein | vLLM | 差 |
|---|---|---|---|
| gemm（8.41GB） | 10.10（844 GB/s） | 9.90（874 GB/s） | +0.20 |
| 非 gemm kernel | 1.02（333 发） | ~0.53 | +0.49 |
| kernel 间 gap | 0.39（580 发×0.67µs） | ~0.21 | +0.18 |

非 gemm 段最贵的三条都不是"发数问题"而是**单价**：`add_norm_zc` 7.33µs 与
`rms_norm_zc` 4.11µs 是 1 block×256 线程的纯延迟（`block_sum_strided` 里
thread 0 的 256 项串行折叠 ≈ 1.1µs，加上两次 `_syncthreads` 与冷读），
`deltanet` 6.43µs。**动这三条要重开位级合同**（o16 golden 的归约树形状），
是下一步的取舍点，不在本刀射程内。

## 5. 验收

- `o40_gemv_tolerance`（新增）：双档几何 × k 齐尾/非齐尾（2056）vs f64
  reference 容差 + 与 cuBLASLt 同题面互拍 rel2 ≤ 5e-3（实测 0）。
- ops_i1 15/15 绿（o16/o20/g7/o35/o36/o39 位级与容差档全过）。
- 端到端贪心：**同配置跨进程本身就不确定**（块池/选型差异，实测 g0 两跑
  文本即不同），故文本差异不作判据；数值判据用上面的位级同。
