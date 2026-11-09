# Agent ITL 31.5ms 的真因：attn 33× 退化（走错路径）+ gemm 发数翻倍（2026-11-09，诊断刀，零生产改动）

> 上一刀判定"ITL 38 vs 15ms 是 prefill 与 decode 争 GPU"。本刀用 nsys 抓 c=16 agent 场景
> （16×3，命中 24%），把每步 31.5ms 拆开。**结论与上一刀不同：GPU 不空（90.2%），
> 差距全在 attention 走了错误路径。**

## 一、先排掉"空洞"假设

稳态窗口（去掉前 2s 预热与末 1s）：

| | 值 |
|---|---|
| kernel 数 | 150,401 |
| 跨度 | 7239ms |
| GPU busy | 6528ms |
| **利用率** | **90.2%** |
| 空隙 >0.3ms 合计 | 408ms（5.6%），中位 0.58ms |

⇒ **ITL 是在算，不是在等**。主机侧没有 64ms 空洞（上一轮"wall=74ms 但段和 10ms"
是 `prof_mark` 自身 85µs/次 × 数百边界造成的插桩开销，不是真实空洞）。

## 二、每步 31.5ms 的分解（busy 6303ms ÷ ~200 步）

| 项 | 我们 | 满载 c=100 | 差 |
|---|---|---|---|
| **attn** | **11.0ms** | 0.33ms | **+10.7（33×）** |
| gemm（cutlass + 小 gemm/gemv） | 14.4ms | 11.7ms | +2.7 |
| delta（dn_chunk_prep/h/o + deltanet） | 4.7ms | 4.96ms | −0.3 |
| norm | 0.6ms | — | |
| conv | 0.4ms | — | |
| 小算子 | 0.4ms | — | |

attn 单发 **1334.8µs**，n=1645；kernel 名 = **`paged_attn_online_r4`**。

## 三、根因：v5/v9 的闸门把混合批整个排除了

```rust
if self.attn_online
    && !has_decode_row      // ← agent 的批全含 decode 行
    && single_span          // ← agent 的批是多 span
    && t >= self.attn_v5_min_t && ...
```

agent 场景每一步都是 **decode 行 + 多请求 prefill 行混批** ⇒ `has_decode_row=true`
且 `single_span=false` ⇒ 三个 v3/v4/v5/v9 分支全部跳过 ⇒ 落回
`run_tile_online_cfg`（老 tile 路径，`paged_attn_online_r4`）。

而这条老路径正是 2026-11-04 被 v5（流量 ÷4）、v9（LDSM+mma，单层 6009 2.7ms）
逐步替换掉的那条——**在单请求长 prefill 上已经快 8-22×，但在 agent 场景从未生效**。

量化：6009 单 span prefill 的 attn 是 16.5ms / 6009 token = 0.34µs/token/层。
agent 每步 176 个 prefill token（分散在 ~8 个 span）⇒ 线性外推 176 × 0.34µs × 8 层
≈ **0.5ms/步**，实测 **11ms/步**。差 22×。

## 四、次因：gemm 发数 500/步（满载 248）

稳态 gemm 类 50,000 发 ÷ 200 步 = **250 发/步**（cutlass Kernel2 26,606 发 +
小 gemm 23,343 发）。单发 55-61µs、16MB 权重 ⇒ **270GB/s**（满载 718GB/s）。

原因同上：gemm 按 span 分组调用，16 个请求各算一遍 ⇒ m≈22 的小 gemm 一串。
L2 把大部分权重读吸收了（否则 16×8.4GB/步 ≈ 149ms），但仍比满载贵 2.7ms。

## 五、下一刀的实现方案

**目标：attn 11 → ~1.5ms，ITL 31.5 → ~22ms，turns/s +43%。**

1. **attn ABI 加行偏移**（`q_off`）：让 v5/v9 能对「行子集」调用。
2. **按 span 拆分**：`forward_batch_mixed` 里把长 span（cnt ≥ 512、非链段）
   逐段调 v9/v5；把 decode 行与短 span 收集成紧凑行表走 `online`。
   需要在 host 侧做一次行 gather（q/pos/seqs/out 各 1 次小拷贝，<1MB）。
3. **位级合同按路径分别保持**：v9 已有 o40 容差档（rel2 ≤ 1.24e-3）、v5 ≡ v4
   逐位（o39）；老路径 o30 位级档不动。端到端以"贪心逐 token 相同"+32K 三轮
   位同复验。

**风险清单**（都踩过）：
- v3 曾在混合批里产生分叉（根因：j 分段依赖行位置，后用 `row_band` 修）。
  v5/v9 的分段是否同样依赖行位置，必须先读 `v5_geom`/`v9_geom` 确认，
  再做「单 span vs 子集」的位级对拍（o37/o38 扩一档）。
- 混合批的 prefill span 常 <512（agent 增量 190-1300 token）⇒ v9 的 64 行对齐
  主段可能只剩 1-3 块 ⇒ 仍需 jsplit 或 v4 兜底。

## 六、回归

工作树干净（HEAD=e783785 + 本报告），生产零改动，GPU 无残留。
现状基线不变：c=16×6 命中 41.6% / TTFT 1865ms / turns/s 2.61 / ITL 38.2ms；
c=16×3 命中 24% / ITL 31.8ms；满载 4321 tps @ 20.85ms。
