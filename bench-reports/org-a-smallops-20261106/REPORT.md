# 组织形态刀 A：小算子集群瘦身（2.21 → 1.5ms/步，满载 +1.7%、轻载 ITL −4.6%）

## 0. 结果
| 口径 | 改前 | 改后 |
|---|---|---|
| 小算子段和（m=100，eager 段表） | 2.20 ms/步 | **1.64 ms/步（−26%）** |
| nsys GPU 两项和 | 20.70 ms/步 | **20.30 ms/步（−0.42）** |
| 满载 tps（图档 c=100，交错 A/B） | 3965.4 / 3943.4 | **4037.4 / 4003.5** |
| 满载 ITL p50 / p99 | 21.82 / 31.6–32.8 | **21.27 / 31.4** |
| 轻载 c=4 图档 ITL p50 | 12.75（293 tps） | **12.12（306 tps，−4.6%）** |
| battery L3q+ 图档满载 | — | **4025.4 @ ITL 21.41 / p99 32.51** |
| 6009 冷 TTFT | 425.8 | 426.8（同档） |

## 1. 发射地图（本刀的核心证据）
在 `examples/qwen_launch_map.rs` + nsys 下把 208 发小 kernel 逐个对到调用点与流量，算「固定开销 = 实测 − 流量/900GB/s」：

| 结论 | 证据 |
|---|---|
| **argmax_rows 的 200µs 不是发射开销，是并行度不足** | 100 块喂 128 SM，单块串行扫 248KB ⇒ 184 GB/s（纯带宽应 55µs） |
| **conv 两发的固定开销算出来 ≤0** | 每发 16MB 却只花 9.6µs（L2 兜住），但「发数减半」实测仍值 −190µs |
| **`silu_mul` / `dn_split_qkv` 固定开销 ≈0（已贴带宽）** | 3.9µs 甚至小于 5.1µs 的纯带宽值 |
| **真正 4–6µs 固定开销的只有 rms/add_norm/embed 这一族，而它们的相邻配对不能合** | 见 §3 证伪 1 |
| **背靠背发射本身已经把固定开销流水化** | `add` 1.95 + `rms_norm_zc` 4.91 = 6.86µs **<** 融合版 `add_norm_zc` 8.09µs |

⇒ **"融合省发射开销"这个笼统判断在本机上大部分不成立**；真正值钱的是「并行度」与「发数减半」两类。

## 2. 三件改动

### 2.1 `argmax_rows` 切片并行（254 → 59~83µs/步，4×）
`grid = rows×8`（每块扫 vocab/8，256 线程、4 路展开求 (max, 最小下标) 落 scratch）→ 二级 `grid = rows` 按**切片升序**合并。
**位级论证**：(val, idx) 的全序 meet 是交换半格（单位元 `(−inf, 0xFFFF_FFFF)`）⇒ 任意切分/合并序结果恒同 ⇒ 与「一块扫全行」逐位同；`reference.rs` 本就全扫，无需镜像。scratch 在 open 时预分配（捕获期零分配）。

### 2.2 `conv1d_dw` + `conv_state_roll` → `conv1d_dw_roll`（464 → 253~280µs/步）
decode 档纯 decode 段（每段 1 token、无投机链）下 `k<st` 读状态槽、`k==st` 读新 token，**累加序与 fma 逐字同老 kernel**；随后原地滚动（读下标恒 > 写下标 ⇒ 原地安全）。老入口保留，判据全在 host 段表（`spans.count` 全 1 且 `spec` 全 0），prefill/混合/投机自动回退两发老路。
新用例 `o36_fused_ops_bitwise`：k=2/3/4 × 反序行 id，逐位比「融合 ≡ 两发」且各自 ≡ `reference`。

### 2.3 norm 归约 scratch → 块内 smem（**性能证伪，按显存收益保留**）
三个 norm kernel 的块归约从全局 scratch 搬进 `NRM_S[256]`（build.rs 的 smem 改写已覆盖），归约序一字未动（闲槽铺 `+0.0` 作精确单位元）。
读数：`rms_norm_zc 249→232`、`add_norm_zc 280→242`、`rms_norm_gated 191→184`，合计 **−30~−60µs（噪声内）**。
**保留理由改为显存/ABI**：删掉 3 份全局 scratch = 默认档 **50MB**、`--ctx-len 8192` 时 **805MB**，且三个 kernel 各少一个形参。性能上记为**证伪**。

## 3. 证伪清单（看似能合、实测不能）
1. **`add` + 下一层 `ln1` 复用 `add_norm_zc`**：`add 1.95 + rms 4.91 = 6.86µs < add_norm_zc 8.09µs` ⇒ 31 层强合 **+38µs/步**。
2. **`silu_mul` 与 up-branch/MLP 后 add 配对**：3.9µs < 纯带宽 5.1µs（固定开销为负），无收益；且形状不同源。
3. **`sigmul` 熔进 `paged_attn` epilogue**：attn 是数值核禁改，且 8 发只值 17µs。
4. **`dn_qk_l2norm` + `dn_split_qkv` 合一**（~70µs）：几何不同（块归约 vs 元素搬运），且属 deltanet 禁改区。
5. **`add`/norm 与 gemm 融合**：gemm 走 cuBLASLt（上一刀已否决改动）。
6. 其余（`embed` q/gate 两发、q-norm/k-norm 两发、`conv` 再吞 `dn_split_qkv`）都在 0.07–0.35% 量级，低于纪律线。

## 4. 剩余账与下一刀
小算子 1.64ms/步里：`paged_attn` 655（数值核）、`splitKreduce` 222（cuBLAS）、conv 261、add_norm 243、rms 232、rms_gated 184、wmma 小 gemm 157、silu 126、l2norm 96、split_qkv 70、add 58、argmax 60、embed 42、rope 21、sigmul 17、kv 13。
⇒ **能压的只剩 gemm 侧的组织**（同一份 `n1` 的 32 次投影批量化/合并小 N 形状），那是下一刀 B。
