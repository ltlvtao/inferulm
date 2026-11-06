# delta 段 65.7 ms vs vLLM 22.7 ms：43 ms 差距的逐项归属

日期：2026-11-06｜wein HEAD：41bb93b（h 诊断刀）｜vLLM 0.32.0 数据：`bench-reports/vs-vllm-kernels-20261102`

口径：**6009 token 冷 prefill，24 个 linear_attention 层，两边都按 chunk 2048**（vLLM
`--max-num-batched-tokens 2048` = 3 个 pass；wein `--chunk-size 2048` = 3 个 pass）。
vLLM 侧是 torch profiler 的 kernel 级实测（GDN 各 kernel 均 72 次发射 = 24 层 × 3 pass，
除以 24 即 µs/层）；wein 侧是 `dn_chunk_probe` 单 span（94 个 BT=64 chunk）+ GPU 段表。

## 1 总量分解（µs/层；括号 = 该侧对应的 kernel）

| 功能 | wein | vLLM | 倍数 | 差额占 43.6 ms 的比 |
|---|---|---|---|---|
| q/k 归一（l2norm） | 85（`dn_qk_l2norm`）| —（并入 norm 桶的 `_fused_qk_rmsnorm_rope_gate`）| — | 2.0 ms（5%）|
| prep：cumsum + A(K Kᵀ) + (I+A)⁻¹ + w/u | **667**（`dn_chunk_prep`）| 345（`chunk_scaled_dot_kkt` 49 + `merge_16x16_to_64x64_inverse` 177 + `recompute_w_u` 79 + cutlass 64×64 40）| 1.9× | 7.7 ms（18%）|
| h：w·S₀ᵀ + δ″ + 状态递推 + 快照 | **1341**（`dn_chunk_h`）| 220（`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`）| **6.1×** | **26.9 ms（62%）** |
| o | **646**（`dn_chunk_o`）| 305（`chunk_fwd_kernel_o`）| 2.1× | 8.2 ms（19%）|
| 合计 | **2739**（= 65.7 ms）| 870（桶计 22.68 ms，差值为未进 top-20 的 cumsum/zeros）| 3.1× | 43.6 ms |

⇒ **h 一个 kernel 就是差距的六成**；prep 与 o 各占约 1/5；l2norm 几乎可以忽略。

## 2 h 的加性分解（本轮实测）

方法：PTX 热替换 + 交错 3 轮，只取 `D2D 带宽标定 ≥ 900 GB/s` 的读数（<900 的发一律作废），
各变体只改一处、其余代码完全一致（sp=4/bdim=256 生产几何）。

| 变体 | 改动 | h µs/层 | 归属 |
|---|---|---|---|
| HEAD | — | 1406 / 1435 / 1441（取 1420）| 基线 |
| no-② | ② 的 tick 循环不执行（fma 链 + k̂/δ″/g 读 + acc 迭代全删）| **730**（@906）| **② = 710** |
| no-snap | 尾段不写快照（96 MB/层）| 1337（@909）| 快照 = 95 |
| no-uload | elementwise 不读 u（改用常数；仍写回 u）| 1345（@913）| u 读 = 90 |
| no-① | ① 的 mma 循环不执行 | ~1270（读数带外，估）| ① ≈ 130 |
| — | 残差 | **≈ 405** | s16 全局写 ~107 + δ″ 写回 u ~109 + gsc/expf/屏障/循环 ~190 |

### 关键发现：h 有 384 MB/层 的**纯设计浪费**

`s16`（bf16 状态镜像）与 `xsc`（① 的 w·S₀ᵀ 中间量）都落在**全局 scratch 的块私有槽**里
（`lib.rs:6426` `let sc = scratch.add(bid * (vrows*kd/2 + bts + bts*vrows))` —— `scratch` 是
kernel 参数的全局缓冲，不是 smem）：

- `xsc`：① 的 `wmma_store_c` 写 8 KB/块/chunk → elementwise 读回 ⇒ **96 MB 写 + 96 MB 读/层**；
- `s16`：② 写完 → 下一个 chunk 的 ① 当 B 片段读 ⇒ **96 MB 写 + ~96 MB 读/层**（读有 4× L1 复用）。

而 **vLLM 这两个中间量都不落 global**：`b_h` 驻寄存器跨 chunk 递推，`v_new = u − w@S₀` 在
kernel 内一次算完直接产出，没有任何"mma 结果 → global → 再读回"的往返。

### 当前设计的流量地板

| 项 | MB/层 | 说明 |
|---|---|---|
| u RMW（δ″ 读 + 写回）| 197 | 写回是 o kernel 需要的，读是必需的 |
| xsc 往返 | 192 | **可消除**（①+elementwise 融合即可）|
| s16 往返 | 192 | **可消除**（搬进 smem 或保持在 accumulator 片段）|
| 快照写 | 96 | 前缀缓存边界，可稀疏化 |
| w hi/lo | 96 | 同 vh 的 4 个 sp 块经 L2 共享 |
| k̂ | 24 | |
| 合计 | **≈ 797** | @900 GB/s ⇒ **885 µs 地板** |

⇒ **885（地板）+ 710（②未被隐藏的标量链）≈ 1595 ≈ 实测 1420-1441**。模型闭合：
h 的每一微秒都有归属，没有"莫名开销"。

## 3 为什么 vLLM 是 220 µs/层

同为 BT=64 的 chunked 递推、同样的数学量，差在三点：

1. **rank-1 更新用 `tl.dot`（tensor core）**：`b_h += kᵀ @ (v_new·g)` 是 [64×128]×[128×64]，
   走 mma；我们是 1024 标量 fma/线程/chunk（262k fma/块/chunk）。这一项就是我们的 ② = 710 µs。
2. **中间量零往返**：`w@S₀` 与 `v_new` 在 kernel 内融合，状态 `b_h` 在寄存器/共享里跨 chunk
   递推；我们每 chunk 两趟 global 往返（xsc + s16）= 384 MB/层。
3. **工作集贴 L2**：2048-token pass 下他们的 h 缓冲 ≈64 MB ≈ L2 72 MB，逐 chunk 写后立刻被
   `chunk_fwd_kernel_o` 读，HBM 账只有 u/v_new/w/k 一线；我们的 scratch（128 块 × 16.3 KB ×
   94 chunk 视窗）利用率没这么高。

## 4 两个假设的实测否决（同日，交错 A/B，只取 D2D≥900 读数）

| 假设 | 做法 | 结果 |
|---|---|---|
| `xsc` 的全局往返是 192 MB/层 HBM 流量 | ① 的 `wmma_store_c` 改落 8 KB 静态 smem（`DNH_XSC`，30976B 静态 + 69632 动态 = 100608 ≤ 101376，占用不变）| head 1335-1519 / xsm 1458-1468 ⇒ **平价偏劣，回退**。原因：8 KB/块/chunk 的工作集（1 MB/波）在 L2，往返只吃 L2 带宽（~5 TB/s），不是 HBM |
| ② 的 710 µs 来自逐 tick 的 δ″ global 载入（~200cy，4 深预取盖不住）| elementwise 顺路把 δ″ 落 8 KB smem 镜像，② 改读 smem（同一 f32 值，位样不变）| head 1335-1517 / dsm 1343-1483 ⇒ **平价，回退**。⇒ 卡的不是 δ″，**k̂ 的逐 tick 载入（8B/lane，4 KB 步长，L2 ~250cy，仅 4 深）才是 ② 的延迟源** |

两个"流量/延迟"改造都不动时间 ⇒ ② 的 710 µs 是"200cy/tick × 64 tick"的**串行
载入-消费链**，而加深预取没有寄存器空间（255 已满）、提占用会 spill。**结论：② 只能
换形态**——要么把 k̂ 片搬进 smem（需腾 16 KB：w 双槽降单槽），要么直接上 tensor core。

## 5 结论与修缮顺序

**差距不是"算法不同"或"我们搬的字节更多"这类不可动的东西**，而是两个可消除的结构：

| 修缮 | 预估 h | 依据 |
|---|---|---|
| ~~①+elementwise 融合杀 xsc~~ | ~~−214 µs~~ | **已证伪**（见 §4：L2 承载，实测平价）|
| ~~δ″ 落 smem 供 ② 读~~ | ~~−200 µs~~ | **已证伪**（见 §4）|
| **k̂ 片搬 smem 供 ② 逐 tick 读**（✅ 已落地，见 §6）| **−57~−128 µs**（实测，非预估）| §4 的推论方向正确但幅度只有预估的 1/3 |
| **② 上 tensor core**（f32 accumulator 片段跨 chunk 不回写）| −560~−610 µs（710 → 100-150）| 终极形态，改动大 |
| 快照按 `snap_stride` 稀疏化（bts=64、stride=256 ⇒ 4 chunk 一次）| −70 µs | 实测 95 µs |

同时 o（2.1×）与 prep（1.9×）也在同一类形态问题上：o 每 chunk 重读快照 + 8 warp/SM 打不满；
prep 的 Gram 是标量 fma（该上 mma），而 (I+A)⁻¹ 我们用前代（vLLM 用 `merge_16x16_to_64x64_inverse`
花 177 µs/层，这项我们可能本来就更快）。

**总预估**：三件（h 融合 + ②tensor core + o/prep 同形态）落地后 delta 段 65.7 → ~30-35 ms，
对 vLLM 22.7 的差距从 2.9× 压到 1.3-1.5×。

## 6 k̂ 片 smem staging 落地（本刀唯一有效改动）

形态：② 的 4 深预取流水整段抽成 `dn_h2pipe!`（参数化 `nh/krow/kstep/urow/gs`，流水形状与
fma 序一字未动），chunk 头顶把本块 k̂ 的两半（token 0..32 / 32..64）**协作预载**进静态
smem：前半 → 新增 `DN_H2_S`（`[u64;1024]` = 8 KB，行距 32），后半 → `DN_H2_K` 高 32 行
（行距 36；该数组属 staged 路，`staged ⇔ bdim=128` 与 `ksm ⇔ bdim=256` 互斥，零新增 smem）。
载入与 ① mma + elementwise 全程并发飞行，② 前既有的 `_syncthreads` 即担保可见 ⇒ **② 内
零新增屏障、零暴露拷贝**。档门 `ksm = VW==4 && bdim==256 && kd==128 && bts==64 && vrows==32`，
其余几何回退全局 k̂。

资源：regs 255 / smem 22784→**30976** / spill 8→4 B（+8192 后 30976+69632 = 100608 ≤ 101376）。

实测（交错 A/B，sp=4/bdim=256，µs/层）：HEAD 有效中位 **1386**（1364.7/1408.1；第三发 1502 @760 作废）
→ new **1329**（1347.8/1328.7/1321.9，三发都更快）；子代理的 12 轮配对里 new **12/12 更快**，
有效中位 1431 → 1303（−128 µs，−9%），反向顺序复验一致。
op 级整段（l2+prep+h+o，BT=64/vs=4）：**2930 → 2806 µs/层 = 70.3 → 67.4 ms/24 层（−4.2%）**。

验收：`ops_i1` 13/13（含 **o20 位级**、**o35 容差八 case 数值与 HEAD 逐字相同**）、
`acceptance` 11/11、`qwen` 23/23、`qwen_bigchunk` 2/2、`qwen_graph` 4/4（并行跑时 qg4 会
因跨进程 GPU 争用假红，单跑绿）。

**为什么只有 −4~9% 而不是预估的 −300~400 µs**：k̂ 每 tick 只有 256 B/块（32 lane × 8 B），
4 深预取本来就能盖住其中的一部分；换 smem 之后剩下的 580 µs 不再由 k̂ 载入主导，回到
"64 tick × 每 tick 的 ALU 发射 + 累加器依赖"本身。⇒ **② 再往下只能换形态（tensor core）**。
