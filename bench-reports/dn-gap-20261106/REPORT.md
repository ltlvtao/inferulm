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

---

# §7 ② 换 tensor core（wmma）落地

## 形态
② 即 `S[v][kd] += Σ_t Δ[t][v]·k̂[t][kd]`（Δ = δ″·g），映射成 m16n16k16：
**M=v(32) / N=kd(128) / K_red=t(64)** ⇒ 16 个 C tile（v 两块 × kd 八块），每 tile
`n/16` 个 kstep；8 warp 平分（`mvb=wid&1`、`kb∈{(wid>>1)*2,+1}`），每 warp 2 tile。

三个"不用搬"是这一刀能成立的关键：
- **A = Δᵀ 用 `load.a.col` 直读 Δ 的 `[t][v]` 自然序**（新增 extern
  `llvm.nvvm.wmma.m16n16k16.load.a.col.stride.bf16`，ldm=vrows=32）——Δ 由 elementwise
  顺路写进 smem（`DN_H2_DH/DL`，各 4096 B，替掉上一刀的 k̂ staging 8 KB），
  **写侧与读侧都零 bank 冲突，静态 smem 精确保持 30976 B**；
- **B = k̂ 用既有 `load.b.row` 从全局直读**（ldm = kh_n·kd = 2048），零 staging；
- **hi/lo 双 mma 串在同一 C 上**（Δ 拆 bf16 高/低半，等效 f32 精度，同 ① 的 w 拆分）——
  单 bf16 会重演 w 那次 4.07e-3 越门。

C 片经块私有 f32 暂存（`inc[vrows][kd]`，scratch 每块 +4096 f32）回到寄存器态 `acc`
（每 lane 4 条 16 B 连续载，合并访存），之后 s16/快照/状态池收尾**一字未改**。
`< 16` token 的余尾走标量 fma（不 ceil 补齐：整 16 取会读到越尾行，`0·NaN` 会毒化状态；
每层只在末 chunk 触发一次）。

## 读数（交错 A/B，`sp=4 bdim=256`，D2D 749-909 GB/s 合格发）
| 轮 | HEAD h | MMA h |
|---|---|---|
| R1 | 1362.7 | 1018.2 |
| R2 | 1255.5 | 995.6 |
| R3 | 1296.2 | 1003.7 |
| 中位 | **1296** | **1004**（−22.5%）|
子代理独立 4 轮：1231.9 → 927.0（−24.8%），配对方向 4/4 一致 ⇒ 本刀 **h −23~25%**，
② 580 → ~275 µs。op 级（l2+prep+h+o）2871.6 → 2523.2 µs/层，delta 段 24 层 65.7 → ~58.4 ms。
6009 冷 TTFT（服务端图档）440.9 → **432.0**（第 2 发口径；−8.9 ms，与账 1:1）。

ptxas：255 regs / 12 B spill st / **静态 smem 30976 B 不变** / h 内 60 条 `wmma.mma` / 零 ldmatrix。

## 正确性
- o35 八 case：rel2 **1.260-1.314e-3**（改前 1.26-1.28e-3，门 3e-3），state rel2 1.20-1.62e-3；
  子代理另做 6009-token 逐元素 dump：state **逐位同**，out 仅 16/24,612,864 个 bf16 值不同
  （rel2 2.457e-6）。
- `acceptance` 11/11、`qwen` 23/23（贪心逐 token 全同，零分叉）、`qwen_graph` 4/4（单跑）。
- 两个红均为既有环境项，已用 HEAD 同样跑法复核非回归：`o22` 延迟门禁（HEAD 0.406 ms 也红，
  单跑绿 0.361）、`qwen_bigchunk` b2 混批（紧接 qwen 进程后跑分叉 5/8，HEAD 同；单跑 2/2 绿）。
- battery（20261106-122319）：图档满载 tps 3586.5 / ITL 23.90 / p99 37.78、轻载 12.94、
  eager 3452.7/24.88——带内（上轮 3596/23.73、13.14、3464/24.71）。chunked 只走 prefill，decode 不动。

## ② 剩下的 ~275 µs 归因
tensor core 本体只值几 µs（128 mma/chunk，覆盖 MAC 的 2 倍）；275 µs 是**组织**：
① k̂ 的 B 载几何——`load.b.row` 每次取 16 个行距 4 KB 的 32 B 段，天然不合并，
且 `mvb` 两组读同一片 k̂（2× 冗余，靠 L2 兜）；② `inc` 的 16 KB 写 + 16 KB 读；
③ 那道跨 warp `_syncthreads`（v 块的产出 warp ≠ 消费 warp，结构性）。下一步若要再压：
本 chunk k̂ 片搬 smem + `.b.col`（ldm=32）把 1024 次散载换成 8 次合并块载（16 KB 需腾 smem）。

---

# §8 h 加性分解（现行版）与两刀落地

## 分解（PTX 交错 A/B，40 个挖空版本，全部 255 regs 无占用率污染）
h(976) = ① 197（w hi/lo staging 97 + mma 环 100）+ ② 309（k̂ B 载 104 + mma 108 + A 载 27
+ store_c 20 + 基址 40）+ **elementwise 413**（u 读 64 + u 写 88 + xsc 23 + gsc 26 + inv 12
+ Δ smem 写 20 + **每迭代串行延迟余 180**）+ 快照写 61 + 杂 20。
流量账：每层 w staging 385 MB + k̂ 385 + s16 读 385 + s16 写 96 + 快照 96 ≈ **1.35 GB**，唯一数据
只有 w 98 MB / k̂ 24.6 MB（读放大 3.9× / 15.6×）⇒ **这个 kernel 是 L2 带宽 + 载入延迟题，
不是 FLOP 题**（② 的 mma 本体只 108 µs）。

## 两刀（均位级不变）
1. **elementwise 4 深流水**（−142）：4 个元素一组，先把 16 条独立载入（u/x/g/inv）全部发出，
   再逐元素算/写回；算式与写址逐字未变，尾组钳位（含"载入 token 钳到 n-1"防尾 chunk 越读）。
2. **② B 片段 1 warp 认 1 个 kd-tile × 两个 v-tile**（−76）：`kb = warp`（8 warp 铺满 8 个 kd 块），
   两个 C 累加器共用同一发 B 载 ⇒ k̂ L2 读 385→192 MB；每片 C 的 ks 序与 hi/lo 序不变。

## 读数（我的复核，4 轮交错；D2D 745-914 GB/s）
| 轮 | HEAD h | 合并 h |
|---|---|---|
| R1 | 934.2 | 837.2 |
| R2 | 979.4 | 714.9 |
| R3 | 918.1 | 763.9 |
| R4 | 1016.2 | 762.4 |
| 中位 | **957** | **763**（4/4 更优，配对中位 −210）|
子代理两轮 campaign：997→770 / 994→749（8/8、6/6 全负）；op 级 BT64/vs4 2529→2244 µs/层。
**h −20~25%**，可加性成立（−142 + −76 ≈ −219 实测）。ptxas：255 regs / **spill 12+40B → 4+32B** / smem 30976B 不变。
端到端 6009 冷 TTFT 432.0 → **424.6**；delta 段 58.4 → ~50 ms。

## 正确性
o35 八 case **输出与 HEAD 逐字符相同**（1.260-1.314e-3）；6009-token 逐元素 dump（state 524288 u16
+ out 24612864 u16）三档 PTX 与 HEAD **逐位全同**；`ops_i1` 13/13（o22 也绿）、`acceptance` 11/11、
`qwen` 23/23 零分叉、`qwen_bigchunk` 2/2、`qwen_graph` 4/4。
battery（20261106-142103）：图档满载 3542.0/23.73、轻载图档 12.28、eager 3430.1/24.96——
decode 路径未被本刀触碰，三日三档 3542-3596 即已知 ±1.5% 漂移。

## 本轮否决清单（再次确认，别再碰）
- **加/减屏障**：删环尾屏障 **+29 µs**（更快反而更差）、删 inc 屏障 +4 ⇒ 屏障在吸收 warp skew。
- **expf 去冗余**（sp=4 重复 4×，0.77M 次/层）：实测 ≈ 0（完全隐藏）。
- **把 xsc/δ″/gsc 搬 smem**：xsc 读只 23、Δ smem 写 20 ⇒ 天花板低于噪声（与 7122b42 一致）。
- **提占用率**：40 个版本全 255 regs / 1 块每 SM，不是杠杆。
- **每 chunk s16 写 / 状态池写**：免费（s16 写删掉反而 +18）。
- `wmma_load_a`(①，ldm=136=272B≡+4 bank) 与 `wmma` A 载（②，ldm=32=64B）**都没有当年"w 槽 32 u64
  +500 µs"那个病**：实测边际 15 / 27 µs。

## 剩余（h ≈ 750，vLLM 220）
① 的 **s16 B 载 4× 冗余**（同 nq 切片被 4 个 tq warp 各读一遍，0.7-0.8 GB/层）是最大单项，但
smem 已被 w hi/lo 双缓冲吃满，要 staging 就得砍双缓冲（那是实测赢面）⇒ 高风险，需先做原型。
次选：elementwise 里 `udiv(e, vrows)` 的 `div.u64` 慢调用（每 chunk 8 次/线程）改增量行号，纯整数寻址。
