# c=1 norm 刀：块归约换 warp 树 + add_norm 一趟融合

日期：2026-11-07　前置 HEAD：04853fb（m=1 直路 GEMV）　对照：vLLM（bf16 状态、无 W8A8）

## 1. 动机（上一刀的账）

c=1 图档单步 11.10ms 里非 gemm 0.87ms（vLLM ~0.53），其中 norm 两项最贵且都是
**1 个 block × 256 线程的纯延迟**：

| kernel | 发/步 | µs/发 |
|---|---|---|
| `add_norm_zc` | 33 | 7.33 |
| `rms_norm_zc` | 48 | 4.11 |

假设：`block_sum_strided` 里 thread 0 顺序折叠 256 个 smem 部分和（`total = s[u] + total`，
4 路展开）≈ 1.1µs 串行链，是这两条的主税；`add_norm_zc` 还有一趟对 `sum_out` 的
回读（写完再读回来累加平方）。

## 2. 刀法（位级合同改成"树形，但两侧仍逐位同"）

- **`block_sum_merge`**：per-thread 部分和（跨步 i 升序，累加序一字未动）→ warp 内
  5 级 `shfl.down` → 各 warp 首槽落块内 smem → **一次** syncthreads → **每个 warp
  各自再 5 级合并**（免第二道屏障与广播槽）→ `shfl.idx` 广播 lane 0 的总和。
  窄块（bdim<32，无整 warp）保留原串行折叠 + 补零到 256。
- **`add_norm_zc`**：写 `sum_out` 的同时把**已收 bf16 的值**平方累加（同跨步、同 i
  升序 ⇒ 与回读 `sum_out` 再累加逐位同），省掉一整趟全局回读。
- **`reference.rs`**：`rms_block_sum` 的 merge 段镜像同一棵树（`merge_parts`）——
  所以 **o16 仍是位级档，没有降容差**（用户批准的"放宽"最终没用上）。

踩到的坑（值得记）：down 树只有 lane 0 拿到总和，我第一版忘了广播，于是**除 lane 0
外每个线程的 inv 都是错的部分和**——o16 报 `3x65[1]`: 0x3fcd vs 0x3f98。窄块档当年
靠 `*sp`（sp[0]）顺带广播，换树时这层语义丢了。修法 = `shfl_idx_f32(u, 0, 31)` 一行。

## 3. 读数（nsys `--cuda-graph-trace=node`，c=1 图档，逐 kernel 中位）

| kernel | 改前 µs/发 | 改后 µs/发 | 段 ms/步 |
|---|---|---|---|
| `add_norm_zc` ×33 | 7.33 | **6.31** | 0.242 → 0.208 |
| `rms_norm_zc` ×48 | 4.11 | **3.74** | 0.197 → 0.179 |

端到端 c=1：ITL p50 11.143/11.134 → **11.131/11.104（−0.02ms，−0.2%）**。

**结论：假设被证伪**——256 项串行折叠的真实代价只有 ~1µs（不是 4µs 里的 1.1µs 之外
还叠了别的），而 `add_norm_zc` 的一趟融合也只值 0.3µs。两处合计 0.05ms/步，在 0.79ms
的总差距里只占 6%。**norm 不是 c=1 的杠杆**，这条道到此为止。

## 4. c=1 差距的最新账（本刀后，11.10 vs vLLM 10.31 = +7.7%）

| 项 | wein | vLLM | 差 |
|---|---|---|---|
| gemm（8.41GB，844 GB/s） | 9.95 | 9.90（874 GB/s） | **+0.05** |
| 非 gemm kernel（333 发） | 0.87 | ~0.53 | **+0.34** |
| kernel 间 gap（580 发 × 0.79µs） | 0.67 | 0.21（~100 发） | **+0.46** |

即：**gemm 已经打平**，剩下的 0.80ms 里 58% 是"我们的发数是人家 5.8 倍"的必然产物。
把发数从 580 压到 ~350 才拿得到那 0.46ms 的一半——这需要跨层合并（例如 24 层线性
注意力层的 5-6 发/层压到 2-3 发），是结构性工程，不是单 kernel 磨刀。

## 5. 验收

- `nrm_probe`（新增）：8 组 (rows, hidden) × {全 1 整数和, 随机向量} 逐一与
  reference 位对齐 = **0 越界**；同时给出 block-per-row 单发单价。
- `o16_norms_golden_bitwise` 仍位级绿（kernels↔reference 同树）；`o36_fused_ops_bitwise`
  （add_norm_zc ≡ add→rms_norm_zc）绿；ops_i1 **15/15**。
- 端到端数值：norm 输出与旧实现差 ≤1 ulp（树换形），链路级容差与 oracle 复验照旧。
