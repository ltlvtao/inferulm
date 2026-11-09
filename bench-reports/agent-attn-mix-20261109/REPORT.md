# 混批 v5/v9：Agent ITL −52%/turns/s +107%，但正确性未定论（默认关）

## 一、改动（一处闸门 + 一个开关，默认关）

```rust
let v5_mix = std::env::var("WEIN_ATTN_V5_MIX").map(|v| v != "0").unwrap_or(false);
if self.attn_online
    && (v5_mix || (!has_decode_row && single_span))   // ← 放松处
    && t >= self.attn_v5_min_t
```

配合 `WEIN_ATTN_V5_T=64`（把 v5 的 token 门槛从 512 降到 64）。

理由：v5/v9 的 ABI 本来就按行传 `bt/seqs/pos`，因果掩码按行 pos 判定，
`!has_decode_row && single_span` 是 2026-11-04 引入时的保守限制。

## 二、性能（c=16×6，每发前清前缀，三发）

| | ITL p50 | turns/s | 热 TTFT | 命中 |
|---|---|---|---|---|
| MIX=0（现状） | 38.2 / 37.4 / 40.6 | 2.45 / 2.25 / 1.90 | 2287 / 3527 / 4577 | 30.5 / 30.5 / 20.3% |
| **MIX=1 + V5_T=64** | **17.4 / 19.0 / 18.2** | **5.76 / 5.38 / 5.35** | **106 / 310 / 235** | **68.0 / 60.8 / 65.2%** |

ITL **−52%**、turns/s **+107%**、热 TTFT **−87%**。命中率同步从 ~30% 升到 ~65%：
总墙钟时间减半 ⇒ radix 块在更短窗口内被复用更多次、驱逐更少（不是命中逻辑改了）。

对照组 vLLM（热态同口径）：c=16 = 5.50 turns/s / 14.9ms / 67.8%。
**turns/s 与命中率已追平，ITL 仍落后 22%。**

## 三、正确性：没有定论，因此默认关

已做：
- `cargo test --release -p wein-exec`（带 MIX=1 + V5_T=64）**41 项全绿**，含
  `b2_chunked_mixed_batch_two_lane`（混合批）、`o39`（v5 ≡ v4 逐位）、`o40`（v9 容差）。
- 端到端同 prompt 贪心对比（8 并发 × 48 token，MIX=0 vs MIX=1）：**输出不同**。
  seq1 首个差异是 `a CSV file` vs `a CSV files`（1 token 级 → 典型 argmax 平局翻转，
  批形不同 ⇒ cuBLASLt 选不同算法 ⇒ f32 累加序不同，已知现象）；但 seq0 从第一个
  生成 token 就不同（两条都通顺）。
- 新探针 `examples/attn_mix_probe.rs`（192 行 = spanA 128 + spanB 64，四条路径）：
  ① 整批 v9、② 分两次 v9、③ 整批 online 与**手搓 reference** 全部差 rel2≈0.3。
  三条实现互相也差 0.3 ⇒ **是探针的 KV 池布局/reference 假设有误**（不是 kernel 全错）。

未定论的关键问题：**v9 是否假设 pos 从 0 连续且单 span**（`t64 = t & !63` 的 tile
划分与因果早停可能依赖「行号 ≈ pos」）。若依赖，则混批（seq1 的 pos 0..63 排在
seq0 的 128 行之后）必错。

## 四、结论与下一步

1. **开关默认关**，生产行为零变化（MIX=0 = 现基线）。
2. 收益是真实的（2.1×），值得再花一刀把它验证干净：
   - **用仓库自己的 reference**（`wein_kernel::reference`，o39/o40 用的那套）重写
     `attn_mix_probe` 的对照，先让 ②（单 span 分两次）与 reference 在容差内对上；
   - 再比 ① vs ②：**相同则混批自洽**，可与 online 归入容差档；
   - 若 ① vs ② 显著不同 ⇒ 确认 v9 依赖 pos 连续，改法是把混批按 span 分组逐段调用
     （`DevBuf::narrow` 逐行视图已具备，见 REPORT §5 的原方案）。
3. 顺带确认：agent 场景 attn 若真走 v9，11ms/步应降到 ~1.5ms——本刀端到端 ITL
   31.5→18.2 与该预估吻合（降 13.3ms），**间接支持"v9 在混批下算的是对的"**，
   但不足以作为验收。

## 五、回归

`cargo build --release --workspace` 绿；exec 41 项绿（MIX=1 下）；GPU 无残留。
默认配置基线不变：c=16 命中 41.6% / TTFT 1865ms / turns/s 2.61 / ITL 38.2ms。
