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

## 三、正确性：**判定为错误**（v9 假设批级单一 block_table）

用仓库自带的 `reference::paged_attn`（支持多 seq，o40 同款）重写探针后，192/960 行
双 span 场景的分组比较给出决定性结果（bpr=32、spanA 512 行、spanB 448 行）：

| 路径 | spanA rel2 | spanB rel2 |
|---|---|---|
| ① 整批 v9 | **1.16e-3（容差内）** | **1.86e-1（错）** |
| ③ 整批 online vs reference | 2.13e-5（位级同档） | 同 |

⇒ **v9 只读批级 block_table 的前 `bpr` 项**，即它假设整个批共享同一张页表
（单 span）。第二个及以后的 span 全部用了第一个 span 的 KV ⇒ 结果错误。

而 agent 场景每一步都是多请求各自 block_table ⇒ **`WEIN_ATTN_V5_MIX=1` 在 agent
场景下算错**（只有批内第一个 span 正确）。第二节的 2.1× 性能提升是"算错得快"，
**不可采用**。`!has_decode_row && single_span` 闸门是**结构性要求，不是保守**。

（注：② 分两次 narrow 调用仍报整体 0.142，说明 narrow 逐段方案还需单独定位——
见 §5 第一步。）

## 三·补、过程留痕（此前一版的判断已被本结果取代）

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

1. **开关默认关**，生产行为零变化（MIX=0 = 现基线）。开关名保留 `WEIN_ATTN_V5_MIX`
   但在文档/报告里标注「**仅用于性能实验，数值错误**」。
2. 正解是 §5 的原方案：**按 span 分组逐段调用**（`DevBuf::narrow` 给每段自己的
   q/bt/seqs/pos/out 视图 ⇒ 每段都是"批级单一 block_table + 单 span"，满足 v9 合同）。
   第一步先让探针的 ② 路径在**分组**口径下对上 reference（当前整体 0.142 说明
   narrow 视图还有一处错，需定位：可能 spanB 的 bt 视图或 pos 基准）。
3. 已知的边际收益估算不变：attn 11ms/步 → 期望 ~1.5ms、ITL 31.5 → ~22ms。

## 五、回归

`cargo build --release --workspace` 绿；exec 41 项绿（MIX=1 下）；GPU 无残留。
默认配置基线不变：c=16 命中 41.6% / TTFT 1865ms / turns/s 2.61 / ITL 38.2ms。

## 六、根因细化：v9 只用 block_table 的**第 0 行**（四个交叉实验）

上一版 §3 的"只读前 bpr 项"精确化为"**只读第 0 行**"：

| 实验 | 构造 | 结果 |
|---|---|---|
| 基线 | spanA(seq0, 页0..31) 在前 / spanB(seq1, 页128..159) 在后 | spanA 对 1.16e-3、spanB 错 1.86e-1 |
| 交换顺序 | spanB 在前 / spanA 在后 | **spanB 仍错、spanA 仍对** ⇒ 与先后无关 |
| spanB 页表倒序 | 31..0 | spanB 仍错（1.92e-1） |
| **spanA 页表也倒序** | 31..0 | **spanA 仍对（1.160e-3）** |

第四条是关键：**页号值是被正确读取的**（倒序也对）⇒ **不存在"忽略 bt 值"的生产 bug**。
四条合起来只在一种解释下自洽：**v9 用 bt 的第 0 行做全批的页表**，与 `seq_ids` 无关。
spanA 恰好是行 0 的页表持有者，所以永远对。

⇒ 生产影响：6009 等单请求场景**无 bug**（单 seq，行 0 正确）；只有"多请求共享一个
混批"时会错。这正是 `single_span` 闸门要挡的东西，但真实原因不是"多 span 不好",
而是 v9 的页表寻址忽略了 `seq_ids`。

⚠ 待查：kernel 里 `slot_of(bt, s, j, bpr, bs)` 明明带 `s`（`seq_ids[t]`），
说明实现与注释不一致。下一步该看 v9 内部传给 `slot_of` 的 `t` 是**块内局部行号**
还是全局行号——若是局部，那 `seq_ids[t]` 读到的就不是本行的 seq。这也是
② narrow 逐段调用仍然错（整体 0.142）的可能原因（每段视图的行 0 未落到 `seq_ids[0]`）。
