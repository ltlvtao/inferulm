# 修 b2：attention 的 j 分段必须与批形无关（batch invariance）

日期：2026-11-07　前置 HEAD：02d8bae　触发：`qwen_bigchunk::b2` 红（混批 small 道 5/8 分叉）

## 1. 第一因（不是 v3）

四条 attention body（`tile_body`/`online_body`/`blk_body`/`qblk_body`）都这样定 j 分片：

```rust
let lspan = udiv(pmax + js, js);   // ceil((pmax+1)/js)，pmax = **块内**所有行的位置最大值
let j0 = sp * lspan; let j1 = min((sp+1)*lspan, pmax+1);
```

`pmax` 由**同块邻居**决定 ⇒ 同一行的 `[j0,j1)` 边界随批内其它请求的位置变 ⇒
`attn_combine` 的归约分组变 ⇒ bf16 末位变 ⇒ 贪心翻 token。

- big 道（pos≈2000）单跑与混批都是块内最大者 ⇒ 边界不变 ⇒ `alone_big == got_big` 永远成立。
- small 道（pos≈40）单跑 `lspan=3`（14 个有效分片），混批（邻居 2000）`lspan=126`（1 个有效分片）⇒ 归约序被邻居改写。

**"`WEIN_ATTN_V3=0` 就绿"是假因**：v3 在这些形状下一次没跑（门控要求 `!has_decode_row && single_span && t≥512`），`WEIN_ATTN_V3` 全文只有一个读取点。真因是同进程的 **cuBLASLt autotune**：`TUNE_M=[1,2,4,...,128]` 每个 m 桶按**计时**选 algo，跨进程会变（同一默认环境连跑 4 次：ok/FAILED/FAILED/FAILED），任何环境变量改动都会扰动计时。二者是两个独立的批形数值依赖：
1. attention 分段（本刀修掉）；
2. GEMM 的 m 桶选型（本仓口径：数值对拍须钉住算法路径 ⇒ b1/b2 在 open 前 `WEIN_TUNE_M=""` 只用 heuristic；库默认不动）。

## 2. 修法：逐行分段（`row_band`）

```rust
fn row_band(p, sp, js) -> (lo, hi) {
    let lspan = ceil((p+1)/js);          // 该行组**自己的**位置
    let (lo, hi) = (sp*lspan, min((sp+1)*lspan, p+1));
    if hi <= lo { (1,1) } else { (lo,hi) }   // 空段归一，不是早退
}
```
- 每行边界只由自己的 pos 决定 ⇒ **批形无关**（o41/o42 位级 64 seed：旧 20 位翻 → 新 0）。
- 块级循环范围取并集 `[jlo, jhi)`（`row_band` 对 pos 单调 ⇒ 极值即并集），**所有线程的屏障次数一致**（这是空段不许早退的原因）。
- 行内 clip 到 `[j0,j1)`；chunk 边界必须落在 `cap`（`ATTN_TILE_CHUNK`）的**绝对倍数**上，`jlo` 只决定"从哪一列开始填表"（若相位锚在并集下界，rescale 边界又变成批相关：实测 o41 翻 2/4096 位）。
- 空段自然落 `(m=−inf, l=0)`，`attn_combine` 按 `l>0` 跳过。
- `js=1`（大 chunk / c≥65 满载）**算术逐位不变**。

两条非显然的坑：① 相位锚错 → 又变批相关；② 把逐行 j 界提前算成块级变量 → `online_r2` 128→152 regs → `cuLaunchKernel` 701（too many resources）。最终版 `online_r2` 保持 128 regs / 0 spill / 0 stack frame。

## 3. 读数（交错配对，同机同脚本）

| 场景 | base 02d8bae | 本刀 | Δ |
|---|---|---|---|
| c=1 ITL p50（短上下文） | 10.995/11.050/10.966 | **11.004/10.924/11.028** | **+0.1%** |
| c=1 tps | 88.83/88.29/88.75 | 88.40/88.95/88.40 | −0.4% |
| 满载 m=100 wall | 22.07 ms | 21.77/21.85 | 不劣 |
| **6k 冷 ITL p50** | 11.77/11.92/12.16 | **12.45/12.46/12.53** | **+4.5%** |
| 6k 冷 TTFT | 421 | 420 | 平 |
| 3000-token probe ITL | 12.81/13.25/13.24 | 13.63/13.94/13.21 | +2~4% |

**中间站**：先按"页表容量 bpr×bs"定 lspan（批形无关，最简），但 pos≪容量时分片塌到 1
⇒ c=1 短上下文 +0.9~1.2%、6k 更差（4.05-4.23 vs 3.98-4.04）。逐行版两个口径都更好，故留逐行。
**对齐 cap 的延伸档更差**（6k 4.32-4.64）：非空分片 16→12，并行度损失大于省下的残块。

## 4. 残余代价的来源（诚实记录）

base 便宜是因为它把 chunk 相位锚在**批相关**的 `j0` 上——那个便宜本身就是本刀要修的 bug。
不变性要求段宽（6k 时 376）不是 `cap` 的倍数 ⇒ 每段至少多跨一个残块（≤cap 死列 + 一次屏障/填表/行内 max 扫），6k 是 2.5 对 2 个 chunk ⇒ +25% 的 chunk 固定开销，实测 attn 段 +12~15%、端到端 ITL +4.5%。

**结论**：接受。同 prompt 的输出不再依赖同批邻居的位置（此前 c<65 的 decode 批会）。

## 5. 验收

- `ops_m0` 22 passed（含位级 o30/o41/o42、容差 o31/o33/o37/o38/o39/o40）；`ops_i1` 15；`acceptance` 11。
- `wein-exec`：qwen 23 / bigchunk 2（V3 默认）+ 2（`WEIN_ATTN_V3=0` 逐位档）/ graph 4 / acceptance 8 全绿。
- 新增 `o41`（单遍档批形不变性，位级，64 seed）、`o42`（两遍档同）。
- `acceptance.rs` 用例计数 68→70 是 HEAD 02d8bae 的既有漂移（`rms_norm_zc`/`add_norm_zc` 两个入口加了没抬总数），本刀无新增 kernel 入口。

## 6. 遗留

- `v3_body`/`v3c_body` 仍是 pmax 分段；v3 门控 `!has_decode_row` 让它见不到混批 decode 行，纯 prefill 批里 t≥512 落 js=1 ⇒ 当前无影响，记作同类残留。
- cuBLASLt m 桶选型导致的**跨进程**位不同（非批形问题）仍在；数值对拍须钉 `WEIN_TUNE_M=""`。
