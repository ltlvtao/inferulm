# decode attention jsplit 上限刀：CAP 16→64 / TARGET 128→512

日期：2026-11-09　基线 commit：8738a0f（mix_budget 刀后）

## 1. 病灶

6k c=1 decode 47 步段表（`qwen_longctx_probe`）：attn **1.42ms/步**（177µs/层），
vs vLLM 0.40ms。gemm 8.34 / logits 1.52 已贴对手，attn 是 6k decode 差距的最大项。

真凶不是读量（rpt=4 读减半反而更慢，199GB/s），是**并行度**：
`ATTN_TILE_SPLIT_CAP=16` + `TARGET_BLOCKS=128` ⇒ 6k decode 恒 64 块 = 2 warp/SM。
读量对 jsplit 不变（每段读自己的 KV 带），块数随 jsplit 线性涨。

## 2. 扫档（attn_ab_probe，dec8 = 8k ctx 单发）

| 档 | ms | GB/s |
|---|---|---|
| CAP16/TGT128（HEAD） | 0.1416 | 474 |
| CAP64/TGT128 | 0.0548-0.0607 | 577-623 |
| CAP64/TGT512 | 同上 | 同上 |
| CAP128/TGT1024 | 持平/dec3 反降 | — |

**定档 CAP=64 / TARGET=512**（`crates/wein-kernel/src/ops/mod.rs`）。
pacc/pml scratch 随 CAP 涨至 67MB+0.5MB（一次性分配）。decode cfg 保持 n4r2。
kernel 侧 js 全参数化无硬编码；row_band 按行自身 pos 分段（o41/o42 合同）。

## 3. 读数

| 指标 | 前 | 后 | vLLM |
|---|---|---|---|
| 6k decode attn 段（47 步均值） | 1.42ms/步 | **0.58** | 0.40 |
| 6k hot ITL p50（服务端图档） | 12.4-12.6 | **11.85**（11.27/12.13/11.85） | 10.67-10.93 |
| 6009 冷 TTFT（exec 探针） | ~600 | 603.3（带内） | 383-418 |
| 段表段和 | — | ≈11.2 ≈ ITL ✓ | — |

gemm 8.34ms/步占 74%——6k decode 剩余差距的主项已转移至 gemm（下刀）。

## 4. 测试与合同

- **o30 位级钉**：形改为 rows>1024（16/4/256×t128；32/16/128×t64）——js=1 时
  ATTEN_TILE 路径逐位同 reference。
- **o31 容差档**：增 (16,4,256,64,300) 组合（js=64 全档），rel2 4.0e-5 / 峰值
  3.3e-3，门 2e-3/1e-2 的 20× 裕量内。
- **b2 混批测试重写**：mixed 轨迹抽闭包，4 个**独立 base** × len 37-40。
  - 硬合同：首 token 4/4 严格同（追平步两侧同形）+ **big 端完整序列 4/4 全同**
    （chunked 路 + 老路并存的状态隔离——互踩必翻 big，实测零翻）。
  - 软合同：small ≥2/4 全同。exec 级贪心对拍有两个固有容差源（混批步 gemm
    m=2002 vs 单跑 m=1 的 cuBLASLt algo 差 + jsplit>1 本身是 o31 容差档）。
  - **判别链**（证明是容差显形非结构违约）：CAP16 同测试绿；同 base 截断的
    4 长度全翻 token3-5（序列相关 ⇒ 同一并列事件重复显形）；换独立 base 后
    2/4 全同、翻的长各 1-2 处（含 token1 = 混合步 gemm 差直接显形）。kernel
    级"批形不改数值"由 o41/o42 位级钉（同 js 下逐位同）——那是强合同。
  - GEOM trace（临时代码，已删）：solo/mixed 几何逐 launch 一致。
- wein-kernel 全套（11+7+5+6+16+22+11+2）、wein-exec 23/23、bigchunk 2/2、
  qwen_graph 3/4（qg4 既有并行 flaky，单跑 3/3 绿）。

## 5. battery 回归（battery-20261109-220737，vs 8738a0f 基线）

| 档 | 前（8738a0f） | 后 | 判 |
|---|---|---|---|
| 图档满载 tps | 4297.7 | **4358.0** | +1.4%（t≤64 行批 jsplit 同享放开） |
| 图档满载 ITL p50/p99 | 20.65 / 32.64 | **20.24 / 32.49** | 带内偏好 |
| eager 满载 tps/ITL | ~4209.8 / 21.54 | 4134.0 / 21.29 | tps 当日漂移带内、ITL 更好 |
| 轻载图档 ITL | 11.23-12.70 | 12.25 | 带内 |

## 6. 遗留

- gemm 8.34ms/步（74%）为 6k decode 剩余主项——m=1 GEMV 已贴天花板，差距在
  m-shape 组织，见 f65bd8f 刀的 m 曲线。
- rope 0.26ms/步未并 gemm epilogue。
- 行数门 32 之上（并发 32-64）未扫。
