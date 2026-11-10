# 混批调度刀：c=8 ITL +22% 的两条调度假设证伪 + v4 余段 kernel 修复收官

日期：2026-11-10 ｜ HEAD 前：44e7deb ｜ 口径：agent_bench c=8×6（ctx32768/kv4096/chunk2048/WEIN_DRAM=1）三轮交错 A/B，服务端口径同 DRAM 刀

## 结论一句话

目标（itl_med 17.0→~14）**未达成**——两条调度层假设（轮转均分、时间闸门下探）均被三轮 A/B 证伪回退；但归因途中挖出的 **v4 余段 kernel 病灶**修复落地：**turns +14%、ttft_hot −6%（最好轮 −22%）、hit/itl 持平**，为净收益。itl 差距 +23% 的真正账在 prefill 单价（kernel 层），不在调度。

## 归因（开工时的数据）

1. **STEP_HIST 直方图**（c=8×6 两发）：混批步仅占步数 5-16% 但占墙钟 25-51%，步均 85-160ms（dec 行 4-7）；纯 decode 步 13.4ms（7.8 行）→ 18-19ms（4-6 行）。itl_med 是轮次单调爬升（12.4→19.1）而非偶发尖峰。
2. **nsys 全程**（agent_c8.nsys-rep，66s 窗口 kernel busy 19.4s）：attn 4990ms（26%）里 **paged_attn_v4 独占 2397ms（608 发 avg 3.94ms）**——v4 门（t≥2048）是 v5 门（t≥512）严格子集，608 发全部来自 **v5/v9 主段切完 64 对齐后的余段**（t4≤48 行），该形状 4× KV 重读放大病态慢（48 行 × 5K 上下文）。
3. 修复：余段整体改走 `run_tile_online_cfg`（行独立、对该形状快）；位级合同从"v4≡旧 tile 位级"降为 online 容差档（o33 rel2≤8e-5），端到端贪心全测绿（qwen 23/23、acceptance、bigchunk、graph）。

## A/B #1：轮转均分 + 闸门 rows 32→2（两改动耦合）——证伪

| 指标（3 轮中位） | old(44e7deb) | new(轮转+rows2+v4修复) |
|---|---|---|
| turns_per_s | 1.734 | 1.735（平） |
| ttft_hot_ms | 413 | **878（翻倍）** |
| itl_med_ms | 16.83 | **17.7（反升）** |
| prefix_hit | 0.689 | 0.696 |

轮转把每会话追平拉长 n×（share≈预算/8≈113 token/步，600-token turn 要 6 步）→ ttft 翻倍；全体同步慢慢追平 ⇒ 混批步更多、decode 行恢复更慢。**FIFO 错峰（多数会话稳 decode、少数逐个追平）结构性优于轮转。** 回退。

## A/B #2：同二进制 flag 分离（g0=闸门关/v4修复，g1=闸门 rows=2 开）——闸门证伪

| 指标（3 轮中位） | old | **g0 = 最终配置** | g1（闸门开） |
|---|---|---|---|
| turns_per_s | 1.734 | **1.974（+14%）** | 1.476（−15%） |
| ttft_hot_ms | 413 | **388（−6%）** | 1263（3×） |
| itl_med_ms | 16.83 | 17.19（带内） | 19.5（r2 达 33.7） |
| prefix_hit | 0.689 | 0.689 | **0.475（0.29-0.52 崩）** |
| itl_p95 | ~18.7 | 19.3 | 41-45 |

闸门失败机理：est 模型 MIX_TOK_MS=0.04ms/tok 低估混批真实 ~0.07ms/tok ⇒ 许可 330 tok/步的步长仍 ~35ms，**压了 prefill 进度没压住步长**；追平期从 1-2 步拖成 3-6 步 ⇒ 各会话中间态长期占 radix/DRAM 槽、decode 行长期陪跑 ⇒ hit 崩 + itl 反升。c=4 回归同向（turns 1.346 vs 基线 1.424）。回退 rows=32（仅满载 c≥32 生效，8738a0f 的满载 p99 收益不动）。

## 最终改动（本刀落地）

1. **v4 余段→online**（qwen.rs 两处：v5 门余段合并单发 online；MIXSEG t4 段并入 pend 连续区间）——**保留**，g0 三轮即最终配置读数。
2. **sched 轮转回退 FIFO** + 注释钉证伪数据；s13 测试改钉 FIFO 独吃语义（chunk==budget 生产比形：先到债主吃满、后到者饿等）。
3. **mix_budget_rows 维持 32**（曾改 2，A/B #2 证伪回退），Default 注释钉结论。
4. 附带仪器：`attn_rows_probe`（decode attn 行数×上下文带宽矩阵，R≥4@5K 断崖 330GB/s 的证据，留后续刀）。

## itl 差距剩余的诚实账

itl_med 17.2 vs vLLM 13.98（+23%）**未收**。纯 decode 步（7-8 行 13.4ms）已平价；差距在（a）混批步的干扰——但调度层两条节流路已证伪，正解是把混批步本身变便宜（本刀 v4 修复已吃掉最大一块，还剩 gemm 组织/delta 小 span）；（b）vLLM prefill 单价 19.2-21.1k tok/s vs 我们约一半——同一 kernel 账（prefill lane/gemm 形态），属"降低 prefill 单价"线，不属调度线。**本刀把"调度层收 itl"这条路正式关闭并留档。**

## 验收清单

- wein-sched 14/14（s13 改钉 FIFO）、acceptance 8/8、qwen 23/23、bigchunk 2/2、graph 4/4
- battery 满载：见 commit message（带内）
- v4 死门（qwen.rs single_span t≥2048 分支）保留不删：WEIN_ATTN_V5=0 时的合法回退路径，运行时零成本
