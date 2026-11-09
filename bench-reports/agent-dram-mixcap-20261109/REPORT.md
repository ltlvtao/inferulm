# Agent 场景 DRAM 分层主动下沉 + 混批封顶 192→4096（2026-11-09）

## 0. 背景与结论速览

上一刀（c1964bf）修掉单 token 分叉与粗栅格锚保留后，c=8 命中仍在 14-25%（vLLM 同口径 67%）。
本刀原目标是"DRAM 分层挂载接力"（W-Agent-5 刀序 2），诊断中发现接力几乎不触发；修好接力后
四格矩阵 A/B 把真正的杠杆暴露出来——**混批 token 封顶 mix_cap=192 才是命中率被压死的根因**。

| 配置（c=8×6 turns, 4096 页, ctx 32768, graph, 2 rep） | turns/s | prefix_hit | ITL med | ttft_hot |
|---|---|---|---|---|
| mix192 + DRAM off | 0.562 / 0.658 | 0.139 / 0.249 | 51.3 / 50.0 | 5145 |
| mix192 + DRAM on  | 0.651 / 0.568 | 0.337 / 0.202 | 46.0 / 52.2 | 3999 |
| mix4096 + DRAM off | 1.472 / 1.531 | 0.364 / 0.395 | 17.7 / 18.0 | 385 / 351 |
| **mix4096 + DRAM on** | **1.517 / 1.547** | **0.513 / 0.528** | 16.3 / 18.0 | — |

- **mix_cap 192→4096**：turns ×2.4、ITL ×2.8、ttft_hot ×13、命中 +13~22pt —— 本轮最大单项。
- **DRAM 分层修复后**：mix4k 下稳定 +3% turns、+13pt 命中、ITL 持平；默认保持关（见 §5）。
- c=4 终验（新默认 mix_cap=4096 + DRAM=1）：**命中 0.6702 ≈ vLLM c=4 的 0.673**，turns/s 1.306，
  ITL med/p95 15.06/19.77，ttft_hot 280.3，0 错误。

## 1. 接力不触发的两个根因

诊断手段：admit 接力规划 debug 块（最深状态 e/对齐/缺块/首缺 + KV 齐 + 状态深度表）+
`DramTier::stats()/state_points()`。两次 c=8 诊断跑（/tmp/agent_dbg-on、agent_dbg2）：

1. **哈希链只登记新插段**：retain 的 blk_hash 登记循环只算 [d..full)，共享段与树覆盖段的锚
   从未有哈希 ⇒ DRAM 状态层的键对不上，全部 `sink-skip`。
2. **状态层只存"被逐出时恰好挂表"的锚**：`release-enter` 169 次全走 owner=None 分支，sink
   从不触发——状态深度表 [196,192,192,191,191,180] 属于别的会话链，同会话回来键对不上。

修前 mount-ok 仅 3-4 次 / 35-49 admit。

## 2. 修复（三处，crates/wein-exec/src/qwen.rs）

1. **哈希全算 + 登记边界**：blk_hash 链改 0..full 全算；登记 `i < shared`（树持有块）与
   `i >= d`（新插段），**跳过 [shared..d)**——该段块已还池、id 会复用，登记它等于日后把
   别人的 KV 当该前缀下沉（静默污染）。
2. **`sink_state()` 统一**（约 3083 行）：三处重复的状态下沉段表构造合并为一函数，
   release_slot 内联段改调用。
3. **主动下沉**：retain 快照循环三分支——`idx < full && (idx < shared || idx >= cover_d)` 挂表
   且**不等驱逐主动沉**（`idx+1 > SNAP_SHALLOW_BLKS(48)` 才沉，浅层共享锚留在 HBM 保粗栅格）；
   `[shared..cover_d)` 冗余重复段不挂表但状态按哈希沉后还槽；`_` 分支补 `snap_owner[slot]=None`。
   另在 clear_prefix_cache / preempted 槽回收处补 owner 清理与挂表反摘。

修后 c=8/4096 页：sink 132 次，mount-ok **8/49 admit**（含 best_e=355 tree_len=31 的大恢复：
树里只剩 31 块，接力找回 355 块）。

## 3. mix_cap 192→4096 与 battery 取舍

`crates/wein-sched/src/lib.rs` 默认值改动。battery 同二进制 CLI 覆盖 A/B（端口 8197）：

| 档 | 满载 tps | ITL p99 | TTFT p99 |
|---|---|---|---|
| mix192（旧默认） | 4336 | **25.9** | 199 |
| mix2048 | 4327 | 43.1 | — |
| **mix4096（新默认）** | **4369（+0.8%）** | 42.8 | **126** |

取舍：满载 ITL p99 25.9→42.5 的罚，换 TTFT p99 199→126 与 Agent 场景 ×2.4。2048 与 4096
同罚但 Agent 收益更小，弃。

全量 battery 归档 `bench-reports/battery-20261109-195103/`：图档满载 tps 4369.8 / ITL 20.74 /
p99 42.46 / TTFT 69.58/125.23；轻载图档 305.1 tps / ITL 11.10；L0 163.5k qps。对基线 c1964bf
（4297.5/21.11/p99 25.47）：吞吐 +1.7% 带内，p99 差即上述取舍。

## 4. c=4 终验（/tmp/agent_final-c4）

新默认 mix_cap=4096 + WEIN_DRAM=1，c=4×6 turns：
turns/s 1.306、命中 0.6702（vLLM c=4 = 0.673）、ITL med/p95 15.06/19.77（vLLM ~15）、
ttft_hot 280.3、tool_ok 0.79、0 错误、wall 18.4s。**命中率追平 vLLM。**

## 5. DRAM 分层默认保持关

收益：mix4k 下 +3% turns、+13pt 命中（0.39→0.52）；代价：~1GB 设备缓冲 + 1.6GB pinned host
（每 215MB 换入 9.43ms，探针见 04e1c92），battery 无关场景零收益。判定：作为长会话/低并发
场景的显式开关（`WEIN_DRAM=1`）比默认开更合理。修好的主动下沉逻辑在开关后面随时可用。

## 6. Dead ends（别再试）

- **等驱逐才下沉状态**：状态层几乎只剩"被逐时恰好挂表"的深锚，同会话键对不上。
- **登记 [shared..d) 冗余块哈希**：块 id 复用 ⇒ 把别人的 KV 当该前缀下沉，静默污染。
- **mix_cap 2048**：满载罚同 4096，Agent 收益更小。
- **2048 页跑 c=8 压测**：8×8500≈68K token > 32K 页池，capacity 错误（5 次），非 bug。

## 7. Open questions

- mix_cap 按时间预算（而非 token）封顶，能否同时拿下 Agent ITL p95（19.8 vs vLLM 15）与满载 p99。
- DRAM 状态槽 64 固定 FIFO：96 张需求会丢旧锚；是否随会话数放大。
- `[shared..cover_d)` 冗余段状态下沉后槽立即还池，其 KV 依赖树内块——若树内块也被逐且未沉，
  该状态条目成死键（占槽不生效，FIFO 自然淘汰，未量化）。

## 附：环境口径

服务：`env WEIN_DRAM=<0|1> [WEIN_RADIX_DEBUG=1] ./target/release/wein-cli --port 8195 --engine core
--exec qwen --model ~/.cache/wein/Qwen3.5-4B --token-delay-ms 0 --graph --ctx-len 32768
--kv-pages <N> --chunk-size 2048 [--mix-cap M]`；客户端 `/home/lvtao/venvs/vllm/bin/python
bench/agent_bench.py --sessions C --turns 6 --payload 512 --json out.json`。4096 页启动后
约 21.1GB 显存。vLLM 参照：c=4 0.673 / c=8 0.670 / c=16 0.678。
