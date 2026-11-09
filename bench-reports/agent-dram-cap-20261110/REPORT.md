# DRAM 分层槽扩容：救 Agent c=8 命中（2026-11-10）

## 动机

终测（final-report-20261109）B 轴：wein agent c=8 turns 1.41（−42% vs vLLM 2.46）、
hit 0.455（vs vLLM 0.666）。定案为容量问题：8 会话 × ~42K token ≈ 336K 需求，
设备页池 4096 页 = 65K token（bf16 KV 下物理放不下 336K×32KB=10.8GB，24G 卡余量仅
~3.2GB——4096 页空载实测 21372 MiB / 24564 MiB）。设备池不可扩 ⇒ 唯一杠杆 =
DRAM 分层容量：旧档块槽 2048（=32K token 副本）只能保住 **1 个会话**的 KV 副本，
多会话 LRU 互逐后 admit 无块可补、恢复点用不上。

## 改动（crates/wein-exec/src/qwen.rs，仅 WEIN_DRAM=1 分支默认值）

- `WEIN_DRAM_BLK` 默认 2048→**8192**（块 KV 副本 32K→128K token，4.0GB pinned host）
- `WEIN_DRAM_ST` 默认 64→**96**（状态锚 96 张 ≈ 16 会话 × 6 轮全覆盖，2.4GB pinned host）
- 验证分配生效：起服务 host available Δ=7221MB（≈6.4GB pinned + 模型加载工作集）

c=8 会话×轮 ≈ 48 张状态锚 < 96 槽；块需求 336K > 128K 仍不全覆盖，但 LRU 窗口
从"1 会话"扩到"3 会话副本 + 命中刷新（tick 重入队）"，活跃会话工作集可整体驻留。

## A/B（同二进制，三轮交错，c=8×6，ctx32768/kv4096/chunk2048/WEIN_DRAM=1）

| 档 | turns/s（中位） | ttft_hot | itl_med | prefix_hit |
|---|---|---|---|---|
| old（BLK2048/ST64） | 1.714 | 442.5ms | 16.7 | 0.565 |
| **new（BLK8192/ST96）** | **1.834（+7%）** | **384.6ms（中位 −13%）** | 17.0（带内） | **0.697（+13pt）** |

- 三轮 3/3 方向一致；new 三发 hit 0.695/0.695/0.701（方差 0.006，极稳）。
- **hit 0.697 ≈ vLLM 同口径 0.666–0.673：追平/略超**。
- c=4 回归（new 档）：turns 1.424 / ttft_hot 268.6 / itl 14.84 / hit 0.6936——与
  终测 c=4（1.359/285/15.01/0.693）同档无回退。

## battery 回归（battery-20261110-002214）

图档满载 tps 4255.2 / ITL p50 20.79 / p99 33.25（基线 4297.7@20.65/32.64 与
4358.0@20.24/32.49，−1%~−2.4% 在当夜漂移 ±2-4% 带内；改动不在 battery 路径——
DRAM 分支默认关）。轻载图档 12.11 带内。

## 测试

wein-exec --tests：acceptance 8/8、dram_tier 3/3、qwen 3/3、bigchunk 绿；
qg4 首轮并行红后单跑 3/3 绿（既有 flaky，与改动无关）。

## 结论

Agent c=8 命中缺口（终测 0.455）的主因确认为 DRAM 块层容量而非机制；扩容后
hit 0.70 追平 vLLM，turns 差距从 −42% 收到约 −25%（1.834 vs 2.46，同场 old 也
1.71，绝对值受环境抬升）。剩余差距 = itl_med 17.0 vs 13.98（+22%，调度/混批
prefill 配额层）与 336K>128K 的副本窗口仍非全覆盖。DRAM 默认仍关
（WEIN_DRAM=1 显式开，pinned host 成本 6.4GB）。
