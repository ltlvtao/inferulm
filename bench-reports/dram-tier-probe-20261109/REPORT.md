# DRAM 分层探针：pinned 搬运判定通过（2026-11-09）

> 刀序 1/3。问题：LRU 驱逐块/快照下沉 DRAM、命中搬回是否划算。
> 判定线：215MB（一张快照 25.2MB + 190 KV 页）换入 < 20ms（重算同段 ~600ms）。

## 一、读数（WSL2，RTX 4090 PCIe 4.0 x16）

| 尺寸 | H2D pinned | D2H pinned | D2H pageable（对照） |
|---|---|---|---|
| 25 MB | 1.12ms (23.4GB/s) | 1.02ms (25.7) | 4.06ms (6.5) |
| **215 MB** | **9.43ms (23.9GB/s)** | **8.63ms (26.1)** | 69.6ms (**3.2**) |
| 1 GB | 44.90ms | 40.97ms | 354ms (3.0) |

- pinned 双向 ≈ PCIe 理论 32GB/s 的 80%，WSL2 半虚拟化无损耗。
- **换入 215MB = 9.43ms，判定通过**：vs 重算 600ms = 64× 赢面。
- pageable D2H 仅 3GB/s（read_to_vec 含大 vec 分配 + 驱动 staging）⇒
  **换出路径必须 pinned**，差 8.7×。

## 二、基建（随本刀入库）

- driver.rs：`cuMemHostAlloc/cuMemFreeHost/cuMemcpyDtoHAsync` 三个 FFI 符号
  + `mem_host_alloc/mem_free_host/memcpy_dtoh_async_pinned` 封装。
- cuda/mod.rs：`HostPinned<'g>`（Drop 释放；`download_on`/`upload_to_on`
  异步区间拷贝，流上保序）+ `Gpu::alloc_host_bytes`。
- examples/dram_tier_probe.rs 留库（含 pageable 对照口径可复跑）。

## 三、下一步（刀序 2）：最小闭环

只做快照槽换出：64 槽驱逐不丢 → pinned 落 DRAM 索引 → admit miss 先查
DRAM → 搬回挂树（链保护 pin 语义复用）。KV 块暂不动。预期 c=16 命中
20% → 40%+（复用点 ×N），换入 1.1ms/张 对 TTFT 不可见。

刀序 3（全量）：KV 块 + 快照整体分层 + 池 2048 页 + 准入联动。
