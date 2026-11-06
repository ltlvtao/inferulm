# 步内宿主路径解剖：0.83ms/步 全部落地（2026-11-06，HEAD 前 = d6b5a0a）

对象：`bench-reports/svc-gap-20261106/` 清单第 3 项（"采样 D2H 与下一步重叠"）——
先量清楚"GPU 空闲那 1.4ms 里主机在干什么"，再决定怎么修。

## 结论一句话

**每步宿主 0.83ms = 图回放 0.50（其中 memset 节点 0.37）+ 输入上传 0.07 + 回读 0.07
+ 引擎循环 0.19**；这 0.50 里的 0.37 是 cuBLASLt 工作区 memset 节点的**发射税**，
但它被 1.9ms 的 gemm 时间优势买回来了（strict 实测净亏）。⇒ 宿主侧已被榨干，
下一个量级不在这一层。

## 一、相位计时（新增两处仪器，均 env 门控、默认零成本）

| 仪器 | 位置 | 输出 |
|---|---|---|
| `WEIN_STEP_PROF2=1` | `qwen.rs` 图档回放路径（上传/launch/sync/回读 四段） | `[graph] N 步均值` |
| `WEIN_LOOP_PROF=1` | `wein-engine` 步循环（前进规划/exec.step/分发/尾部） | `[loop] 100 步均值` |

c=100 / n=3000 / max_tokens=32 / 图档，4350 tps：

```
[loop]  前进/规划 0.19-0.34 + exec.step 21.9 + 分发 0.045 + 尾部 0.005 = 22.1ms
[graph] 上传 59-88 + launch 490-509 + sync 19613-19845 + 回读 62-74 = 20265-20498µs
```

⇒ 引擎侧（tokens 分发/scheduler 记账/retire）只 0.05ms，**不是**空档来源；
空档全在 exec 的 4 个宿主段里（合计 0.66ms）+ 引擎前进规划 0.19ms。

## 二、四个宿主段逐个定案

### 1. `cuGraphLaunch` 490-509µs —— 真凶是 memset 节点（新证据）

`nsys cuda_api_sum` 说 `cuGraphLaunch` 只 **20.3µs/发**（475 发/9.62ms，max 485µs），
而应用侧量到 500µs ⇒ 差额在驱动内部、CUPTI 不记 API。

新探针 `examples/graph_nodes_bench.rs`（留库）逐类节点量**宿主提交单价**
（单发计时，每发前先 sync——连续 launch 同 exec 会被驱动反压串起来，量到排队）：

| 节点类型 | launch 宿主 µs/节点 |
|---|---|
| 自写 kernel（`add` 4 参 / `rms_norm_zc` 7 参，每节点参数互不同） | **0.08-0.11** |
| cuBLASLt gemm（`run_wt` 100×2560×2560） | **0.20-0.36** |
| memcpy D2D（`copy_range_on` 4KB） | **1.27** |
| **memset（`zero_range_on` 4KB）** | **5.36** |

按每步节点构成（nsys `cuda_gpu_trace` 普查 702.9 节点/步）算：

```
68.9 memset × 5.36 = 369µs
42.0 memcpy × 1.27 =  53µs
223  gemm   × 0.22 =  49µs
370  kernel × 0.10 =  37µs
               合计 ≈ 508µs   ← 与实测 500µs 对上
```

- **`cuGraphUpload` 无效**：探针与真模型图各测一遍（launch 前 upload、实例化后
  upload 一次都试过）——500µs 不变，strict 档 192µs 也不变 ⇒ 不是"图没上传"。
- `WEIN_LT_WS0=strict`（禁一切带工作区 algo，新增档）：launch **500→192µs** ✓
  但 **GPU 21.7ms（+1.9ms）、tps 4350→4166** ⇒ 那些 algo 省下的 gemm 时间远
  超发射税，**默认 3% 滞回是对的，不动**。
- memset 节点来源全是 cuBLASLt 工作区（45.8% 前驱 `add_norm_zc`、45.8% 前驱
  cutlass gemm）；1KB/1KB 大小。⇒ 无法从我们侧消除（要么换 gemm 实现）。

### 2. 输入上传 59-88µs / 回读 62-74µs —— pageable 内存税（**唯一可白捡的 0.1ms**）

`nsys cuda_api_sum`：`cuMemcpyHtoDAsync_v2` 4052 发 **11.8µs/发**（6.7 发/步）、
`cuMemcpyDtoH_v2` 603 发 **56µs/发**。两处都是 **pageable 源/目的**：异步拷贝
从页锁外内存发出时驱动要内部暂存并与流序同步，所以"异步"拷贝实际逐发阻塞
~12µs。改 pinned（`cuMemHostAlloc`）可降到 ~2-5µs/发：**上传 79→20µs、
回读 56→15µs，合计 −0.1ms/步（−0.45%）**。
注意：上传有 6 个调用点且同一流上排队，共用一块 pinned 暂存会被后续调用提前
覆写 ⇒ 要么按调用点分槽（一次性 16KB 竞技场），要么合并成一次批量上传。

### 3. memcpy 节点 42 发/步（21×1MiB + 21×48KiB）—— radix 状态快照，属设计

`push_snapshot_ops` 在 `(pos+1) % grid == 0` 的块边界存整行递推状态
（1MiB）+ conv 状态（48KiB），每请求每 `snap_grid_dec` token 一次；实测
~0.9 快照/步 ⇒ 42 节点/53µs 宿主 + 22MB/24µs GPU。想省：`--snap-grid-dec`
调大（频次线性降）、或换成自写 bulk-copy kernel（42 节点 → 1 节点）。
属"前缀缓存要不要这么密"的策略问题，不是 bug。

### 4. 引擎前进/规划 0.19-0.34µs×1000 —— `advance`+`pump`+`plan`

`pump` 里的 admitted/released 已在本轮 `d6b5a0a` 从 95 发降到 1 发；剩余
0.19ms 是 100 请求的调度记账、块表账、日志/指标刷新。要和 GPU 重叠就得
设备端造下一步输入（overlap scheduler 级重构），不在本刀范围。

## 三、判定

1. **宿主侧这条路走完**：可兑现的只剩 pinned 暂存 0.1ms（0.45%）。
2. 剩余差距全部落在 GPU 像素级：norm/eltw 家族 1.75ms（vLLM 0.645）里的
   ~0.53ms 是归约树形态问题 ⇒ 需批准破位级合同（`bench-reports/svc-gap-20261106/`）。
3. **副产品知识**：以后凡"图里多一发 kernel 值不值"用 0.10µs/节点算；
   多一个 memset 节点要按 **5.4µs** 算、memcpy 按 1.27µs——这也解释了
   `d6b5a0a`（48 发 memset→1 发 kernel）为什么值 +6.5% tps：它在**每请求
   流转点**上，而流转点每步 1-1.5 次。

## 四、改动清单（本刀）

- `crates/wein-exec/src/qwen.rs`：`WEIN_STEP_PROF2` 四段计时（默认关）；
  图实例化后 `exec.upload()`（实测无效果，保留为 API 例证）
- `crates/wein-engine/src/core.rs`：`WEIN_LOOP_PROF` 四段计时（默认关）
- `crates/wein-kernel/src/cuda/driver.rs` + `mod.rs`：`cuGraphUpload` 绑定与
  `GraphExec::upload`（探针用）
- `crates/wein-kernel/src/cuda/cublaslt.rs`：`WEIN_LT_WS0=strict` 档（默认关，
  实测净亏，留作 `ws0` 判据的证据）
- `crates/wein-kernel/examples/graph_nodes_bench.rs`：节点单价探针（新）
