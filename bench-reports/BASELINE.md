# wein 性能基线制度（BASELINE.md）

> 原则：**后续每一层实现都会把性能往下拖。没有基线，我们既不知道掉了多少，
> 更不知道掉在谁身上——也就永远不知道设计还有没有优化空间。**
> 因此：每层有专属指标（它的"账"）、每次变更跑电池、回归必须归因到层。

## 1. 归因阶梯

每层只对自己的指标负责；上一层掉数字，先问下一层要账，逐层分解到唯一责任方。

| 层 | 测什么 | 引擎形态 | 专属指标（账） | 状态 |
|---|---|---|---|---|
| **L0 路由上限** | accept/parse/route/小响应 | 无关（GET /v1/models） | QPS、p50/p99 延迟 | ✅ 在测 |
| **L1 serve 层开销** | HTTP+SSE+detok+channel 的纯成本 | 零延迟 mock（`--token-delay-ms 0`） | TTFT p50/p99、ITL mean | ✅ 在测 |
| **L1' 节拍泵参照** | 有节拍生成下的端到端 | 1ms/token mock | ITL mean − 1ms = 每 token 可观察开销 | ✅ 在测（真实引擎接入后退役为 L2） |
| **L2 engine core** | 调度决策 + 批组装开销 | 真调度器 + 脚本执行器（`--engine core --token-delay-ms 0`） | TTFT p50/p99、ITL mean | ✅ 在测（schedule_ns 细粒度插桩待 exec 层，现用 EngineStats.steps 计数） |
| **L2' 节拍泵(调度器)** | 真调度器下的端到端 | 1ms/步 | ITL mean（与 L1' 之差 = 调度层叠加开销） | ✅ 在测 |
| **L2m 内存层记账** | BlockPool 水位/radix/DeferredRelease 的热路径成本 | `--engine core --memory --token-delay-ms 0` | ITL mean（与 L2 之差 = 内存层叠加开销） | ✅ 在测 |
| L3 exec/图 | 提交与回放开销 | 真执行层 | launch→replay 时延、图命中率 | ⏳ |
| L4 kernel | 算子真算力 | 真算子 | 每 op 耗时（图内探针） | ⏳ |

## 2. 噪声纪律

1. **只认 battery 报告**：`bench/battery.sh [release|debug]`，报告落 `bench-reports/battery-<时间戳>/`，
   带 git rev + profile + 时间戳；手工单跑的数字只作调试参考，不入演化表。
2. **repeat=3 取中位数**（wein-bench `--repeat`），治单次抖动。
3. 同机、同 profile、同参数对比；debug 与 release 的数字**永不混比**。
4. 每个集成里程碑（I1-I4）与任何可能影响性能的 PR 必须跑 battery，演化表追加一行。

## 3. 已知测量陷阱（第一条就是它自己抓出来的）

- **ITL 看 mean 不看 p50**：单进程 bench 客户端在测量路径内，100 条并发流的帧解析
  挤在同一 runtime，交付被量化成"团+洞"（首测：p50=3µs 而 mean=1.70ms 双峰）。
  p50/p99 仍记录（作相对回归信号），但**归因结论以 mean 为准**。
  需要绝对 ITL 时换多进程压测端（v2 工具）。
- **L0 是 loopback 数字**：同机回环，绝对值虚高；作层间比例与回归用，不作部署承诺。
- **mock 的定时器抖动属于 mock**：L1'/L1 之差才是 serve 层的账；mock 内部 timer wheel
  的合并行为（ITL p99 ~17ms）不是 serve 层劣化。
- **L2m 与 L2 的对照才是内存层的账**：内存层（radix 匹配/块分配/水位判定）
  在每个 token 的热路径上；首测 ITL mean +13µs/token、tps 持平（−0.2%，噪声内）。
  内存层块容量给宽裕值（8192×16）——量的是记账开销，不是抢占行为的账。
- **L2 与 L1 的 tps 不可直接比**：mock 是每请求独立任务逐 token 唤醒；EngineCore 整批
  同步交付（32 步管 100 流），天然平滑——L2 tps 309k 对 L1 tps 81k 的优势是模型差，
  不是 serve 层回归。同节拍对照只看 L2' vs L1'。

## 4. 未来插桩（随实现落地，设计现在就定死）

| 插桩点 | 位置 | 形态 |
|---|---|---|
| engine core 每步耗时 | 调度循环 | span：schedule_ns / batch_size / admitted / preempted |
| exec 提交与回放 | exec 线程 | span：plan_ns / submit_ns / replay_ns / graph_hit |
| kernel 图内计时 | 图捕获体内 | DeepInfer graph_timer 思想：图内事件探针，量图内真实耗时而非墙钟差 |
| 输出 ring 水位 | serve↔engine 边界 | 高水位计数（背压信号，02 §13-4） |

## 5. 门禁（smoke 阈值随基线演化修订，只许收紧）

```bash
# L1（零延迟引擎）——serve 层的账
--smoke "ttft_p99_ms<=30;itl_mean<=2;errors_max<=0"
# L0——路由上限
qps >= 100000（人工核对 SUMMARY）
```

## 6. 基线演化表

| 日期 | rev | profile | L0 QPS | L1 TTFT p50/p99 (ms) | L1 ITL mean (ms) | L1' ITL mean (ms) | L2 TTFT p50/p99 (ms) | L2 ITL mean (ms) | L2' ITL mean (ms) | L2m ITL mean (ms) | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-28 | 30c3b01 | release | 170,528 | 10.15 / 12.46 | 0.89 | 1.70 | — | — | — | — | 首基线；ITL 双峰发现（§3） |
| 2026-10-28 | 6200950+core | release | 170,415 | 10.84 / 12.25 | — | — | 1.11 / 2.68 | 0.027 | 1.69 | — | engine core 首测（battery-20261028-144250）；L2'≈L1'：调度层叠加≈0；L2/L1 tps 模型差见 §3 |
| 2026-10-28 | +memory | release | 168,289 | 10.56 / 12.21 | — | — | 1.18 / 4.98 | 0.022 | 1.67 | 0.035 | 内存层首测（battery-20261028-152856）；L2m−L2 = +13µs/token，tps 持平（312.6k vs 313.1k）；宽裕块容量——纯记账账 |

## 7. kernel 侧微基准（设备端首测，随 exec/kernel 里程碑追加）

| 日期 | rev | 设备 | 指标 | 值 | 备注 |
|---|---|---|---|---|---|
| 2026-10-28 | +kernel | RTX 4090 (sm_89) @WSL 驱动 610.57 | launch 均摊 | **4.69 µs/次** | 1 万次连续提交+末尾一次 sync；k8 金丝雀 |
| 2026-10-28 | +kernel | 同上 | PTX JIT 冷/热（跨进程） | 冷 170ms → 热 96ms | 含进程启动+cuInit 固定开销；驱动缓存文件落盘复用（k7，§13-2 ①） |
| 2026-10-28 | +kernel | 同上 | bf16 点积位对齐 | GPU≡CPU 位位相等 | 512 线程×64 段（k6，§13-2 ②）；PTX `.target sm_70` JIT→sm_89 |
| 2026-10-28 | +gemm | 同上 | GEMM 吞吐 canary（2048³ bf16） | **125.6 TFLOPS** | cuBLASLt 12.9.2（pip wheel 提取，vendor/）；峰值 ~165 的 76%；g6 金丝雀，下限门禁 10 防非 TC 回退 |

launch 4.69µs 是 L3（exec 层）落地时的对照基线：纯 Rust→PTX 路线在 WSL 上的
单 kernel 提交成本在此，exec 步的图捕获/回放收益按此尺度衡量。
