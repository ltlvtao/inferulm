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
| L3 exec/图 | 提交与回放开销 | 真执行层（`--engine core --exec toy`，真 GPU 步；图档 NONE） | TTFT p50/p99、ITL mean、tps | ✅ 在测 |
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
- **L3 与 L2 的 tps 不可直接比**：L2 执行器是零成本脚本；L3 是真 GPU 前向
  （每步 16 launch + GEMM×7 + 上传/回读，x7 金丝雀 ~0.6ms/步）。L3 的账 =
  exec 步真算成本 + 批并发摊薄（批 100 时 ITL mean 0.80ms/流 ≈ 整批一步）。
  电池还顺带抓到过长寿性缺陷（块表行不回收，第 513 次 admit 炸引擎任务，
  run2/3 全空——x8 回归钉）。

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
| 2026-10-28 | +exec | release | 168,890 | 10.20 / 12.31 | — | — | 1.15 / 3.08 | 0.041 | 1.72 | 0.043 | exec 真身首测（battery-20261028-215639）：**L3 TTFT p50 6.76ms / p99 14.27ms / ITL mean 0.80ms / tps 47.1k**（toy 真 GPU 步，其余层噪声带内）；行回收缺陷修复后三重复全健康 |
| 2026-10-29 | +qwen | release | 165,559 | 10.78 / 12.10 | — | — | 1.05 / 4.59 | 0.036 | 1.65 | 0.058 | 真模型竖切首测（battery-20261029-002042，I1）：**L3q TTFT p50 72.35ms / ITL p50 30.91ms / tps 115.7 / errors 0**（Qwen3.5-4B 真权重，轻载 c=4/n=8/12tok——满负荷 profile 撞 exec 批界不适用）；ITL≈q9 单步账（33.7ms 中位），TTFT=追平+首步；conv 状态行间接缺陷（span 序号≠状态行号）q8 双胞胎实验定位，o19 升级钉住 |
| 2026-10-29 | +graph | release | 169,691 | — | — | — | 1.13 / 2.18 | 0.027 | 1.64 | 0.037 | W2 图档首测（battery-20261029-013037，`--graph` FULL_DECODE 桶{1..32}回放）：**L3q+ ITL p50 28.79ms vs L3q 31.28ms（−8.0%）、tps 122.1 vs 115.4（+5.8%）、TTFT p50 72.29 vs 75.64ms、errors 0**；单步对照 qg3 图先测防热身漂移（§7）；混合步/prefill 仍 eager（sink 行吸收桶 padding）；其余层噪声带内 |
| 2026-10-29 | +chat | release | 162,763 | — | — | — | 1.10 / 5.47 | 0.025 | 1.68 | 0.050 | W5 chat 模板（battery-20261029-091642）：**语义变更——bench 流量走 chat 端点吃到真模板**，prompt 4→11 token（占位 `user: bench\n` → `<\|im_start\|>…<think>\n`，实测 token 数）：L3q TTFT 72→112ms 是追平多算 7 token 的账（**非性能回归**；隔离复跑复现 + token 数对账归因），ITL 32.91ms / tps 100.2；同轮同语义 L3q+ 图档 ITL 31.94ms（**−3.0%**，方向与 §6 上行一致）；stop_token_ids 热路径零开销（total_tokens 两轮同 96——无提前 eos）；其余层噪声带内 |
| 2026-10-29 | +fix | release | 163,838 | — | — | — | 1.21 / 4.12 | 0.032 | 1.66 | 0.046 | 递推偏差双修复复跑（battery-20261029-112815）：GQA 头映射 `qh%kvh`→`qh//group`（PTX kernel/CPU 参考/f64 镜像三处）+ conv 状态存值 `qkvc`→原始 `qkv`（exec 调用点/镜像两处）——**纯正确性修复零性能账**：L3q TTFT 117.79 / ITL 33.48 / tps 99.2、L3q+ ITL 31.24 / tps 104.2，与 W5 语义基线（091642）带内；外部对拍 wein≡vLLM 48 token × 5 prompt 全同（`bench-reports/vs-vllm-correct/`）；L0 六轮 143k–169k，SUMMARY 引用的首套冷启 142.9k 偏低（wall 0.12s 小样本），表中取六轮中位 |
| 2026-10-29 | +full | release | 174,069 | — | — | — | 1.17 / 3.75 | 0.035 | 1.69 | 0.085 | **满负荷 L3q 首测**（battery-20261029-121058，c=100/n=500/mt=32 与 L0-L3 同 profile）：L3q 满负荷 TTFT 804.08 / ITL 141.61 / tps 574.1，L3q+ 图档满负荷 795.98 / **138.99 / 581.4**（桶 {1,2,4,8,16,32,64,100,128}）。配套 engine 准入重构（e5）：`Executor::row_capacity` 钳 max_running + 建行挪到调度准入（先回收后建行）——超容排队不炸批界，**副产 busy 档换手提速：L3 tps 49.5k→91.4k（+84%）、L2' 28.2k→45.7k（+62%）**，每请求指标逐位不变（ITL/TTFT/token 数同）、单波对照（n=c=100 新旧无差）⇒ 全部来自换手时延（Finished 交付提前一个步内位置）；轻载 L3q/L3q+ 带内；L2/L2m 289-313k 带内展宽（零延迟纯 CPU 档对机器态敏感）；批扩展曲线与图桶 padding 教训见 §7 |

## 7. kernel 侧微基准（设备端首测，随 exec/kernel 里程碑追加）

| 日期 | rev | 设备 | 指标 | 值 | 备注 |
|---|---|---|---|---|---|
| 2026-10-28 | +kernel | RTX 4090 (sm_89) @WSL 驱动 610.57 | launch 均摊 | **4.69 µs/次** | 1 万次连续提交+末尾一次 sync；k8 金丝雀 |
| 2026-10-28 | +kernel | 同上 | PTX JIT 冷/热（跨进程） | 冷 170ms → 热 96ms | 含进程启动+cuInit 固定开销；驱动缓存文件落盘复用（k7，§13-2 ①） |
| 2026-10-28 | +kernel | 同上 | bf16 点积位对齐 | GPU≡CPU 位位相等 | 512 线程×64 段（k6，§13-2 ②）；PTX `.target sm_70` JIT→sm_89 |
| 2026-10-28 | +gemm | 同上 | GEMM 吞吐 canary（2048³ bf16） | **125.6 TFLOPS** | cuBLASLt 12.9.2（pip wheel 提取，vendor/）；峰值 ~165 的 76%；g6 金丝雀，下限门禁 10 防非 TC 回退 |
| 2026-10-28 | +exec | 同上 | exec 步时延 canary（批 4 纯 decode） | **中位 611 µs/步** | 一步 = 5 上传（alloc+H2D）+ 16 launch + GEMM×7 + sync + logits 回读；x7 金丝雀；对照 launch 4.69µs——exec 层优化账（W2：常驻上传缓冲/图捕获/切片回读） |
| 2026-10-29 | +graph | 同上 | 图档步时延 canary（toy，批 50 步中位） | eager 680 µs → 图 528 µs（**1.29×**） | w3 金丝雀；toy 步 launch 债占比高（16 kernel/步），图收益 = 全 launch + 5 次分配/上传转常驻覆写 |
| 2026-10-29 | +graph | 同上 | 图档步时延 canary（Qwen3.5-4B，单请求 decode 20 步中位） | eager 26.9 ms → 图 25.1 ms（**−6.7%**） | qg3 金丝雀，图先测（热身漂移偏向后测的 eager）；大模型步 GEMM 主导（8.4GB 权重读/token），图收益 = 32 层 launch 债（~160 kernel/步）+ 14 次分配/同步上传；端到端 L3q+ ITL −8.0%（§6） |
| 2026-10-29 | +full | 同上 | 批扩展曲线（`qwen-sweep-20261029-121226`，n=c 同步起步纯 decode 稳态） | eager ITL 36.99 → 62.05 → 80.96 → 108.09 → 142.76 → 169.34 ms（c=4→128） | m×32 而步时仅 ×4.6（批摊销有效）；**m=100 带宽账 8.4GB/142.76ms ≈ 59GB/s**，vLLM 同卡 692GB/s——每步 ×12 的账不在显存带宽（权重只值 ~12ms），在每行 host 侧工作与 kernel 形状（cuBLASLt 按 m 选 algo、逐行上传/argmax），满负荷优化的路线图数据；tps 峰值 605（c=128 eager） |
| 2026-10-29 | +full | 同上 | 图档收益 vs 批（同曲线） | −4.4% / **−7.1% / −7.5%** / −2.7% / −2.3% / −0.1%（c=4→128） | 收益峰在批 16-32（launch 债占比最高处），大 m 收窄（W2 结论的满负荷延伸）；**桶 padding 教训**：桶 128 对 m=100 多算 28 行 → ITL +16%（反劣于 eager），桶粒度须贴工作点（默认桶集 {1,2,4,8,16,32,64,100,128}，`WEIN_GRAPH_BUCKETS` 可覆盖）；c=128 图档 tps 538.6 < eager 605.2——sink 让位 1 行 = 少 1 并发槽 |

launch 4.69µs 是 L3（exec 层）落地时的对照基线：纯 Rust→PTX 路线在 WSL 上的
单 kernel 提交成本在此，exec 步的图捕获/回放收益按此尺度衡量。
