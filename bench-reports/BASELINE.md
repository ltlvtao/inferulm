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
| 2026-10-29 | +deltanet | release | 158,378 | — | — | — | 1.09 / 2.67 | 0.040 | 1.68 | 0.042 | **deltanet 状态访存重写首战**（battery-20261029-194337）：kernel 4 趟 6 访存融合 3 趟 + f32x4 向量（算术序逐操作数不变——decay 写读往返改更新趟同式重算，位不变），o22 金丝雀 3.08→~0.75ms/launch（§7）。**L3q 满负荷 ITL 141.61→89.67 / tps 574.1→985.7（+72%），L3q+ 图档 138.99→87.61 / 581.4→1001.8——首破千 tps**；轻载 L3q 31.00 / L3q+ 29.98 带内（m=4 时 delta 段占比小，收益随批扩大：曲线 −6%→−38%，§7）；位对齐零漂移（o20/o16/kernel 44 测试 + exec q 系全绿）+ **wein≡vLLM 48tok×5prompt 逐字节全同 oracle 复验**；其余层带内（L0 158k、L3 93.9k） |
| 2026-10-29 | +rms | release | 165,014 | — | — | — | 1.15 / 3.54 | 0.028 | 1.65 | 0.059 | **rms 系块归约重写**（battery-20261029-205735）：row-per-thread 串行扫整行（rows=m 时全 GPU 仅 ~m 线程在跑，DRAM 延迟喂不饱，692µs/launch）→ block-per-row 跨步合并 + **kernel 参数携带全局 scratch 块归约**（stdarch nvptx 无 shfl intrinsic / `asm!` 未实现 / `link_section=".shared"` 被后端忽略——三条跨线程通道全断后的第四条路，§7；`argmax_rows` 同款 blockIdx 私有槽 + syncthreads + 线程 0 顺序合并，reference 同序镜像），o23 金丝雀 692→11.5µs/launch（60×，§7）。**L3q 满负荷 ITL 89.67→51.33 / tps 985.7→1724.7（+75%），L3q+ 图档 87.61→49.76 / 1001.8→1776.2**；轻载 L3q ITL 31.00→20.62 / L3q+ 29.98→18.97；归约序变更位对齐纪律走通（kernel↔reference 同步换序，o16/o4 位一致；q5 f64 镜像自动跟随）+ **wein≡vLLM oracle 复验逐字节全同**（resp diff 仅时间戳）；其余层带内（L0 165k、L3 91.7k） |
| 2026-10-30 | +attn | release | 170,091 | — | — | — | 1.14 / 2.64 | 0.042 | 1.69 | 0.040 | **paged_attn 块协作重写（访存攻坚三刀收官）**（battery-20261030-095232）：一线程一 (t,qh)（m=100 全 GPU 仅 3200 线程 + 每线程 256-f32 本地数组 + 加权 V 趟每 e 重走页表 128×）→ **一 block 一 (t,qh)（block=hd 线程）四相位协作**：A 记分 j 属主单线程 d 升序（u64 载 4×bf16）/ B max 纯比较冗余算 / C 分母线程 0 串行 j 升序 / D 加权 V 每 e 独立 j 升序 warp 合并——**逐元素累加序与 reference 全同，o12 位对齐零漂移（参考侧零改动零重生成）**；scratch 参数同 rms 模式（[16384,258] f32，T_MAX×qh 容量）。o24 金丝雀 1.083→0.031ms（35×，§7）。**L3q 满负荷 ITL 51.33→34.64 / tps 1724.7→2394.8（+39%），L3q+ 49.76→33.08 / 1776.2→2492.2**；轻载 20.62→14.32 / 18.97→12.84；**副产 L3（toy 同算子）tps 91.7k→112.0k（+22%）**；wein≡vLLM oracle 逐字节全同；其余层带内（L0 170k）。**三刀总账：m=100 步 143→34.82ms（4.1×）、满负荷 tps 574.1→2394.8（4.2×）、对 vLLM ×12→×2.9@c=100**；delta/gemm 双双贴各自带宽地板——下一杠杆换层（多流重叠/状态池精度，§7） |

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
| 2026-10-29 | +prof | 同上 | 步时分解段表（`WEIN_STEP_PROFILE` 事件插桩，m=100 纯 decode 20 步直驱，pos≤24） | delta **70.65**（53%）· norm **39.11**（29%）· gemm 11.54（8.7%）· attn 8.09（6%）· logits 1.55 ms/步 | **"每行 host 侧工作"假设证伪**：图档 139ms ≈ eager 143ms 已证纯 GPU 时间；gemm 11.5ms 载 8.4GB ≈ 730GB/s——cuBLASLt 满血非嫌疑人。账在 kernel 访存形态：deltanet 状态池 9.6GB/步（r+w）地板 ~14ms 实测 70.7（每线程串行扫 S 行、跨线程 512B stride 不合并）；rms 系 372µs/拍 vs µs 级工作量（row-per-thread 串行扫 2560）。三处压地板 → m=100 步 135 → ~30ms 级。插桩自知之明：事件 GPU 成本 3-4µs/次（步尾校准突发扣除），host 提交 ~50µs/次（m=4 段表含饥饿间隙，大 m GPU-bound 段表干净） |
| 2026-10-29 | +deltanet | 同上 | deltanet 单 launch 金丝雀（m=100×32 头×128² 状态 200MB，o22，门禁 1.5ms） | **3.077 → ~0.75 ms/launch（4.1×）** | 4 趟 6 访存（decay r+w / kvm r / 更新 r+w / 输出 r）融合 **3 趟**（kvm 读 + 更新读 + 写——decay 中间值不落 DRAM，更新趟同式重算位不变）+ `core::simd::f32x4` 向量加载（`ld/st.global.v4.b32` 实发射；kd%4==0 守卫 + 标量回退；cuMemAlloc 256B 对齐是前提）；3 趟 600MB @0.75ms ≈ 800GB/s **带宽打满**——shared-mem 保 S0 两趟变 r+w（400MB→~0.5ms 地板）复杂度/收益不取。v4 与标量化融合版同速（噪声带内 <10%），融合才是主账；位对齐零漂移由 o20/o16 + vLLM oracle 复验兜底 |
| 2026-10-29 | +deltanet | 同上 | 步时分解复测（同 +prof 口径）+ 批扩展曲线（`qwen-sweep-20261029-194247`） | delta 70.65→**17.22**（−75.6%）· norm **38.64 升任主账（49.7%）**· gemm 11.23 · attn 7.82 · 整步 143→**80.4ms** | 曲线：eager ITL 34.73/52.03/61.76/73.16/**90.45**/105.45ms（c=4→128，对旧曲线 −6%→−38%），c=100 tps 969、c=128 **1060 峰值**；图档收益 −2.3~-4.1% 带内；gemm/attn 分毫不动（改谁没改谁）；对 vLLM 差距 ×12→×7.5@c=100。下一刀账已现形：norm 系 warp-per-row（38.64ms）→ paged_attn（7.82ms） |
| 2026-10-29 | +rms | 同上 | rms 单 launch 时延金丝雀（o23：(100,2560) 与 (1600,128) zc+gated 各 50 拍均摊，门禁 60µs） | **692 → 11.5 µs/launch（60×）**；(1600,128) 12.7µs | row-per-thread（rows=100 时全 GPU 100 线程、串行扫 2560 元素）→ **block-per-row**：跨步合并访存（线程 t 占 {t,t+bdim,…}）+ 块归约。**工具链约束归档**（stdarch nvptx 跨线程通道三断）：① 无 shfl intrinsic；② `asm!` 未实现（E0658 + 无寄存器类，`#![feature(asm)]` 亦不破）；③ `#[link_section=".shared"]` 被 rustc 后端忽略——static 落 `.global` 模块级、并发 block 互踩（PTX grep 证实，未上机即废弃）。第四条路 = **kernel 参数携带 scratch**（`argmax_rows` 已验证同款）：op resolve 时预分配 `[8192,256]` f32（q-norm 满批 2048 行 ×4 余量；capture 期零分配），block 按 blockIdx 私有槽写部分和 → `_syncthreads`（CUDA 语义 = 块内含 global 的可见性屏障）→ 线程 0 顺序合并 u 升序 `total = s[u]+total`——reference.rs 同式同序标量镜像（归约序变更两侧同步，o16 位一致维持） |
| 2026-10-29 | +rms | 同上 | 步时分解复测（同 +prof 口径）+ 批扩展曲线（`qwen-sweep-20261029-205652`） | norm 38.64→**0.64**（−98.3%，105 打点/步全过带宽地板）· delta 17.17（地板带内）· gemm 11.39 · attn 7.90 · 整步 80.4→**42.71ms** | 曲线：eager ITL 23.80/34.35/36.99/43.81/**51.54**/56.90ms（c=4→128，对 deltanet 后曲线 −31%→−46%），c=100 tps 1725、c=128 峰 **1935**；图档 −3.7~−6.1%（批小处收益大——launch 债占比）；轻载 ITL 31→20.6（c=4 段表 m=4 时 norm 105 打点同病同治）；对 vLLM 差距 ×7.5→**×4.3@c=100**。段表剩账：delta 17.17（42.8% 带宽地板不动）· gemm 11.39（cuBLASLt 满血）· **attn 7.90（19.7%，带宽地板 ~0.13ms——60× 差距，下一刀 paged_attn 访存形态）** |
| 2026-10-30 | +attn | 同上 | paged_attn 单 launch 时延金丝雀（o24：tokens=100×qh32/kv16/hd128，pos=23 对位段表工作点 / pos=63 深序列，20 拍均摊，门禁 0.30/0.60ms） | **1.083→0.031 / 2.657→0.070 ms/launch（35×/38×）** | 旧形态一线程一 (t,qh)：rows=m 时全 GPU 仅 3200 线程（喂不饱 DRAM 延迟，同 rms 病根）+ `sc[256]` 1KB/线程本地数组 + 加权 V 趟 e-outer/j-inner 对每 e 重走页表与 p 读（128× 冗余槽位计算）。新形态一 block 一 (t,qh)（block=hd 线程）四相位：**A** j 属主单线程整条点积（d 升序逐元素同参考；u64 载 4×bf16——hd%4 守卫+标量回退，PTX 实发射 `ld.global.b64`）；**B** 各线程冗余扫 max（纯比较值同）；**C** 线程 0 串行 j 升序 `p_j + denom`（操作数序逐字同参考）；**D** 线程 τ=e 认领输出维 j 升序加权 V（warp 内连续 e → 每 j 一次 256B 合并载入；slot_of 从 128×pos/线程降到 pos/线程）。p/sc 广播走 scratch kernel 参数（rms 同款 [16384,258] f32 块私有槽 + 3× syncthreads）。实测 31/70µs vs 带宽地板 25/65µs——**贴线**；本地数组清零（PTX 零 .local） |
| 2026-10-30 | +attn | 同上 | 步时分解复测（同 +prof 口径）+ 批扩展曲线（`qwen-sweep-20261030-095214`） | attn 7.90→**0.35**（−95.6%，8 打点=8 full-attn 层）· delta 17.14（地板）· gemm 11.42（满血）· norm 0.66 · 整步 42.71→**34.82ms** | 曲线：eager ITL 14.23/17.53/20.97/27.40/**34.81**/39.83ms（c=4→128，对 rms 后曲线 −40%→−30%），c=100 tps 2349、c=128 峰 **2570**；图档 −3.1~−8.9%；轻载 ITL 20.6→14.3；对 vLLM 差距 ×4.3→**×2.9@c=100**。**访存攻坚三刀（deltanet/rms/paged_attn）收官总账：m=100 步 143→34.82ms（4.1×）、满负荷 tps 574→2395/图档 2492（4.2×）**。段表剩账双双贴各自带宽地板：delta 17.14ms = 24 层状态池 4.8GB r+w @~800GB/s；gemm 11.42ms = 8.4GB 权重读 @736GB/s——kernel 形态已到形态内极限，下一杠杆换层：**多流重叠**（delta 与 gemm 同流串行，理论步时 → max(17.1,11.4)+杂项 ≈ 22ms 级）或 **状态池 bf16 化**（4.8→2.4GB，数值合同须重开）或 PIECEWISE 图档 |
| 2026-10-30 | +overlap | 同上 | 多流重叠决定性实验（`overlap_probe`：o22 真形 delta 0.63GB/iter + 线性层全量 GEMM 簇 0.25GB/iter——qkv/z/a/b/out+MLP 与 8.4GB/步权重账对齐；各 3 热身 + 20 拍） | delta 单跑 808 / gemm 单跑 698 / serial 772 → dual **831 GB/s 聚合（省 7.1%，22.77→21.15ms）**；理想重排 15.6ms 未达 | **证伪（DRAM 单墙）**：delta 与 gemm 双双 HBM-bound，双流并发只把聚合带宽 772→831（1008 峰值 82%）——重叠换不来带宽，只换来调度形态；且 probe **无依赖约束，是上限估计**（exec 层内链 qkv→conv→delta→out→MLP 全串行，真可重叠窗更小），≤7% 对 exec 双流重构 + 图 fork/join 捕获的工程不成比例。**机制面归档不归零**：driver +`cuStreamWaitEvent`、`Stream::wait_event`（跨流 join 边），gr6 双测（fork/join 直发位一致 + **跨流图捕获**——s1 经事件 join 进 s0 的捕获区间、跨流边烤进拓扑、回放位一致）——昇腾/多墙设备走重叠时此路已验证。真杠杆顺位更新：① **状态池 bf16 化**（池 4.8→2.4GB、流量 14.4→7.2GB → delta 17.1→~8.6ms、步时 →~26ms 级/tps ~3170，数值合同须重开位对齐→容差）；② **GEMM 每形状钉 algo**（probe 副产：gemm 簇单跑仅 698 vs delta 808 GB/s——heuristic 差 13%，≈1.4ms 步时，厂商库容差合同内免费）；③ PIECEWISE 图档（混合步） |

launch 4.69µs 是 L3（exec 层）落地时的对照基线：纯 Rust→PTX 路线在 WSL 上的
单 kernel 提交成本在此，exec 步的图捕获/回放收益按此尺度衡量。
