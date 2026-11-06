# 服务级差距解剖：kernel 之外那 4%（2026-11-06）

对象：c=100 连续到达的稳态服务口径（battery L3q full 同款负载），
HEAD = 296a4be（= 834ff80 + kernel 级账报告）。

## 0. 结论摘要

1. 同 harness 同 n 口径：**wein 4105.2 tps vs vLLM 4305.1 tps（−4.6%）**，
   ITL 均值 23.10 vs 20.58。此前 battery 的 4015 是 n=500（5 波）短跑，
   被爬坡主导，不能和 vLLM 的 n=4000 直接比。
2. 稳态服务端 **GPU 利用率 91.9%**，每步 21.4ms 里 19.7ms 在干活，
   **1.7ms 是空档**。同一模型同一批量的进程内探针（qwen_launch_map）
   空档只有 0.19ms ⇒ **差距不在 kernel，在服务步的组织**。
3. 每步 888 个 GPU 活动：711 kernel + **141 memset（45.3MB）** +
   **36 D2D memcpy（19.5MB）** + 7 H2D + 1 D2H（采样回传）。
   memset/D2D 的主要来源是**状态池行的回收/领用零化**，而且回收零一遍、
   领用再零一遍（代码注释自称"幂等"）。
4. 单请求短 prompt（~40 token）TTFT = **44.0ms**（1.1ms/token），
   而 c=100 批量 prefill 是 14.5µs/token（75×）⇒ 小 m prefill 走 eager，
   732 发发射/固定开销主导。

## 1. 同口径纠正（第一个发现：之前的对比口径不齐）

| 负载 | tps | ITL 均值 | ITL p50 | TTFT p50 |
|---|---|---|---|---|
| wein c=100 n=500 max_tok=32 | 4033.3 | 22.48 | 21.18 | 61.4 |
| wein c=100 n=**4000** max_tok=32 | **4105.2** | 23.10 | 21.34 | 58.3 |
| wein c=100 n=600 max_tok=180 | 3829.6 | 22.01 | 21.65 | 46.5 |
| vLLM（vs-vllm-kernels2，n=4000） | **4305.1** | 20.58 | 19.10 | 101.9 |

- vLLM 那一发是 n=4000（40 波，接近稳态），我们过去引用的 4015 是 n=500。
  补齐到 n=4000 后是 4105，差 4.6%（不是 6.6%）。
- 我们把 TTFT 赢下来（58 vs 102），代价是长跑 tps 输 4.6%。
- max_tokens 拉到 180（到达速率降 5.6×）后 ITL 均值只降到 22.0 —— 说明
  prefill 混流只贡献约 1.1ms/步，**还剩约 1.8ms/步与 prefill 无关**。

## 2. 稳态时间轴（nsys，c=100 n=600 max_tokens=180，`--cuda-graph-trace=node`）

稳态窗 9532ms / 445 步 / 399080 个活动：

| 项 | 每步 |
|---|---|
| 步周期 | 21.4ms |
| GPU busy（全部活动） | 19.7ms（util 91.9%） |
| **空档** | **1.74ms** |
| kernel | 711.5 发 |
| memset | **141.0 发 / 45.3MB** |
| memcpy D2D | **35.6 发 / 19.5MB** |
| memcpy H2D | 6.8 发 / 16KB |
| memcpy D2H | 1.02 发 / 2KB（采样 token 回传） |
| argmax | 2.05 发（p1 + p2 两级） |

空档归属（按前后活动聚合，>0.05ms）：

| 空档 | 次数 | 合计 | 每次 |
|---|---|---|---|
| D2H → H2D（主机在等采样结果 + 做步间工作） | 225 | 76.6ms | 0.34ms |
| memset → memset（零化之间主机跟不上） | 352 | 47.5ms | 0.135ms |
| argmax → D2H | 350 | 26.0ms | 0.074ms |
| D2D → H2D | 65 | 20.1ms | 0.31ms |
| D2D → D2D | 131 | 16.8ms | 0.13ms |

细缝：392527 个 <0.05ms 的空档合计 496.9ms = **1.12ms/步**（平均 1.27µs/发）。
进程内探针同一批 730 发的细缝只 0.26µs/发 —— 服务步里有大量 **eager 活动
夹在 graph replay 里**，把流水打断（主机逐发提交）。

## 3. 141 次 memset 的来源（临时计数器取证，已回退补丁）

给 `DrvBuf::zero_on` / `zero_range_on` 加 `WEIN_ZDBG` 计数，跑 c=100 n=300
max_tokens=60 后统计：

```
12264 zero_range on 134217728 len 1048576     ← 状态池 24 层 × 1 行
12264 zero_range on   6291456 len   49152     ← conv 池 24 层 × 1 行
   24 zero_on 134217728 / 6291456 / 3145728 / 67108864   ← 启动期各池整块清零
```

- `134217728` = 128MB = 128 行 × **1MiB**（`vh·kd·vd·2` = 32·128·128·2 恰好 1MiB），
  24 个缓冲 = 24 个 linear 层的状态池（共 3GB）。
- `6291456` = 6MB = 128 行 × **48KiB**（`conv_dim·(k−1)·2` = 8192·3·2 = 49152），
  24 个 = conv 状态池。
- 所以一次 `zero_row_state(row)` = **48 发 memset / 25.2MB**。
- 每次请求命中两次：`released()` 里零一遍（qwen.rs:4834/4837），
  `admitted()` 里领用时再零一遍（qwen.rs:4334，注释自称"幂等"）。
- 12264 / 48 = 255.5 次 `zero_row_state` / 约 240 步 ≈ 1.5 次请求流转/步 ⇒
  **每步 70~140 发 memset、38MB 写带宽**，与时间轴实测 45.3MB/步吻合。

直接代价：≈38MB/900GB/s = 42µs 带宽 + 70~140 发 eager 发射（每发 1~2µs）
≈ **0.15~0.25ms/步**。间接代价更贵：这些 eager 发夹在 graph replay 之间，
正是 3 节里 1.12ms/步细缝的一部分。

## 4. 单请求短 prefill 的 44ms

| 场景 | 每 token |
|---|---|
| c=1，~40 token prompt（wein-bench TTFT p50 43.97ms） | 1.10ms |
| c=100 批量 prefill（首波 100 请求 / 4000 token，TTFT p50 58.3ms） | 14.5µs |

75× 的差 ⇒ 小 m prefill 走 eager：一步 732 发，主机发射主导
（m=40 的 gemm 本身只有几百 µs）。连续到达时每步混进 0.23~1.3 个小 prefill，
这部分是 2 节里"prefill 混流 ≈1.1ms/步"的来源。

## 5. 修缮清单（按性价比，全部位级安全）

| # | 措施 | 预期 | 风险 |
|---|---|---|---|
| 1 | 去掉 `released()` 侧的 `zero_row_state`（领用侧已零） | ✓ 每步 −35 发 / −19MB，−0.1~0.15ms | 先确认"链 arena 直接借 free_rows"路径是否依赖回收侧零化 |
| 2 | 状态池按层合并成一个缓冲（或写一个跨缓冲行零 kernel），48 发 → 1~2 发 | −0.1~0.2ms | 中（池布局/快照原语要同步改） |
| 3 | 采样 D2H 与下一步重叠（token 闭环留在 GPU 侧，主机结果迟一步取） | −0.3~0.5ms | 中高（引擎步循环 + 停止条件时序） |
| 4 | 小 m prefill 上 CUDA 图（桶 1/2/4/8/16/32/64），或把 prefill 的 732 发压到 ~1/3 | 混流步 −1ms、TTFT 同降 | 中高（prefill 路径有 chunk/块表/投机分支） |
| 5 | 141→~50 发后复核细缝（1.12ms/步有多少是发射引起） | 待测 | — |

净潜力：**0.4~0.9ms/步（2~4%）**，即 4105 → 4250~4400，对 vLLM 4305 打平或反超。
这条路线**不需要破任何数值合同**，优先级高于 norm 归约树（0.46ms，需批准）。

## 6. 复现

```
# 服务端（图档）
./target/release/wein-cli --port 8211 --engine core --exec qwen \
  --model ~/.cache/wein/Qwen3.5-4B --token-delay-ms 0 --graph
# 同口径负载（与 vLLM 侧一致）
./target/release/wein-bench --url http://127.0.0.1:8211 --model ~/.cache/wein/Qwen3.5-4B \
  --concurrency 100 --num 4000 --max-tokens 32 --warmup
# 时间轴（图内 kernel 必须 --cuda-graph-trace=node）
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none -o /tmp/svc ...
nsys stats --report cuda_gpu_trace --format csv /tmp/svc.nsys-rep
```
