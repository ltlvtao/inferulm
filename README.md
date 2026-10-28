# wein

国产硬件上的 vLLM+SGLang：纯 Rust 核心 + PTX（NVIDIA）/ AscendC（昇腾）双平台算子层。

设计文档见 [`../design/`](../design/)（01-strategy / 02-architecture）。
开发方法：**TDD 外→内**——验收用例先于实现，从 serve 层向内核层推进。

## 当前状态（M0 第五层：trait Op 注册表 + wein-platform 最小面）

- `wein-serve`：OpenAI-compatible HTTP/SSE（chat/completions + completions + models），增量 detokenizer（UTF-8 边界安全），stop 参数截断 + 引擎取消，断连自动取消
- `wein-sched`：统一 token 预算调度器（纯逻辑零依赖）——无 prefill/decode phase（token 债统一建模）、chunked prefill、抢占 = recompute（债 = prompt + 已生成）、FIFO 准入、BatchBuffer 增量 op、DeferredRelease；容量侧可插拔（`CapacityModel`，默认聚合 token 模型）
- `wein-engine`：边界协议（Submit/Cancel + 事件流）+ **EngineCore**（调度器 + 可插拔 `Executor` 组装）+ MockEngine（对照组）；`EngineStats` 为 L2/L2m 插桩载体
- `wein-memory`：paged KV 布局 / BlockPool（LIFO + rc + DeferredRelease + 水位）/ radix 前缀缓存（pin 驱逐免疫，LRU 级联驱逐）/ req_to_token / `MemState` 容量语义——零依赖 crate，经引擎侧 `PoolCapacity` 适配器接入 `CapacityModel` 钩子
- `wein-kernel`：cuda 冒烟（§13-2 三线全过）——`kernels/` 子 crate 纯 Rust→PTX（nightly + abi_ptx/stdarch_nvptx feature，无 nvcc）+ dlopen(libcuda) 最小 Driver API 面（零 toolkit 依赖）+ 高层 Gpu/Module/Func/Stream/DevBuf；**trait Op 注册表**（§4）：启动时按 `HardwareProfile` resolve 一次 → `OpTable` 具体类型槽位（无 Box<dyn>/字符串查表），失败 = Degrade 三元组；**M0 算子清单六算子齐**（rms_norm/rope/silu/W8A8 动态量化/KV scatter/varlen paged attention，全部 GPU↔CPU golden 位对齐）+ CPU 参考实现（`reference`，独立代码路径）+ golden vector 框架（每算子一文件，cuda/asc 共享，§4）；**exp/RNE 自控数值路径**（exp2 多项式、round-half-even 位实现——禁 MUFU 近似与默认 round）
- `wein-platform`：三层能力模型（§3）——`Device` PCI vendor 零依赖探测（无任何 vendor SDK）/ `HardwareProfile` 不可变画像（supports + block_size 平台注入：CUDA 16 / 昇腾 128）/ `Degrade` 声明式降级三元组（不隐式硬改配置的类型化载体）
- `wein-cli`：`wein serve --mock` 起 mock 服务
- `wein-bench`：并发性能验收工具（TTFT/ITL/吞吐 + smoke 阈值门禁）

## 常用命令

```bash
cargo test -p wein-serve          # 验收套件（A 契约 / B 并发与 abort）
cargo test                        # 全部（含单元测试）

# 手工体验
cargo run -p wein-cli -- --mock --port 8080
curl -N localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}'

# 并发性能验收（今天打 mock，之后打真实引擎——同一把尺子）
cargo run -p wein-bench -- --url http://127.0.0.1:8080 --concurrency 100 --num 200 \
  --max-tokens 32 --repeat 3 --warmup \
  --smoke "ttft_p99_ms<=100;itl_p99_ms<=20;tps_min>=5000;errors_max<=0"

# 性能基线电池（归因阶梯 L0/L1/L1'/L2/L2'/L2m，报告入 bench-reports/battery-*/，方法论见 bench-reports/BASELINE.md）
bench/battery.sh release
```

## TDD 推进路线（外→内，每层验收先行）

1. **serve** ✅：MockEngine 钉死边界协议（24 验收用例）
2. **engine core** ✅：wein-sched 调度器（8 同步验收）+ EngineCore 协议验收（3）+ serve 套件换引擎 parity 复跑（8）——同一协议，行为零漂移
3. **memory** ✅：wein-memory（布局/BlockPool/radix/req_to_token/MemState，验收 m1-m10）+ `CapacityModel` 钩子接入（引擎适配器 + im1-im4 集成验收 + serve 套件 parity 复跑 p9）——抢占触发从聚合 token 模型换成 BlockPool 水位，调度策略零改动
4. **kernel**（冒烟 ✅，trait Op ✅，exec 接入待做）：cuda crate 冒烟实验完成——验收 k1-k8 全绿（PTX 产出物/指令降落/驱动探测/显存往返/vec_add 位对齐/**bf16 点积位对齐**/JIT 缓存跨进程复用/launch 4.69µs）；**风险登记**：路线锁 nightly + 两个未稳定 feature（abi_ptx #38788、stdarch_nvptx #111199），不达标时启用 cubin 旁路（架构不变）
5. **trait Op 注册表** ✅：`OpTable` 启动时 resolve（arch 档位裁决，o1/o3 确定性；o2 无卡 Err = Degrade 三元组）+ 首算子 `rms_norm`（同一 kernel 双 launch 档位，o5 两档 golden 位一致）+ golden vector 框架（o6 缺失生成/回读对拍/漂移须显式重生成；asc 接入复用同一文件）+ `wein-platform` 最小面（pf1-pf3）；**数值对齐纪律**：只正确舍入原语、行内顺序累加、bf16→f32 精确——o4 GPU↔CPU 参考位对齐一次达成
6. **M0 算子清单** ✅（o7-o13）：五算子齐入表——rope（half-split，cos/sin 表为输入图捕获友好）/ silu（sigmoid 走**自控 exp2 多项式**，两侧同字面量同 Horner 序）/ W8A8 动态量化（逐 token amax→RNE 半偶，i8+f32 scale 双输出位对齐）/ KV scatter（页表寻址跨块写，与 memory 层 token 记账同构；"未触及槽位不碰"语义在对拍中钉死）/ varlen paged attention（GQA、因果、两遍法 correctness-first，W2 flash 分块）；exp/RNE 均自控实现（禁 MUFU 近似/默认 round）——**o8/o9/o12 位对齐一次达成**；k2 抓住运行数整除的 panic 桩 call（换 unchecked_div/rem）与数组下标检查（换裸指针）
7. 逐层替换 mock，验收套件始终不变——后续：cuBLASLt 最小 FFI（GEMM）、`Executor` 从 ScriptedExecutor 换真实 exec 线程、Qwen3.5 竖切（I1）

## 规约

- 依赖只许向下：`serve → engine → {sched, memory, exec, model} → kernel → platform`（02-architecture §2）
- 引擎边界只经 `EngineHandle::{submit, cancel}` + 事件流，禁止旁路
- 每个集成里程碑跑一次 `wein-bench` 并把 JSON 报告归档
