# wein

国产硬件上的 vLLM+SGLang：纯 Rust 核心 + PTX（NVIDIA）/ AscendC（昇腾）双平台算子层。

设计文档见 [`../design/`](../design/)（01-strategy / 02-architecture）。
开发方法：**TDD 外→内**——验收用例先于实现，从 serve 层向内核层推进。

## 当前状态（M0 第七层：exec 真身——DeviceSession + toy 模型 + cuBLASLt）

- `wein-serve`：OpenAI-compatible HTTP/SSE（chat/completions + completions + models），增量 detokenizer（UTF-8 边界安全），stop 参数截断 + 引擎取消，断连自动取消；chat 模板经 `Tokenizer::render_chat_template` 钩子注入（W5 真身，无模板回退占位），eos 停止走引擎 stop_token_ids
- `wein-sched`：统一 token 预算调度器（纯逻辑零依赖）——无 prefill/decode phase（token 债统一建模）、chunked prefill、抢占 = recompute（债 = prompt + 已生成）、FIFO 准入、BatchBuffer 增量 op、DeferredRelease；容量侧可插拔（`CapacityModel`，默认聚合 token 模型）
- `wein-engine`：边界协议（Submit/Cancel + 事件流）+ **EngineCore**（调度器 + 可插拔 `Executor` 组装）+ MockEngine（对照组）；`EngineStats` 为 L2/L2m 插桩载体
- `wein-memory`：paged KV 布局 / BlockPool（LIFO + rc + DeferredRelease + 水位）/ radix 前缀缓存（pin 驱逐免疫，LRU 级联驱逐）/ req_to_token / `MemState` 容量语义——零依赖 crate，经引擎侧 `PoolCapacity` 适配器接入 `CapacityModel` 钩子
- `wein-kernel`：cuda 冒烟（§13-2 三线全过）——`kernels/` 子 crate 纯 Rust→PTX（nightly + abi_ptx/stdarch_nvptx feature，无 nvcc）+ dlopen(libcuda) 最小 Driver API 面（零 toolkit 依赖）+ 高层 Gpu/Module/Func/Stream/DevBuf；**trait Op 注册表**（§4）：启动时按 `HardwareProfile` resolve 一次 → `OpTable` 具体类型槽位（无 Box<dyn>/字符串查表），失败 = Degrade 三元组；**M0 算子清单七算子齐**（rms_norm/rope/silu/W8A8 动态量化/KV scatter/varlen paged attention 全 GPU↔CPU golden 位对齐；**GEMM 走 cuBLASLt**（§4 标准算子走厂商库）：dlopen 零构建依赖、形状粒度 algo 缓存、容差对齐模式）+ CPU 参考实现（`reference`，独立代码路径）+ golden vector 框架（每算子一文件，cuda/asc 共享，§4）；**exp/RNE 自控数值路径**（exp2 多项式、round-half-even 位实现——禁 MUFU 近似与默认 round）
- `wein-platform`：三层能力模型（§3）——`Device` PCI vendor 零依赖探测（无任何 vendor SDK）/ `HardwareProfile` 不可变画像（supports + block_size 平台注入：CUDA 16 / 昇腾 128）/ `Degrade` 声明式降级三元组（不隐式硬改配置的类型化载体）
- `wein-exec`：exec 层真身——`DeviceSession`（探测→画像→`OpTable::resolve`→流，一次装配全程只读）+ `ToyExecutor`（单块 toy 模型：M0 算子清单全走真路径、KV 进 BlockPool 分页池、host 贪心 argmax；实现 engine 的 `Executor`，组装经 `EngineCore::spawn_with` 发生在 cli 层）+ 独立 CPU 镜像（x2 数值诚实对照端）；调度合同钉死：追平步段尾采样 / 纯 decode 前向 seq 尾 / 空 prompt 落 BOS / 抢占清前向游标重放
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

# 性能基线电池（归因阶梯 L0/L1/L1'/L2/L2'/L2m/L3，报告入 bench-reports/battery-*/，方法论见 bench-reports/BASELINE.md）
bench/battery.sh release
```

## TDD 推进路线（外→内，每层验收先行）

1. **serve** ✅：MockEngine 钉死边界协议（24 验收用例）
2. **engine core** ✅：wein-sched 调度器（8 同步验收）+ EngineCore 协议验收（3）+ serve 套件换引擎 parity 复跑（8）——同一协议，行为零漂移
3. **memory** ✅：wein-memory（布局/BlockPool/radix/req_to_token/MemState，验收 m1-m10）+ `CapacityModel` 钩子接入（引擎适配器 + im1-im4 集成验收 + serve 套件 parity 复跑 p9）——抢占触发从聚合 token 模型换成 BlockPool 水位，调度策略零改动
4. **kernel**（冒烟 ✅，trait Op ✅，exec 接入待做）：cuda crate 冒烟实验完成——验收 k1-k8 全绿（PTX 产出物/指令降落/驱动探测/显存往返/vec_add 位对齐/**bf16 点积位对齐**/JIT 缓存跨进程复用/launch 4.69µs）；**风险登记**：路线锁 nightly + 两个未稳定 feature（abi_ptx #38788、stdarch_nvptx #111199），不达标时启用 cubin 旁路（架构不变）
5. **trait Op 注册表** ✅：`OpTable` 启动时 resolve（arch 档位裁决，o1/o3 确定性；o2 无卡 Err = Degrade 三元组）+ 首算子 `rms_norm`（同一 kernel 双 launch 档位，o5 两档 golden 位一致）+ golden vector 框架（o6 缺失生成/回读对拍/漂移须显式重生成；asc 接入复用同一文件）+ `wein-platform` 最小面（pf1-pf3）；**数值对齐纪律**：只正确舍入原语、行内顺序累加、bf16→f32 精确——o4 GPU↔CPU 参考位对齐一次达成
6. **M0 算子清单** ✅（o7-o13）：五算子齐入表——rope（half-split，cos/sin 表为输入图捕获友好）/ silu（sigmoid 走**自控 exp2 多项式**，两侧同字面量同 Horner 序）/ W8A8 动态量化（逐 token amax→RNE 半偶，i8+f32 scale 双输出位对齐）/ KV scatter（页表寻址跨块写，与 memory 层 token 记账同构；"未触及槽位不碰"语义在对拍中钉死）/ varlen paged attention（GQA、因果、两遍法 correctness-first，W2 flash 分块）；exp/RNE 均自控实现（禁 MUFU 近似/默认 round）——**o8/o9/o12 位对齐一次达成**；k2 抓住运行数整除的 panic 桩 call（换 unchecked_div/rem）与数组下标检查（换裸指针）
7. **cuBLASLt 最小 FFI** ✅（g1-g6）：dlopen libcublasLt（pip wheel 提取到 `vendor/`，`WEIN_CUBLASLT` 可覆盖）+ bf16 入出/f32 累加（tensor core）+ 每形状 desc/layout/heuristic 解析一次缓存 + 库缺 = Degrade；**容差对齐**（f64 参考 ± 1 bf16 ulp，实测最差 69% 容差占用）+ 重跑位确定 + 2048³ 吞吐 canary **125.6 TFLOPS**（BASELINE §7）——golden 框架自此两模式：自控算子位对齐 / 厂商库算子容差
8. **exec 真身** ✅（x1-x7）：三骨架算子（embed gather / 残差 add / 门控 mul，o14 位对齐）补齐十槽；`ToyExecutor` 把 OpTable + cuBLASLt + BlockPool 分页 KV 接进引擎（`spawn_with` 组装在 cli，依赖单向无环）；**x2 设备↔CPU 镜像逐位相等**（K=128 的 TC f32 累加误差被 bf16 网格吸收，容差 0% 占用）；x3 双执行器确定性；x4 位置敏感（rope 真接线）；x5 跨页 KV（40 token → 3 页）；x6 引擎集成零漂移 + 取消 Aborted；x7 步时延金丝雀（批 4 decode 中位 ~1.3ms，对照 launch 4.69µs——exec 层优化账本翻开）。**真发现 ×2**：① CUDA context 线程局部——engine 任务跨 worker 迁移须 `bind_thread`；② `step_delay=0` 时 `sleep(ZERO)` 立即就绪不 yield，单线程 runtime 里引擎任务饿死同行任务（x6 用 multi_thread flavor 钉死）
9. **wein-model + Qwen3.5 竖切** ✅（q1-q11，I1）：`wein-model`（config/safetensors 惰性按层加载 + HfTokenizer，q1-q4）+ kernel 增补 8 入口（rope_pt/sigmul/rms_norm_zc/qk-norm 同族、conv1d_dw+conv_state_roll、deltanet、rms_norm_gated、argmax_rows，o15-o21/g7）+ **Qwen3.5-4B 混合骨干真权重执行器**（24 linear-attn gated-deltanet 层 + 8 full-attn GQA 层，bf16 全链路分位舍入与 HF 对齐；分页 KV / 递推状态池 / conv 状态池按请求行分账，行回收零化）——q5 逐步 logits 对独立 f64 CPU 镜像（容差 81% 占用 + 贪心决策逐位一致）、q6 引擎零漂移、q7 真文本 "→ Paris"、q8 状态隔离、q9 decode 33.7ms/步、q10 行回收、q11 serve 端到端（HTTP → Paris）；battery **L3q** 归档（BASELINE §6）。**真发现 ×3**：① cuBLASLt 按 m 选 algo——跨批组成 ulp 漂移可翻低边际贪心位（q8 改同型对拍，W2 每形状钉 algo 后收紧）；② conv 状态滚动 kernel 拿 span 序号当状态行号——单请求永行 0 故 q5/q7/q10 全绿，双胞胎实验定位后 o19 升级非平凡 row_id 间接钉死；③ 零化区间越界（池字节 vs 单行字节）q10 现形
10. **W2 图捕获** ✅（gr/w/qg 系列，NONE→FULL_DECODE）：机制层 `Stream::begin/end_capture` + `Graph/GraphExec`（Driver API，exec update 参数重绑就位）+ **捕获区间守卫**（debug 构建捕获中隐式同步/分配直接 panic——ZoS `forbid_cuda_sync` 的 Rust 化，gr5）；gr1 回放≡直发位一致、gr2 重放读活输入、gr3 exec update 重绑、gr4 cuBLASLt 入图（workspace 常驻 + 形状缓存命中即纯 launch）。exec 层 **FULL_DECODE 图档**：桶集合共用单输入存储（常驻 DevBuf 每步整段覆写，各桶图录 narrow view）、纯 decode 步路由 ≥m 最小桶回放、批尾 **sink 行**（块表末行+专属页+专属状态行）吸收 padding、混合步 eager——toy（w1-w4：位一致/sink 吸收/1.29× 步时延/引擎零漂移）+ Qwen3.5-4B（qg1-qg3：10 token 逐位对齐、m=3→桶 4 零漂移、**decode 25.1ms vs eager 26.9ms（−6.7%）**）；`--graph` CLI 旗标 + battery **L3q+** 图档 suite（BASELINE §7）。**真发现 ×2**：① THREAD_LOCAL 捕获下并行线程的隐式同步（legacy stream）会击穿捕获态（906→901）——图测试全串行、引擎集成独立二进制；② 桶共享输入存储是 **b_max 整段覆写**语义——按选中桶组数组会在小桶回放时上传失配（w4 抓获）
11. **W5 chat 模板真身** ✅（ct/cs/e4/q12）：`wein-model::ChatTemplate`——tokenizer_config.json `chat_template`（chat_template.jinja 文件回退）→ **minijinja 3.0** 渲染（HF 同款环境：trim_blocks/lstrip_blocks + raise_exception global；**Python str 方法仿真层**：HF 模板按 Jinja2+Python 书写，startswith/endswith/split/rstrip/lstrip 经 `set_unknown_method_callback` 补齐——llama.cpp minja 同思路），golden 对拍 Python jinja2 六用例逐字节一致（单 user/system 前置/多轮/历史 `<think>` 剥离/enable_thinking=false/无 user 拒绝 400）；serve `Tokenizer` trait 增 `render_chat_template`/`eos_id` 钩子（默认 None——mock 路径零漂移，桥接在 cli 层 §2 单向）；引擎 `SamplingParams.stop_token_ids`（vLLM 语义：命中不外吐 + Finished(Stop)，决策在 token 发射处——sched 看到 token 时它已出站）；chat/completions 两端点默认停于 eos。cs1-cs4（模板进引擎/拒绝 400/eos 提前停）+ e4（事件序不变量扩展）+ q12（真模型端到端：chat prompt_tokens == encode(模板渲染) 精确对齐）
12. 后续：满负荷 L3q profile（批界扩展后）、PIECEWISE 图档（§7 谱系延伸）、tools/tool_calls 消息分支（模板已支持，serve DTO 未开）

## 规约

- 依赖只许向下：`serve/cli → engine → {sched, memory} → kernel → platform`；exec 实现 engine 的 trait（exec→engine 单向），组装在 serve/cli 层（02-architecture §2 无环）
- 引擎边界只经 `EngineHandle::{submit, cancel}` + 事件流，禁止旁路
- 每个集成里程碑跑一次 `wein-bench` 并把 JSON 报告归档
