# wein

纯 Rust 的 LLM 推理框架——从 device kernel 到 serving 全栈：kernel 以 Rust 直编 PTX（无 nvcc、无 CUDA C），Driver/cuBLASLt 全部 dlopen，产物是**单一静态二进制**。首个完整支持的模型是 Qwen3.5-4B（混合架构：24 层 gated DeltaNet 线性注意力 + 8 层 GQA 全注意力）。

在 RTX 4090 上与 vLLM（同权重、同 bf16 KV/递推状态、双方均无量化）对比如下（成对交错 3 轮中位，2026-11，口径与原始数据见 `bench-reports/final-report-20261109/`）：

| 指标 | wein | vLLM | 差 |
|---|---|---|---|
| 满载 c=100 tps | 4343 | 4425 | −1.9%（平价） |
| 满载 ITL p99 | **23.3ms** | 36.8ms | **−37%** |
| 满载 TTFT p99 | **~114ms** | ~200ms | **−43%** |
| 3009/6009 token 冷 TTFT | 641/1340ms | 603/1360ms | 平价 |
| 6009 token 热 TTFT（前缀命中） | **31ms** | 53ms | **−42%** |
| 模型声明口径（f32 递推状态） | 4343 | 3516 | **+24%** |
| Agent 多轮工具调用 c=4 前缀命中 | **0.694** | 0.665 | +4pt |

> vLLM 口径：`--mamba-ssm-cache-dtype bfloat16 --gpu-memory-utilization 0.85`（27 档扫参后的最优配置）；wein 口径：`--graph --ctx-len 32768 --kv-pages 4096 --chunk-size 2048`。

## 特性

- **kernel 全 Rust**：PTX 由 rustc 直出，含 wmma/mma.sync tensor core、cp.async 双缓冲、shared memory、warp shuffle；后端缺口（smem 段、cp.async、L2 预取）由 build.rs 的 PTX 改写管线补齐（`crates/wein-kernel/build.rs`）
- **deltanet chunked 递推**（BT=64 分块流水，wmma tensor core）与 **attention v9**（手写 PTX：ldmatrix + mma.sync + 99KB 动态 smem 双缓冲）/ v5（wmma 大瓦片）/ online（单遍在线 softmax）三档按批形接管
- **CUDA Graph**：FULL_DECODE 桶图回放 + 混合批分段路径
- **radix 前缀缓存**（递推状态快照锚点，16-token 对齐）+ **DRAM 分层**（块副本 128K token + 状态槽，host pinned，容量轴扩到显存之外）
- **MTP 投机解码**（opt-in，低并发门控，接受率自适应）与**设备端采样器**（temperature/top-k/top-p，图内 RNG；greedy 与 argmax 逐位同）
- **OpenAI 兼容 API**：`/v1/chat/completions`（SSE 流式 + tool calls）、`/v1/completions`、`/v1/models`、`/health`、`/metrics`（Prometheus）、`/reset_prefix_cache`；SIGTERM drain、结构化错误
- 32K 上下文端到端（6009 token 冷启动与 vLLM 平价，热恢复快 42%）

## 构建

依赖：Rust nightly（`rust-src` + `llvm-bitcode-linker` 组件；nvptx64 目标经 build-std 编 core）+ 运行时 NVIDIA 驱动（`libcuda.so.1`；WSL 下在 `/usr/lib/wsl/lib`）。**无需 CUDA toolkit**——PTX 由 rustc 直出，驱动 JIT；cuBLASLt 运行时 dlopen。

```bash
rustup toolchain install nightly --component rust-src --component llvm-bitcode-linker
cargo build --release --workspace
```

## 运行

```bash
./target/release/wein-cli --port 8191 \
  --engine core --exec qwen \
  --model /path/to/Qwen3.5-4B \
  --token-delay-ms 0 --graph \
  --ctx-len 32768 --kv-pages 4096 --chunk-size 2048
```

```bash
curl -N localhost:8191/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.5-4b", "messages": [{"role": "user", "content": "你好"}],
  "max_tokens": 64, "stream": true
}'
```

采样参数：`temperature` / `top_k` / `top_p`（缺省 greedy，与 argmax 逐位一致）。

## Workspace

| crate | 职责 |
|---|---|
| `wein-platform` | 硬件抽象层（vendor 零依赖探测、能力画像、降级声明） |
| `wein-kernel` | 算子层：`kernels/` 子 crate（Rust→PTX）+ driver/cuBLASLt dlopen 绑定 + Op 注册表 |
| `wein-exec` | 执行器：Qwen3.5 真权重前向、CUDA Graph 捕获/回放、形状调优 |
| `wein-memory` | KV 页池、radix 前缀缓存、递推状态快照、DRAM 分层 |
| `wein-engine` | 步循环引擎、请求状态机、快照恢复 |
| `wein-sched` | 连续批调度（混批时间预算、chunked prefill、抢占重算） |
| `wein-serve` | OpenAI 兼容 HTTP/SSE、健康指标、drain |
| `wein-model` | safetensors 装载、chat 模板（minijinja） |
| `wein-bench` | 压测客户端（flood/battery/长上下文/Agent 多轮） |
| `wein-cli` | 入口 |

## 工程方法

- **TDD 外→内**：验收用例先于实现；kernel↔CPU reference 位级对齐优先，结构性换形走容差档 + oracle 复验
- **每一刀优化都有假设—测量—证伪/落地记录**：`bench-reports/`（含 battery 归档与 A/B 原始数据），M0 起的完整工程日志在 `docs/M0-LOG.md`
- 代表性证伪留档：多流重叠（DRAM 单墙）、自写 skinny GEMM（流量代数 2× 慢于 cuBLASLt）、调度层轮转/时间闸门（FIFO 错峰结构性占优）

## Roadmap

- 昇腾 NPU 后端（`wein-platform` 双平台轴已备）
- W8A8 量化（对称对比：对手开 fp8 时）
- MTP 默认开启策略

## License

Apache-2.0
