# vLLM 长上下文基线 · 与 wein 并排对照

2026-10-31｜只测量，未改 wein 任何代码。GPU 一次一进程（起服前确认无 wein/vLLM 进程，跑完杀进程组并等显存回落）。
wein 侧读数**未重测**：本轮值由调用方提供，上一轮基线取 `bench-reports/battery-20261031-134113/` 与
`bench-reports/ctx-a-20261031/`（`A-DELTANET-PREFILL.md`）。

## 1. 配置清单（全部取自运行期权威口径 `vllm:cache_config_info` + 启动日志）

| 项 | 值 |
|---|---|
| GPU / 驱动 | NVIDIA GeForce RTX 4090 24GB / 610.88（单卡，无 MIG） |
| vLLM / torch | 0.32.0 / 2.13.0+cu130（V1 引擎，`EngineCore` 单进程） |
| 模型 | `~/.cache/wein/Qwen3.5-4B-text`（滤视觉塔的文本骨干副本，bf16），`--hf-overrides {"architectures":["Qwen3_5ForCausalLM"]}` |
| `--max-model-len` | **8192**（另跑一档 512 = 旧基线口径，仅短上下文，见 §2 注） |
| `--max-num-batched-tokens` | **2048**（= vLLM 对 24GB 非 H100 + OpenAI server 档的内置默认，`arg_utils.py:2991-3003`；显式钉住以便记录） |
| `--max-num-seqs` | 256（同上，默认值） |
| `--gpu-memory-utilization` | 0.85 → 可用 KV 10.33 GiB / **238,312 token** / `num_gpu_blocks=640` |
| `--enable-prefix-caching` | **True（默认）** |
| mamba cache align | **align（默认）**：`Mamba cache mode is set to 'align' ... by default when prefix caching is enabled` |
| `--mamba-ssm-cache-dtype` | **auto → float32**（模型 config `mamba_ssm_dtype: float32`）；bfloat16 为**非默认**档，见 §4 |
| 缓存块几何 | `block_size=528`（注意力页抬到与 mamba 页等大，`hash`/`mamba_block_size=528`） |
| 其它默认 | chunked prefill **on**、`cudagraph_mode=FULL_AND_PIECEWISE`（51 桶，≤512）、`compilation mode=VLLM_COMPILE`（torch.compile/inductor，compile range 1–2048） |
| 环境修补 | 沿用 `bench/vs_vllm.sh`：`unset *_proxy`、剥 `/mnt/c` 的 CUDA、`VLLM_USE_FLASHINFER_SAMPLER=0` |
| 仅本次加的 | `VLLM_SERVER_DEV_MODE=1`——只为 `POST /reset_prefix_cache`（冷跑前清缓存），不进引擎性能路径 |
| 命令 | 每档落在 `<cfg>/cmd.txt`；日志 `<cfg>/server.log`；运行期配置 `<cfg>/cache-config.txt` |
| 版本控制 | wein `git 1da2b26`；启动到就绪 ≈ 107 s（含 12.6 s torch.compile + 40 s warmup + 13 s 图捕获） |

Prompt 用 tokenizer 精确建：目标 3000 / 6000 token 的纯文本段落填充，实测 `prompt_tokens` = **3009 / 6009**
（多出的 9 个是 chat 模板）。贪心 `temperature=0`、`max_tokens=32`、流式，单请求。
每项 **3 次取中位**；冷跑每次前后抓 `/metrics` 证明 `prefix_cache_hits=0`。

## 2. 表 1 · 短上下文 sanity（wein-bench 同尺，c/n 与旧基线对齐）

| 指标 | vLLM（本次，3 次中位） | wein 本轮（图档） | wein 上一轮 `battery-20261031-134113`（图档） | 备注 |
|---|---|---|---|---|
| 轻载 c=4 n=8 mt=12 · ITL p50 | **12.19 ms** | 12.60 ms | 12.72 ms | vLLM 低 3.3%，同量级 |
| 轻载 · ITL p99 | **14.15 ms** | 26.90 ms | 27.95 ms | vLLM 尾延迟明显更好 |
| 轻载 · TTFT p50 / p99 | 56.61 / 71.24 ms | **41.21 / 43.85 ms** | 42.21 / 43.75 ms | wein 快 15 ms（serve 层薄） |
| 轻载 · tps | 249.8 | **262.8** | 250.2 | 差 4.9% |
| 满负荷 c=100 n=500 mt=32 · ITL p50 | 25.41 ms | **24.15 ms** | 23.31 ms | 差 5.2%，同量级 |
| 满负荷 · ITL p99 | 52.40 ms | **39.03 ms** | 42.55 ms | vLLM p99 差 34%（54 token 桶无关，vLLM 走 512 桶） |
| 满负荷 · TTFT p50 / p99 | 171.49 / **213.50** ms | 72.06 / 239.54 ms | 73.52 / 263.32 ms | p50 wein 快 2.4×，p99 vLLM 好 |
| 满负荷 · tps | 3306.9 | **3537.1** | 3542.7 | vLLM 低 6.5% |
| 满负荷 · eager（无图档）TTFT p50 | — | 69.26 ms | 73.91 ms | 对照行，非本次测量轴 |

**环境没有漂移**：ITL p50 两档都在旧基线 ±5% 内。TTFT p50 偏高 50 ms 的**来源已定位**：同一脚本
`--max-model-len 512`（= 旧 `full-c100.json` 口径）复跑得 TTFT p50 **138.42 ms**、p99 218.5、ITL p50 25.81、
tps 3314（`maxlen512/short-full-c100.json`）——即 8192 与 512 的差（171.5→138.4）来自 KV/mamba 页几何变大，
剩下 ~17 ms 是单次跑 vs 3 次中位的口径差。旧基线 `full-c100.json` = 121.2 / 228.4 / 25.23 / 3430.6。

## 3. 表 2 · 长上下文冷跑（单请求，贪心，mt=32，每次清前缀缓存后测）

| 指标 | vLLM | wein 本轮（单请求贪心） | wein 上一轮 `ctx-a-20261031` | 备注 |
|---|---|---|---|---|
| 3000 token · TTFT（含 prefill） | **197.5 ms**（223.0/196.4/197.5） | 1483 ms | 未实测；按 ctx-a 线性外推 delta≈753 ms + attention | **vLLM 快 7.5×** |
| 3000 token · 首 32 token ITL p50 | **10.89 ms** | 本轮未落盘 | 250-token 段 delta 62.65/64.39 ms，8k 外推 2.84 s | wein 长上下文 ITL 缺档（见 §6） |
| 6000 token · TTFT（含 prefill） | **394.8 ms**（405.6/393.2/394.8） | 3756 ms | 同上外推 delta≈1.51 s | **vLLM 快 9.5×** |
| 6000 token · 首 32 token ITL p50 | **10.67 ms** | 本轮未落盘 | — | vLLM 长/短上下文 ITL 几乎不变 |
| 实测 prompt_tokens | 3009 / 6009 | 3000 / 6000（调用方口径） | — | tokenizer 精确计数 |
| TTFT 折算 prefill 速率 | **15.2k tok/s**（3009/0.198、6009/0.395） | **2.0k / 1.6k tok/s** | ctx-a 探针：每 token 每层 10.6 µs | 7.5–9.5× |
| 冷跑前缀命中 | 0/3009、0/6009 | — | — | `/metrics` 逐次核对 |

## 4. 表 3 · 长上下文前缀命中（同 prompt 连发两次，第二次记 TTFT）

| 指标 | vLLM（默认 float32 SSM 缓存） | wein 本轮 | wein 上一轮 | 备注 |
|---|---|---|---|---|
| 3000 token · 第二次 TTFT | **37.5 ms**（37.5/37.9/36.9） | 114.9 ms | — | **vLLM 快 3.1×**；37 ms ≈ 1 个 decode 步 |
| 3000 token · 命中 token | 2640 / 3009（87.8%） | 2816 | — | vLLM 按 528 对齐 ⇒ floor(3009/528)×528 |
| 6000 token · 第二次 TTFT | **37.3 ms**（37.3/37.6/37.0） | 348 ms | — | **vLLM 快 9.3×** |
| 6000 token · 命中 token | **5808 / 6009（96.7%）** | 2816 / 6000（46.9%） | — | 命中长度不同构，见 §6 |
| 命中后首 32 token ITL p50 | 10.72 / 10.75 ms | 未落盘 | — | 与冷跑 ITL 无差 |
| 命中是否真的发生 | `/metrics` `prefix_cache_hits_total` 逐次 +2640/+5808 | 调用方口径 | — | 不是推断，是指标直读 |

### 4b. 非默认变体（均标注"非默认"，同 3 次中位）

| 档 | 3000 冷 / 热 TTFT | 6000 冷 / 热 TTFT | 命中 token | 备注 |
|---|---|---|---|---|
| `--mamba-ssm-cache-dtype bfloat16` | 217.0 / **32.0** ms | 401.3 / **34.4** ms | 2992/3009、5984/6009 | **非默认**：SSM 状态页减半 ⇒ `block_size` 528→**272**，命中粒度更细（+1.5 个点），冷跑无收益（噪声内） |
| `--max-num-batched-tokens 256`（对齐 wein chunk 256） | — | **876.3** / 40.4 ms | 5808/6009 | **非默认**：6000 token 切成 24 步，vLLM 自己慢 2.2×（394.8→876.3） |

## 5. 结论性归因

1. **差距全在 prefill，不在 decode。** 满负荷 ITL/tps 两端同量级（25.41 vs 24.15 ms、3306.9 vs 3537.1 tok/s，
   vLLM 低 5–7%），而冷跑 3000/6000 token 的 TTFT 是 197.5/394.8 ms 对 1483/3756 ms——**7.5× 与 9.5×**，
   折算 prefill 速率 15.2k vs 2.0k tok/s。
2. **不是 chunk 划分造成的。** 把 vLLM 的 `--max-num-batched-tokens` 压到 256（与 wein `chunk_size=256` 同档），
   6000 token 冷跑 876 ms——vLLM 自己慢了 2.2×，但仍是 wein 3756 ms 的 **1/4.3**。所以差距主体是
   **每 token 的 prefill kernel 单价**（ctx-a 量到的每 token 每层 10.6 µs vs vLLM 全部 32 层加起来约 3 µs 级），
   剩余约 2× 才是分块策略。
3. **前缀命中的绝对差同样来自冷跑底数。** 命中后 vLLM 37 ms（≈1 个 decode 步 + 调度），wein 114.9/348 ms；
   但真正的问题在命中长度：vLLM 6000 token 命中 5808（96.7%），wein 只命中 2816（46.9%），
   即 wein 的热跑仍要 prefill 掉一半 prompt——这一项比"热跑快几倍"更值得看。
4. **长上下文 decode 稳定性：vLLM 平，wein 无读数。** vLLM 单请求首 32 token ITL p50 在 3000 / 6000 上下文下
   10.89 / 10.67 ms，与短上下文单请求同档——因为 32 层里只有 8 层是全注意力（`full_attention_interval=4`），
   其余是线性注意力，上下文长度几乎不进 decode 单价。wein 侧长上下文 ITL 本轮没有落盘，这一格是空的。
5. **vLLM 默认开着、wein 没有对应物的优化**：`VLLM_COMPILE`（torch.compile/inductor）+ CUDA graph
   `FULL_AND_PIECEWISE` 回放（51 桶 ≤512）、2048-token 的 prefill chunk、prefix caching 在 **528-token**
   mamba-align 检查点（这是它能把 6000 token 命中到 5808 的直接原因）。非默认的 bf16 SSM 缓存只改命中粒度、
   不改冷跑——即长上下文冷跑的 7.5–9.5× 与 SSM 缓存精度无关。

## 6. 若要公平比较还需要什么（测量口径，不是 wein 待办）

- **wein 侧长上下文 ITL 未落盘**：本轮只有 TTFT。要并排 §3 的 ITL 行，需要 wein 侧同样逐 token 时间戳。
- **wein 侧运行参数没随读数存档**：`--ctx-len` 取值、`max_tokens`、prompt 文本都不在 `bench-reports/` 里，
  3000/6000 只是口头口径；本次 vLLM 侧 prompt 文本与 token 数（3009/6009）已存进 `long-*.json`。
- **命中长度不同构**：wein 两次都是 2816（`NUM_PAGES×BLOCK` 几何），vLLM 2640 / 5808（528 对齐）。
  要比较"命中路径的效率"，得先按相同命中 token 数对齐请求。
- **chunk 档要同时报两档**：2048（vLLM 默认）与 256（wein 同档），单看任一档都会把 2.2× 的分块成本
  算到对方头上。
- **采样轴未对齐**：vLLM 侧沿用旧脚本的 `VLLM_USE_FLASHINFER_SAMPLER=0`（flashinfer JIT 与 cu130 头不兼容），
  wein 是自带 sampler；本报告不测采样开销，别把这一项算进 ITL 差。
- **未锁时钟**：没有 `nvidia-smi -lgc` / persistence，跨轮次 ±5% 属正常；同一档内 3 次中位的离散度
  冷跑 ≤7%（223 vs 196），跨批次比较请以 5% 为阈。

## 7. 文件

| 文件 | 内容 |
|---|---|
| `bench/vs_vllm_longctx.sh` | 起服 → 压测 → 停服驱动（`default` / `bf16ssm` / `chunk256` / `maxlen512` 四档） |
| `bench/longctx_probe.py` | 长上下文单请求探针（tokenizer 精确建 prompt、冷/热、逐 token 时间戳、metrics 前后对照） |
| `default/{short-light-c4,short-c100-n8,short-full-c100}.json` | 短上下文 3 次原始 run + 中位 |
| `default/long-{3000,6000}.json` | 长上下文冷/热各 3 次原始时序 + `metrics_delta` |
| `bf16ssm/long-{3000,6000}.json`、`chunk256/long-6000.json`、`maxlen512/short-*.json` | 变体档 |
| `<cfg>/{config.txt,cmd.txt,cache-config.txt,server-flags.txt,server.log,metrics-after-start.txt}` | 每档命令、时间戳、运行期配置、启动日志 |

原始时间戳：`default` 2026-10-31T15:20:02+08:00 起、末档 `maxlen512` 15:25:48（各档 `config.txt` 的 `ts:`）；
四档均在同一台机、同一 GPU、串行单进程，相邻两档之间确认 `pgrep vllm` 为空且显存回落到 1.5–1.8 GB。
